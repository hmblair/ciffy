"""
Training utilities for PCA + Flow models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from .model import PCAFlow
from .data import compute_pca


def _compute_aligned_rmsd(coords1: np.ndarray, coords2: np.ndarray, n_atoms: int) -> float:
    """
    Compute aligned RMSD between coordinate arrays.

    Args:
        coords1: First coordinates, shape (N, n_atoms*3) or (N, n_atoms*3 + extras).
        coords2: Second coordinates, same shape as coords1.
        n_atoms: Number of atoms (used to extract just coordinate portion).

    Returns:
        Mean aligned RMSD across all instances.
    """
    from ciffy import rmsd

    # Extract just the coordinate portion (first n_atoms*3 dims)
    coord_dims = n_atoms * 3
    c1 = coords1[:, :coord_dims].reshape(-1, n_atoms, 3)
    c2 = coords2[:, :coord_dims].reshape(-1, n_atoms, 3)

    # Compute per-instance aligned RMSD
    rmsds = rmsd(c1, c2)
    return float(np.mean(rmsds))


def _gaussianity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Compute loss penalizing deviation from standard normal N(0, I).

    This encourages the latent space to be Gaussian, which improves
    sampling quality since we sample from N(0, I) at inference time.

    Args:
        z: Latent vectors (N, d).

    Returns:
        Scalar loss value.
    """
    # Penalize non-zero mean
    mean_loss = z.mean(dim=0).pow(2).mean()
    # Penalize non-unit variance
    var_loss = (z.var(dim=0) - 1).pow(2).mean()
    return mean_loss + var_loss


def train_pca_flow(
    train_data: np.ndarray,
    test_data: np.ndarray | None = None,
    n_atoms: int | None = None,
    latent_dim: int = 12,
    n_layers: int = 8,
    hidden_dim: int = 64,
    bound: float | None = None,
    coupling_type: str = "affine",
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    gaussianity_weight: float = 0.0,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[PCAFlow, dict]:
    """
    Train a PCA + Flow model on data with proper train/test evaluation.

    This is the core training function used by ResidueFlowModel.from_structures().
    It handles PCA computation, model creation, and the training loop.

    Args:
        train_data: (n_train, d) training data array.
        test_data: (n_test, d) test data array for evaluation. If None, uses train_data.
        n_atoms: Number of atoms (for aligned RMSD). If None, inferred from data dims.
        latent_dim: Number of PCA components / latent dimensions.
        n_layers: Number of flow layers.
        hidden_dim: Hidden dimension in coupling networks.
        bound: Tanh bound for decode (in std devs). Prevents extrapolation.
        coupling_type: Type of coupling layer ('affine' or 'spline').
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        gaussianity_weight: Weight for Gaussianity regularization loss. Encourages
            latent space to match N(0, I), improving sampling quality. Default 0.0.
        device: Device to train on.
        verbose: Print progress.

    Returns:
        flow: Trained PCAFlow model.
        info: Dictionary with training info including:
            - pca_rmsd: PCA reconstruction RMSD on train data
            - train_rmsd: Flow reconstruction RMSD on train data
            - test_rmsd: Flow reconstruction RMSD on test data (aligned)
            - var_explained: Variance explained by PCA
            - losses: Training loss history
            - n_params: Number of model parameters
    """
    # Handle 3D input (n_instances, n_atoms, 3) by flattening
    if train_data.ndim == 3:
        if n_atoms is None:
            n_atoms = train_data.shape[1]
        train_data = train_data.reshape(len(train_data), -1)

    if test_data is not None and test_data.ndim == 3:
        test_data = test_data.reshape(len(test_data), -1)

    # Infer n_atoms if not provided (assume pure coordinates, no extras)
    if n_atoms is None:
        # Assume data is n_atoms*3 (pure coordinates)
        n_atoms = train_data.shape[1] // 3

    n_train = len(train_data)
    batch_size = min(batch_size, n_train)

    # Compute PCA on training data only
    V, mean, singular_values, var_explained = compute_pca(train_data, n_components=latent_dim)
    pca_var = var_explained[latent_dim - 1]

    # Compute PCA reconstruction RMSD on training data
    pca_coords = (train_data - mean) @ V.T
    recon = pca_coords @ V + mean
    pca_rmsd = _compute_aligned_rmsd(train_data, recon, n_atoms)

    if verbose:
        print(f"PCA: {latent_dim} dims, {pca_var*100:.1f}% var, RMSD={pca_rmsd:.4f}Å")

    # Create model
    V_tensor = torch.from_numpy(V).float()
    mean_tensor = torch.from_numpy(mean).float()
    flow = PCAFlow(
        V_tensor, mean_tensor,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        bound=bound,
        coupling_type=coupling_type,
    ).to(device)

    # Prepare training data
    X_train = torch.from_numpy(train_data).float().to(device)

    # Training loop
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    losses = []

    for epoch in range(n_epochs):
        flow.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch = X_train[perm[i:i + batch_size]]

            optimizer.zero_grad()
            z, log_det = flow(batch)
            log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
            nll = -(log_pz + log_det).mean()

            # Add Gaussianity regularization if enabled
            if gaussianity_weight > 0:
                gauss_loss = _gaussianity_loss(z)
                loss = nll + gaussianity_weight * gauss_loss
            else:
                loss = nll

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}")

    # Import metrics functions
    from ..metrics import compute_nll, compute_latent_moments

    # Evaluate on training data
    flow.eval()
    with torch.no_grad():
        X_train_recon = flow.decode(flow.encode(X_train)).cpu().numpy()
    train_rmsd = _compute_aligned_rmsd(train_data, X_train_recon, n_atoms)

    # Compute training set metrics
    train_nll = compute_nll(flow, train_data)
    train_moments = compute_latent_moments(flow, train_data)

    # Evaluate on test data (held-out)
    if test_data is not None and len(test_data) > 0:
        X_test = torch.from_numpy(test_data).float().to(device)
        with torch.no_grad():
            X_test_recon = flow.decode(flow.encode(X_test)).cpu().numpy()
        test_rmsd = _compute_aligned_rmsd(test_data, X_test_recon, n_atoms)
        test_nll = compute_nll(flow, test_data)
        test_moments = compute_latent_moments(flow, test_data)
    else:
        test_rmsd = train_rmsd
        test_nll = train_nll
        test_moments = train_moments

    if verbose:
        if test_data is not None and len(test_data) > 0:
            print(f"Final: train_RMSD={train_rmsd:.4f}Å, test_RMSD={test_rmsd:.4f}Å (PCA={pca_rmsd:.4f}Å)")
            print(f"       train_NLL={train_nll:.4f}, test_NLL={test_nll:.4f}")
            print(f"       test_gaussianity={test_moments.gaussianity_score():.3f}")
        else:
            print(f"Final: RMSD={train_rmsd:.4f}Å (PCA={pca_rmsd:.4f}Å)")
            print(f"       NLL={train_nll:.4f}, gaussianity={train_moments.gaussianity_score():.3f}")

    info = {
        "pca_rmsd": pca_rmsd,
        "train_rmsd": train_rmsd,
        "test_rmsd": test_rmsd,
        "flow_rmsd": test_rmsd,  # Legacy alias (now uses test RMSD)
        "var_explained": pca_var,
        "losses": losses,
        "n_params": sum(p.numel() for p in flow.parameters()),
        # New metrics
        "train_nll": train_nll,
        "test_nll": test_nll,
        "train_gaussianity": train_moments.gaussianity_score(),
        "test_gaussianity": test_moments.gaussianity_score(),
        "train_moments": train_moments,
        "test_moments": test_moments,
    }

    return flow, info
