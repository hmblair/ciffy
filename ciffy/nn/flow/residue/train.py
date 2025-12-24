"""
Training utilities for PCA + Flow models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from .model import PCAFlow
from .data import compute_pca


def train_pca_flow(
    data: np.ndarray,
    latent_dim: int = 12,
    n_layers: int = 8,
    hidden_dim: int = 64,
    bound: float | None = 3.0,
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[PCAFlow, dict]:
    """
    Train a PCA + Flow model on data.

    This is the core training function used by ResidueFlowModel.from_structures().
    It handles PCA computation, model creation, and the training loop.

    Args:
        data: (n_instances, d) flat data array. Can be coordinates, extended
              representations, or any flat feature vectors.
        latent_dim: Number of PCA components / latent dimensions.
        n_layers: Number of flow layers.
        hidden_dim: Hidden dimension in coupling networks.
        bound: Tanh bound for decode (in std devs). Prevents extrapolation.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        device: Device to train on.
        verbose: Print progress.

    Returns:
        flow: Trained PCAFlow model.
        info: Dictionary with training info (pca_rmsd, var_explained, losses).
    """
    # Handle 3D input (n_instances, n_atoms, 3) by flattening
    if data.ndim == 3:
        data = data.reshape(len(data), -1)

    n_instances = len(data)
    batch_size = min(batch_size, n_instances)

    # Compute PCA
    V, mean, singular_values, var_explained = compute_pca(data, n_components=latent_dim)
    pca_var = var_explained[latent_dim - 1]

    # Compute PCA reconstruction RMSD
    pca_coords = (data - mean) @ V.T
    recon = pca_coords @ V + mean
    pca_rmsd = float(np.sqrt(((data - recon) ** 2).mean()))

    if verbose:
        print(f"PCA: {latent_dim} dims, {pca_var*100:.1f}% var, RMSD={pca_rmsd:.4f}")

    # Create model
    V_tensor = torch.from_numpy(V).float()
    mean_tensor = torch.from_numpy(mean).float()
    flow = PCAFlow(
        V_tensor, mean_tensor,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        bound=bound,
    ).to(device)

    # Prepare data
    X = torch.from_numpy(data).float().to(device)

    # Training
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    losses = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n_instances)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_instances, batch_size):
            batch = X[perm[i:i + batch_size]]

            optimizer.zero_grad()
            loss = -flow.log_prob(batch).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}")

    # Compute final reconstruction RMSD
    flow.eval()
    with torch.no_grad():
        X_recon = flow.decode(flow.encode(X))
        flow_rmsd = float(torch.sqrt(((X_recon - X) ** 2).mean()).item())

    if verbose:
        print(f"Final: RMSD={flow_rmsd:.4f} (PCA={pca_rmsd:.4f})")

    info = {
        "pca_rmsd": pca_rmsd,
        "flow_rmsd": flow_rmsd,
        "var_explained": pca_var,
        "losses": losses,
        "n_params": sum(p.numel() for p in flow.parameters()),
    }

    return flow, info
