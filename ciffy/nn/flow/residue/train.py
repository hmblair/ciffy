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
    coords: np.ndarray,
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
    Train a PCA + Flow model on coordinate data.

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array.
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
    n_instances = len(coords)

    # Compute PCA
    V, mean, singular_values, var_explained = compute_pca(coords, n_components=latent_dim)
    pca_var = var_explained[latent_dim - 1]

    # Compute PCA reconstruction RMSD
    coords_flat = coords.reshape(n_instances, -1)
    pca_coords = (coords_flat - mean) @ V.T
    recon_flat = pca_coords @ V + mean
    pca_rmsd = float(np.sqrt(((coords_flat - recon_flat) ** 2).mean()))

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
    ).to(device)

    # Prepare data
    X = torch.from_numpy(coords).float().to(device)

    # Training
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(n_instances)

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_instances, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = X[batch_idx]

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

    # Compute final reconstruction RMSD (should match PCA RMSD)
    flow.eval()
    with torch.no_grad():
        X_flat = X.reshape(n_instances, -1)
        X_recon = flow.decode(flow.encode(X))  # Returns (N, d) flat
        flow_rmsd = float(torch.sqrt(((X_recon - X_flat) ** 2).mean()).item())

    if verbose:
        print(f"Final: RMSD={flow_rmsd:.4f}Å (PCA={pca_rmsd:.4f}Å)")

    info = {
        "pca_rmsd": pca_rmsd,
        "flow_rmsd": flow_rmsd,
        "var_explained": pca_var,
        "losses": losses,
        "n_params": sum(p.numel() for p in flow.parameters()),
    }

    return flow, info
