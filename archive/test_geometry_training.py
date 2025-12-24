#!/usr/bin/env python3
"""
Test geometry-penalized training for PCAFlow.

This script compares:
1. Standard MLE training (baseline)
2. MLE + geometry penalty training
3. MLE + geometry + contrastive training

The goal is to show that geometry penalty reduces bond length errors.
"""

import numpy as np
import torch
from pathlib import Path

# Add ciffy to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciffy.biochemistry import Residue
from ciffy.nn.flow.residue.data import extract_residues_with_links, compute_pca
from ciffy.nn.flow.residue.train import train_pca_flow_with_geometry
from ciffy.nn.flow.residue.geometry import GeometryLoss, compute_reference_geometry
from ciffy.nn.flow.residue.model import PCAFlow


def test_geometry_training():
    """Test geometry-penalized training on real data."""
    print("=" * 70)
    print("Testing Geometry-Penalized Training")
    print("=" * 70)

    # Load CIF files
    cif_dir = Path(__file__).parent.parent / "tests" / "data"
    cif_paths = list(cif_dir.glob("*.cif"))[:50]

    if len(cif_paths) < 5:
        print("Not enough CIF files found, need at least 5")
        return

    print(f"Using {len(cif_paths)} CIF files")

    # Extract data
    residue = Residue.A
    coords, transforms, atoms = extract_residues_with_links(
        cif_paths, residue, min_coverage=0.9, verbose=True
    )

    print(f"\nExtracted {len(coords)} residue pairs")
    print(f"Coords shape: {coords.shape}, Transforms shape: {transforms.shape}")

    # Print reference geometry
    print("\nReference bond lengths from training data:")
    ref_geom = compute_reference_geometry(coords, atoms, residue)
    for name, (mean_len, std_len) in sorted(ref_geom.items()):
        print(f"  {name}: {mean_len:.4f} ± {std_len:.4f} Å")

    # =======================================================================
    # Train with geometry penalty
    # =======================================================================
    print("\n" + "=" * 70)
    print("Training with geometry penalty (λ_geom=1.0)")
    print("=" * 70)

    flow_geom, info_geom = train_pca_flow_with_geometry(
        coords=coords,
        transforms=transforms,
        atoms=atoms,
        residue=residue,
        latent_dim=8,
        n_layers=8,
        hidden_dim=64,
        bound=3.0,
        n_epochs=200,
        batch_size=256,
        lr=1e-3,
        geometry_weight=1.0,
        contrastive_weight=0.0,
        device="cpu",
        verbose=True,
    )

    # =======================================================================
    # Evaluate geometry at different z values
    # =======================================================================
    print("\n" + "=" * 70)
    print("Evaluating bond geometry at different z values")
    print("=" * 70)

    n_atoms = len(atoms)
    geom_loss_fn = GeometryLoss(atoms, residue)

    # Get P-OP1 and P-OP2 bond indices for detailed analysis
    name_to_col = {}
    for member in residue:
        if member.value in {a: i for i, a in enumerate(atoms)}:
            name_to_col[member.name] = {a: i for i, a in enumerate(atoms)}[member.value]

    p_idx = name_to_col.get("P")
    op1_idx = name_to_col.get("OP1")
    op2_idx = name_to_col.get("OP2")

    if p_idx is not None and op1_idx is not None and op2_idx is not None:
        print("\nP-OP1/OP2 bond lengths at different z values:")
        print(f"Reference: P-OP1 = 1.485 ± 0.010 Å, P-OP2 = 1.485 ± 0.010 Å")

        for z_scale in [0, 1, 2, 3]:
            z = torch.zeros(1, 8)
            if z_scale > 0:
                z[0, 0] = z_scale  # Vary first latent dim

            with torch.no_grad():
                decoded = flow_geom.decode(z)
                coords_decoded = decoded[:, :n_atoms * 3].reshape(-1, n_atoms, 3)

                p_op1_dist = torch.norm(coords_decoded[0, p_idx] - coords_decoded[0, op1_idx]).item()
                p_op2_dist = torch.norm(coords_decoded[0, p_idx] - coords_decoded[0, op2_idx]).item()

                p_op1_z = abs(p_op1_dist - 1.485) / 0.010
                p_op2_z = abs(p_op2_dist - 1.485) / 0.010

                geom_loss = geom_loss_fn(coords_decoded).item()

            print(f"  z[0]={z_scale}: P-OP1={p_op1_dist:.4f}Å ({p_op1_z:.1f}σ), "
                  f"P-OP2={p_op2_dist:.4f}Å ({p_op2_z:.1f}σ), geom_loss={geom_loss:.4f}")

    # =======================================================================
    # Sample geometry distribution
    # =======================================================================
    print("\n" + "=" * 70)
    print("Sampling geometry distribution (100 samples from N(0,1))")
    print("=" * 70)

    with torch.no_grad():
        z_samples = torch.randn(100, 8)
        decoded_samples = flow_geom.decode(z_samples)
        coords_samples = decoded_samples[:, :n_atoms * 3].reshape(100, n_atoms, 3)

        if p_idx is not None and op1_idx is not None:
            p_op1_dists = torch.norm(
                coords_samples[:, p_idx] - coords_samples[:, op1_idx], dim=-1
            ).numpy()
            print(f"P-OP1 from samples: {p_op1_dists.mean():.4f} ± {p_op1_dists.std():.4f} Å")
            print(f"  Reference: 1.485 ± 0.010 Å")
            print(f"  Z-score: {abs(p_op1_dists.mean() - 1.485) / 0.010:.1f}σ")

        if p_idx is not None and op2_idx is not None:
            p_op2_dists = torch.norm(
                coords_samples[:, p_idx] - coords_samples[:, op2_idx], dim=-1
            ).numpy()
            print(f"P-OP2 from samples: {p_op2_dists.mean():.4f} ± {p_op2_dists.std():.4f} Å")
            print(f"  Reference: 1.485 ± 0.010 Å")
            print(f"  Z-score: {abs(p_op2_dists.mean() - 1.485) / 0.010:.1f}σ")

    return flow_geom, info_geom


def compare_mle_vs_geometry():
    """Compare standard MLE training vs geometry-penalized training."""
    print("\n" + "=" * 70)
    print("Comparing MLE-only vs Geometry-Penalized Training")
    print("=" * 70)

    # Load CIF files
    cif_dir = Path(__file__).parent.parent / "tests" / "data"
    cif_paths = list(cif_dir.glob("*.cif"))[:50]

    if len(cif_paths) < 5:
        print("Not enough CIF files")
        return

    residue = Residue.A
    coords, transforms, atoms = extract_residues_with_links(
        cif_paths, residue, min_coverage=0.9, verbose=False
    )
    print(f"Dataset: {len(coords)} residue pairs")

    n_atoms = len(atoms)
    n_instances = len(coords)
    latent_dim = 8

    # Create extended representation
    coords_flat = coords.reshape(n_instances, -1)
    extended = np.concatenate([coords_flat, transforms], axis=1)

    # Compute PCA
    V, mean, _, var_explained = compute_pca(extended, n_components=latent_dim)
    V_tensor = torch.from_numpy(V).float()
    mean_tensor = torch.from_numpy(mean).float()

    # Create geometry loss function
    geom_loss_fn = GeometryLoss(atoms, residue)

    # =======================================================================
    # MLE-only training
    # =======================================================================
    print("\n--- MLE-only training ---")
    flow_mle = PCAFlow(V_tensor.clone(), mean_tensor.clone(), n_layers=8, hidden_dim=64, bound=3.0)
    X = torch.from_numpy(extended).float()

    optimizer = torch.optim.Adam(flow_mle.parameters(), lr=1e-3)
    for epoch in range(100):
        optimizer.zero_grad()
        loss = -flow_mle.log_prob(X).mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}: MLE loss = {loss.item():.4f}")

    # Evaluate MLE model
    flow_mle.eval()
    with torch.no_grad():
        z_zero = torch.zeros(1, latent_dim)
        decoded = flow_mle.decode(z_zero)
        coords_decoded = decoded[:, :n_atoms * 3].reshape(-1, n_atoms, 3)
        mle_geom_at_zero = geom_loss_fn(coords_decoded).item()

        z_samples = torch.randn(100, latent_dim)
        decoded_samples = flow_mle.decode(z_samples)
        coords_samples = decoded_samples[:, :n_atoms * 3].reshape(100, n_atoms, 3)
        mle_geom_samples = geom_loss_fn(coords_samples).item()

    print(f"  Geometry loss at z=0: {mle_geom_at_zero:.4f}")
    print(f"  Geometry loss (100 samples): {mle_geom_samples:.4f}")

    # =======================================================================
    # Geometry-penalized training
    # =======================================================================
    print("\n--- Geometry-penalized training (λ=1.0) ---")
    flow_geom, info = train_pca_flow_with_geometry(
        coords=coords,
        transforms=transforms,
        atoms=atoms,
        residue=residue,
        latent_dim=8,
        n_layers=8,
        hidden_dim=64,
        bound=3.0,
        n_epochs=100,
        geometry_weight=1.0,
        verbose=False,
    )

    geom_at_zero = info["geom_at_zero"]
    geom_samples = info["geom_samples"]

    print(f"  Geometry loss at z=0: {geom_at_zero:.4f}")
    print(f"  Geometry loss (100 samples): {geom_samples:.4f}")

    # =======================================================================
    # Summary
    # =======================================================================
    print("\n" + "=" * 70)
    print("Summary: Geometry Loss Comparison")
    print("=" * 70)
    print(f"                        MLE-only    Geometry-penalized")
    print(f"  At z=0:               {mle_geom_at_zero:8.4f}    {geom_at_zero:8.4f}")
    print(f"  100 samples:          {mle_geom_samples:8.4f}    {geom_samples:8.4f}")

    improvement_zero = (mle_geom_at_zero - geom_at_zero) / mle_geom_at_zero * 100
    improvement_samples = (mle_geom_samples - geom_samples) / mle_geom_samples * 100
    print(f"  Improvement at z=0:   {improvement_zero:+.1f}%")
    print(f"  Improvement samples:  {improvement_samples:+.1f}%")


if __name__ == "__main__":
    # Test geometry training
    test_geometry_training()

    # Compare MLE vs geometry
    compare_mle_vs_geometry()
