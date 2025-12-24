#!/usr/bin/env python
"""
Train ResidueFlow models for all four RNA residue types.

Usage:
    python scripts/train_residue_flows.py
"""

from pathlib import Path
import numpy as np

from ciffy.biochemistry import Residue
from ciffy.nn.residue_flow import ResidueFlowModel, ResidueFlowConfig
from ciffy.nn.residue_flow.data import extract_residues, align_to_frame, check_bond_lengths


def get_cif_files() -> list[Path]:
    """Get all CIF files from test data."""
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    return list(data_dir.glob("*.cif"))


def train_residue_type(
    residue: Residue,
    cif_paths: list[Path],
    config: ResidueFlowConfig,
    n_epochs: int = 200,
) -> dict:
    """Train a model for a single residue type and return results."""
    print(f"\n{'='*60}")
    print(f"Training {residue.name}")
    print(f"{'='*60}")

    # Extract and analyze data
    try:
        coords, atoms = extract_residues(cif_paths, residue, min_coverage=0.9, verbose=True)
    except ValueError as e:
        print(f"  Skipping {residue.name}: {e}")
        return {"error": str(e)}

    coords = align_to_frame(coords, atoms, residue)

    # Check bond lengths in raw data
    bond_stats = check_bond_lengths(coords, atoms, residue)
    print(f"\nData bond stats: {bond_stats['bond']} = {bond_stats['mean']:.3f} ± {bond_stats['std']:.3f} Å")

    # Train model
    from ciffy.nn.residue_flow.data import compute_pca
    from ciffy.nn.residue_flow.train import train_pca_flow
    import torch

    # Train
    flow, info = train_pca_flow(
        coords,
        latent_dim=config.latent_dim,
        n_layers=config.n_layers,
        hidden_dim=config.hidden_dim,
        n_epochs=n_epochs,
        device="cpu",
        verbose=True,
    )

    # Sample and check quality
    with torch.no_grad():
        samples = flow.sample(100)
    sample_bonds = check_bond_lengths(samples.numpy(), atoms, residue)

    print(f"\nSample bond stats: {sample_bonds['bond']} = {sample_bonds['mean']:.3f} ± {sample_bonds['std']:.3f} Å")

    results = {
        "n_instances": len(coords),
        "n_atoms": len(atoms),
        "var_explained": info["var_explained"],
        "pca_rmsd": info["pca_rmsd"],
        "flow_rmsd": info["flow_rmsd"],
        "n_params": info["n_params"],
        "data_bond_mean": bond_stats["mean"],
        "data_bond_std": bond_stats["std"],
        "sample_bond_mean": sample_bonds["mean"],
        "sample_bond_std": sample_bonds["std"],
    }

    return results


def main():
    cif_paths = get_cif_files()
    print(f"Found {len(cif_paths)} CIF files")

    config = ResidueFlowConfig(
        latent_dim=12,
        n_layers=8,
        hidden_dim=64,
        min_coverage=0.9,
    )

    # RNA residue types
    residues = [Residue.A, Residue.C, Residue.G, Residue.U]
    results = {}

    for residue in residues:
        results[residue.name] = train_residue_type(residue, cif_paths, config)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Residue':<10} {'N':<8} {'Atoms':<8} {'Var%':<8} {'RMSD':<10} {'Bond Mean':<12} {'Bond Std':<10}")
    print("-" * 80)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<10} ERROR: {res['error']}")
        else:
            print(
                f"{name:<10} "
                f"{res['n_instances']:<8} "
                f"{res['n_atoms']:<8} "
                f"{res['var_explained']*100:.1f}%{'':<4} "
                f"{res['pca_rmsd']:.4f}Å{'':<3} "
                f"{res['sample_bond_mean']:.3f}Å{'':<5} "
                f"{res['sample_bond_std']:.4f}Å"
            )


if __name__ == "__main__":
    main()
