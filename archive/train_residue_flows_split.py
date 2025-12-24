#!/usr/bin/env python
"""
Train ResidueFlow models with structure-level train/test split.
"""

from pathlib import Path
import numpy as np
import torch

from ciffy.biochemistry import Residue
from ciffy.nn.residue_flow.data import extract_residues, align_to_frame, check_bond_lengths, compute_pca
from ciffy.nn.residue_flow.train import train_pca_flow


def train_and_evaluate(
    residue: Residue,
    train_paths: list[Path],
    test_paths: list[Path],
    latent_dim: int = 12,
    n_epochs: int = 200,
) -> dict:
    """Train on train set, evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"{residue.name}: {len(train_paths)} train / {len(test_paths)} test structures")
    print(f"{'='*60}")

    # Extract training data
    try:
        train_coords, atoms = extract_residues(train_paths, residue, min_coverage=0.9, verbose=False)
        train_coords = align_to_frame(train_coords, atoms, residue)
        print(f"Train: {len(train_coords)} instances, {len(atoms)} atoms")
    except ValueError as e:
        print(f"  Skipping: {e}")
        return {"error": str(e)}

    # Extract test data (using same atom set)
    try:
        test_coords, test_atoms = extract_residues(test_paths, residue, min_coverage=0.9, verbose=False)
        test_coords = align_to_frame(test_coords, test_atoms, residue)

        # Filter to common atoms
        common_atoms = set(atoms) & set(test_atoms)
        if len(common_atoms) < len(atoms):
            print(f"  Warning: test has different atoms, using {len(common_atoms)} common")
            # Reextract with stricter coverage to get matching atoms
            # For simplicity, skip if atoms don't match
            if set(atoms) != set(test_atoms):
                # Map test coords to train atom order
                test_atom_to_col = {a: i for i, a in enumerate(test_atoms)}
                train_atom_to_col = {a: i for i, a in enumerate(atoms)}

                # Only keep atoms present in both
                common = sorted(common_atoms)
                new_test = np.zeros((len(test_coords), len(common), 3), dtype=np.float32)
                new_train = np.zeros((len(train_coords), len(common), 3), dtype=np.float32)

                for i, a in enumerate(common):
                    new_test[:, i] = test_coords[:, test_atom_to_col[a]]
                    new_train[:, i] = train_coords[:, train_atom_to_col[a]]

                test_coords = new_test
                train_coords = new_train
                atoms = common

        print(f"Test: {len(test_coords)} instances")
    except ValueError as e:
        print(f"  No test data: {e}")
        test_coords = None

    # Train
    flow, info = train_pca_flow(
        train_coords,
        latent_dim=latent_dim,
        n_layers=8,
        hidden_dim=64,
        n_epochs=n_epochs,
        verbose=True,
    )

    results = {
        "n_train": len(train_coords),
        "n_atoms": len(atoms),
        "var_explained": info["var_explained"],
        "train_rmsd": info["pca_rmsd"],
    }

    # Evaluate on test set
    if test_coords is not None and len(test_coords) > 0:
        flow.eval()
        with torch.no_grad():
            X_test = torch.from_numpy(test_coords).float()

            # Project to PCA space and back (test PCA generalization)
            test_flat = test_coords.reshape(len(test_coords), -1)
            mean = flow.mean.numpy()
            V = flow.V.numpy()

            pca_test = (test_flat - mean) @ V.T
            recon_flat = pca_test @ V + mean
            pca_test_rmsd = float(np.sqrt(((test_flat - recon_flat) ** 2).mean()))

            # Full flow reconstruction
            X_test_recon = flow.decode(flow.encode(X_test))
            flow_test_rmsd = float(torch.sqrt(((X_test_recon - X_test) ** 2).mean()).item())

        results["n_test"] = len(test_coords)
        results["test_rmsd"] = flow_test_rmsd
        results["pca_test_rmsd"] = pca_test_rmsd

        print(f"Test RMSD: {flow_test_rmsd:.4f}Å (PCA: {pca_test_rmsd:.4f}Å)")

    # Sample quality
    with torch.no_grad():
        samples = flow.sample(100)
    sample_bonds = check_bond_lengths(samples.numpy(), atoms, residue)
    results["sample_bond_mean"] = sample_bonds["mean"]
    results["sample_bond_std"] = sample_bonds["std"]

    return results


def main():
    # Load structures
    data_dir = Path("/Users/hmblair/academic/data/rna-libraries/pdb/structures/pdb130")
    all_paths = sorted(data_dir.glob("*.cif"))[:100]  # First 100 for quick iteration

    # 80/20 structure-level split
    np.random.seed(42)
    indices = np.random.permutation(len(all_paths))
    split_idx = int(0.8 * len(all_paths))

    train_paths = [all_paths[i] for i in indices[:split_idx]]
    test_paths = [all_paths[i] for i in indices[split_idx:]]

    print(f"Structures: {len(train_paths)} train / {len(test_paths)} test")

    # Train each residue type
    residues = [Residue.A, Residue.C, Residue.G, Residue.U]
    results = {}

    for residue in residues:
        results[residue.name] = train_and_evaluate(
            residue, train_paths, test_paths,
            latent_dim=12, n_epochs=200,
        )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (Structure-level split)")
    print(f"{'='*80}")
    print(f"{'Res':<5} {'N_train':<8} {'N_test':<8} {'Var%':<7} {'Train RMSD':<12} {'Test RMSD':<12}")
    print("-" * 80)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<5} ERROR: {res['error']}")
        else:
            test_rmsd = f"{res.get('test_rmsd', 0):.4f}Å" if 'test_rmsd' in res else "N/A"
            print(
                f"{name:<5} "
                f"{res['n_train']:<8} "
                f"{res.get('n_test', 0):<8} "
                f"{res['var_explained']*100:.1f}%{'':<2} "
                f"{res['train_rmsd']:.4f}Å{'':<5} "
                f"{test_rmsd}"
            )


if __name__ == "__main__":
    main()
