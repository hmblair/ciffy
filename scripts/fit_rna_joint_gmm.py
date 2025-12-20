#!/usr/bin/env python3
"""
Fit a Gaussian Mixture Model to RNA backbone dihedrals (7D joint distribution).

This script:
1. Loads RNA structures from a directory of CIF files
2. Extracts 7D dihedral angle tuples: (alpha, beta, gamma, delta, epsilon, zeta, chi)
3. Filters by nucleotide type (purine vs pyrimidine)
4. Fits a GMM to the empirical distribution
5. Saves the fitted parameters to ciffy/data/gmm/rna/{purine,pyrimidine}.npz

The joint GMM captures correlations between dihedral angles within a residue,
which independent 1D GMMs cannot capture.

Usage:
    python scripts/fit_rna_joint_gmm.py [--pdb-dir PATH] [--n-components N] [--output-dir PATH]

Example:
    python scripts/fit_rna_joint_gmm.py --pdb-dir tests/data --n-components 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import ciffy
from ciffy import DihedralType
from ciffy.biochemistry import Molecule, Residue, RibonucleicAcid as RNA
from ciffy.utils.gmm import GaussianMixtureModel


def extract_rna_joint_dihedrals(
    pdb_dir: str | Path,
    nucleotide_filter: str | None = None,
) -> np.ndarray:
    """
    Extract 7D joint dihedral tuples from RNA structures.

    Extracts all 7 backbone dihedrals (alpha, beta, gamma, delta, epsilon, zeta)
    plus the glycosidic chi angle from each RNA residue. Filters out incomplete
    residues (e.g., terminal residues with NaN dihedrals).

    Args:
        pdb_dir: Directory containing .cif files.
        nucleotide_filter: 'purine', 'pyrimidine', or None (all). Filter by
                          nucleotide type before extraction.

    Returns:
        (N, 7) array of joint dihedral tuples in radians with NaN values removed.
        Columns: [alpha, beta, gamma, delta, epsilon, zeta, chi]
    """
    pdb_dir = Path(pdb_dir)
    cif_files = sorted(pdb_dir.glob("*.cif"))

    if not cif_files:
        raise ValueError(f"No .cif files found in {pdb_dir}")

    print(f"Found {len(cif_files)} CIF files")

    if nucleotide_filter:
        print(f"Filtering to {nucleotide_filter} nucleotides")

    all_dihedrals = []
    structures_processed = 0

    for cif_path in cif_files:
        try:
            polymer = ciffy.load(str(cif_path))

            # Get RNA chains only
            rna = polymer.by_type(RNA)
            if rna.size() == 0:
                continue

            # Extract all 7 backbone dihedrals for all residues
            alpha = rna.dihedral(DihedralType.ALPHA)
            beta = rna.dihedral(DihedralType.BETA)
            gamma = rna.dihedral(DihedralType.GAMMA)
            delta = rna.dihedral(DihedralType.DELTA)
            epsilon = rna.dihedral(DihedralType.EPSILON)
            zeta = rna.dihedral(DihedralType.ZETA)

            # Chi angle: try pyrimidine first, fall back to purine
            try:
                chi = rna.dihedral(DihedralType.CHI_PYRIMIDINE)
            except (KeyError, AttributeError, ValueError):
                try:
                    chi = rna.dihedral(DihedralType.CHI_PURINE)
                except (KeyError, AttributeError, ValueError):
                    # If neither works, skip this structure
                    continue

            # Stack into (N, 7) array
            dihedrals = np.column_stack([alpha, beta, gamma, delta, epsilon, zeta, chi])

            # Filter by nucleotide type if specified
            if nucleotide_filter is not None:
                purine_residues = [Residue.A, Residue.G, Residue.DA, Residue.DG]
                pyrimidine_residues = [Residue.C, Residue.U, Residue.T, Residue.DC, Residue.DU, Residue.DT]

                if nucleotide_filter == "purine":
                    # Keep only rows for purine nucleotides
                    mask = np.array([rna.sequence[i] in purine_residues for i in range(len(rna.sequence))])
                    dihedrals = dihedrals[mask]
                elif nucleotide_filter == "pyrimidine":
                    # Keep only rows for pyrimidine nucleotides
                    mask = np.array([rna.sequence[i] in pyrimidine_residues for i in range(len(rna.sequence))])
                    dihedrals = dihedrals[mask]

            # Remove rows with any NaN (incomplete terminal residues)
            valid_mask = ~np.isnan(dihedrals).any(axis=1)
            dihedrals = dihedrals[valid_mask]

            if len(dihedrals) > 0:
                all_dihedrals.append(dihedrals)
                print(f"  {cif_path.name}: {len(dihedrals)} valid 7D tuples")
                structures_processed += 1

        except Exception as e:
            print(f"  {cif_path.name}: Warning - {e}")
            continue

    if not all_dihedrals:
        raise ValueError("No valid RNA dihedrals extracted from any structure")

    combined = np.vstack(all_dihedrals)
    print(
        f"\nTotal: {len(combined)} 7D dihedral tuples "
        f"from {structures_processed} structures"
    )

    return combined


def fit_and_save_gmm(
    dihedrals: np.ndarray,
    output_path: str | Path,
    n_components: int = 10,
    seed: int = 42,
) -> GaussianMixtureModel:
    """
    Fit 7D GMM to RNA dihedral data and save to file.

    Args:
        dihedrals: (N, 7) array of 7D dihedral tuples in radians.
        output_path: Output .npz file path.
        n_components: Number of GMM components.
        seed: Random seed for reproducibility.

    Returns:
        Fitted GaussianMixtureModel.
    """
    print(f"\nFitting 7D GMM with {n_components} components...")

    rng = np.random.default_rng(seed)
    gmm = GaussianMixtureModel.fit(dihedrals, n_components=n_components, rng=rng)

    print(f"\nFitted GMM:")
    print(f"  Number of components: {gmm.n_components}")
    print(f"  Dimensionality: {gmm.n_features}")
    print(f"  Total parameters: {gmm.n_components * (gmm.n_features + gmm.n_features * gmm.n_features // 2 + 1)}")
    print(f"\n  Component weights:")
    for i in range(gmm.n_components):
        weight_pct = gmm.weights[i] * 100
        print(f"    Component {i}: {weight_pct:.1f}%")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmm.save(output_path)
    print(f"\nSaved to: {output_path}")

    return gmm


def main():
    parser = argparse.ArgumentParser(
        description="Fit joint GMM to RNA backbone dihedrals (7D distribution)"
    )
    parser.add_argument(
        "--pdb-dir",
        default="tests/data",
        help="Directory containing .cif files (default: tests/data)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of GMM components (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="ciffy/data/gmm/rna",
        help="Output directory for .npz files (default: ciffy/data/gmm/rna)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Train purine GMM
    print("=" * 70)
    print("TRAINING PURINE GMM (A, G)")
    print("=" * 70)
    purine_data = extract_rna_joint_dihedrals(args.pdb_dir, nucleotide_filter="purine")
    if len(purine_data) > 0:
        fit_and_save_gmm(
            purine_data,
            output_dir / "purine.npz",
            args.n_components,
            args.seed,
        )
    else:
        print("No purine data found, skipping purine GMM training")

    # Train pyrimidine GMM
    print("\n" + "=" * 70)
    print("TRAINING PYRIMIDINE GMM (C, U)")
    print("=" * 70)
    pyrimidine_data = extract_rna_joint_dihedrals(
        args.pdb_dir, nucleotide_filter="pyrimidine"
    )
    if len(pyrimidine_data) > 0:
        fit_and_save_gmm(
            pyrimidine_data,
            output_dir / "pyrimidine.npz",
            args.n_components,
            args.seed,
        )
    else:
        print("No pyrimidine data found, skipping pyrimidine GMM training")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
