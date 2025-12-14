#!/usr/bin/env python3
"""
Fit a Gaussian Mixture Model to Ramachandran (phi/psi) data from PDB structures.

This script:
1. Loads protein structures from a directory of CIF files
2. Extracts phi and psi dihedral angles using ciffy
3. Fits a GMM to the empirical distribution
4. Saves the fitted parameters to ciffy/data/ramachandran_gmm.npz

Usage:
    python scripts/fit_ramachandran_gmm.py [--pdb-dir PATH] [--n-components N] [--output PATH]

Example:
    python scripts/fit_ramachandran_gmm.py --pdb-dir tests/data --n-components 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import ciffy
from ciffy import DihedralType, PROTEIN
from ciffy.utils.gmm import GaussianMixtureModel


def extract_phi_psi(pdb_dir: str | Path) -> np.ndarray:
    """
    Extract phi/psi angles from all CIF files in a directory.

    Args:
        pdb_dir: Directory containing .cif files.

    Returns:
        (N, 2) array of (phi, psi) pairs in radians, with NaN values removed.
    """
    pdb_dir = Path(pdb_dir)
    cif_files = list(pdb_dir.glob("*.cif"))

    if not cif_files:
        raise ValueError(f"No .cif files found in {pdb_dir}")

    print(f"Found {len(cif_files)} CIF files")

    all_phi_psi = []

    for cif_path in cif_files:
        try:
            polymer = ciffy.load(str(cif_path))

            # Get protein chains only
            protein = polymer.by_type(PROTEIN)
            if protein.size() == 0:
                continue

            # Extract phi and psi
            phi = protein.dihedral(DihedralType.PHI)
            psi = protein.dihedral(DihedralType.PSI)

            # Combine into (N, 2) array
            phi_psi = np.column_stack([phi, psi])

            # Remove rows with any NaN
            valid_mask = ~np.isnan(phi_psi).any(axis=1)
            phi_psi = phi_psi[valid_mask]

            if len(phi_psi) > 0:
                all_phi_psi.append(phi_psi)
                print(f"  {cif_path.name}: {len(phi_psi)} valid phi/psi pairs")

        except Exception as e:
            print(f"  {cif_path.name}: Error - {e}")
            continue

    if not all_phi_psi:
        raise ValueError("No valid phi/psi data extracted from any structure")

    combined = np.vstack(all_phi_psi)
    print(f"\nTotal: {len(combined)} phi/psi pairs from {len(all_phi_psi)} structures")

    return combined


def fit_and_save_gmm(
    phi_psi: np.ndarray,
    output_path: str | Path,
    n_components: int = 5,
    seed: int = 42,
) -> GaussianMixtureModel:
    """
    Fit GMM to phi/psi data and save to file.

    Args:
        phi_psi: (N, 2) array of phi/psi pairs in radians.
        output_path: Output .npz file path.
        n_components: Number of GMM components.
        seed: Random seed for reproducibility.

    Returns:
        Fitted GMM.
    """
    print(f"\nFitting GMM with {n_components} components...")

    rng = np.random.default_rng(seed)
    gmm = GaussianMixtureModel.fit(phi_psi, n_components=n_components, rng=rng)

    print(f"Fitted GMM:")
    for i in range(gmm.n_components):
        phi_mean_deg = np.degrees(gmm.means[i, 0])
        psi_mean_deg = np.degrees(gmm.means[i, 1])
        weight_pct = gmm.weights[i] * 100
        print(f"  Component {i}: weight={weight_pct:.1f}%, "
              f"phi={phi_mean_deg:.1f}°, psi={psi_mean_deg:.1f}°")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmm.save(output_path)
    print(f"\nSaved to: {output_path}")

    return gmm


def main():
    parser = argparse.ArgumentParser(
        description="Fit GMM to Ramachandran data from PDB structures"
    )
    parser.add_argument(
        "--pdb-dir",
        default="tests/data",
        help="Directory containing .cif files (default: tests/data)"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of GMM components (default: 5)"
    )
    parser.add_argument(
        "--output",
        default="ciffy/data/ramachandran_gmm.npz",
        help="Output .npz file (default: ciffy/data/ramachandran_gmm.npz)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Extract phi/psi from structures
    phi_psi = extract_phi_psi(args.pdb_dir)

    # Fit and save GMM
    fit_and_save_gmm(phi_psi, args.output, args.n_components, args.seed)


if __name__ == "__main__":
    main()
