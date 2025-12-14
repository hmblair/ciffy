#!/usr/bin/env python3
"""
Fit Gaussian Mixture Models to RNA backbone dihedral angles from PDB structures.

This script:
1. Loads RNA structures from a directory of CIF files
2. Extracts all 7 backbone dihedrals (alpha, beta, gamma, delta, epsilon, zeta, chi)
3. Fits a separate GMM to each dihedral's empirical distribution
4. Saves the fitted parameters to ciffy/data/rna_dihedrals.npz

Usage:
    python scripts/fit_rna_dihedrals.py [--pdb-dir PATH] [--n-components N] [--output PATH]

Example:
    python scripts/fit_rna_dihedrals.py --pdb-dir tests/data --n-components 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import ciffy
from ciffy import DihedralType, Molecule
from ciffy.utils.gmm import GaussianMixtureModel


# RNA backbone dihedrals in order
RNA_DIHEDRALS = [
    DihedralType.ALPHA,
    DihedralType.BETA,
    DihedralType.GAMMA,
    DihedralType.DELTA,
    DihedralType.EPSILON,
    DihedralType.ZETA,
    DihedralType.CHI_PURINE,
    DihedralType.CHI_PYRIMIDINE,
]


def extract_rna_dihedrals(pdb_dir: str | Path) -> dict[DihedralType, np.ndarray]:
    """
    Extract all backbone dihedrals from RNA structures in a directory.

    Args:
        pdb_dir: Directory containing .cif files.

    Returns:
        Dict mapping DihedralType -> (N,) array of valid angles in radians.
    """
    pdb_dir = Path(pdb_dir)
    cif_files = list(pdb_dir.glob("*.cif"))

    if not cif_files:
        raise ValueError(f"No .cif files found in {pdb_dir}")

    print(f"Found {len(cif_files)} CIF files")

    # Collect all angles per dihedral type
    all_angles: dict[DihedralType, list[np.ndarray]] = {d: [] for d in RNA_DIHEDRALS}

    for cif_path in cif_files:
        try:
            polymer = ciffy.load(str(cif_path))

            # Get RNA chains only
            rna = polymer.by_type(Molecule.RNA)
            if rna.size() == 0:
                continue

            print(f"  {cif_path.name}: {rna.size(ciffy.RESIDUE)} RNA residues")

            # Extract each dihedral type
            for dtype in RNA_DIHEDRALS:
                try:
                    angles = rna.dihedral(dtype)
                    # Filter out NaN values
                    valid = angles[~np.isnan(angles)]
                    if len(valid) > 0:
                        all_angles[dtype].append(valid)
                except Exception:
                    # Some dihedrals may not be computable for all structures
                    pass

        except Exception as e:
            print(f"  {cif_path.name}: Error - {e}")
            continue

    # Combine all angles per dihedral type
    result = {}
    for dtype in RNA_DIHEDRALS:
        if all_angles[dtype]:
            combined = np.concatenate(all_angles[dtype])
            result[dtype] = combined
            print(f"  {dtype.value}: {len(combined)} valid angles")
        else:
            print(f"  {dtype.value}: No valid angles found")

    return result


def fit_and_save_gmms(
    angles: dict[DihedralType, np.ndarray],
    output_path: str | Path,
    n_components: int = 3,
    seed: int = 42,
) -> None:
    """
    Fit separate 1D GMM to each dihedral type and save to file.

    Args:
        angles: Dict mapping DihedralType -> (N,) array of angles in radians.
        output_path: Output .npz file path.
        n_components: Number of GMM components per dihedral.
        seed: Random seed for reproducibility.
    """
    print(f"\nFitting GMMs with {n_components} components each...")

    rng = np.random.default_rng(seed)

    # Store GMM parameters for each dihedral
    gmm_params = {}

    for dtype, data in angles.items():
        if len(data) < n_components * 10:
            print(f"  {dtype.value}: Skipping (insufficient data: {len(data)} samples)")
            continue

        # Reshape to 2D for GMM (N, 1)
        data_2d = data.reshape(-1, 1)

        # Fit GMM
        gmm = GaussianMixtureModel.fit(data_2d, n_components=n_components, rng=rng)

        # Store parameters with dihedral name as key
        gmm_params[f"{dtype.value}_means"] = gmm.means
        gmm_params[f"{dtype.value}_covariances"] = gmm.covariances
        gmm_params[f"{dtype.value}_weights"] = gmm.weights

        # Print summary
        print(f"  {dtype.value}:")
        for i in range(gmm.n_components):
            mean_deg = np.degrees(gmm.means[i, 0])
            std_deg = np.degrees(np.sqrt(gmm.covariances[i, 0, 0]))
            weight_pct = gmm.weights[i] * 100
            print(f"    Component {i}: weight={weight_pct:.1f}%, "
                  f"mean={mean_deg:.1f}°, std={std_deg:.1f}°")

    # Save all parameters to single npz file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **gmm_params)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit GMMs to RNA backbone dihedrals from PDB structures"
    )
    parser.add_argument(
        "--pdb-dir",
        default="tests/data",
        help="Directory containing .cif files (default: tests/data)"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of GMM components per dihedral (default: 3)"
    )
    parser.add_argument(
        "--output",
        default="ciffy/data/rna_dihedrals.npz",
        help="Output .npz file (default: ciffy/data/rna_dihedrals.npz)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Extract dihedrals from structures
    angles = extract_rna_dihedrals(args.pdb_dir)

    if not angles:
        print("No RNA dihedral data found!")
        return

    # Fit and save GMMs
    fit_and_save_gmms(angles, args.output, args.n_components, args.seed)


if __name__ == "__main__":
    main()
