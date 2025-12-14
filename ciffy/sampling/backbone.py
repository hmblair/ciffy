"""
Backbone dihedral sampling from empirical Ramachandran distributions.

Provides functions to sample realistic phi/psi/omega angles from
Gaussian Mixture Models fitted to PDB data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils.gmm import GaussianMixtureModel

if TYPE_CHECKING:
    from ..polymer import Polymer

# Path to pre-fitted GMM parameters
_DATA_DIR = Path(__file__).parent.parent / "data"
_RAMA_GMM_PATH = _DATA_DIR / "ramachandran_gmm.npz"

# Lazy-loaded GMM (loaded on first use)
_RAMA_GMM: GaussianMixtureModel | None = None


def _get_rama_gmm() -> GaussianMixtureModel:
    """Get the pre-fitted Ramachandran GMM, loading if necessary."""
    global _RAMA_GMM
    if _RAMA_GMM is None:
        if not _RAMA_GMM_PATH.exists():
            raise FileNotFoundError(
                f"Ramachandran GMM not found at {_RAMA_GMM_PATH}. "
                "Run scripts/fit_ramachandran_gmm.py to generate it."
            )
        _RAMA_GMM = GaussianMixtureModel.load(_RAMA_GMM_PATH)
    return _RAMA_GMM


def sample_protein_dihedrals(
    n_residues: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample phi, psi, omega angles for n residues.

    Uses a Gaussian Mixture Model fitted to empirical Ramachandran
    distributions from PDB structures.

    Args:
        n_residues: Number of residues to sample angles for.
        rng: Random number generator for reproducibility.

    Returns:
        Tuple of (phi, psi, omega) arrays, each of shape (n_residues,).
        - phi: Backbone phi angles in radians. First residue is NaN.
        - psi: Backbone psi angles in radians. Last residue is NaN.
        - omega: Backbone omega angles in radians (~pi for trans).
            Last residue is NaN.
    """
    if rng is None:
        rng = np.random.default_rng()

    gmm = _get_rama_gmm()

    # Sample phi/psi pairs from GMM
    samples = gmm.sample(n_residues, rng)  # (n_residues, 2)
    phi = samples[:, 0].copy()
    psi = samples[:, 1].copy()

    # First residue has no phi (N-terminus), last has no psi (C-terminus)
    phi[0] = np.nan
    psi[-1] = np.nan

    # Omega: predominantly trans (~180 degrees) with small variance
    # Trans peptide bonds are ~99% of all peptide bonds
    omega = rng.normal(np.pi, 0.05, n_residues)
    omega[-1] = np.nan  # Last residue has no omega

    return phi, psi, omega


def randomize_backbone(
    polymer: "Polymer",
    seed: int | None = None,
) -> "Polymer":
    """
    Randomize backbone dihedrals using empirical Ramachandran distributions.

    Samples phi/psi angles from a Gaussian Mixture Model fitted to PDB
    data and applies them to the polymer structure using the NERF
    reconstruction algorithm.

    Args:
        polymer: Polymer to randomize. Must be a protein.
        seed: Random seed for reproducibility.

    Returns:
        The polymer with randomized backbone dihedrals.

    Example:
        >>> import ciffy
        >>> from ciffy.sampling import randomize_backbone
        >>> polymer = ciffy.from_sequence("MGKLF")
        >>> polymer = randomize_backbone(polymer, seed=42)
    """
    from ..types import DihedralType, Scale

    rng = np.random.default_rng(seed)
    n_residues = polymer.size(Scale.RESIDUE)

    if n_residues == 0:
        return polymer

    # Sample dihedrals
    phi, psi, omega = sample_protein_dihedrals(n_residues, rng)

    # Apply dihedrals (filter out NaN values)
    phi_valid = phi[~np.isnan(phi)]
    psi_valid = psi[~np.isnan(psi)]
    omega_valid = omega[~np.isnan(omega)]

    polymer.set_dihedral(DihedralType.PHI, phi_valid)
    polymer.set_dihedral(DihedralType.PSI, psi_valid)
    polymer.set_dihedral(DihedralType.OMEGA, omega_valid)

    # Force coordinate reconstruction
    _ = polymer.coordinates

    return polymer
