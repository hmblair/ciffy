"""
Backbone dihedral sampling from empirical distributions.

Provides functions to sample realistic backbone dihedral angles from
Gaussian Mixture Models fitted to PDB data. Supports both proteins
and nucleic acids (RNA/DNA).
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
_RNA_GMM_PATH = _DATA_DIR / "rna_dihedrals.npz"

# Lazy-loaded GMMs (loaded on first use)
_RAMA_GMM: GaussianMixtureModel | None = None
_RNA_GMMS: dict[str, GaussianMixtureModel] | None = None


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


def _get_rna_gmms() -> dict[str, GaussianMixtureModel]:
    """Get the pre-fitted RNA dihedral GMMs, loading if necessary."""
    global _RNA_GMMS
    if _RNA_GMMS is None:
        if not _RNA_GMM_PATH.exists():
            raise FileNotFoundError(
                f"RNA dihedral GMMs not found at {_RNA_GMM_PATH}. "
                "Run scripts/fit_rna_dihedrals.py to generate it."
            )
        data = np.load(_RNA_GMM_PATH)

        # Reconstruct GMMs from stored parameters
        _RNA_GMMS = {}
        dihedral_names = set()
        for key in data.files:
            # Keys are like "alpha_means", "alpha_covariances", "alpha_weights"
            name = key.rsplit("_", 1)[0]
            dihedral_names.add(name)

        for name in dihedral_names:
            if f"{name}_means" in data:
                _RNA_GMMS[name] = GaussianMixtureModel(
                    means=data[f"{name}_means"],
                    covariances=data[f"{name}_covariances"],
                    weights=data[f"{name}_weights"],
                )

    return _RNA_GMMS


# =============================================================================
# Protein Sampling
# =============================================================================

def sample_protein_dihedrals(
    n_residues: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample phi, psi, omega angles for n protein residues.

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


# =============================================================================
# RNA Sampling
# =============================================================================

def sample_rna_dihedrals(
    n_residues: int,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Sample backbone dihedrals for n RNA residues.

    Uses Gaussian Mixture Models fitted to empirical distributions
    from PDB RNA structures.

    Args:
        n_residues: Number of residues to sample angles for.
        rng: Random number generator for reproducibility.

    Returns:
        Dict mapping dihedral name -> (n_residues,) array in radians.
        Keys: alpha, beta, gamma, delta, epsilon, zeta, chi
        Terminal residues have NaN where the dihedral cannot be defined.
    """
    if rng is None:
        rng = np.random.default_rng()

    gmms = _get_rna_gmms()
    result = {}

    # Sample each backbone dihedral
    for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
        if name in gmms:
            samples = gmms[name].sample(n_residues, rng)
            result[name] = samples[:, 0].copy()  # 1D GMM, take first column
        else:
            # Fallback: use uniform distribution if GMM not available
            result[name] = rng.uniform(-np.pi, np.pi, n_residues)

    # Chi (glycosidic) - use chi_pyrimidine as it has more data
    # TODO: Handle purine vs pyrimidine based on residue type
    if "chi_pyrimidine" in gmms:
        samples = gmms["chi_pyrimidine"].sample(n_residues, rng)
        result["chi"] = samples[:, 0].copy()
    elif "chi_purine" in gmms:
        samples = gmms["chi_purine"].sample(n_residues, rng)
        result["chi"] = samples[:, 0].copy()
    else:
        result["chi"] = rng.uniform(-np.pi, np.pi, n_residues)

    # Set terminal NaN values
    # Alpha: requires O3' from previous residue (first residue has no alpha)
    result["alpha"][0] = np.nan
    # Epsilon: requires P from next residue (last residue has no epsilon)
    result["epsilon"][-1] = np.nan
    # Zeta: requires O5' from next residue (last residue has no zeta)
    result["zeta"][-1] = np.nan

    return result


# =============================================================================
# Unified Interface
# =============================================================================

def randomize_backbone(
    polymer: "Polymer",
    seed: int | None = None,
) -> "Polymer":
    """
    Randomize backbone dihedrals using empirical distributions.

    Automatically detects the molecule type (protein or RNA) and samples
    appropriate backbone dihedrals from Gaussian Mixture Models fitted
    to PDB data.

    Args:
        polymer: Polymer to randomize. Supports proteins and RNA.
        seed: Random seed for reproducibility.

    Returns:
        The polymer with randomized backbone dihedrals.

    Example:
        >>> import ciffy
        >>> from ciffy.sampling import randomize_backbone
        >>>
        >>> # Works for proteins
        >>> protein = ciffy.from_sequence("MGKLF")
        >>> protein = randomize_backbone(protein, seed=42)
        >>>
        >>> # Works for RNA
        >>> rna = ciffy.from_sequence("acgu")
        >>> rna = randomize_backbone(rna, seed=42)
    """
    from ..types import DihedralType, Molecule, Scale
    from ..biochemistry import Residue

    rng = np.random.default_rng(seed)
    n_residues = polymer.size(Scale.RESIDUE)

    if n_residues == 0:
        return polymer

    # Detect molecule type from first residue
    first_res_idx = int(polymer.sequence[0])
    try:
        first_res = Residue(first_res_idx)
        mol_type = first_res.molecule_type
    except (ValueError, AttributeError):
        mol_type = Molecule.PROTEIN  # Default to protein

    if mol_type == Molecule.PROTEIN:
        # Sample protein dihedrals
        phi, psi, omega = sample_protein_dihedrals(n_residues, rng)

        # Apply dihedrals (filter out NaN values)
        polymer.set_dihedral(DihedralType.PHI, phi[~np.isnan(phi)])
        polymer.set_dihedral(DihedralType.PSI, psi[~np.isnan(psi)])
        polymer.set_dihedral(DihedralType.OMEGA, omega[~np.isnan(omega)])

    elif mol_type in (Molecule.RNA, Molecule.DNA):
        # Sample RNA/DNA dihedrals
        dihedrals = sample_rna_dihedrals(n_residues, rng)

        # Map names to DihedralType
        name_to_type = {
            "alpha": DihedralType.ALPHA,
            "beta": DihedralType.BETA,
            "gamma": DihedralType.GAMMA,
            "delta": DihedralType.DELTA,
            "epsilon": DihedralType.EPSILON,
            "zeta": DihedralType.ZETA,
            "chi": DihedralType.CHI_PYRIMIDINE,  # Use pyrimidine for now
        }

        for name, dtype in name_to_type.items():
            if name in dihedrals:
                values = dihedrals[name]
                valid = values[~np.isnan(values)]
                if len(valid) > 0:
                    try:
                        polymer.set_dihedral(dtype, valid)
                    except (ValueError, KeyError):
                        # Dihedral type may not be defined for this polymer
                        pass

    # Force coordinate reconstruction
    _ = polymer.coordinates

    return polymer
