"""
Backbone dihedral sampling from empirical distributions.

Provides functions to sample realistic backbone dihedral angles from
Gaussian Mixture Models fitted to PDB data. Supports both proteins
and nucleic acids (RNA/DNA).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils.gmm import GaussianMixtureModel
from .gmm_registry import GMMRegistry

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..biochemistry import Molecule

# Path to pre-fitted GMM parameters (backward compatibility)
_DATA_DIR = Path(__file__).parent.parent / "data"
_RAMA_GMM_PATH = _DATA_DIR / "ramachandran_gmm.npz"
_RNA_GMM_PATH = _DATA_DIR / "rna_dihedrals.npz"

# Global registry instance
_registry = GMMRegistry()


@lru_cache(maxsize=1)
def _get_rama_gmm() -> GaussianMixtureModel:
    """Get the pre-fitted Ramachandran GMM, loading if necessary.

    Uses the GMM registry to load from gmm/protein/default.npz, with fallback
    to the legacy ramachandran_gmm.npz for backward compatibility.

    Thread-safe via lru_cache - only loads once even with concurrent access.
    """
    from ..biochemistry import Molecule

    try:
        # Try to load from new registry system
        return _registry.get_gmm(Molecule.PROTEIN)
    except (ValueError, FileNotFoundError):
        # Fallback to old path for backward compatibility
        if not _RAMA_GMM_PATH.exists():
            raise FileNotFoundError(
                f"Ramachandran GMM not found. Expected at either "
                f"ciffy/data/gmm/protein/default.npz or {_RAMA_GMM_PATH}. "
                "Run scripts/fit_ramachandran_gmm.py to generate it."
            )
        return GaussianMixtureModel.load(_RAMA_GMM_PATH)


@lru_cache(maxsize=1)
def _get_rna_gmms() -> dict[str, GaussianMixtureModel]:
    """Get the pre-fitted RNA dihedral GMMs, loading if necessary.

    Thread-safe via lru_cache - only loads once even with concurrent access.
    """
    if not _RNA_GMM_PATH.exists():
        raise FileNotFoundError(
            f"RNA dihedral GMMs not found at {_RNA_GMM_PATH}. "
            "Run scripts/fit_rna_dihedrals.py to generate it."
        )
    data = np.load(_RNA_GMM_PATH)

    # Reconstruct GMMs from stored parameters
    gmms = {}
    dihedral_names = set()
    for key in data.files:
        # Keys are like "alpha_means", "alpha_covariances", "alpha_weights"
        name = key.rsplit("_", 1)[0]
        dihedral_names.add(name)

    for name in dihedral_names:
        if f"{name}_means" in data:
            gmms[name] = GaussianMixtureModel(
                means=data[f"{name}_means"],
                covariances=data[f"{name}_covariances"],
                weights=data[f"{name}_weights"],
            )

    return gmms


# =============================================================================
# Clash Detection & Exception Classes
# =============================================================================


class ClashSamplingError(Exception):
    """Raised when autoregressive sampling fails due to persistent steric clashes."""

    pass


@lru_cache(maxsize=1)
def _load_vdw_radii() -> dict[str, float]:
    """Load Van der Waals radii from data file.

    Returns:
        Dictionary mapping element names to VDW radii in Angstroms.
    """
    import json

    vdw_path = _DATA_DIR / "vdw_radii.json"
    if not vdw_path.exists():
        raise FileNotFoundError(f"VDW radii file not found at {vdw_path}")

    with open(vdw_path) as f:
        data = json.load(f)

    # Remove comment key if present
    data.pop("comment", None)
    return data


def _has_clash(
    polymer: "Polymer",
    current_residue_idx: int,
    vdw_reduction: float = 0.4,
) -> bool:
    """
    Check if current residue clashes with any previous residues (excluding immediate predecessor).

    Uses VDW radii to determine clash threshold for each atom pair.
    Excludes the immediately previous residue (residue i-1) since it has bonded
    interactions (peptide bond) with the current residue.

    Args:
        polymer: Polymer with reconstructed coordinates.
        current_residue_idx: Index of newly sampled residue.
        vdw_reduction: Tolerance (Angstroms) to subtract from VDW sum.
                      Default 0.4 Å allows tight packing.

    Returns:
        True if clash detected, False otherwise.
    """
    from ..backend.ops import cdist
    from ..biochemistry import Element
    from ..types import Scale

    if current_residue_idx <= 2:
        # Only 0, 1, or 2 previous residues; skip clash checks
        # (residue 0 has no previous, residue 1 is bonded to residue 0,
        #  residue 2 shares spatial regions with residue 1 via backbone)
        return False

    # Extract current residue atoms
    current_atoms = polymer.by_residue_index(current_residue_idx)

    # Extract previous residues EXCEPT the immediately adjacent ones (which are bonded)
    # Check residues 0 to current_idx-3 (skip current_idx-2 and current_idx-1)
    previous_indices = list(range(current_residue_idx - 2))
    if len(previous_indices) == 0:
        return False

    previous_atoms = polymer.by_residue_index(previous_indices)

    # Filter to heavy atoms only (exclude hydrogen)
    curr_mask = current_atoms.elements != Element.H
    prev_mask = previous_atoms.elements != Element.H

    current_atoms = current_atoms[curr_mask]
    previous_atoms = previous_atoms[prev_mask]

    if current_atoms.size() == 0 or previous_atoms.size() == 0:
        # No heavy atoms to check
        return False

    # Get coordinates and elements
    curr_coords = current_atoms.coordinates  # (N_curr, 3)
    prev_coords = previous_atoms.coordinates  # (N_prev, 3)
    curr_elements = current_atoms.elements
    prev_elements = previous_atoms.elements

    # Compute pairwise distances
    distances = cdist(curr_coords, prev_coords)

    # Load VDW radii
    vdw_radii = _load_vdw_radii()

    # Check for clashes
    for i, curr_elem_idx in enumerate(curr_elements):
        # Get element symbol from index
        from ..biochemistry import ELEMENT_NAMES

        curr_elem = ELEMENT_NAMES.get(int(curr_elem_idx), "X")
        curr_radius = vdw_radii.get(curr_elem, vdw_radii.get("default", 1.7))

        for j, prev_elem_idx in enumerate(prev_elements):
            prev_elem = ELEMENT_NAMES.get(int(prev_elem_idx), "X")
            prev_radius = vdw_radii.get(prev_elem, vdw_radii.get("default", 1.7))

            # Clash threshold: sum of VDW radii minus tolerance
            threshold = curr_radius + prev_radius - vdw_reduction

            if distances[i, j] < threshold:
                return True

    return False


# =============================================================================
# Helper Functions for Autoregressive Sampling
# =============================================================================


def _sample_single_residue_protein(
    gmm: GaussianMixtureModel,
    residue_idx: int,
    n_residues: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Sample (phi, psi, omega) for a single protein residue.

    Handles terminal residue constraints:
    - First residue: phi = NaN
    - Last residue: psi = NaN, omega = NaN
    - Other residues: all angles sampled

    Args:
        gmm: 2D GMM for phi/psi
        residue_idx: Index of residue being sampled (0-indexed)
        n_residues: Total number of residues
        rng: Random number generator

    Returns:
        Tuple of (phi, psi, omega) in radians
    """
    # Sample phi/psi from 2D GMM
    phi_psi = gmm.sample(1, rng)[0]  # (2,)
    phi = phi_psi[0]
    psi = phi_psi[1]

    # Handle terminal constraints
    if residue_idx == 0:
        phi = np.nan  # N-terminus
    if residue_idx == n_residues - 1:
        psi = np.nan  # C-terminus

    # Sample omega (predominantly trans)
    omega = rng.normal(np.pi, 0.05)
    if residue_idx == n_residues - 1:
        omega = np.nan  # C-terminus

    return phi, psi, omega


def _sample_single_residue_rna(
    registry: GMMRegistry,
    residue_idx: int,
    n_residues: int,
    residue_type: "Residue",
    rng: np.random.Generator,
) -> dict["DihedralTypeHint", float]:
    """
    Sample all backbone dihedrals for a single RNA residue.

    Uses residue-type-specific GMMs via registry (purine/pyrimidine grouping).
    Handles terminal residue constraints.

    Args:
        registry: GMM registry for residue-type lookup
        residue_idx: Index of residue being sampled (0-indexed)
        n_residues: Total number of residues
        residue_type: Residue enum (e.g., Residue.A)
        rng: Random number generator

    Returns:
        Dict mapping DihedralType -> scalar value in radians (or NaN)
    """
    from ..types import DihedralType, Molecule
    from ..biochemistry import Residue as ResidueEnum

    # Get residue-type-specific 7D GMM via registry
    try:
        gmm = registry.get_gmm(Molecule.RNA, residue_type)
    except (ValueError, FileNotFoundError, KeyError):
        # Fallback: sample from independent 1D GMMs
        # KeyError: raised when NPZ file has wrong structure (old format)
        gmms = _get_rna_gmms()
        result = {}
        backbone_dihedrals = [
            (DihedralType.ALPHA, "alpha"),
            (DihedralType.BETA, "beta"),
            (DihedralType.GAMMA, "gamma"),
            (DihedralType.DELTA, "delta"),
            (DihedralType.EPSILON, "epsilon"),
            (DihedralType.ZETA, "zeta"),
        ]
        for dtype, gmm_key in backbone_dihedrals:
            if gmm_key in gmms:
                sample = gmms[gmm_key].sample(1, rng)[0, 0]
                result[dtype] = sample
            else:
                result[dtype] = rng.uniform(-np.pi, np.pi)

        # Chi angle
        if "chi_pyrimidine" in gmms:
            chi_sample = gmms["chi_pyrimidine"].sample(1, rng)[0, 0]
            result[DihedralType.CHI_PYRIMIDINE] = chi_sample
        else:
            result[DihedralType.CHI_PYRIMIDINE] = rng.uniform(-np.pi, np.pi)

        # Set terminal NaN
        if residue_idx == 0:
            result[DihedralType.ALPHA] = np.nan
        if residue_idx == n_residues - 1:
            result[DihedralType.EPSILON] = np.nan
            result[DihedralType.ZETA] = np.nan

        return result

    # Sample from 7D GMM
    sample_7d = gmm.sample(1, rng)[0]  # (7,)

    result = {
        DihedralType.ALPHA: sample_7d[0],
        DihedralType.BETA: sample_7d[1],
        DihedralType.GAMMA: sample_7d[2],
        DihedralType.DELTA: sample_7d[3],
        DihedralType.EPSILON: sample_7d[4],
        DihedralType.ZETA: sample_7d[5],
        DihedralType.CHI_PYRIMIDINE: sample_7d[6],
    }

    # Set terminal NaN
    if residue_idx == 0:
        result[DihedralType.ALPHA] = np.nan
    if residue_idx == n_residues - 1:
        result[DihedralType.EPSILON] = np.nan
        result[DihedralType.ZETA] = np.nan

    return result


def _apply_protein_dihedrals_partial(
    polymer: "Polymer",
    phi_values: list[float],
    psi_values: list[float],
    omega_values: list[float],
) -> None:
    """
    Apply protein dihedrals to polymer, handling NaN terminal values.

    Filters out NaN values and applies dihedrals using set_dihedral,
    then reconstructs coordinates.

    Args:
        polymer: Polymer to modify (in-place)
        phi_values: List of phi angles (may contain NaN)
        psi_values: List of psi angles (may contain NaN)
        omega_values: List of omega angles (may contain NaN)
    """
    from ..types import DihedralType

    # Filter non-NaN values
    phi_valid = np.array([v for v in phi_values if not np.isnan(v)])
    psi_valid = np.array([v for v in psi_values if not np.isnan(v)])
    omega_valid = np.array([v for v in omega_values if not np.isnan(v)])

    # Apply dihedrals to polymer
    if len(phi_valid) > 0:
        polymer.set_dihedral(DihedralType.PHI, phi_valid)

    if len(psi_valid) > 0:
        polymer.set_dihedral(DihedralType.PSI, psi_valid)

    if len(omega_valid) > 0:
        polymer.set_dihedral(DihedralType.OMEGA, omega_valid)

    # Force coordinate reconstruction
    _ = polymer.coordinates


# =============================================================================
# Unified Helper Functions for Langevin Sampling
# =============================================================================


def _get_pairwise_distances(
    polymer: "Polymer",
    current_res_idx: int,
    back_residues: int = 2,
) -> np.ndarray:
    """
    Get pairwise distances for current residue vs previous residues.

    Excludes the immediately adjacent residues (bonded neighbors).

    Args:
        polymer: Polymer with reconstructed coordinates.
        current_res_idx: Index of current residue.
        back_residues: Number of residues to exclude from the back (default 2).

    Returns:
        Flattened 1D array of pairwise distances, or empty array if no previous residues.
    """
    if current_res_idx <= back_residues:
        return np.array([])

    curr_atoms = polymer.by_residue_index(current_res_idx)
    prev_indices = list(range(current_res_idx - back_residues))
    prev_atoms = polymer.by_residue_index(prev_indices)

    return _filter_and_compute_distances(curr_atoms, prev_atoms)


def _filter_and_compute_distances(
    curr_atoms: "Polymer",
    prev_atoms: "Polymer",
) -> np.ndarray:
    """
    Filter atoms to heavy atoms only and compute pairwise distances.

    Args:
        curr_atoms: Current residue atom collection.
        prev_atoms: Previous residue(s) atom collection.

    Returns:
        Flattened 1D array of pairwise distances, or empty array if no atoms.
    """
    from ..biochemistry import Element
    from ..backend.ops import cdist

    # Filter to heavy atoms only
    curr_mask = curr_atoms.elements != Element.H
    prev_mask = prev_atoms.elements != Element.H

    curr_heavy = curr_atoms[curr_mask]
    prev_heavy = prev_atoms[prev_mask]

    if curr_heavy.size() == 0 or prev_heavy.size() == 0:
        return np.array([])

    distances = cdist(curr_heavy.coordinates, prev_heavy.coordinates)
    return distances.flatten()


def _apply_dihedrals(
    polymer: "Polymer",
    dihedral_dict: dict,
) -> None:
    """
    Apply dihedrals to polymer from a dictionary mapping DihedralType to values.

    Automatically filters NaN values, applies to polymer, and reconstructs coordinates.
    Works for any number of dihedral types (protein, RNA, or custom).

    Args:
        polymer: Polymer to modify (in-place).
        dihedral_dict: Dict mapping DihedralType -> list of angle values (may contain NaN).
    """
    from ..types import DihedralType

    for dtype, values in dihedral_dict.items():
        valid_values = np.array([v for v in values if not np.isnan(v)])
        if len(valid_values) > 0:
            try:
                polymer.set_dihedral(dtype, valid_values)
            except (ValueError, KeyError):
                # Dihedral type may not be defined for this polymer
                pass

    # Force coordinate reconstruction
    _ = polymer.coordinates


def _apply_terminal_constraints(
    candidate_dihedrals: dict,
    res_idx: int,
    n_residues: int,
    molecule_type: "Molecule",
) -> None:
    """
    Apply terminal NaN constraints based on molecule type.

    Modifies candidate_dihedrals in-place to set terminal dihedrals to NaN
    where they cannot be defined (N-terminus, C-terminus, etc.).

    Args:
        candidate_dihedrals: Dict mapping DihedralType -> scalar value.
        res_idx: Current residue index.
        n_residues: Total number of residues.
        molecule_type: Molecule type (PROTEIN, RNA, or DNA).
    """
    from ..types import DihedralType, Molecule

    if molecule_type == Molecule.PROTEIN:
        if res_idx == 0:
            candidate_dihedrals[DihedralType.PHI] = np.nan
        if res_idx == n_residues - 1:
            candidate_dihedrals[DihedralType.PSI] = np.nan
            candidate_dihedrals[DihedralType.OMEGA] = np.nan

    elif molecule_type in (Molecule.RNA, Molecule.DNA):
        if res_idx == 0:
            candidate_dihedrals[DihedralType.ALPHA] = np.nan
        if res_idx == n_residues - 1:
            candidate_dihedrals[DihedralType.EPSILON] = np.nan
            candidate_dihedrals[DihedralType.ZETA] = np.nan


# =============================================================================
# Protein Autoregressive Sampling
# =============================================================================


def sample_protein_autoregressive(
    polymer: "Polymer",
    max_attempts: int = 100,
    vdw_reduction: float = 0.4,
    seed: int | None = None,
) -> "Polymer":
    """
    Sample backbone dihedrals autoregressively with clash detection.

    Samples phi, psi, omega angles one residue at a time, checking for steric
    clashes with all previously sampled residues. Rejects and resamples if a
    clash is detected (up to max_attempts per residue).

    Args:
        polymer: Polymer to sample. Must have valid coordinates.
        max_attempts: Maximum rejection sampling attempts per residue.
        vdw_reduction: VDW tolerance in Angstroms (default 0.4). Increase to
                      be more permissive with atomic overlap.
        seed: Random seed for reproducibility.

    Returns:
        Polymer with clash-free sampled backbone dihedrals.

    Raises:
        ClashSamplingError: If sampling fails after max_attempts for any residue.

    Example:
        >>> import ciffy
        >>> from ciffy.sampling import sample_protein_autoregressive
        >>> protein = ciffy.from_sequence("MGKLF")
        >>> protein = sample_protein_autoregressive(protein, seed=42)
    """
    from ..types import Scale, Molecule
    from ..biochemistry import Residue as ResidueEnum

    rng = np.random.default_rng(seed)
    n_residues = polymer.size(Scale.RESIDUE)

    if n_residues == 0:
        return polymer

    # Detect residue type from first residue
    first_res_idx = int(polymer.sequence[0])
    try:
        first_res = ResidueEnum(first_res_idx)
    except (ValueError, AttributeError):
        first_res = ResidueEnum.ALA  # Default to ALA

    phi_list = []
    psi_list = []
    omega_list = []

    for res_idx in range(n_residues):
        # Get residue type
        res_enum_val = int(polymer.sequence[res_idx])
        try:
            res_enum = ResidueEnum(res_enum_val)
        except (ValueError, AttributeError):
            res_enum = ResidueEnum.ALA

        # Get residue-specific GMM via registry
        gmm = _registry.get_gmm(Molecule.PROTEIN, res_enum)

        # Rejection sampling loop
        for attempt in range(max_attempts):
            # Sample candidate angles for this residue
            phi, psi, omega = _sample_single_residue_protein(
                gmm, res_idx, n_residues, rng
            )

            # Temporarily apply angles to check clash
            candidate_phi = phi_list + [phi]
            candidate_psi = psi_list + [psi]
            candidate_omega = omega_list + [omega]

            _apply_protein_dihedrals_partial(
                polymer,
                candidate_phi,
                candidate_psi,
                candidate_omega,
            )

            # Check clash (skip for first residue)
            if res_idx == 0 or not _has_clash(
                polymer, res_idx, vdw_reduction=vdw_reduction
            ):
                # Accept this residue
                phi_list.append(phi)
                psi_list.append(psi)
                omega_list.append(omega)
                break
        else:
            # Exhausted max_attempts
            raise ClashSamplingError(
                f"Failed to sample residue {res_idx} without clash after "
                f"{max_attempts} attempts. Try increasing vdw_reduction "
                f"(current: {vdw_reduction}) to be more permissive."
            )

    return polymer


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

# Import DihedralType for type hints - actual import happens in functions to avoid circular imports
if TYPE_CHECKING:
    from ..types import DihedralType as DihedralTypeHint


def sample_rna_dihedrals(
    n_residues: int,
    rng: np.random.Generator | None = None,
) -> dict["DihedralTypeHint", np.ndarray]:
    """
    Sample backbone dihedrals for n RNA residues.

    Uses Gaussian Mixture Models fitted to empirical distributions
    from PDB RNA structures.

    Args:
        n_residues: Number of residues to sample angles for.
        rng: Random number generator for reproducibility.

    Returns:
        Dict mapping DihedralType -> (n_residues,) array in radians.
        Keys: ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA, CHI_PYRIMIDINE
        Terminal residues have NaN where the dihedral cannot be defined.
    """
    from ..types import DihedralType

    if rng is None:
        rng = np.random.default_rng()

    gmms = _get_rna_gmms()
    result: dict[DihedralType, np.ndarray] = {}

    # Map DihedralType to GMM key names (GMM files use lowercase string keys)
    backbone_dihedrals = [
        (DihedralType.ALPHA, "alpha"),
        (DihedralType.BETA, "beta"),
        (DihedralType.GAMMA, "gamma"),
        (DihedralType.DELTA, "delta"),
        (DihedralType.EPSILON, "epsilon"),
        (DihedralType.ZETA, "zeta"),
    ]

    # Sample each backbone dihedral
    for dtype, gmm_key in backbone_dihedrals:
        if gmm_key in gmms:
            samples = gmms[gmm_key].sample(n_residues, rng)
            result[dtype] = samples[:, 0].copy()  # 1D GMM, take first column
        else:
            # Fallback: use uniform distribution if GMM not available
            result[dtype] = rng.uniform(-np.pi, np.pi, n_residues)

    # Chi (glycosidic) - use chi_pyrimidine as it has more data
    # TODO: Handle purine vs pyrimidine based on residue type
    if "chi_pyrimidine" in gmms:
        samples = gmms["chi_pyrimidine"].sample(n_residues, rng)
        result[DihedralType.CHI_PYRIMIDINE] = samples[:, 0].copy()
    elif "chi_purine" in gmms:
        samples = gmms["chi_purine"].sample(n_residues, rng)
        result[DihedralType.CHI_PYRIMIDINE] = samples[:, 0].copy()
    else:
        result[DihedralType.CHI_PYRIMIDINE] = rng.uniform(-np.pi, np.pi, n_residues)

    # Set terminal NaN values
    # Alpha: requires O3' from previous residue (first residue has no alpha)
    result[DihedralType.ALPHA][0] = np.nan
    # Epsilon: requires P from next residue (last residue has no epsilon)
    result[DihedralType.EPSILON][-1] = np.nan
    # Zeta: requires O5' from next residue (last residue has no zeta)
    result[DihedralType.ZETA][-1] = np.nan

    return result


# =============================================================================
# RNA Autoregressive Sampling
# =============================================================================


def sample_rna_autoregressive(
    polymer: "Polymer",
    max_attempts: int = 100,
    vdw_reduction: float = 0.4,
    seed: int | None = None,
) -> "Polymer":
    """
    Sample backbone dihedrals autoregressively with clash detection for RNA.

    Samples all 7 backbone dihedrals (alpha, beta, gamma, delta, epsilon, zeta, chi)
    one residue at a time using residue-type-specific GMMs (purine/pyrimidine).
    Rejects and resamples if a clash is detected with previously sampled residues.

    Args:
        polymer: RNA polymer to sample. Must have valid coordinates.
        max_attempts: Maximum rejection sampling attempts per residue.
        vdw_reduction: VDW tolerance in Angstroms (default 0.4).
        seed: Random seed for reproducibility.

    Returns:
        RNA polymer with clash-free sampled backbone dihedrals.

    Raises:
        ClashSamplingError: If sampling fails after max_attempts for any residue.

    Example:
        >>> import ciffy
        >>> from ciffy.sampling import sample_rna_autoregressive
        >>> rna = ciffy.from_sequence("acgu")
        >>> rna = sample_rna_autoregressive(rna, seed=42)
    """
    from ..types import DihedralType, Scale
    from ..biochemistry import Residue as ResidueEnum

    rng = np.random.default_rng(seed)
    n_residues = polymer.size(Scale.RESIDUE)

    if n_residues == 0:
        return polymer

    # Initialize dihedral lists
    dihedral_lists = {
        DihedralType.ALPHA: [],
        DihedralType.BETA: [],
        DihedralType.GAMMA: [],
        DihedralType.DELTA: [],
        DihedralType.EPSILON: [],
        DihedralType.ZETA: [],
        DihedralType.CHI_PYRIMIDINE: [],
    }

    for res_idx in range(n_residues):
        # Get residue type
        res_enum_val = int(polymer.sequence[res_idx])
        try:
            res_enum = ResidueEnum(res_enum_val)
        except (ValueError, AttributeError):
            res_enum = ResidueEnum.A  # Default to adenine

        # Rejection sampling loop
        for attempt in range(max_attempts):
            # Sample candidate dihedrals for this residue
            candidate_dihedrals = _sample_single_residue_rna(
                _registry, res_idx, n_residues, res_enum, rng
            )

            # Temporarily apply angles to check clash
            temp_dihedral_lists = {
                k: v + [candidate_dihedrals[k]] for k, v in dihedral_lists.items()
            }

            # Apply dihedrals to polymer
            for dtype, values in temp_dihedral_lists.items():
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    try:
                        polymer.set_dihedral(dtype, np.array(valid_values))
                    except (ValueError, KeyError):
                        # Dihedral type may not be defined
                        pass

            # Force coordinate reconstruction
            _ = polymer.coordinates

            # Check clash (skip for first residue)
            if res_idx == 0 or not _has_clash(
                polymer, res_idx, vdw_reduction=vdw_reduction
            ):
                # Accept this residue
                for dtype in dihedral_lists:
                    dihedral_lists[dtype].append(candidate_dihedrals[dtype])
                break
        else:
            # Exhausted max_attempts
            raise ClashSamplingError(
                f"Failed to sample RNA residue {res_idx} without clash after "
                f"{max_attempts} attempts. Try increasing vdw_reduction "
                f"(current: {vdw_reduction}) to be more permissive."
            )

    return polymer


# =============================================================================
# Unified Interface
# =============================================================================


def _detect_molecule_type(polymer: "Polymer") -> "Molecule":
    """
    Detect molecule type (protein/RNA/DNA) from polymer.

    Uses multiple heuristics to handle ambiguous residue codes:
    1. Check multiple residues (not just first)
    2. Use unambiguous protein codes (A, P, S, T, etc.) if present
    3. Use unambiguous RNA codes (U) if present
    4. Fall back to first residue's molecule type
    5. Default to PROTEIN if all else fails

    Args:
        polymer: Polymer to classify

    Returns:
        Molecule.PROTEIN, Molecule.RNA, or Molecule.DNA
    """
    from ..types import Molecule
    from ..biochemistry import Residue

    # Check multiple residues for unambiguous codes
    protein_only_codes = {1, 2, 7, 8, 14, 16, 18, 20}  # C,D,H,I,L,N,Q,R
    rna_only_codes = {21}  # U (uracil, unambiguous RNA)
    dna_only_codes = {22}  # T (thymine, DNA-specific)

    for res_idx in polymer.sequence[:min(10, len(polymer.sequence))]:
        res_int = int(res_idx)
        if res_int in rna_only_codes or res_int == 22:  # U or T
            return Molecule.RNA if res_int == 21 else Molecule.DNA
        if res_int in protein_only_codes:
            return Molecule.PROTEIN

    # Fall back to first residue's molecule type
    first_res_idx = int(polymer.sequence[0])
    try:
        first_res = Residue(first_res_idx)
        mol_type = first_res.molecule_type
        # Trust the molecule type if it's not ambiguous
        if mol_type in (Molecule.PROTEIN, Molecule.RNA, Molecule.DNA):
            return mol_type
    except (ValueError, AttributeError):
        pass

    # Default to protein
    return Molecule.PROTEIN


def sample_autoregressive(
    polymer: "Polymer",
    max_attempts: int = 100,
    vdw_reduction: float = 0.4,
    seed: int | None = None,
) -> "Polymer":
    """
    Sample backbone dihedrals autoregressively with clash detection.

    Unified dispatcher that detects molecule type and calls the appropriate
    autoregressive sampling function (protein or RNA/DNA).

    Uses **Langevin dynamics** (default, recommended) to jointly optimize
    GMM likelihood + clash avoidance. This approach:
    - Never rejects (always accepts)
    - Handles tight VDW constraints that rejection sampling can't
    - Efficiently explores the joint space of distributions and constraints

    For backwards compatibility, pass `use_langevin=False` to get the old
    rejection-sampling behavior.

    Args:
        polymer: Polymer to sample (protein or RNA/DNA).
        max_attempts: Ignored (kept for backwards compatibility).
        vdw_reduction: VDW tolerance in Angstroms (default 0.4). Increase to
                      be more permissive with atomic overlap.
        seed: Random seed for reproducibility.

    Returns:
        Polymer with clash-minimized sampled backbone dihedrals.

    Raises:
        ValueError: If molecule type is not supported.

    Example:
        >>> import ciffy
        >>> from ciffy.sampling import sample_autoregressive
        >>>
        >>> # Works for proteins (uses Langevin)
        >>> protein = ciffy.from_sequence("MGKLF")
        >>> protein = sample_autoregressive(protein, seed=42)
        >>>
        >>> # Works for RNA (uses Langevin, no more rejections!)
        >>> rna = ciffy.from_sequence("acgu")
        >>> rna = sample_autoregressive(rna, seed=42)
    """
    from ..types import Molecule

    # Detect molecule type using robust heuristics
    mol_type = _detect_molecule_type(polymer)

    if mol_type == Molecule.PROTEIN:
        return sample_protein_autoregressive_langevin(
            polymer, vdw_reduction=vdw_reduction, seed=seed
        )
    elif mol_type in (Molecule.RNA, Molecule.DNA):
        return sample_rna_autoregressive_langevin(
            polymer, vdw_reduction=vdw_reduction, seed=seed
        )
    else:
        raise ValueError(f"Unsupported molecule type for autoregressive sampling: {mol_type}")


# =============================================================================
# Langevin Dynamics Autoregressive Sampling (Phase 3b)
# =============================================================================


def sample_protein_autoregressive_langevin(
    polymer: "Polymer",
    n_langevin_steps: int = 50,
    step_size: float = 0.01,
    temperature: float = 1.0,
    lambda_clash: float = 1000.0,
    vdw_reduction: float = 0.4,
    seed: int | None = None,
) -> "Polymer":
    """
    Sample protein backbone dihedrals using Langevin dynamics with clash avoidance.

    Uses autoregressive Langevin sampling: for each residue, jointly optimizes
    GMM log-likelihood + clash penalties using Langevin dynamics.

    This approach efficiently explores the joint space of distributions and
    constraints, avoiding the rejection sampling failures seen with tight
    VDW constraints.

    Args:
        polymer: Protein polymer to sample. Must have valid coordinates.
        n_langevin_steps: Number of Langevin steps per residue (default 50).
        step_size: Step size for Langevin discretization (default 0.01).
        temperature: Temperature for thermal noise (default 1.0).
        lambda_clash: Weight of clash penalty term (default 1000).
        vdw_reduction: VDW tolerance in Angstroms (default 0.4).
        seed: Random seed for reproducibility.

    Returns:
        Protein with sampled backbone dihedrals (clash-minimized).

    Example:
        >>> import ciffy
        >>> from ciffy.sampling import sample_protein_autoregressive_langevin
        >>> protein = ciffy.from_sequence("MGKLF")
        >>> protein = sample_protein_autoregressive_langevin(protein, seed=42)
    """
    from ..types import Scale, Molecule, DihedralType
    from ..biochemistry import Residue as ResidueEnum
    from .energy import GMMEnergy, ClashEnergy, CompositeEnergy
    from .langevin import langevin_dynamics

    rng = np.random.default_rng(seed)
    n_residues = polymer.size(Scale.RESIDUE)

    if n_residues == 0:
        return polymer

    phi_list = []
    psi_list = []
    omega_list = []

    for res_idx in range(n_residues):
        # Get residue type
        res_enum_val = int(polymer.sequence[res_idx])
        try:
            res_enum = ResidueEnum(res_enum_val)
        except (ValueError, AttributeError):
            res_enum = ResidueEnum.ALA

        # Get residue-specific GMM via registry
        gmm = _registry.get_gmm(Molecule.PROTEIN, res_enum)

        # Define polymer evaluator: applies candidate angles and returns pairwise distances
        def make_protein_evaluator(poly, res_idx, prev_phi, prev_psi, prev_omega, rng):
            def evaluator(angles: np.ndarray) -> np.ndarray:
                """Apply angles to polymer and return pairwise distances for clash energy."""
                from ..types import DihedralType

                # angles is (phi, psi) for this residue
                phi, psi = float(angles[0]), float(angles[1])
                omega = float(rng.normal(np.pi, 0.05))

                # Build full angle lists for all residues up to and including current
                full_phi = prev_phi + [phi]
                full_psi = prev_psi + [psi]
                full_omega = prev_omega + [omega]

                # Apply dihedrals using helper function
                _apply_dihedrals(poly, {
                    DihedralType.PHI: full_phi[1:],  # Skip first (undefined at N-terminus)
                    DihedralType.PSI: full_psi[:-1],  # Skip last (undefined at C-terminus)
                    DihedralType.OMEGA: full_omega[:-1],  # Skip last
                })

                # Return pairwise distances (helper function handles back_residues exclusion)
                return _get_pairwise_distances(poly, res_idx, back_residues=2)

            return evaluator

        # Create energy functions: GMM + clash (1/r²)
        gmm_energy = GMMEnergy(gmm)
        polymer_evaluator = make_protein_evaluator(polymer, res_idx, phi_list, psi_list, omega_list, rng)
        clash_energy = ClashEnergy(polymer_evaluator, lambda_clash=lambda_clash, r_min=0.5)
        composite_energy = CompositeEnergy([gmm_energy, clash_energy])

        # Sample initial angles from GMM
        phi_psi_sample = gmm.sample(1, rng)[0]

        # Run Langevin dynamics on joint GMM + clash energy
        refined_angles = langevin_dynamics(
            composite_energy,
            phi_psi_sample,
            n_steps=n_langevin_steps,
            step_size=step_size,
            temperature=temperature,
            rng=rng,
        )

        phi = refined_angles[0]
        psi = refined_angles[1]
        omega = rng.normal(np.pi, 0.05)

        # Build candidate dihedral dict and apply terminal constraints
        candidate_dihedrals = {
            DihedralType.PHI: phi,
            DihedralType.PSI: psi,
            DihedralType.OMEGA: omega,
        }
        _apply_terminal_constraints(candidate_dihedrals, res_idx, n_residues, Molecule.PROTEIN)

        # Accept angles for this residue
        phi_list.append(candidate_dihedrals[DihedralType.PHI])
        psi_list.append(candidate_dihedrals[DihedralType.PSI])
        omega_list.append(candidate_dihedrals[DihedralType.OMEGA])

    # Apply final dihedrals to polymer
    _apply_dihedrals(polymer, {
        DihedralType.PHI: phi_list[1:],  # Skip first (undefined at N-terminus)
        DihedralType.PSI: psi_list[:-1],  # Skip last (undefined at C-terminus)
        DihedralType.OMEGA: omega_list[:-1],  # Skip last
    })

    return polymer


def sample_rna_autoregressive_langevin(
    polymer: "Polymer",
    n_langevin_steps: int = 50,
    step_size: float = 0.01,
    temperature: float = 1.0,
    lambda_clash: float = 1000.0,
    vdw_reduction: float = 0.4,
    seed: int | None = None,
) -> "Polymer":
    """
    Sample RNA backbone dihedrals using Langevin dynamics with clash avoidance.

    Uses autoregressive Langevin sampling for all 7 backbone dihedrals,
    jointly optimizing GMM likelihood + clash avoidance via energy composition.

    For each residue, samples 7D (alpha, beta, gamma, delta, epsilon, zeta, chi)
    angles from the joint energy: E = -log(GMM) + λ*Σ(1/r²) for all atom pairs.

    Args:
        polymer: RNA polymer to sample. Must have valid coordinates.
        n_langevin_steps: Number of Langevin steps per residue (default 50).
        step_size: Step size for Langevin discretization (default 0.01).
        temperature: Temperature for thermal noise (default 1.0).
        lambda_clash: Weight of clash penalty term (default 1000).
        vdw_reduction: VDW tolerance in Angstroms (default 0.4).
        seed: Random seed for reproducibility.

    Returns:
        RNA with sampled backbone dihedrals (clash-minimized).

    Example:
        >>> import ciffy
        >>> from ciffy.sampling import sample_rna_autoregressive_langevin
        >>> rna = ciffy.from_sequence("acgu")
        >>> rna = sample_rna_autoregressive_langevin(rna, seed=42)
    """
    from ..types import DihedralType, Scale, Molecule
    from ..biochemistry import Residue as ResidueEnum
    from .energy import GMMEnergy, ClashEnergy, CompositeEnergy
    from .langevin import langevin_dynamics

    rng = np.random.default_rng(seed)
    n_residues = polymer.size(Scale.RESIDUE)

    if n_residues == 0:
        return polymer

    # Initialize dihedral lists
    dihedral_lists = {
        DihedralType.ALPHA: [],
        DihedralType.BETA: [],
        DihedralType.GAMMA: [],
        DihedralType.DELTA: [],
        DihedralType.EPSILON: [],
        DihedralType.ZETA: [],
        DihedralType.CHI_PYRIMIDINE: [],
    }

    for res_idx in range(n_residues):
        # Get residue type
        res_enum_val = int(polymer.sequence[res_idx])
        try:
            res_enum = ResidueEnum(res_enum_val)
        except (ValueError, AttributeError):
            res_enum = ResidueEnum.A

        # Try to get residue-specific 7D GMM via registry
        # If not available, fall back to independent sampling
        try:
            gmm = _registry.get_gmm(Molecule.RNA, res_enum)
            # Check if it's a 7D GMM by checking dimensions
            if gmm.means.shape[1] != 7:
                # Not a 7D GMM, use independent sampling instead
                raise ValueError("RNA GMM is not 7D")
        except (ValueError, FileNotFoundError, KeyError):
            gmm = None

        # Define RNA evaluator: applies candidate 7D angles and returns pairwise distances
        def make_rna_evaluator(poly, res_idx, prev_dihedrals, rng):
            def evaluator(angles_7d: np.ndarray) -> np.ndarray:
                """Apply 7D angles to polymer and return pairwise distances for clash energy."""
                # angles_7d is [alpha, beta, gamma, delta, epsilon, zeta, chi]
                alpha, beta, gamma, delta, epsilon, zeta, chi = angles_7d

                # Build full dihedral lists including current angles
                full_dihedrals = {
                    DihedralType.ALPHA: prev_dihedrals[DihedralType.ALPHA] + [alpha],
                    DihedralType.BETA: prev_dihedrals[DihedralType.BETA] + [beta],
                    DihedralType.GAMMA: prev_dihedrals[DihedralType.GAMMA] + [gamma],
                    DihedralType.DELTA: prev_dihedrals[DihedralType.DELTA] + [delta],
                    DihedralType.EPSILON: prev_dihedrals[DihedralType.EPSILON] + [epsilon],
                    DihedralType.ZETA: prev_dihedrals[DihedralType.ZETA] + [zeta],
                    DihedralType.CHI_PYRIMIDINE: prev_dihedrals[DihedralType.CHI_PYRIMIDINE] + [chi],
                }

                # Apply dihedrals using helper function
                _apply_dihedrals(poly, full_dihedrals)

                # Return pairwise distances (helper function handles back_residues exclusion)
                return _get_pairwise_distances(poly, res_idx, back_residues=2)

            return evaluator

        # Use Langevin dynamics if 7D GMM available, else use independent sampling
        if gmm is not None:
            # Create energy functions: GMM + clash (1/r²)
            gmm_energy = GMMEnergy(gmm)
            polymer_evaluator = make_rna_evaluator(polymer, res_idx, dihedral_lists, rng)
            clash_energy = ClashEnergy(polymer_evaluator, lambda_clash=lambda_clash, r_min=0.5)
            composite_energy = CompositeEnergy([gmm_energy, clash_energy])

            # Sample initial 7D angles from GMM
            angles_7d_sample = gmm.sample(1, rng)[0]

            # Run Langevin dynamics on joint GMM + clash energy
            refined_angles_7d = langevin_dynamics(
                composite_energy,
                angles_7d_sample,
                n_steps=n_langevin_steps,
                step_size=step_size,
                temperature=temperature,
                rng=rng,
            )

            # Extract individual dihedral angles
            candidate_dihedrals = {
                DihedralType.ALPHA: refined_angles_7d[0],
                DihedralType.BETA: refined_angles_7d[1],
                DihedralType.GAMMA: refined_angles_7d[2],
                DihedralType.DELTA: refined_angles_7d[3],
                DihedralType.EPSILON: refined_angles_7d[4],
                DihedralType.ZETA: refined_angles_7d[5],
                DihedralType.CHI_PYRIMIDINE: refined_angles_7d[6],
            }
        else:
            # Fallback to independent sampling (e.g., when 7D GMMs not available)
            candidate_dihedrals = _sample_single_residue_rna(
                _registry, res_idx, n_residues, res_enum, rng
            )

        # Apply terminal NaN constraints
        _apply_terminal_constraints(candidate_dihedrals, res_idx, n_residues, Molecule.RNA)

        # Accept dihedrals for this residue
        for dtype in dihedral_lists:
            dihedral_lists[dtype].append(candidate_dihedrals[dtype])

    # Apply final dihedrals to polymer using helper
    _apply_dihedrals(polymer, dihedral_lists)

    return polymer


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
        # Sample RNA/DNA dihedrals (returns dict[DihedralType, np.ndarray])
        dihedrals = sample_rna_dihedrals(n_residues, rng)

        for dtype, values in dihedrals.items():
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
