"""
Structure comparison metrics: RMSD, TM-score, and lDDT.
"""

from __future__ import annotations
from functools import singledispatch
from typing import TYPE_CHECKING, overload

import numpy as np

from ..backend import Array, is_torch, is_numpy, svd, svdvals, det, multiply, has_nan, has_inf, sqrt, clamp, clone
from ..biochemistry import Scale, Molecule

if TYPE_CHECKING:
    import torch
    from ..polymer import Polymer


def _get_representative_coords(polymer: "Polymer", mol_type: Molecule) -> Array:
    """
    Get representative atom coordinates for TM-score calculation.

    For proteins, selects Cα atoms. For RNA/DNA, selects C1' atoms.

    Args:
        polymer: The polymer to extract coordinates from.
        mol_type: Molecule type determining which atoms to select.

    Returns:
        Coordinates array of shape (n_residues, 3).

    Raises:
        ValueError: If representative atoms don't match residue count
            (e.g., missing atoms from unresolved residues).
    """
    from ..biochemistry.constants import Sugar, ProteinBackbone

    n_residues = polymer.size(Scale.RESIDUE)

    if mol_type == Molecule.PROTEIN:
        atom_name = "Cα"
        rep_atoms = polymer.by_atom(ProteinBackbone.CA.index())
    else:
        atom_name = "C1'"
        rep_atoms = polymer.by_atom(Sugar.C1p.index())

    n_found = rep_atoms.size()
    if n_found != n_residues:
        raise ValueError(
            f"Missing {atom_name} atoms: found {n_found} but polymer has {n_residues} residues. "
            f"This may indicate unresolved residues or non-standard residue types."
        )

    return rep_atoms.coordinates


def tm_score(
    pred: Polymer,
    ref: Polymer,
    molecule_type: Molecule | None = None,
) -> float:
    """
    Compute TM-score between two structures.

    TM-score is a length-normalized structural similarity metric
    ranging from 0 to 1, where 1 indicates identical structures.
    Scores > 0.5 generally indicate same fold.

    Args:
        pred: Predicted structure.
        ref: Reference structure (used for length normalization).
        molecule_type: Molecule type for d_0 calculation. If None,
            auto-detected from ref.molecule_types (raises ValueError
            if not available).

    Returns:
        TM-score value between 0 and 1.

    Raises:
        ValueError: If representative atoms are missing (unresolved residues
            or non-standard residue types without Cα/C1').

    Note:
        Uses representative atoms per residue:
        - Protein: Cα atoms
        - RNA/DNA: C1' atoms

        Uses molecule-type-specific normalization:
        - Protein: d_0 = 1.24 * (L - 15)^(1/3) - 1.8
        - RNA/DNA: d_0 = 0.6 * sqrt(L - 5) - 2.5
    """
    from .alignment import kabsch_align

    # Determine molecule type (needed for representative atom selection)
    mol_type = molecule_type if molecule_type is not None else _get_molecule_type(ref)

    # Get representative atom coordinates (Cα for protein, C1' for RNA/DNA)
    pred_coords = _get_representative_coords(pred, mol_type)
    ref_coords = _get_representative_coords(ref, mol_type)

    # Length for normalization (from reference)
    L = ref_coords.shape[0]

    if pred_coords.shape[0] != L:
        raise ValueError(
            f"Structure sizes must match: pred has {pred_coords.shape[0]} residues, "
            f"ref has {L} residues"
        )

    # Compute d_0 based on molecule type
    d_0 = _compute_d0(L, mol_type)

    # Align predicted structure onto reference using Kabsch algorithm
    # kabsch_align places pred_aligned at ref centroid, so compare to ref_coords directly
    pred_aligned, _, _ = kabsch_align(pred_coords, ref_coords, center=True)

    # Compute per-residue distances
    if is_torch(ref_coords):
        import torch
        distances = torch.sqrt(((pred_aligned - ref_coords) ** 2).sum(dim=1))
        tm = (1.0 / (1.0 + (distances / d_0) ** 2)).sum() / L
        return tm.item()
    else:
        distances = np.sqrt(((pred_aligned - ref_coords) ** 2).sum(axis=1))
        tm = (1.0 / (1.0 + (distances / d_0) ** 2)).sum() / L
        return float(tm)


def _compute_d0(L: int, mol_type: Molecule) -> float:
    """
    Compute the d_0 normalization parameter for TM-score.

    Args:
        L: Length of the structure (number of residues).
        mol_type: Molecule type for formula selection.

    Returns:
        d_0 value (minimum 0.5 for very small structures).
    """
    if mol_type == Molecule.PROTEIN:
        # Protein: d_0 = 1.24 * (L - 15)^(1/3) - 1.8
        inner = L - 15
        if inner >= 0:
            d_0 = 1.24 * (inner ** (1/3)) - 1.8
        else:
            d_0 = 1.24 * (-((-inner) ** (1/3))) - 1.8
    else:
        # RNA/DNA: d_0 = 0.6 * sqrt(L - 5) - 2.5
        inner = max(0, L - 5)
        d_0 = 0.6 * np.sqrt(inner) - 2.5

    # Ensure d_0 is positive (minimum 0.5 for very small structures)
    return float(max(d_0, 0.5))


def lddt(
    pred: Polymer,
    ref: Polymer,
    cutoff: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> tuple[float, Array]:
    """
    Compute lDDT score between two structures.

    lDDT (Local Distance Difference Test) measures local structural
    similarity by comparing inter-atomic distances. Unlike RMSD,
    it's robust to domain movements.

    Args:
        pred: Predicted structure.
        ref: Reference structure (defines which pairs to consider).
        cutoff: Only consider atom pairs within this distance in reference.
        thresholds: Distance difference thresholds for scoring.

    Returns:
        Tuple of (global_lddt, per_residue_lddt).
        - global_lddt: Single score between 0 and 1.
        - per_residue_lddt: Array of shape (num_residues,) with per-residue scores.
    """
    # Get coordinates
    pred_coords = pred.coordinates
    ref_coords = ref.coordinates

    n_atoms = pred_coords.shape[0]
    if ref_coords.shape[0] != n_atoms:
        raise ValueError(
            f"Structure sizes must match: pred has {n_atoms}, "
            f"ref has {ref_coords.shape[0]} atoms"
        )

    # Compute distance matrices
    pred_dists = pred.pairwise_distances()
    ref_dists = ref.pairwise_distances()

    if is_torch(pred_coords):
        import torch

        # Create mask for pairs within cutoff in reference (excluding diagonal)
        mask = (ref_dists < cutoff) & (ref_dists > 0)

        # Compute distance differences
        dist_diff = torch.abs(pred_dists - ref_dists)

        # Score each threshold
        scores = torch.zeros_like(dist_diff)
        for thresh in thresholds:
            scores += (dist_diff < thresh).float()
        scores = scores / len(thresholds)

        # Apply mask and compute per-atom scores
        masked_scores = scores * mask.float()
        pair_counts = mask.float().sum(dim=1)

        # Avoid division by zero
        pair_counts = pair_counts.clamp(min=1)
        per_atom_lddt = masked_scores.sum(dim=1) / pair_counts

        # Aggregate to per-residue
        per_residue_lddt = pred.reduce(per_atom_lddt, Scale.RESIDUE)

        # Global score (average over atoms with valid pairs)
        valid_atoms = mask.any(dim=1)
        if valid_atoms.any():
            global_lddt = per_atom_lddt[valid_atoms].mean().item()
        else:
            global_lddt = 0.0

        return global_lddt, per_residue_lddt
    else:
        # NumPy implementation
        # Create mask for pairs within cutoff in reference (excluding diagonal)
        mask = (ref_dists < cutoff) & (ref_dists > 0)

        # Compute distance differences
        dist_diff = np.abs(pred_dists - ref_dists)

        # Score each threshold
        scores = np.zeros_like(dist_diff)
        for thresh in thresholds:
            scores += (dist_diff < thresh).astype(float)
        scores = scores / len(thresholds)

        # Apply mask and compute per-atom scores
        masked_scores = scores * mask.astype(float)
        pair_counts = mask.astype(float).sum(axis=1)

        # Avoid division by zero
        pair_counts = np.maximum(pair_counts, 1)
        per_atom_lddt = masked_scores.sum(axis=1) / pair_counts

        # Aggregate to per-residue
        per_residue_lddt = pred.reduce(per_atom_lddt, Scale.RESIDUE)

        # Global score (average over atoms with valid pairs)
        valid_atoms = mask.any(axis=1)
        if valid_atoms.any():
            global_lddt = float(per_atom_lddt[valid_atoms].mean())
        else:
            global_lddt = 0.0

        return global_lddt, per_residue_lddt


def _get_molecule_type(polymer: Polymer) -> Molecule:
    """Get the predominant molecule type of a polymer.

    Raises:
        ValueError: If molecule_types is not available on the polymer.
    """
    from ..biochemistry._generated_molecule import molecule_type

    try:
        mol_types = polymer.molecule_types
    except AttributeError:
        mol_types = None
    if mol_types is None:
        raise ValueError("Cannot determine molecule type: molecule_types not available on this polymer")
    if is_torch(mol_types):
        mol_types = mol_types.cpu().numpy()

    # Get most common type (excluding UNKNOWN and OTHER)
    unique, counts = np.unique(mol_types, return_counts=True)
    for idx in np.argsort(-counts):
        mol = molecule_type(int(unique[idx]))
        if mol not in (Molecule.UNKNOWN, Molecule.OTHER, Molecule.WATER, Molecule.ION):
            return mol

    # Fallback to PROTEIN if no valid type found
    return Molecule.PROTEIN


# =============================================================================
# RMSD (Root Mean Square Deviation)
# =============================================================================

def _rmsd_coords(coords1: Array, coords2: Array, scale=None, eps: float = 0.0) -> Array:
    """
    Compute Kabsch-aligned RMSD between coordinate sets.

    Supports both single pairs (N, 3) and batched inputs (B, N, 3).

    Args:
        coords1: First coordinates, shape (N, 3) or (B, N, 3).
        coords2: Second coordinates, shape (N, 3) or (B, N, 3).
        scale: Unused (for compatibility with Polymer dispatch).
        eps: Small value added before sqrt for gradient stability.

    Returns:
        RMSD value(s). Shape () for 2D input, (B,) for 3D input.
    """
    if type(coords1) != type(coords2):
        raise TypeError(f"Both inputs must be same type, got {type(coords1).__name__} and {type(coords2).__name__}")

    if coords1.shape != coords2.shape:
        raise ValueError(f"Shape mismatch: {coords1.shape} vs {coords2.shape}")

    if coords1.ndim == 2:
        if coords1.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {coords1.shape}")
    elif coords1.ndim == 3:
        if coords1.shape[2] != 3:
            raise ValueError(f"Expected shape (B, N, 3), got {coords1.shape}")
    else:
        raise ValueError(f"Expected 2D (N, 3) or 3D (B, N, 3) array, got {coords1.ndim}D")

    # Handle both 2D and 3D by adding batch dim if needed
    squeeze = coords1.ndim == 2
    if squeeze:
        coords1 = coords1[None, :, :]  # (1, N, 3)
        coords2 = coords2[None, :, :]

    # Center coordinates: mean over atoms (axis=1)
    if is_torch(coords1):
        centroid1 = coords1.mean(dim=1, keepdim=True)
        centroid2 = coords2.mean(dim=1, keepdim=True)
    else:
        centroid1 = coords1.mean(axis=1, keepdims=True)
        centroid2 = coords2.mean(axis=1, keepdims=True)

    c1 = coords1 - centroid1  # (B, N, 3)
    c2 = coords2 - centroid2  # (B, N, 3)

    # Cross-covariance: H = c1^T @ c2, batched as (B, 3, N) @ (B, N, 3) -> (B, 3, 3)
    if is_torch(c1):
        H = c1.transpose(-2, -1) @ c2
    else:
        H = np.swapaxes(c1, -2, -1) @ c2

    # Batched SVD
    try:
        U, S, Vt = svd(H)
    except Exception as e:
        if has_nan(H) or has_inf(H):
            raise ValueError(
                "RMSD failed: covariance matrix contains NaN or infinity. "
                "Check input coordinates for invalid values."
            ) from e
        raise ValueError(
            "RMSD failed: SVD did not converge. "
            "This may indicate extreme coordinate values causing overflow."
        ) from e

    # Optimal rotation: R = V @ U^T -> (B, 3, 3)
    if is_torch(Vt):
        R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)
    else:
        R = np.swapaxes(Vt, -2, -1) @ np.swapaxes(U, -2, -1)

    # Handle reflection case (det(R) < 0)
    d = det(R)
    Vt = clone(Vt)
    if is_torch(Vt):
        mask = d < 0
        Vt[mask, -1, :] *= -1
        R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)
    else:
        mask = d < 0
        Vt[mask, -1, :] *= -1
        R = np.swapaxes(Vt, -2, -1) @ np.swapaxes(U, -2, -1)

    # Apply rotation: aligned = c1 @ R^T + centroid2
    if is_torch(R):
        aligned = c1 @ R.transpose(-2, -1) + centroid2
    else:
        aligned = c1 @ np.swapaxes(R, -2, -1) + centroid2

    # Compute RMSD
    diff = aligned - coords2
    if is_torch(diff):
        msd = (diff ** 2).mean(dim=(-2, -1))  # Mean over N and 3
    else:
        msd = (diff ** 2).mean(axis=(-2, -1))

    result = sqrt(clamp(msd, min_val=0.0) + eps)

    # Remove batch dim if input was 2D
    if squeeze:
        if is_torch(result):
            result = result.squeeze(0)
        else:
            result = result.squeeze()

    return result


def coordinate_covariance(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale",
) -> Array:
    """
    Compute coordinate covariance matrices between two polymers.

    The covariance is computed by taking the outer product of coordinates
    and reducing at the specified scale.

    Args:
        polymer1: First polymer structure.
        polymer2: Second polymer structure (must have same atom count).
        scale: Scale at which to compute covariance (e.g., MOLECULE).

    Returns:
        Array of covariance matrices, one per scale unit.
    """
    outer_prod = multiply(
        polymer1.coordinates[:, None, :],
        polymer2.coordinates[:, :, None],
    )
    return polymer1.reduce(outer_prod, scale)


def _rmsd_polymer(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale | None" = None,
    eps: float = 0.0,
) -> Array:
    """
    Compute Kabsch distance (aligned RMSD) between polymer structures.

    Uses singular value decomposition to find the optimal rotation
    that minimizes the distance between the two structures. The
    polymers must have the same number of atoms and atom ordering.

    Args:
        polymer1: First polymer structure.
        polymer2: Second polymer structure.
        scale: Scale at which to compute distance. Default is MOLECULE.
        eps: Small value added before sqrt for gradient stability.

    Returns:
        Array of RMSD values (Angstroms), one per scale unit.

    Note:
        Single-atom molecules (ions, water) produce degenerate covariance
        matrices, leading to numerical instability in the SVD. Use .poly()
        to exclude non-polymer atoms before computing RMSD if your structure
        contains such molecules.
    """
    if scale is None:
        scale = Scale.MOLECULE

    # Center both structures
    polymer1_c, _ = polymer1.center(scale)
    polymer2_c, _ = polymer2.center(scale)

    # Compute coordinate covariance
    cov = coordinate_covariance(polymer1_c, polymer2_c, scale)

    # SVD to find optimal rotation
    try:
        sigma = svdvals(cov)
    except Exception as e:
        if has_nan(cov) or has_inf(cov):
            raise ValueError(
                "RMSD failed: covariance matrix contains NaN or infinity. "
                "Check input coordinates for invalid values."
            ) from e
        raise ValueError(
            "RMSD failed: SVD did not converge. "
            "This may indicate extreme coordinate values causing overflow."
        ) from e
    cov_det = det(cov)

    # Handle reflection case
    if is_torch(sigma):
        import torch
        sigma = sigma.clone()
        sigma[cov_det < 0, -1] = -sigma[cov_det < 0, -1]
    else:
        sigma = sigma.copy()
        sigma[cov_det < 0, -1] = -sigma[cov_det < 0, -1]
    sigma = sigma.mean(-1)

    # Get variances of both point clouds
    var1 = polymer1_c.moment(2, scale).mean(-1)
    var2 = polymer2_c.moment(2, scale).mean(-1)

    # Compute Kabsch distance (RMSD)
    msd = var1 + var2 - 2 * sigma
    return sqrt(clamp(msd, min_val=0.0) + eps)


# =============================================================================
# Unified RMSD interface using singledispatch
# =============================================================================

# Type stubs for static type checking
@overload
def rmsd(a: Array, b: Array, scale: None = None, eps: float = 0.0) -> Array: ...

@overload
def rmsd(a: "Polymer", b: "Polymer", scale: "Scale | None" = None, eps: float = 0.0) -> Array: ...


@singledispatch
def _rmsd_dispatch(a, b, scale=None, eps=0.0):
    """Internal singledispatch for rmsd."""
    raise TypeError(
        f"rmsd() not supported for type {type(a).__name__}. "
        f"Expected Polymer or numpy/torch array."
    )


_rmsd_dispatch.register(np.ndarray, _rmsd_coords)

# Register torch.Tensor if available
try:
    import torch
    _rmsd_dispatch.register(torch.Tensor, _rmsd_coords)
except ImportError:
    pass


# Register Polymer type - must be done after Polymer is importable
# We use a lazy registration pattern to avoid circular imports
_polymer_registered = False


def _ensure_polymer_registered():
    """Lazily register Polymer type with rmsd dispatcher."""
    global _polymer_registered
    if _polymer_registered:
        return

    from ..polymer import Polymer
    _rmsd_dispatch.register(Polymer, _rmsd_polymer)
    _polymer_registered = True


def rmsd(
    a: "Array | Polymer",
    b: "Array | Polymer",
    scale: "Scale | None" = None,
    eps: float = 0.0,
) -> Array:
    """
    Compute Kabsch-aligned RMSD between structures or coordinate arrays.

    This function dispatches based on the input type:
    - Polymer objects: Uses optimized hierarchical computation
    - numpy/torch arrays: Direct coordinate-based computation

    Args:
        a: First structure (Polymer) or coordinates (array).
        b: Second structure (Polymer) or coordinates (array).
        scale: For Polymer inputs, the scale at which to compute RMSD
            (default: MOLECULE). Ignored for array inputs.
        eps: Small value added before sqrt for gradient stability.
            Useful when using RMSD as a training loss.

    Returns:
        For Polymer: Array of RMSD values, one per scale unit.
        For arrays (N, 3): Scalar RMSD value (0-d array).
        For arrays (B, N, 3): Array of B RMSD values.

    Note:
        For torch inputs, gradients flow through the computation.
        Use eps > 0 (e.g., 1e-8) when training to avoid gradient
        instability near RMSD = 0.

    Examples:
        >>> import ciffy
        >>> from ciffy.operations import rmsd
        >>> # Polymer RMSD
        >>> p1 = ciffy.load("struct1.cif")
        >>> p2 = ciffy.load("struct2.cif")
        >>> rmsd_val = rmsd(p1, p2)
        >>>
        >>> # Array RMSD (single pair)
        >>> coords1 = np.random.randn(100, 3)
        >>> coords2 = np.random.randn(100, 3)
        >>> rmsd_val = rmsd(coords1, coords2)
        >>>
        >>> # Batched array RMSD
        >>> batch1 = np.random.randn(32, 100, 3)
        >>> batch2 = np.random.randn(32, 100, 3)
        >>> rmsd_vals = rmsd(batch1, batch2)  # shape (32,)
        >>>
        >>> # Training with gradient stability
        >>> loss = rmsd(pred_coords, target_coords, eps=1e-8)
        >>> loss.backward()
    """
    _ensure_polymer_registered()
    return _rmsd_dispatch(a, b, scale, eps)


# =============================================================================
# Radius of Gyration
# =============================================================================

def rg(
    polymer: "Polymer",
    scale: "Scale" = Scale.MOLECULE,
) -> Array:
    """
    Compute radius of gyration (Rg) of a structure.

    Radius of gyration measures the compactness of a structure, defined as
    the root-mean-square distance of atoms from their centroid:

        Rg = sqrt(sum(|r_i - r_cm|^2) / N)

    Equivalently, Rg = sqrt(Var_x + Var_y + Var_z) = sqrt(trace(cov)).

    Args:
        polymer: The polymer structure.
        scale: Scale at which to compute Rg (default: MOLECULE).
            - MOLECULE: Single Rg for the entire structure.
            - CHAIN: Rg per chain.
            - RESIDUE: Rg per residue.

    Returns:
        Array of Rg values, one per unit at the specified scale.
        For MOLECULE scale, returns a 1-element array.

    Examples:
        >>> import ciffy
        >>> p = ciffy.load("structure.cif")
        >>> r = ciffy.rg(p)  # Single value for whole structure
        >>> rg_per_chain = ciffy.rg(p, ciffy.CHAIN)  # Per chain
    """
    coords = polymer.coordinates
    n_units = polymer.size(scale)

    # Compute centroid per unit
    centroids = polymer.reduce(coords, scale)  # (n_units, 3)

    # Expand centroids back to atom level
    centroids_expanded = polymer.expand(centroids, scale)  # (n_atoms, 3)

    # Squared distances from centroid
    diff = coords - centroids_expanded
    sq_dist = (diff * diff).sum(axis=-1)  # (n_atoms,)

    # Mean squared distance per unit
    mean_sq_dist = polymer.reduce(sq_dist, scale)  # (n_units,)

    return sqrt(mean_sq_dist)


# =============================================================================
# Clash Detection
# =============================================================================

def clashes(
    polymer: "Polymer",
    vdw_scale: float = 0.6,
    exclude_bonds: int = 3,
    heavy_only: bool = True,
    residue_cutoff: float | None = 8.0,
) -> Array:
    """
    Detect steric clashes in a polymer structure.

    A clash occurs when two atoms are closer than the sum of their
    van der Waals radii (scaled by vdw_scale), excluding atoms that
    are connected by a small number of covalent bonds.

    Args:
        polymer: Structure to check for clashes.
        vdw_scale: Scale factor for VDW radii (default 0.6 means atoms
            can overlap 40% before it's considered a clash).
        exclude_bonds: Exclude atom pairs connected by this many bonds
            or fewer (default 3 excludes 1-2, 1-3, 1-4 interactions).
        heavy_only: If True, only consider heavy (non-hydrogen) atoms.
        residue_cutoff: Pre-filter to residues with non-sequential neighbors
            within this distance (Angstroms). Set to None to disable.
            Default 8.0 significantly speeds up large structures.

    Returns:
        (C, 2) array of clashing atom index pairs (indices into filtered polymer
        if residue_cutoff is used).

    Note:
        This uses O(N²) memory for distance and exclusion matrices.
        The residue_cutoff pre-filter reduces N for large structures.

    Examples:
        >>> import ciffy
        >>> p = ciffy.load("structure.cif")
        >>> pairs = ciffy.clashes(p)
        >>> print(f"{len(pairs)} clashes found")
    """
    from ..backend.ops import cdist, triu, argwhere
    from ..backend import to_numpy

    # Optionally filter to heavy atoms
    if heavy_only:
        polymer = polymer.heavy()

    # Residue-level pre-filter: keep only residues with close non-sequential neighbors
    if residue_cutoff is not None:
        polymer = _filter_close_residues(polymer, residue_cutoff, exclude_seq=3)

    n_atoms = polymer.size()
    if n_atoms == 0:
        if is_torch(polymer.coordinates):
            import torch
            return torch.zeros((0, 2), dtype=torch.int64)
        return np.zeros((0, 2), dtype=np.int64)

    coords = polymer.coordinates
    radii = polymer.vdw_radii()

    # Pairwise distances
    dists = cdist(coords, coords)

    # VDW threshold matrix: (r_i + r_j) * scale
    thresholds = (radii[:, None] + radii[None, :]) * vdw_scale

    # Build exclusion mask via sparse neighbor expansion
    exclusion = _build_exclusion_mask(polymer.bonds, n_atoms, exclude_bonds, coords)

    # Find clashes: close AND not excluded
    is_clash = (dists < thresholds) & ~exclusion

    # Upper triangle only (avoid double-counting and self-clashes)
    is_clash = triu(is_clash, diagonal=1)
    return argwhere(is_clash)


def _filter_close_residues(
    polymer: "Polymer",
    cutoff: float,
    exclude_seq: int = 3,
) -> "Polymer":
    """
    Filter to residues that have non-sequential neighbors within cutoff.

    Args:
        polymer: Input polymer.
        cutoff: Distance threshold in Angstroms.
        exclude_seq: Exclude sequential neighbors within ±exclude_seq residues.

    Returns:
        Filtered polymer with only residues that could potentially clash.
    """
    from ..backend.ops import cdist
    from ..backend import to_numpy

    n_res = polymer.size(Scale.RESIDUE)
    if n_res == 0:
        return polymer

    # Get residue centroids
    _, centroids = polymer.center(Scale.RESIDUE)

    # Compute residue-residue distances
    res_dists = to_numpy(cdist(centroids, centroids))

    # Build sequential neighbor mask (exclude ±exclude_seq within same chain)
    chain_lengths = to_numpy(polymer.lengths)
    chain_boundaries = np.cumsum(np.concatenate([[0], chain_lengths]))

    # Map residue -> chain
    chain_membership = np.zeros(n_res, dtype=np.int32)
    for c_idx in range(len(chain_lengths)):
        start, end = chain_boundaries[c_idx], chain_boundaries[c_idx + 1]
        chain_membership[start:end] = c_idx

    # Create sequential neighbor mask
    seq_neighbor = np.zeros((n_res, n_res), dtype=bool)
    for offset in range(-exclude_seq, exclude_seq + 1):
        if offset == 0:
            np.fill_diagonal(seq_neighbor, True)
        else:
            idx = np.arange(max(0, -offset), min(n_res, n_res - offset))
            idx_offset = idx + offset
            same_chain = chain_membership[idx] == chain_membership[idx_offset]
            seq_neighbor[idx[same_chain], idx_offset[same_chain]] = True

    # Find residues with close non-sequential neighbors
    close = (res_dists < cutoff) & ~seq_neighbor
    has_close_neighbor = close.any(axis=1)

    # Filter polymer
    if has_close_neighbor.all():
        return polymer
    return polymer.select(has_close_neighbor, Scale.RESIDUE)


def _build_exclusion_mask(
    bonds: Array,
    n_atoms: int,
    max_bonds: int,
    ref: Array,
) -> Array:
    """
    Build boolean exclusion mask for atoms within max_bonds of each other.

    Uses sparse neighbor expansion: O(n × degree^max_bonds) instead of O(n³).

    Args:
        bonds: (B, 2) bond pairs.
        n_atoms: Total number of atoms.
        max_bonds: Maximum bond distance to exclude.
        ref: Reference array for backend matching.

    Returns:
        (n_atoms, n_atoms) boolean mask where True = within max_bonds.
    """
    from ..backend import to_numpy, is_torch
    from ..backend.ops import zeros

    # Work in numpy for the sparse expansion, convert at end
    bonds_np = to_numpy(bonds)

    # Build adjacency list (CSR-like)
    # neighbors[i] = list of atoms bonded to i
    neighbors = [[] for _ in range(n_atoms)]
    for i, j in bonds_np:
        neighbors[i].append(j)
        neighbors[j].append(i)

    # For each atom, find all atoms within max_bonds
    # reachable[i] = set of atoms reachable from i within max_bonds
    exclusion = np.zeros((n_atoms, n_atoms), dtype=bool)

    for atom in range(n_atoms):
        # BFS up to max_bonds depth
        visited = {atom}
        frontier = {atom}
        for _ in range(max_bonds):
            next_frontier = set()
            for node in frontier:
                for neighbor in neighbors[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        # Mark all visited atoms as excluded
        for other in visited:
            exclusion[atom, other] = True

    # Convert to target backend
    if is_torch(ref):
        import torch
        return torch.from_numpy(exclusion).to(ref.device)
    return exclusion
