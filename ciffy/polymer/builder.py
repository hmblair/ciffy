"""
Chain building utilities for polymer construction and generative models.

This module provides functions for:

- **Residue expansion**: Get atom data with terminal filtering (expand_residue)
- **Frame resolution**: Resolve linking frames for positioning (_resolve_frame_indices)
- **Chain assembly**: Batch-assemble chains from coordinates and transforms (assemble_chain)

The chain assembly uses cumulative SE(3) transforms for 10-20x speedup over
sequential positioning, with optional torch.compile for additional GPU acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ..biochemistry import Molecule, Residue, atom_to_element
from ..biochemistry.linking import LINKING_BY_TYPE, LinkingDefinition, NUCLEIC_ACID_LINK, PEPTIDE_LINK
from ..backend import Array, is_torch, zeros_like, bmm, cat, stack, transpose, empty
from ..geometry import rodrigues
from ..utils import atoms_to_col_map

if TYPE_CHECKING:
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class MoleculeConfig:
    """
    Configuration for a molecule type.

    Bundles molecule-specific parameters for chain building.
    """
    linking: LinkingDefinition | None
    start_terminal_atoms: frozenset[str]  # 5'/N-terminal only
    end_terminal_atoms: frozenset[str]    # 3'/C-terminal only


@dataclass
class FrameIndices:
    """
    Pre-resolved frame column indices for a residue type.

    Uses arrays with -1 sentinel for missing perp_ref, following ciffy's
    philosophy of arrays over Python data types. This enables efficient
    frame computation without Python attribute lookups.

    Attributes:
        prev_cols: shape (3,) int32 array [origin, z_ref, perp_ref] for outgoing frame.
            -1 indicates missing perp_ref.
        prev_z_toward: If True, Z points from z_ref toward origin.
        next_cols: shape (3,) int32 array [origin, z_ref, perp_ref] for incoming frame.
            -1 indicates missing perp_ref.
        next_z_toward: If True, Z points from z_ref toward origin.
    """
    prev_cols: np.ndarray   # shape (3,), dtype=int32, -1 = no perp_ref
    prev_z_toward: bool
    next_cols: np.ndarray   # shape (3,), dtype=int32, -1 = no perp_ref
    next_z_toward: bool


# =============================================================================
# MOLECULE CONFIGURATIONS
# =============================================================================

_MOLECULE_CONFIGS: dict[Molecule, MoleculeConfig] = {
    Molecule.RNA: MoleculeConfig(
        linking=NUCLEIC_ACID_LINK,
        start_terminal_atoms=frozenset({'OP3', 'HOP3'}),
        end_terminal_atoms=frozenset({'HO3p'}),
    ),
    Molecule.DNA: MoleculeConfig(
        linking=NUCLEIC_ACID_LINK,
        start_terminal_atoms=frozenset({'OP3', 'HOP3'}),
        end_terminal_atoms=frozenset({'HO3p'}),
    ),
    Molecule.PROTEIN: MoleculeConfig(
        linking=PEPTIDE_LINK,
        start_terminal_atoms=frozenset({'H2', 'H3'}),
        end_terminal_atoms=frozenset({'OXT', 'HXT'}),
    ),
}


def get_molecule_config(mol_type: Molecule) -> MoleculeConfig:
    """Get configuration for a molecule type."""
    if mol_type not in _MOLECULE_CONFIGS:
        raise ValueError(f"Unsupported molecule type: {mol_type.name}")
    return _MOLECULE_CONFIGS[mol_type]


# =============================================================================
# RESIDUE EXPANSION (cached ideal coordinates and atom data)
# =============================================================================

@lru_cache(maxsize=64)
def _expand_residue_cached(residue_idx: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], np.ndarray]:
    """
    Get cached atom data for a residue type.

    Returns tuple of (atom_indices, element_indices, atom_names, ideal_coords).
    The numpy array is cached; callers should copy if mutating.
    """
    try:
        residue = Residue.from_index(residue_idx)
    except (ValueError, KeyError):
        raise ValueError(f"Invalid residue index: {residue_idx}")

    if residue.atoms is None:
        raise ValueError(f"No atom definitions for residue {residue.name}")

    atom_indices = []
    element_indices = []
    atom_names = []

    for member in residue.atoms:
        atom_indices.append(member.value)
        atom_names.append(member.name)
        element_indices.append(atom_to_element(member))

    return tuple(atom_indices), tuple(element_indices), tuple(atom_names), residue.ideal


def expand_residue(
    residue: Residue,
    start_terminal: bool = True,
    end_terminal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get atom data for a residue type with optional terminal filtering.

    Args:
        residue: Residue type.
        start_terminal: If True, include 5'/N-terminal atoms (OP3, HOP3 for RNA).
        end_terminal: If True, include 3'/C-terminal atoms (HO3' for RNA).

    Returns:
        Tuple of (atoms, elements, coords) as numpy arrays.

    Example:
        >>> # Internal residue (no terminal atoms)
        >>> atoms, elements, coords = expand_residue(Residue.A, start_terminal=False, end_terminal=False)
        >>> # First residue (5' terminal only)
        >>> atoms, elements, coords = expand_residue(Residue.A, start_terminal=True, end_terminal=False)
        >>> # Last residue (3' terminal only)
        >>> atoms, elements, coords = expand_residue(Residue.U, start_terminal=False, end_terminal=True)
    """
    atom_indices, element_indices, atom_names, coords = _expand_residue_cached(residue.value)

    # Check if filtering is needed
    if start_terminal and end_terminal:
        # No filtering - return all atoms as arrays
        return (
            np.array(atom_indices, dtype=np.int64),
            np.array(element_indices, dtype=np.int64),
            coords.copy(),
        )

    # Get molecule config for terminal atoms
    mol_type = residue.molecule_type
    if mol_type not in _MOLECULE_CONFIGS:
        # Unknown molecule type - return all atoms
        return (
            np.array(atom_indices, dtype=np.int64),
            np.array(element_indices, dtype=np.int64),
            coords.copy(),
        )

    config = _MOLECULE_CONFIGS[mol_type]

    # Build set of atoms to exclude
    exclude: set[str] = set()
    if not start_terminal:
        exclude.update(config.start_terminal_atoms)
    if not end_terminal:
        exclude.update(config.end_terminal_atoms)

    if not exclude:
        # Nothing to exclude
        return (
            np.array(atom_indices, dtype=np.int64),
            np.array(element_indices, dtype=np.int64),
            coords.copy(),
        )

    # Filter atoms
    keep_indices = [i for i, name in enumerate(atom_names) if name not in exclude]

    return (
        np.array([atom_indices[i] for i in keep_indices], dtype=np.int64),
        np.array([element_indices[i] for i in keep_indices], dtype=np.int64),
        coords[keep_indices].copy(),
    )


def linear_extend_transform(
    prev_coords: Array,
    prev_atoms: Array,
    prev_residue: Residue,
    next_atoms: Array,
    next_residue: Residue,
) -> np.ndarray:
    """
    Calculate SE(3) transform for linear chain extension.

    Computes the transform that positions the next residue along the backbone
    axis with proper spacing, maintaining the same orientation as the previous
    residue.

    Args:
        prev_coords: (n_atoms, 3) coordinates of previous residue.
        prev_atoms: Atom type indices of previous residue.
        prev_residue: Previous residue type.
        next_atoms: Atom type indices of next residue.
        next_residue: Next residue type.

    Returns:
        (6,) SE(3) transform [axis-angle (3), translation (3)] as numpy array.
        The axis-angle is [0, 0, 0] (no rotation), and translation is
        [0, 0, spacing] where spacing is the backbone span + bond length.

    Example:
        >>> atoms1, elements1, coords1 = expand_residue(Residue.A)
        >>> atoms2, elements2, coords2 = expand_residue(Residue.C, start_terminal=False)
        >>> transform = linear_extend_transform(coords1, atoms1, Residue.A, atoms2, Residue.C)
        >>> # Use with Polymer.extend() or position_residue_fast()
    """
    from ..biochemistry.linking import LINKING_BY_TYPE

    link_def = LINKING_BY_TYPE.get(prev_residue.molecule_type)

    if link_def is None:
        # No linking definition - use default spacing
        spacing = 6.0
    else:
        # Build atom_to_col mappings
        prev_atom_to_col = atoms_to_col_map(tuple(int(a) for a in prev_atoms))
        next_atom_to_col = atoms_to_col_map(tuple(int(a) for a in next_atoms))

        # Get linking atoms
        prev_link_atom = getattr(prev_residue, link_def.prev_atom)  # e.g., O3' for RNA
        next_link_atom = getattr(next_residue, link_def.next_atom)  # e.g., P for RNA

        # Get P position of previous residue to calculate backbone span
        prev_p_atom = getattr(prev_residue, link_def.next_atom)  # P atom

        if prev_link_atom.value in prev_atom_to_col and prev_p_atom.value in prev_atom_to_col:
            prev_link_pos = prev_coords[prev_atom_to_col[prev_link_atom.value]]
            prev_p_pos = prev_coords[prev_atom_to_col[prev_p_atom.value]]
            # Backbone span is distance from P to O3' plus bond length
            backbone_span = float(np.linalg.norm(prev_link_pos - prev_p_pos))
            spacing = backbone_span + link_def.bond_length
        else:
            # Missing atoms - use default spacing
            spacing = 6.0

    # Return identity rotation + Z-axis translation
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, spacing], dtype=np.float32)


# =============================================================================
# FRAME RESOLUTION (cached for fast positioning)
# =============================================================================

def _infer_link_definition(atom_set: set[int]) -> LinkingDefinition | None:
    """
    Infer the linking definition from present backbone atoms.

    This enables robust frame resolution for modified residues by detecting
    the polymer type from which backbone atoms are present.

    Args:
        atom_set: Set of atom type values present in the residue.

    Returns:
        LinkingDefinition for nucleic acid or peptide, or None if neither.
    """
    from ..biochemistry.linking import BACKBONE_ATOM_VALUES

    # Check for nucleic acid backbone (P, O3', C3', etc.)
    nucleic_required = {
        BACKBONE_ATOM_VALUES["P"],
        BACKBONE_ATOM_VALUES["O3p"],
        BACKBONE_ATOM_VALUES["C3p"],
    }
    if nucleic_required.issubset(atom_set):
        return NUCLEIC_ACID_LINK

    # Check for protein backbone (N, CA, C)
    protein_required = {
        BACKBONE_ATOM_VALUES["N"],
        BACKBONE_ATOM_VALUES["CA"],
        BACKBONE_ATOM_VALUES["C"],
    }
    if protein_required.issubset(atom_set):
        return PEPTIDE_LINK

    return None


@lru_cache(maxsize=256)
def _resolve_frame_indices(residue_idx: int, atom_indices: tuple[int, ...]) -> FrameIndices:
    """
    Resolve frame column indices for a (residue_type, atom_subset) pair.

    This function uses unified backbone atom values to resolve frames,
    making it robust to modified residues with standard backbones.
    The residue_idx is used as a cache key and for fallback lookup,
    but frame resolution uses the fixed backbone values directly.

    Args:
        residue_idx: Residue enum value (int), used for caching and fallback.
        atom_indices: Tuple of atom type values in the residue's coordinate array.

    Returns:
        FrameIndices with pre-resolved column indices for both incoming and
        outgoing frames.

    Raises:
        ValueError: If required linking atoms are missing from atom_indices.
    """
    atom_to_col = atoms_to_col_map(atom_indices)
    atom_set = set(atom_indices)

    # First, try to infer linking type from backbone atoms present
    # This works for modified residues not in the whitelist
    link_def = _infer_link_definition(atom_set)

    # Fallback: use residue enum if inference failed
    if link_def is None:
        try:
            residue = Residue.from_index(residue_idx)
            link_def = LINKING_BY_TYPE.get(residue.molecule_type)
        except (ValueError, KeyError):
            pass  # Modified residue not in whitelist

    if link_def is None:
        # Non-polymer residue (ligand, etc.) - no linking frames
        return FrameIndices(
            prev_cols=np.array([-1, -1, -1], dtype=np.int32),
            prev_z_toward=True,
            next_cols=np.array([-1, -1, -1], dtype=np.int32),
            next_z_toward=True,
        )

    # Use value-based resolution (works for any residue with standard backbone)
    try:
        prev_tuple = link_def.prev_frame.resolve_by_value(atom_to_col)
        next_tuple = link_def.next_frame.resolve_by_value(atom_to_col)
    except KeyError as e:
        raise ValueError(f"Missing backbone atom for frame resolution: {e}")

    prev_cols = np.array([
        prev_tuple[0],
        prev_tuple[1],
        prev_tuple[2] if prev_tuple[2] is not None else -1,
    ], dtype=np.int32)

    next_cols = np.array([
        next_tuple[0],
        next_tuple[1],
        next_tuple[2] if next_tuple[2] is not None else -1,
    ], dtype=np.int32)

    return FrameIndices(
        prev_cols=prev_cols,
        prev_z_toward=link_def.prev_frame.z_toward_origin,
        next_cols=next_cols,
        next_z_toward=link_def.next_frame.z_toward_origin,
    )


# =============================================================================
# CHAIN ASSEMBLY
# =============================================================================


def _cumulative_matmul(matrices: Array) -> Array:
    """
    Compute cumulative matrix product: M[0], M[0]@M[1], M[0]@M[1]@M[2], ...

    Args:
        matrices: (n, 4, 4) homogeneous transformation matrices.

    Returns:
        (n, 4, 4) cumulative products.
    """
    n = len(matrices)

    # Use list accumulation to avoid in-place ops (preserves autograd)
    results = [matrices[0]]
    for i in range(1, n):
        results.append(results[i - 1] @ matrices[i])

    # Stack into tensor
    if is_torch(matrices):
        import torch
        return torch.stack(results, dim=0)
    else:
        import numpy as np
        return np.stack(results, axis=0)


# Lazy-initialized compiled version for GPU acceleration
_cumulative_matmul_compiled = None


def _get_cumulative_matmul_compiled():
    """Get the torch.compiled version of cumulative matmul (lazy init)."""
    global _cumulative_matmul_compiled
    if _cumulative_matmul_compiled is None:
        import torch
        _cumulative_matmul_compiled = torch.compile(
            _cumulative_matmul,
            mode="reduce-overhead",
        )
    return _cumulative_matmul_compiled


def _normalize_axis_angles(axis_angles: Array) -> Array:
    """
    Normalize axis-angles to keep rotation angles in [-π, π].

    Large rotation angles (near 180° or beyond) cause numerical instability
    in the Rodrigues formula and gradient explosion. This function wraps
    angles to the principal range while preserving the rotation direction.

    Args:
        axis_angles: (n, 3) axis-angle vectors.

    Returns:
        (n, 3) normalized axis-angle vectors with angles in [-π, π].
    """
    import math

    if is_torch(axis_angles):
        import torch
        # Compute angle magnitudes
        angles = torch.norm(axis_angles, dim=1, keepdim=True)  # (n, 1)

        # Avoid division by zero
        safe_angles = torch.where(angles < 1e-8, torch.ones_like(angles), angles)
        axes = axis_angles / safe_angles  # Unit axes

        # Wrap angles to [-π, π]
        angles_wrapped = torch.remainder(angles + math.pi, 2 * math.pi) - math.pi

        # Reconstruct axis-angles
        return axes * angles_wrapped
    else:
        # NumPy path
        angles = np.linalg.norm(axis_angles, axis=1, keepdims=True)
        safe_angles = np.where(angles < 1e-8, np.ones_like(angles), angles)
        axes = axis_angles / safe_angles
        angles_wrapped = np.remainder(angles + math.pi, 2 * math.pi) - math.pi
        return axes * angles_wrapped


def _apply_cumulative_transforms(
    coords: Array,
    transforms: Array,
    compile: bool = False,
) -> Array:
    """
    Apply cumulative SE(3) transforms to batched coordinates.

    Args:
        coords: (n_residues, n_atoms, 3) coordinate arrays.
        transforms: (n_residues, 6) SE(3) transforms [axis-angle, translation].
        compile: If True and using CUDA, use torch.compile for speedup.

    Returns:
        (n_residues, n_atoms, 3) transformed coordinates.
    """
    n_residues = len(transforms)
    n_atoms = coords.shape[1]

    # Normalize axis-angles to avoid rodrigues singularities
    axis_angles = _normalize_axis_angles(transforms[:, :3])

    # Build SE(3) matrices: rotation from Rodrigues, translation direct
    Rs = rodrigues(axis_angles)

    # Create (n, 4, 4) homogeneous transform matrices
    T = empty((n_residues, 4, 4), like=coords)
    T[:] = 0
    T[:, :3, :3] = Rs
    T[:, :3, 3] = transforms[:, 3:]
    T[:, 3, 3] = 1.0

    # Cumulative product of transforms
    if compile and is_torch(coords) and coords.is_cuda:
        T_cumul = _get_cumulative_matmul_compiled()(T)
    else:
        T_cumul = _cumulative_matmul(T)

    # Build homogeneous coordinates (n, n_atoms, 4)
    ones = empty((n_residues, n_atoms, 1), like=coords)
    ones[:] = 1.0
    coords_h = cat([coords, ones], axis=2)

    # Apply batched transform: (n, 4, 4) @ (n, 4, n_atoms) -> (n, 4, n_atoms)
    coords_h_t = transpose(coords_h, (0, 2, 1))  # (n, n_atoms, 4) -> (n, 4, n_atoms)
    result_h_t = bmm(T_cumul, coords_h_t)        # (n, 4, n_atoms)
    result_h = transpose(result_h_t, (0, 2, 1))  # (n, n_atoms, 4)

    return result_h[:, :, :3]


def _validate_assembly_inputs(
    residue_coords: list[Array],
    transforms: list[Array],
) -> None:
    """
    Validate inputs to assemble_chain.

    Raises:
        ValueError: If inputs have invalid shapes or mismatched backends.
    """
    n_residues = len(residue_coords)

    if len(transforms) != n_residues:
        raise ValueError(
            f"Length mismatch: got {n_residues} coordinate arrays "
            f"but {len(transforms)} transforms"
        )

    for i, coords in enumerate(residue_coords):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"residue_coords[{i}] has shape {coords.shape}, expected (n_atoms, 3)"
            )

    for i, t in enumerate(transforms):
        if t.shape != (6,):
            raise ValueError(
                f"transforms[{i}] has shape {t.shape}, expected (6,)"
            )

    first_is_torch = is_torch(residue_coords[0])
    for i, coords in enumerate(residue_coords[1:], 1):
        if is_torch(coords) != first_is_torch:
            backend_0 = "torch" if first_is_torch else "numpy"
            backend_i = "torch" if is_torch(coords) else "numpy"
            raise ValueError(
                f"Mixed backends: residue_coords[0] is {backend_0} "
                f"but residue_coords[{i}] is {backend_i}"
            )


def assemble_chain(
    residue_coords: list[Array],
    transforms: list[Array],
    *,
    compile: bool = False,
) -> Array:
    """
    Assemble a chain from per-residue coordinates and transforms.

    Uses cumulative SE(3) transforms for 10-20x speedup over sequential positioning.
    Coordinates are padded to uniform size, transformed in batch, then unpadded.

    Args:
        residue_coords: List of (n_atoms, 3) coordinate arrays per residue,
            in their canonical/local frame.
        transforms: List of (6,) SE(3) transforms [axis-angle, translation].
            Transform[i] positions residue i relative to residue i-1.
        compile: If True and using CUDA, use torch.compile for ~5x additional speedup.

    Returns:
        (N, 3) concatenated positioned coordinates for the entire chain.

    Example:
        >>> coords = [model.decode(z)[0] for z in latents]
        >>> transforms = [model.decode(z)[1] for z in latents]
        >>> positioned = assemble_chain(coords, transforms, compile=True)
    """
    if len(residue_coords) == 0:
        return np.empty((0, 3), dtype=np.float32)

    _validate_assembly_inputs(residue_coords, transforms)

    atom_counts = [c.shape[0] for c in residue_coords]
    max_atoms = max(atom_counts)
    uniform_size = all(c == max_atoms for c in atom_counts)

    n_residues = len(residue_coords)

    # Pad (if needed) and stack coordinates
    if uniform_size:
        coords_padded = stack(residue_coords)
    else:
        coords_padded = empty((n_residues, max_atoms, 3), like=residue_coords[0])
        coords_padded[:] = 0
        for i, (coords, n) in enumerate(zip(residue_coords, atom_counts)):
            coords_padded[i, :n] = coords

    # Stack transforms
    transforms_stacked = stack(transforms)

    # Apply cumulative transforms (backend-agnostic)
    result_padded = _apply_cumulative_transforms(
        coords_padded, transforms_stacked, compile=compile
    )

    # Unpad (if needed) and concatenate
    if uniform_size:
        return result_padded.reshape(-1, 3)
    else:
        result_list = [result_padded[i, :n] for i, n in enumerate(atom_counts)]
        return cat(result_list, axis=0)

