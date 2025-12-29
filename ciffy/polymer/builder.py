"""
Utilities for chain building.

Provides:
- expand_residue(): Get atom arrays with terminal filtering
- linear_extend_transform(): Compute SE(3) transform for ideal chain extension
- assemble_chain(): Position coordinates for ML models (returns coords only)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ..biochemistry import Scale, Molecule, Residue, atom_to_element
from ..biochemistry.linking import LINKING_BY_TYPE, LinkingDefinition, NUCLEIC_ACID_LINK, PEPTIDE_LINK
from ..backend import Array, is_torch
from ..utils import atoms_to_col_map

if TYPE_CHECKING:
    from ..biochemistry.atom import AtomGroup


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
# CHAIN ASSEMBLY (standalone function for generative models)
# =============================================================================

# =============================================================================
# CHAIN ASSEMBLY - OPTIMIZED WITH CUMULATIVE TRANSFORMS
# =============================================================================


def _rodrigues_torch(axis_angles: "torch.Tensor") -> "torch.Tensor":
    """
    Convert axis-angle vectors to rotation matrices using Rodrigues formula.

    Args:
        axis_angles: (n, 3) axis-angle rotation vectors.

    Returns:
        (n, 3, 3) rotation matrices.
    """
    import torch

    n = len(axis_angles)
    device = axis_angles.device
    dtype = axis_angles.dtype

    angles = torch.norm(axis_angles, dim=1, keepdim=True)
    safe_angles = torch.where(angles < 1e-8, torch.ones_like(angles), angles)
    axes = axis_angles / safe_angles

    # Skew-symmetric matrices K
    K = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1)
    sin_a = torch.sin(angles).unsqueeze(-1)
    cos_a = torch.cos(angles).unsqueeze(-1)

    return eye + sin_a * K + (1 - cos_a) * (K @ K)


def _rodrigues_numpy(axis_angles: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle vectors to rotation matrices using Rodrigues formula.

    Args:
        axis_angles: (n, 3) axis-angle rotation vectors.

    Returns:
        (n, 3, 3) rotation matrices.
    """
    n = len(axis_angles)

    angles = np.linalg.norm(axis_angles, axis=1, keepdims=True)
    safe_angles = np.where(angles < 1e-8, 1.0, angles)
    axes = axis_angles / safe_angles

    # Skew-symmetric matrices K
    K = np.zeros((n, 3, 3), dtype=axis_angles.dtype)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    eye = np.eye(3, dtype=axis_angles.dtype)[None, :, :].repeat(n, axis=0)
    sin_a = np.sin(angles)[:, :, None]
    cos_a = np.cos(angles)[:, :, None]

    return eye + sin_a * K + (1 - cos_a) * (K @ K)


def _cumulative_matmul(matrices: Array) -> Array:
    """
    Compute cumulative matrix product: M[0], M[0]@M[1], M[0]@M[1]@M[2], ...

    Args:
        matrices: (n, 4, 4) homogeneous transformation matrices.

    Returns:
        (n, 4, 4) cumulative products.
    """
    n = len(matrices)

    if is_torch(matrices):
        import torch
        result = torch.zeros_like(matrices)
    else:
        result = np.zeros_like(matrices)

    result[0] = matrices[0]
    for i in range(1, n):
        result[i] = result[i - 1] @ matrices[i]

    return result


# Compiled version for GPU (lazy initialization)
_cumulative_matmul_compiled = None


def _get_compiled_cumulative_matmul():
    """Get or create the compiled cumulative matmul function."""
    global _cumulative_matmul_compiled
    if _cumulative_matmul_compiled is None:
        import torch
        _cumulative_matmul_compiled = torch.compile(
            _cumulative_matmul,
            mode="reduce-overhead",
        )
    return _cumulative_matmul_compiled


def _ensure_compiled():
    """Ensure the compiled cumulative matmul is initialized."""
    global _cumulative_matmul_compiled
    if _cumulative_matmul_compiled is None:
        _cumulative_matmul_compiled = _get_compiled_cumulative_matmul()


def _apply_cumulative_transforms_torch(
    coords: "torch.Tensor",
    transforms: "torch.Tensor",
    compile: bool = False,
) -> "torch.Tensor":
    """Apply cumulative SE(3) transforms to batched coordinates (torch path)."""
    import torch

    n_residues = len(transforms)

    # Build SE(3) matrices
    Rs = _rodrigues_torch(transforms[:, :3])
    T = torch.zeros(n_residues, 4, 4, device=coords.device, dtype=coords.dtype)
    T[:, :3, :3] = Rs
    T[:, :3, 3] = transforms[:, 3:]
    T[:, 3, 3] = 1.0

    # Cumulative product
    if compile and coords.is_cuda:
        _ensure_compiled()
        T_cumul = _cumulative_matmul_compiled(T)
    else:
        T_cumul = _cumulative_matmul(T)

    # Apply to all coordinates: (n, 4, 4) @ (n, n_atoms, 4).T
    n_atoms = coords.shape[1]
    ones = torch.ones(n_residues, n_atoms, 1, device=coords.device, dtype=coords.dtype)
    coords_h = torch.cat([coords, ones], dim=2)

    # Batch transform
    result_h = torch.bmm(T_cumul, coords_h.transpose(1, 2)).transpose(1, 2)
    return result_h[:, :, :3]


def _apply_cumulative_transforms_numpy(
    coords: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    """Apply cumulative SE(3) transforms to batched coordinates (numpy path)."""
    n_residues = len(transforms)

    # Build SE(3) matrices
    Rs = _rodrigues_numpy(transforms[:, :3])
    T = np.zeros((n_residues, 4, 4), dtype=coords.dtype)
    T[:, :3, :3] = Rs
    T[:, :3, 3] = transforms[:, 3:]
    T[:, 3, 3] = 1.0

    # Cumulative product
    T_cumul = _cumulative_matmul(T)

    # Apply to all coordinates
    n_atoms = coords.shape[1]
    ones = np.ones((n_residues, n_atoms, 1), dtype=coords.dtype)
    coords_h = np.concatenate([coords, ones], axis=2)

    # Batch transform using einsum
    result_h = np.einsum('nij,nmj->nmi', T_cumul, coords_h)
    return result_h[:, :, :3]


def assemble_chain(
    residue_coords: list[Array],
    transforms: list[Array],
    residues: list[Residue] | None = None,
    atom_subsets: list[tuple[int, ...]] | None = None,
    *,
    compile: bool = False,
) -> Array:
    """
    Assemble a chain from per-residue coordinates and transforms.

    Uses optimized cumulative transform computation (10-20x faster than
    sequential positioning). Coordinates are padded to uniform size if needed,
    then transformed in batch, then unpadded.

    For GPU tensors, set compile=True for additional 5-7x speedup via
    kernel fusion (only effective on CUDA).

    Args:
        residue_coords: List of (n_atoms, 3) coordinate arrays per residue.
            These are the decoded/predicted coordinates in canonical frame.
        transforms: List of (6,) SE(3) transforms per residue [axis-angle, translation].
            Transform[i] positions residue i relative to residue i-1.
        residues: (Unused, kept for API compatibility) List of Residue enum values.
        atom_subsets: (Unused, kept for API compatibility) List of atom index tuples.
        compile: If True and using CUDA tensors, use torch.compile for speedup.

    Returns:
        (N, 3) concatenated positioned coordinates for the entire chain.

    Example:
        >>> residue_coords = [model.decode(latent)[0] for latent in latents]
        >>> transforms = [model.decode(latent)[1] for latent in latents]
        >>> positioned = assemble_chain(residue_coords, transforms, compile=True)
    """
    if len(residue_coords) == 0:
        if is_torch(residue_coords):
            import torch
            return torch.empty(0, 3)
        return np.empty((0, 3), dtype=np.float32)

    # Input validation
    n_residues = len(residue_coords)
    if len(transforms) != n_residues:
        raise ValueError(
            f"Length mismatch: got {n_residues} coordinate arrays "
            f"but {len(transforms)} transforms"
        )

    # Validate coordinate shapes
    for i, coords in enumerate(residue_coords):
        if coords.ndim != 2:
            raise ValueError(
                f"residue_coords[{i}] has shape {coords.shape}, "
                f"expected (n_atoms, 3)"
            )
        if coords.shape[1] != 3:
            raise ValueError(
                f"residue_coords[{i}] has shape {coords.shape}, "
                f"expected (n_atoms, 3)"
            )

    # Validate transform shapes
    for i, t in enumerate(transforms):
        if t.shape != (6,):
            raise ValueError(
                f"transforms[{i}] has shape {t.shape}, expected (6,)"
            )

    # Check consistent backend
    first_is_torch = is_torch(residue_coords[0])
    for i, coords in enumerate(residue_coords[1:], 1):
        if is_torch(coords) != first_is_torch:
            raise ValueError(
                f"Mixed backends: residue_coords[0] is "
                f"{'torch' if first_is_torch else 'numpy'} but "
                f"residue_coords[{i}] is {'torch' if is_torch(coords) else 'numpy'}"
            )

    # Get atom counts per residue
    atom_counts = [c.shape[0] for c in residue_coords]
    max_atoms = max(atom_counts)
    use_torch = first_is_torch

    if use_torch:
        import torch

        device = residue_coords[0].device
        dtype = residue_coords[0].dtype

        # Pad and stack coordinates
        if all(c == max_atoms for c in atom_counts):
            # Uniform size - no padding needed
            coords_padded = torch.stack(residue_coords)
            needs_unpad = False
        else:
            # Variable sizes - pad to max
            coords_padded = torch.zeros(n_residues, max_atoms, 3, device=device, dtype=dtype)
            for i, (coords, n) in enumerate(zip(residue_coords, atom_counts)):
                coords_padded[i, :n] = coords
            needs_unpad = True

        # Stack transforms
        transforms_stacked = torch.stack(transforms)

        # Apply cumulative transforms
        result_padded = _apply_cumulative_transforms_torch(
            coords_padded, transforms_stacked, compile=compile
        )

        # Unpad and concatenate
        if needs_unpad:
            result_list = [result_padded[i, :n] for i, n in enumerate(atom_counts)]
            return torch.cat(result_list, dim=0)
        else:
            return result_padded.reshape(-1, 3)

    else:
        # NumPy path
        dtype = residue_coords[0].dtype

        # Pad and stack coordinates
        if all(c == max_atoms for c in atom_counts):
            coords_padded = np.stack(residue_coords)
            needs_unpad = False
        else:
            coords_padded = np.zeros((n_residues, max_atoms, 3), dtype=dtype)
            for i, (coords, n) in enumerate(zip(residue_coords, atom_counts)):
                coords_padded[i, :n] = coords
            needs_unpad = True

        # Stack transforms
        transforms_stacked = np.stack(transforms)

        # Apply cumulative transforms
        result_padded = _apply_cumulative_transforms_numpy(coords_padded, transforms_stacked)

        # Unpad and concatenate
        if needs_unpad:
            result_list = [result_padded[i, :n] for i, n in enumerate(atom_counts)]
            return np.concatenate(result_list, axis=0)
        else:
            return result_padded.reshape(-1, 3)

