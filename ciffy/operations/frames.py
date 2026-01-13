"""
Frame and transform operations on polymer structures.

Primary API:
- decompose(polymer, source, target) -> Transforms
- compose(polymer, transforms) -> Polymer

The Transforms dataclass carries frame metadata, so compose doesn't
need separate frame arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..backend import Array
from ..backend import ops
from ..biochemistry import Scale

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..biochemistry.linking import FrameDefinition


# =============================================================================
# Core API
# =============================================================================


@dataclass
class Transforms:
    """Inter-residue SE(3) transforms with frame metadata.

    Attributes:
        data: (n_residues, 7) array of [quaternion(4), translation(3)].
            transforms[0] is identity (first residue has no predecessor).
            transforms[i] describes residue i relative to residue i-1.
        source: Frame definition for residue i-1 (where transform originates).
        target: Frame definition for residue i (where transform lands).

    Example:
        >>> transforms = operations.decompose(polymer)
        >>> rebuilt = operations.compose(polymer, transforms)
    """

    data: Array
    source: "FrameDefinition"
    target: "FrameDefinition"

    def __len__(self) -> int:
        return self.data.shape[0]


def decompose(
    polymer: "Polymer",
    source: "FrameDefinition | None" = None,
    target: "FrameDefinition | None" = None,
) -> Transforms:
    """Extract inter-residue SE(3) transforms from a polymer.

    Args:
        polymer: Single-chain polymer structure.
        source: Frame on residue i-1 where transform originates.
            Defaults to O3P_FRAME (O3' atom) for nucleic acids.
        target: Frame on residue i where transform lands.
            Defaults to P_FRAME (P atom) for nucleic acids.

    Returns:
        Transforms object containing:
        - data: (n_residues, 7) transforms as [quaternion, translation]
        - source/target: The frame definitions used

    Example:
        >>> from ciffy import operations
        >>> transforms = operations.decompose(polymer)
        >>> # transforms.data[i] positions residue i relative to i-1
    """
    from ..biochemistry.linking import O3P_FRAME, P_FRAME
    from ..geometry.transforms import frame_from_positions, rotation_matrix_to_quaternion

    # Default frames for nucleic acids
    if source is None:
        source = O3P_FRAME
    if target is None:
        target = P_FRAME

    n_residues = polymer.size(Scale.RESIDUE)

    # Get frame positions using gather
    src_positions = gather(polymer, [source.origin, source.axis_ref, source.plane_ref])
    src_origins, src_Rs = frame_from_positions(src_positions)

    if source is target:
        tgt_origins, tgt_Rs = src_origins, src_Rs
    else:
        tgt_positions = gather(polymer, [target.origin, target.axis_ref, target.plane_ref])
        tgt_origins, tgt_Rs = frame_from_positions(tgt_positions)

    # Compute transforms between consecutive residues
    if n_residues < 2:
        data = ops.zeros_nd((n_residues, 7), like=polymer.coordinates)
        if n_residues == 1:
            # Set identity quaternion for single residue
            if hasattr(data, 'clone'):
                data = data.clone()
                data[0, 0] = 1.0
            else:
                data = data.copy()
                data[0, 0] = 1.0
        return Transforms(data=data, source=source, target=target)

    # Relative rotation: R_rel = R_src[i-1].T @ R_tgt[i]
    src_Rs_prev = src_Rs[:-1]
    tgt_Rs_curr = tgt_Rs[1:]
    R_rel = ops.transpose(src_Rs_prev, (0, 2, 1)) @ tgt_Rs_curr

    quaternions = rotation_matrix_to_quaternion(R_rel)

    # Translation in source frame coords
    t_world = tgt_origins[1:] - src_origins[:-1]
    t_local = (ops.transpose(src_Rs_prev, (0, 2, 1)) @ t_world[..., None]).squeeze(-1)

    transforms_data = ops.cat([quaternions, t_local], axis=-1)

    # Prepend identity for first residue
    identity = ops.zeros_nd((1, 7), like=polymer.coordinates)
    if hasattr(identity, 'clone'):
        identity = identity.clone()
        identity[0, 0] = 1.0
    else:
        identity = identity.copy()
        identity[0, 0] = 1.0

    data = ops.cat([identity, transforms_data], axis=0)

    return Transforms(data=data, source=source, target=target)


def compose(
    polymer: "Polymer",
    transforms: Transforms,
) -> "Polymer":
    """Build a polymer by chaining residues with transforms.

    Takes per-residue coordinates from the input polymer and positions
    them using the transforms. The first residue stays at origin;
    subsequent residues are positioned relative to their predecessor.

    Args:
        polymer: Polymer providing per-residue coordinates and metadata.
        transforms: Transforms object from decompose() or prediction.

    Returns:
        Polymer with global coordinates (chain assembled in world frame).

    Example:
        >>> transforms = operations.decompose(reference)
        >>> rebuilt = operations.compose(reference, transforms)
        >>> ciffy.rmsd(rebuilt, reference) < 0.01  # True

        >>> # With predicted transforms
        >>> predicted = Transforms(pred_data, source=O3P_FRAME, target=P_FRAME)
        >>> structure = operations.compose(template, predicted)
    """
    from ..geometry.transforms import (
        apply_relative_transform,
        extract_frame_positions,
        frame_from_positions,
        rigid_align,
    )

    source = transforms.source
    target = transforms.target

    n_residues = polymer.size(Scale.RESIDUE)
    if len(transforms) != n_residues:
        raise ValueError(
            f"transforms has {len(transforms)} entries but polymer has "
            f"{n_residues} residues"
        )

    coords = polymer.coordinates
    counts = polymer.counts(Scale.RESIDUE)

    global_coords = ops.clone(coords)

    # Compute atom offsets
    atom_offsets = [0]
    for i in range(n_residues):
        atom_offsets.append(atom_offsets[-1] + int(counts[i]))

    # Chain residues together
    for i in range(1, n_residues):
        # Get previous residue in global coords
        prev_start, prev_end = atom_offsets[i - 1], atom_offsets[i]
        prev_global = global_coords[prev_start:prev_end]
        prev_atoms = polymer.atoms[prev_start:prev_end]

        # Compute source frame in previous residue
        src_positions = extract_frame_positions(prev_global, prev_atoms, source)
        src_origin, src_R = frame_from_positions(src_positions)

        # Apply transform to get target frame position
        target_origin, target_R = apply_relative_transform(
            src_origin, src_R, transforms.data[i]
        )

        # Get current residue (still in local coords)
        curr_start, curr_end = atom_offsets[i], atom_offsets[i + 1]
        curr_local = coords[curr_start:curr_end]
        curr_atoms = polymer.atoms[curr_start:curr_end]

        # Compute target frame in current residue's local coords
        tgt_positions = extract_frame_positions(curr_local, curr_atoms, target)
        tgt_local_origin, tgt_local_R = frame_from_positions(tgt_positions)

        # Align to global position
        global_coords[curr_start:curr_end] = rigid_align(
            curr_local, tgt_local_origin, tgt_local_R, target_origin, target_R
        )

    return polymer.copy(coordinates=global_coords)


# =============================================================================
# Legacy API (to be removed)
# =============================================================================


def frames(
    polymer: "Polymer",
    frame: "FrameDefinition | None" = None,
) -> Array:
    """
    Compute local coordinate frames for each residue.

    Args:
        polymer: Polymer structure to analyze.
        frame: FrameDefinition specifying origin, axis_ref, and plane_ref.
            Defaults to GLYCOSIDIC_FRAME (C1' origin, Z toward N9/N1).
            Common frames from ciffy.biochemistry.linking:
            - GLYCOSIDIC_FRAME: For nucleotides (C1' origin, Z toward N9/N1)
            - PROTEIN_BACKBONE_FRAME: For proteins (CA origin, Z toward N)

    Returns:
        (n_residues, 7) frames as [quaternion (4), origin (3)] where
        quaternion encodes the rotation as (w, x, y, z).

    Raises:
        ValueError: If required frame atoms are missing from any residue.

    Example:
        >>> from ciffy import operations
        >>> frames = operations.frames(polymer.strip())
        >>> # frames[i, :4] is quaternion rotation for residue i
        >>> # frames[i, 4:] is origin position for residue i
    """
    from ..geometry.transforms import rotation_matrix_to_quaternion

    origins, Rs = polymer._compute_frame_matrices(frame)
    quaternions = rotation_matrix_to_quaternion(Rs)
    return ops.cat([quaternions, origins], axis=1)


def align_to_frame(
    polymer: "Polymer",
    frame: "FrameDefinition | None" = None,
    *,
    return_origins: bool = False,
) -> "tuple[Polymer, Array] | tuple[Polymer, Array, Array]":
    """
    Align all residues to a specified local coordinate frame.

    This is the GLOBAL FRAME paradigm - it puts each residue in a consistent
    local frame independent of global position. Use for per-residue analysis
    and ML models that operate on individual residues.

    Args:
        polymer: Polymer structure to align.
        frame: FrameDefinition specifying origin, axis_ref, and plane_ref
            AtomGroups. Defaults to GLYCOSIDIC_FRAME (C1' origin, Z toward
            N9/N1) which works for all nucleotides. Common frames from
            ciffy.biochemistry.linking:
            - GLYCOSIDIC_FRAME: For nucleotides (C1' origin, Z toward N9/N1)
            - PROTEIN_BACKBONE_FRAME: For proteins (CA origin, Z toward N)
        return_origins: If True, also return the frame origins needed for
            unalign(). Default False for backward compatibility.

    Returns:
        If return_origins=False (default):
            Tuple of (aligned_polymer, Rs) where:
            - aligned_polymer: New Polymer with aligned coordinates
            - Rs: (n_residues, 3, 3) rotation matrices used for alignment

        If return_origins=True:
            Tuple of (aligned_polymer, Rs, origins) where:
            - origins: (n_residues, 3) frame origin positions needed for unalign()

    Raises:
        ValueError: If required frame atoms are missing from any residue.

    Example:
        >>> from ciffy import operations
        >>> # Basic usage
        >>> aligned, Rs = operations.align_to_frame(polymer.strip())
        >>> # Rs[i] is the rotation matrix for residue i

        >>> # Full usage with origins for unalign()
        >>> aligned, Rs, origins = operations.align_to_frame(polymer, return_origins=True)
        >>> restored = operations.unalign(aligned, Rs, origins)
        >>> ciffy.rmsd(restored, polymer) < 0.001  # True
    """
    origins, Rs = polymer._compute_frame_matrices(frame)

    # Expand origins and rotations to atom level for vectorized alignment
    membership = polymer.membership(Scale.RESIDUE)
    origins_expanded = origins[membership]
    Rs_expanded = Rs[membership]

    # Apply alignment: (coords - origin) @ R
    centered = polymer.coordinates - origins_expanded
    aligned_coords = (centered[:, None, :] @ Rs_expanded).squeeze(1)

    aligned_polymer = polymer.copy(coordinates=aligned_coords)

    if return_origins:
        return aligned_polymer, Rs, origins
    return aligned_polymer, Rs


def unalign(
    polymer: "Polymer",
    Rs: Array,
    origins: Array,
) -> "Polymer":
    """
    Reverse the alignment operation: restore residues to their original positions.

    This is the inverse of align_to_frame(). Takes a polymer with local-frame
    coordinates (each residue centered at origin) and restores the original
    global positions.

    Args:
        polymer: Aligned polymer (from align_to_frame).
        Rs: (n_residues, 3, 3) rotation matrices from align_to_frame().
        origins: (n_residues, 3) frame origin positions from align_to_frame(return_origins=True).

    Returns:
        Polymer with coordinates restored to original global frame.

    Example:
        >>> from ciffy import operations
        >>> aligned, Rs, origins = operations.align_to_frame(polymer, return_origins=True)
        >>> restored = operations.unalign(aligned, Rs, origins)
        >>> ciffy.rmsd(restored, polymer) < 0.001  # True
    """
    n_residues = polymer.size(Scale.RESIDUE)
    if Rs.shape[0] != n_residues:
        raise ValueError(
            f"Rs has {Rs.shape[0]} rows but polymer has {n_residues} residues"
        )
    if origins.shape[0] != n_residues:
        raise ValueError(
            f"origins has {origins.shape[0]} rows but polymer has {n_residues} residues"
        )

    # Expand to atom level
    membership = polymer.membership(Scale.RESIDUE)
    origins_expanded = origins[membership]
    Rs_expanded = Rs[membership]

    # Reverse alignment: coords @ R.T + origin
    Rs_T = ops.transpose(Rs_expanded, (0, 2, 1))
    unaligned_coords = (polymer.coordinates[:, None, :] @ Rs_T).squeeze(1) + origins_expanded

    return polymer.copy(coordinates=unaligned_coords)


def local_transforms(
    polymer: "Polymer",
    source_frame: "FrameDefinition",
    target_frame: "FrameDefinition",
) -> Array:
    """
    Compute inter-residue SE(3) transforms using specified source and target frames.

    This is the LOCAL FRAME paradigm - each transform[i] describes how residue i
    is positioned relative to residue i-1. Use for chain reconstruction and
    backbone sampling.

    Args:
        polymer: Polymer structure to analyze.
        source_frame: Frame definition for residue i-1 (5'/N-terminal side).
            REQUIRED - no default to make frame choice explicit.
        target_frame: Frame definition for residue i (3'/C-terminal side).
            REQUIRED - no default to make frame choice explicit.

    Returns:
        (n_residues, 7) array where:
        - transforms[0] = zeros (first residue has no predecessor)
        - transforms[i] = SE(3) from source_frame[i-1] TO target_frame[i]
        - Each transform is [quaternion (4), translation (3)]

    Raises:
        TypeError: If source_frame or target_frame is None (fail-fast).

    Common Frame Pairs:
        - GLYCOSIDIC→GLYCOSIDIC: For ML models (same frame, canonical orientation)
        - O3P_FRAME→P_FRAME: For physical backbone modeling (actual bond geometry)

    Example:
        >>> from ciffy import operations
        >>> from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME
        >>> transforms = operations.local_transforms(polymer, GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
    """
    from ..geometry.transforms import rotation_matrix_to_quaternion

    # Fail-fast: require both frames
    if source_frame is None:
        raise TypeError("source_frame is required (got None)")
    if target_frame is None:
        raise TypeError("target_frame is required (got None)")

    n_residues = polymer.size(Scale.RESIDUE)

    # Compute frames for source and target
    src_origins, src_Rs = polymer._compute_frame_matrices(source_frame)
    if source_frame is target_frame:
        # Optimization: same frame for both, reuse
        tgt_origins, tgt_Rs = src_origins, src_Rs
    else:
        tgt_origins, tgt_Rs = polymer._compute_frame_matrices(target_frame)

    # Compute transforms: transforms[i] = source_frame[i-1] -> target_frame[i]
    if n_residues < 2:
        return ops.zeros_nd((n_residues, 7), like=polymer.coordinates)

    # Relative rotation: R_rel = R1.T @ R2 for consecutive pairs
    src_Rs_prev = src_Rs[:-1]  # frames 0..n-2
    tgt_Rs_curr = tgt_Rs[1:]   # frames 1..n-1
    R_rel = ops.transpose(src_Rs_prev, (0, 2, 1)) @ tgt_Rs_curr  # (n-1, 3, 3)

    # Convert to quaternion (vectorized)
    quaternions = rotation_matrix_to_quaternion(R_rel)  # (n-1, 4)

    # Translation in source frame coords: t_local = R1.T @ (origin2 - origin1)
    t_world = tgt_origins[1:] - src_origins[:-1]  # (n-1, 3)
    t_local = (ops.transpose(src_Rs_prev, (0, 2, 1)) @ t_world[..., None]).squeeze(-1)

    # Concatenate rotation (quaternion) and translation
    transforms_1_to_n = ops.cat([quaternions, t_local], axis=-1)  # (n-1, 7)

    # Prepend identity quaternion [1,0,0,0] + zero translation for first residue
    identity_first = ops.zeros_nd((1, 7), like=polymer.coordinates)
    if hasattr(identity_first, 'clone'):  # torch
        identity_first = identity_first.clone()
        identity_first[0, 0] = 1.0
    else:  # numpy
        identity_first = identity_first.copy()
        identity_first[0, 0] = 1.0
    transforms = ops.cat([identity_first, transforms_1_to_n], axis=0)  # (n, 7)

    return transforms


def apply_local_transforms(
    polymer: "Polymer",
    transforms: Array,
    source_frame: "FrameDefinition",
    target_frame: "FrameDefinition",
) -> "Polymer":
    """
    Chain residues together using inter-residue transforms.

    This is the LOCAL FRAME paradigm - takes a polymer with LOCAL coordinates
    (each residue in its own frame, typically aligned to target_frame) and
    chains them using the transforms.

    Args:
        polymer: Polymer with local coordinates (from align_to_frame).
        transforms: (n_residues, 7) array from local_transforms().
            Each row is [quaternion (4), translation (3)].
            transforms[i] describes how residue i is positioned relative to i-1.
        source_frame: Frame definition for residue i-1 (must match local_transforms).
        target_frame: Frame definition for residue i (must match local_transforms).

    Returns:
        Polymer with GLOBAL coordinates (chain assembled in world frame).

    Raises:
        TypeError: If source_frame or target_frame is None.

    Example:
        >>> from ciffy import operations
        >>> from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME
        >>> transforms = operations.local_transforms(polymer, GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        >>> aligned, _ = operations.align_to_frame(polymer, frame=GLYCOSIDIC_FRAME)
        >>> rebuilt = operations.apply_local_transforms(aligned, transforms, GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        >>> ciffy.rmsd(rebuilt, polymer) < 0.01  # True
    """
    from ..geometry.transforms import (
        apply_relative_transform,
        extract_frame_positions,
        frame_from_positions,
        rigid_align,
    )

    # Fail-fast: require both frames
    if source_frame is None:
        raise TypeError("source_frame is required (got None)")
    if target_frame is None:
        raise TypeError("target_frame is required (got None)")

    n_residues = polymer.size(Scale.RESIDUE)
    if transforms.shape[0] != n_residues:
        raise ValueError(
            f"transforms has {transforms.shape[0]} rows but polymer has "
            f"{n_residues} residues"
        )

    coords = polymer.coordinates
    counts = polymer.counts(Scale.RESIDUE)

    # Build global coordinates
    global_coords = ops.clone(coords)

    # Compute atom offsets
    atom_offsets = [0]
    for i in range(n_residues):
        atom_offsets.append(atom_offsets[-1] + int(counts[i]))

    # First residue stays at origin
    for i in range(1, n_residues):
        # Get previous residue's atoms (now in global coords)
        prev_start, prev_end = atom_offsets[i - 1], atom_offsets[i]
        prev_global = global_coords[prev_start:prev_end]
        prev_atoms = polymer.atoms[prev_start:prev_end]

        # Compute source_frame in previous residue (in global coords)
        src_positions = extract_frame_positions(prev_global, prev_atoms, source_frame)
        src_origin, src_R = frame_from_positions(src_positions)

        # Apply transforms[i] to get target_frame position for current residue
        target_origin, target_R = apply_relative_transform(
            src_origin, src_R, transforms[i]
        )

        # Get current residue's atoms (still in local coords)
        curr_start, curr_end = atom_offsets[i], atom_offsets[i + 1]
        curr_local = coords[curr_start:curr_end]
        curr_atoms = polymer.atoms[curr_start:curr_end]

        # Compute target_frame in current residue's local coords
        tgt_positions = extract_frame_positions(curr_local, curr_atoms, target_frame)
        tgt_local_origin, tgt_local_R = frame_from_positions(tgt_positions)

        # Align local target_frame to global target_frame position
        global_coords[curr_start:curr_end] = rigid_align(
            curr_local, tgt_local_origin, tgt_local_R, target_origin, target_R
        )

    return polymer.copy(coordinates=global_coords)


def gather(
    polymer: "Polymer",
    groups: list,
) -> Array:
    """
    Gather coordinates for specific atoms from each residue.

    For each atom group, finds the matching atom in each residue and
    returns their coordinates. Useful for extracting frame atoms.

    Args:
        polymer: Polymer structure to analyze.
        groups: List of AtomGroups (e.g., [Sugar.C1p, PurineBase.N9]).

    Returns:
        (n_residues, len(groups), 3) coordinate array.

    Raises:
        ValueError: If any residue doesn't have exactly one atom
            matching each group.

    Example:
        >>> from ciffy import operations
        >>> from ciffy.biochemistry.constants import Sugar, PurineBase
        >>> positions = operations.gather(polymer, [Sugar.C1p, PurineBase.N9, PurineBase.C4])
    """
    n_groups = len(groups)
    n_residues = polymer.size(Scale.RESIDUE)
    membership = polymer.membership(Scale.RESIDUE)

    indices = ops.empty((n_residues, n_groups), like=polymer.atoms, dtype='int64')

    for i, group in enumerate(groups):
        values = ops.to_backend(group.index(), polymer.atoms)
        mask = ops.isin(polymer.atoms, values)
        atom_idx = ops.nonzero_1d(mask)

        if len(atom_idx) != n_residues:
            raise ValueError(
                f"Group {i}: expected {n_residues} matches (one per residue), "
                f"got {len(atom_idx)}. Check for missing or duplicate atoms."
            )

        residue_idx = membership[atom_idx]
        indices[residue_idx, i] = atom_idx

    return polymer.coordinates[indices]


def sort_atoms(polymer: "Polymer") -> "Polymer":
    """
    Sort atoms within each residue by atom type enum value.

    This creates a canonical atom ordering that is consistent regardless
    of the original CIF file ordering. Useful for ensuring training and
    inference use the same atom order.

    Args:
        polymer: Polymer structure to sort.

    Returns:
        New Polymer with all atom-level fields reordered so atoms within
        each residue are sorted by their enum value.

    Example:
        >>> from ciffy import operations
        >>> aligned, _ = operations.align_to_frame(polymer)
        >>> canonical = operations.sort_atoms(aligned)
    """
    from ..backend.ops import argsort
    from ..biochemistry import Atom

    if polymer.size(Scale.RESIDUE) == 0:
        return polymer.copy()

    # Vectorized segment argsort
    membership = polymer.membership(Scale.RESIDUE)
    offset = Atom.count() + 1
    combined_key = membership * offset + polymer.atoms
    sort_indices = argsort(combined_key)

    # Reorder all atom-level fields
    overrides = {}
    for name, field in polymer._get_fields().items():
        if field.scale == Scale.ATOM:
            overrides[name] = field.data[sort_indices]

    return polymer.copy(**overrides)


__all__ = [
    # New API
    "Transforms",
    "decompose",
    "compose",
    # Utilities
    "gather",
    "sort_atoms",
    # Legacy (to be removed)
    "frames",
    "align_to_frame",
    "unalign",
    "local_transforms",
    "apply_local_transforms",
]
