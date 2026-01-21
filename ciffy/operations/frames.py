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
    "Transforms",
    "decompose",
    "compose",
    "gather",
    "sort_atoms",
]
