"""Utilities for packing variable-length residue data for batched operations."""

from __future__ import annotations

from ciffy.backend import ops
from ciffy.backend.core import Array


def pack_by_residue(
    features: Array,
    counts: Array,
) -> tuple[Array, Array]:
    """
    Pack atom-level features into residue-level batches with padding.

    Converts (N_atoms, d) features into (n_residues, max_atoms, d) with
    padding for efficient batched attention within residues.

    Note: Assumes atoms are already grouped by residue (as in Polymer).

    Args:
        features: (N_atoms, d) atom-level features.
        counts: (n_residues,) number of atoms per residue.

    Returns:
        packed: (n_residues, max_atoms, d) padded features.
        mask: (n_residues, max_atoms) boolean mask (True = valid).
    """
    n_residues = len(counts)
    max_atoms = int(counts.max())
    d = features.shape[-1]

    # Build mask
    positions = ops.unsqueeze(ops.arange(max_atoms, like=features), 0)
    mask = positions < ops.unsqueeze(counts, 1)

    # Scatter features into packed tensor
    packed = ops.zeros_nd((n_residues, max_atoms, d), like=features)
    packed[mask] = features

    return packed, mask


def unpack_by_residue(
    packed: Array,
    mask: Array,
) -> Array:
    """
    Unpack residue-level batched features back to atom-level.

    Inverse of pack_by_residue.

    Args:
        packed: (n_residues, max_atoms, d) padded features.
        mask: (n_residues, max_atoms) boolean mask (True = valid).

    Returns:
        features: (N_atoms, d) atom-level features.
    """
    return packed[mask]
