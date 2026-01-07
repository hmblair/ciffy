"""Utilities for packing variable-length residue data for batched operations."""

from __future__ import annotations

import torch


def pack_by_residue(
    features: torch.Tensor,
    counts: torch.Tensor,
    membership: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack atom-level features into residue-level batches with padding.

    Converts (N_atoms, d) features into (n_residues, max_atoms, d) with
    padding for efficient batched attention within residues.

    Args:
        features: (N_atoms, d) atom-level features.
        counts: (n_residues,) number of atoms per residue.
        membership: (N_atoms,) residue index for each atom.

    Returns:
        packed: (n_residues, max_atoms, d) padded features.
        mask: (n_residues, max_atoms) boolean mask (True = valid).
        sort_idx: Indices to sort atoms by residue.
        positions: Position of each atom within its residue.
        membership_sorted: Membership after sorting.
    """
    n_atoms = features.shape[0]
    n_residues = len(counts)
    max_atoms = counts.max().item()
    d = features.shape[-1]
    device = features.device

    sort_idx = torch.argsort(membership)
    features_sorted = features[sort_idx]

    cumsum = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        counts.cumsum(0)[:-1]
    ])
    membership_sorted = membership[sort_idx]
    positions = torch.arange(n_atoms, device=device) - cumsum[membership_sorted]

    packed = torch.zeros(n_residues, max_atoms, d, device=device, dtype=features.dtype)
    packed[membership_sorted, positions] = features_sorted

    mask = torch.arange(max_atoms, device=device).unsqueeze(0) < counts.unsqueeze(1)

    return packed, mask, sort_idx, positions, membership_sorted


def unpack_by_residue(
    packed: torch.Tensor,
    mask: torch.Tensor,
    sort_idx: torch.Tensor,
    positions: torch.Tensor,
    membership_sorted: torch.Tensor,
) -> torch.Tensor:
    """
    Unpack residue-level batched features back to atom-level.

    Inverse of pack_by_residue.

    Args:
        packed: (n_residues, max_atoms, d) padded features.
        mask: (n_residues, max_atoms) boolean mask (True = valid).
        sort_idx: Indices used to sort atoms by residue.
        positions: Position of each atom within its residue.
        membership_sorted: Membership after sorting.

    Returns:
        features: (N_atoms, d) atom-level features in original order.
    """
    n_atoms = len(sort_idx)
    d = packed.shape[-1]
    device = packed.device

    # Extract valid positions
    features_sorted = packed[membership_sorted, positions]

    # Unsort to original order
    features = torch.empty(n_atoms, d, device=device, dtype=packed.dtype)
    features[sort_idx] = features_sorted

    return features
