"""
Structure comparison metrics: TM-score and lDDT.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch
from ..types import Scale, Molecule

if TYPE_CHECKING:
    from ..polymer import Polymer


def tm_score(
    pred: Polymer,
    ref: Polymer,
    scale: Scale = Scale.RESIDUE,
) -> float:
    """
    Compute TM-score between two structures.

    TM-score is a length-normalized structural similarity metric
    ranging from 0 to 1, where 1 indicates identical structures.
    Scores > 0.5 generally indicate same fold.

    Args:
        pred: Predicted structure.
        ref: Reference structure (used for length normalization).
        scale: Scale at which to compute (typically RESIDUE for Cα).

    Returns:
        TM-score value between 0 and 1.

    Note:
        Uses molecule-type-specific normalization:
        - Protein: d_0 = 1.24 * (L - 15)^(1/3) - 1.8
        - RNA/DNA: d_0 = 0.6 * sqrt(L - 5) - 2.5
    """
    from .alignment import kabsch_align

    # Get coordinates at specified scale
    if scale == Scale.ATOM:
        pred_coords = pred.coordinates
        ref_coords = ref.coordinates
    else:
        pred_coords = pred.reduce(pred.coordinates, scale)
        ref_coords = ref.reduce(ref.coordinates, scale)

    # Length for normalization (from reference)
    L = ref_coords.shape[0]

    if pred_coords.shape[0] != L:
        raise ValueError(
            f"Structure sizes must match: pred has {pred_coords.shape[0]}, "
            f"ref has {L} at scale {scale.name}"
        )

    # Determine molecule type for d_0 calculation
    mol_type = _get_molecule_type(ref)

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
    """Get the predominant molecule type of a polymer."""
    from ..biochemistry._generated_molecule import molecule_type

    mol_types = polymer.molecule_type
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
