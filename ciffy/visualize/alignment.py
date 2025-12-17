"""
Sequence alignment utilities for visualization.

Provides functions for aligning data sequences to polymer sequences when
they don't match exactly (e.g., different numbering or missing residues).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer


def needleman_wunsch(
    seq1: str,
    seq2: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> Tuple[str, str]:
    """
    Perform Needleman-Wunsch global sequence alignment.

    A simple pure-Python implementation for aligning two sequences.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        match: Score for matching characters.
        mismatch: Score for mismatching characters.
        gap: Score for gaps.

    Returns:
        Tuple of (aligned_seq1, aligned_seq2) with '-' for gaps.

    Example:
        >>> aln1, aln2 = needleman_wunsch("ACGU", "ACU")
        >>> print(aln1)  # "ACGU"
        >>> print(aln2)  # "AC-U"
    """
    n, m = len(seq1), len(seq2)

    # Initialize score matrix
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        score[i, 0] = i * gap
    for j in range(m + 1):
        score[0, j] = j * gap

    # Fill score matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = score[i - 1, j - 1] + match
            else:
                diag = score[i - 1, j - 1] + mismatch
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            score[i, j] = max(diag, up, left)

    # Traceback
    aln1, aln2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                s = match
            else:
                s = mismatch

            if score[i, j] == score[i - 1, j - 1] + s:
                aln1.append(seq1[i - 1])
                aln2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and score[i, j] == score[i - 1, j] + gap:
            aln1.append(seq1[i - 1])
            aln2.append('-')
            i -= 1
        else:
            aln1.append('-')
            aln2.append(seq2[j - 1])
            j -= 1

    return ''.join(reversed(aln1)), ''.join(reversed(aln2))


def _map_values_through_alignment(
    values: np.ndarray,
    aln1: str,
    aln2: str,
    gap_value: float = 0.0,
) -> np.ndarray:
    """
    Map values from seq1 to seq2 through an alignment.

    Args:
        values: Values corresponding to non-gap positions in aln1.
        aln1: Aligned first sequence (with gaps).
        aln2: Aligned second sequence (with gaps).
        gap_value: Value to use for positions that are gaps in seq1.

    Returns:
        Values mapped to non-gap positions in seq2.
    """
    # Count non-gap characters in seq1
    n_seq1 = sum(1 for c in aln1 if c != '-')
    if values.shape[0] != n_seq1:
        raise ValueError(
            f"Values length ({values.shape[0]}) must match number of "
            f"non-gap characters in seq1 ({n_seq1})"
        )

    mapped = []
    val_idx = 0

    for c1, c2 in zip(aln1, aln2):
        if c2 == '-':
            # Gap in seq2 - skip this position
            if c1 != '-':
                val_idx += 1
            continue

        if c1 == '-':
            # Gap in seq1 - use gap value for seq2
            mapped.append(gap_value)
        else:
            # Both have characters - use the value
            mapped.append(values[val_idx])
            val_idx += 1

    return np.array(mapped)


def align_values(
    data: np.ndarray,
    data_seq: str,
    polymer: "Polymer",
    chain: int = 0,
    gap_value: float = 0.0,
) -> np.ndarray:
    """
    Align values to a polymer's sequence.

    Uses Needleman-Wunsch alignment to map values from a data sequence
    to the polymer's sequence. Useful when the data sequence has different
    numbering or missing residues compared to the structure.

    Args:
        data: Per-residue values corresponding to data_seq.
        data_seq: Sequence string for the data.
        polymer: Target polymer structure.
        chain: Chain index to align to (default 0).
        gap_value: Value to use for gaps in the data sequence.

    Returns:
        Values aligned to polymer's residue sequence for the specified chain.

    Example:
        >>> import ciffy
        >>> polymer = ciffy.load("structure.cif")
        >>> # Data has slightly different sequence
        >>> data_seq = "ACGU"
        >>> data_values = np.array([0.1, 0.2, 0.3, 0.4])
        >>> aligned = ciffy.visualize.align_values(
        ...     data_values, data_seq, polymer, chain=0
        ... )
    """
    data = np.asarray(data)

    if len(data_seq) != data.shape[0]:
        raise ValueError(
            f"Data sequence length ({len(data_seq)}) must match data "
            f"length ({data.shape[0]})"
        )

    # Get polymer sequence for the specified chain
    chain_polymer = polymer.by_index(chain)
    polymer_seq = chain_polymer.sequence_str()

    # Normalize sequences (uppercase, T->U for RNA)
    data_seq_norm = data_seq.upper().replace('T', 'U')
    polymer_seq_norm = polymer_seq.upper().replace('T', 'U')

    # Check if sequences match exactly
    if data_seq_norm == polymer_seq_norm:
        return data.copy()

    # Perform alignment
    aln_data, aln_polymer = needleman_wunsch(data_seq_norm, polymer_seq_norm)

    # Map values through alignment
    return _map_values_through_alignment(data, aln_data, aln_polymer, gap_value)
