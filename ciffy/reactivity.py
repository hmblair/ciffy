"""
Reactivity data loading and matching for SHAPE/DMS experiments.

This module provides utilities for associating per-residue reactivity data
(from chemical probing experiments like SHAPE or DMS) with 3D structures.

Example:
    >>> from ciffy.reactivity import ReactivityIndex
    >>> import ciffy
    >>>
    >>> # Build index from dataset
    >>> index = ReactivityIndex()
    >>> index.add("sample1", "GGGACGUACGUAAA", reactivity_array)
    >>>
    >>> # Match against structure (works with numpy, torch, or GPU polymers)
    >>> polymer = ciffy.load("structure.cif").chain("A")
    >>> result = index.match(polymer)
    >>> if result is not None:
    ...     polymer.annotate("shape", result.reactivity)  # Already correct backend
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .backend.core import Array
from .backend.ops import convert_backend

if TYPE_CHECKING:
    from .polymer import Polymer

__all__ = ["ReactivityIndex", "ReactivityMatch"]


@dataclass
class ReactivityMatch:
    """Result of a successful reactivity match.

    Attributes:
        name: Identifier of the matched entry.
        reactivity: Reactivity values aligned to the polymer's residues.
            Shape is (n_residues,) or (n_residues, n_channels) for multi-channel data.
            Backend and device match the polymer used in the match() call.
        offset: Position in the probed sequence where the structure sequence starts.
            Only set when structure is a substring of probed sequence.
        span: Fraction of structure residues with reactivity data (0-1).
            1.0 means all residues have data, <1.0 means some positions are NaN-padded.
    """

    name: str
    reactivity: Array  # numpy or torch, matching polymer backend
    offset: int
    span: float


class ReactivityIndex:
    """Index for matching reactivity data to structures by sequence.

    Stores (name, sequence, reactivity) entries and provides lookup by
    exact substring matching against polymer sequences.

    The matching logic handles two cases:
    1. Structure sequence is a substring of probed sequence (common case):
       Reactivity is sliced to match the structure.
    2. Probed sequence is a substring of structure sequence:
       Reactivity is padded with NaN for positions without data.

    Example:
        >>> index = ReactivityIndex()
        >>> index.add("exp1", "GGGACGUACGUAAA", np.array([0.1, 0.2, ...]))
        >>> index.add("exp2", "CUAGCUAGCU", np.array([0.5, 0.3, ...]))
        >>>
        >>> polymer = ciffy.load("structure.cif").chain("A")
        >>> result = index.match(polymer)
        >>> if result is not None:
        ...     polymer.annotate("shape", result.reactivity)
    """

    def __init__(self) -> None:
        """Initialize an empty reactivity index."""
        self._entries: list[tuple[str, str, np.ndarray]] = []

    def add(self, name: str, sequence: str, reactivity: np.ndarray) -> None:
        """Add a reactivity entry to the index.

        Args:
            name: Identifier for this entry (e.g., experiment ID, PDB chain).
            sequence: The probed sequence (RNA/DNA letters, case-insensitive).
            reactivity: Per-residue reactivity values. Shape should be (L,) or
                (L, C) where L is len(sequence) and C is number of channels.

        Raises:
            ValueError: If sequence length doesn't match reactivity length.
        """
        sequence = sequence.upper().replace("T", "U")
        reactivity = np.asarray(reactivity)

        if reactivity.shape[0] != len(sequence):
            raise ValueError(
                f"Reactivity length ({reactivity.shape[0]}) doesn't match "
                f"sequence length ({len(sequence)}) for entry '{name}'"
            )

        self._entries.append((name, sequence, reactivity))

    def __len__(self) -> int:
        """Return number of entries in the index."""
        return len(self._entries)

    def match(self, polymer: "Polymer") -> ReactivityMatch | None:
        """Find matching reactivity data for a polymer.

        Searches all entries for exact substring matches between the polymer's
        sequence and the probed sequences.

        Args:
            polymer: Structure to match against. Uses polymer.sequence_str().
                The returned reactivity will match the polymer's backend and device.

        Returns:
            ReactivityMatch if exactly one match is found, None otherwise
            (no matches or multiple matches).

        Note:
            The polymer should typically be stripped of unresolved residues
            (polymer.strip()) before matching, as missing residues would
            cause sequence mismatches.
        """
        structure_seq = polymer.sequence_str().upper().replace("T", "U")
        matches: list[tuple[str, np.ndarray, int, float]] = []

        for name, probed_seq, reactivity in self._entries:
            result = self._try_match(structure_seq, probed_seq, reactivity)
            if result is not None:
                aligned_reactivity, offset, span = result
                matches.append((name, aligned_reactivity, offset, span))

        if len(matches) == 1:
            name, reactivity, offset, span = matches[0]
            # Convert to polymer's backend/device
            reactivity = convert_backend(reactivity, polymer.coordinates)
            return ReactivityMatch(
                name=name,
                reactivity=reactivity,
                offset=offset,
                span=span,
            )

        return None

    def match_all(self, polymer: "Polymer") -> list[ReactivityMatch]:
        """Find all matching reactivity entries for a polymer.

        Like match(), but returns all matches instead of requiring exactly one.

        Args:
            polymer: Structure to match against.
                The returned reactivities will match the polymer's backend and device.

        Returns:
            List of ReactivityMatch objects (may be empty).
        """
        structure_seq = polymer.sequence_str().upper().replace("T", "U")
        matches: list[ReactivityMatch] = []
        ref = polymer.coordinates  # Reference for backend/device conversion

        for name, probed_seq, reactivity in self._entries:
            result = self._try_match(structure_seq, probed_seq, reactivity)
            if result is not None:
                aligned_reactivity, offset, span = result
                # Convert to polymer's backend/device
                aligned_reactivity = convert_backend(aligned_reactivity, ref)
                matches.append(
                    ReactivityMatch(
                        name=name,
                        reactivity=aligned_reactivity,
                        offset=offset,
                        span=span,
                    )
                )

        return matches

    def _try_match(
        self,
        structure_seq: str,
        probed_seq: str,
        reactivity: np.ndarray,
    ) -> tuple[np.ndarray, int, float] | None:
        """Try to match structure sequence against probed sequence.

        Returns:
            (aligned_reactivity, offset, span) if match found, None otherwise.
        """
        n_structure = len(structure_seq)
        n_probed = len(probed_seq)

        # Case 1: Structure is substring of probed (common - probed has padding)
        offset = probed_seq.find(structure_seq)
        if offset != -1:
            # Slice reactivity to match structure
            aligned = reactivity[offset : offset + n_structure]
            return aligned.copy(), offset, 1.0

        # Case 2: Probed is substring of structure (probed covers partial structure)
        offset = structure_seq.find(probed_seq)
        if offset != -1:
            # Pad reactivity with NaN
            if reactivity.ndim == 1:
                aligned = np.full(n_structure, np.nan, dtype=reactivity.dtype)
                aligned[offset : offset + n_probed] = reactivity
            else:
                # Multi-channel: (n_residues, n_channels)
                aligned = np.full(
                    (n_structure, reactivity.shape[1]),
                    np.nan,
                    dtype=reactivity.dtype,
                )
                aligned[offset : offset + n_probed] = reactivity

            span = n_probed / n_structure
            return aligned, offset, span

        return None
