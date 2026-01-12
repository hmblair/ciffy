"""
Secondary structure utilities for RNA.

Functions for converting between dot-bracket notation and base pair arrays.

Dot-bracket notation uses:
- '.' for unpaired bases
- '(' and ')' for base pairs (matching parentheses)
- '[', ']', '{', '}', '<', '>' for pseudoknots (extended notation)

Base pair arrays are (n, 2) integer arrays where each row [i, j] represents
a base pair between positions i and j (0-indexed, i < j).

Example:
    >>> from ciffy.rna import dotbracket_to_pairs, pairs_to_dotbracket
    >>>
    >>> pairs = dotbracket_to_pairs("((...))")
    >>> pairs
    array([[0, 6],
           [1, 5]])
    >>>
    >>> pairs_to_dotbracket(pairs, length=7)
    '((...))'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer

__all__ = ["dotbracket_to_pairs", "pairs_to_dotbracket", "secondary_structure"]

# Bracket pairs for extended notation (pseudoknots)
_OPEN_BRACKETS = "([{<"
_CLOSE_BRACKETS = ")]}>"
_BRACKET_PAIRS = dict(zip(_CLOSE_BRACKETS, _OPEN_BRACKETS))


def dotbracket_to_pairs(dotbracket: str) -> np.ndarray:
    """Convert dot-bracket notation to base pair array.

    Args:
        dotbracket: Secondary structure in dot-bracket notation.
            Uses '.' for unpaired, '()' for pairs, and optionally
            '[]', '{}', '<>' for pseudoknots.

    Returns:
        Array of shape (n_pairs, 2) where each row [i, j] is a base pair
        with i < j. Returns empty array of shape (0, 2) if no pairs.

    Raises:
        ValueError: If brackets are unbalanced.

    Example:
        >>> dotbracket_to_pairs("((..))")
        array([[0, 5],
               [1, 4]])
    """
    # Stack for each bracket type
    stacks: dict[str, list[int]] = {b: [] for b in _OPEN_BRACKETS}
    pairs: list[tuple[int, int]] = []

    for i, char in enumerate(dotbracket):
        if char in _OPEN_BRACKETS:
            stacks[char].append(i)
        elif char in _CLOSE_BRACKETS:
            open_bracket = _BRACKET_PAIRS[char]
            if not stacks[open_bracket]:
                raise ValueError(
                    f"Unbalanced bracket '{char}' at position {i}: "
                    f"no matching '{open_bracket}'"
                )
            j = stacks[open_bracket].pop()
            pairs.append((j, i))  # j < i always
        elif char != '.':
            raise ValueError(f"Invalid character '{char}' at position {i}")

    # Check for unclosed brackets
    for bracket, stack in stacks.items():
        if stack:
            raise ValueError(
                f"Unclosed bracket '{bracket}' at position(s): {stack}"
            )

    if not pairs:
        return np.empty((0, 2), dtype=np.int64)

    # Sort by first position
    pairs.sort()
    return np.array(pairs, dtype=np.int64)


def pairs_to_dotbracket(pairs: np.ndarray, length: int) -> str:
    """Convert base pair array to dot-bracket notation.

    Uses extended notation ([{<) for pseudoknots when needed.

    Args:
        pairs: Array of shape (n_pairs, 2) where each row [i, j] is a base
            pair. Pairs should have i < j.
        length: Total sequence length.

    Returns:
        Dot-bracket string of the given length.

    Raises:
        ValueError: If pairs are invalid (out of bounds, i >= j) or
            contain more pseudoknot levels than supported (max 4).

    Example:
        >>> pairs_to_dotbracket(np.array([[0, 5], [1, 4]]), length=6)
        '((..))'
    """
    pairs = np.asarray(pairs)

    if pairs.size == 0:
        return "." * length

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (n, 2), got {pairs.shape}")

    # Validate pairs
    if np.any(pairs[:, 0] >= pairs[:, 1]):
        bad = pairs[pairs[:, 0] >= pairs[:, 1]]
        raise ValueError(f"Pairs must have i < j, got: {bad.tolist()}")

    if np.any(pairs < 0) or np.any(pairs >= length):
        raise ValueError(f"Pair indices must be in [0, {length})")

    # Assign bracket types to avoid crossing (pseudoknot detection)
    result = ["."] * length
    bracket_types: list[int] = []  # Which bracket type each pair uses

    # Sort pairs by opening position
    sorted_indices = np.argsort(pairs[:, 0])
    sorted_pairs = pairs[sorted_indices]

    for i, (left, right) in enumerate(sorted_pairs):
        # Find which bracket type to use (first one that doesn't cross)
        bracket_idx = 0
        for prev_idx in range(i):
            prev_left, prev_right = sorted_pairs[prev_idx]
            prev_bracket = bracket_types[prev_idx]
            # Check if this pair crosses the previous one
            # Crossing: prev_left < left < prev_right < right
            if prev_left < left < prev_right < right:
                # Need different bracket type than prev
                if prev_bracket >= bracket_idx:
                    bracket_idx = prev_bracket + 1

        if bracket_idx >= len(_OPEN_BRACKETS):
            raise ValueError(
                f"Too many nested pseudoknots at pair ({left}, {right}). "
                f"Maximum {len(_OPEN_BRACKETS)} bracket types supported."
            )

        bracket_types.append(bracket_idx)
        result[left] = _OPEN_BRACKETS[bracket_idx]
        result[right] = _CLOSE_BRACKETS[bracket_idx]

    return "".join(result)


def secondary_structure(polymer: "Polymer", min_loop_size: int = 3) -> str:
    """Extract secondary structure from polymer connections as dot-bracket.

    Determines base pairs from hydrogen bond connections in the polymer
    and returns the secondary structure in dot-bracket notation.

    Args:
        polymer: RNA polymer with connections loaded. Must be loaded with
            ``ciffy.load(file, skip=[])`` or ``skip=["descriptions"]`` to
            include connections.
        min_loop_size: Minimum number of residues between paired positions.
            Pairs with |i - j| <= min_loop_size are filtered out. Default 3.

    Returns:
        Dot-bracket string of length equal to the number of residues.
        Uses extended notation ([{<) for pseudoknots if present.

    Raises:
        ValueError: If polymer has no connections loaded.

    Example:
        >>> import ciffy
        >>> polymer = ciffy.load("structure.cif", skip=[])
        >>> from ciffy.rna import secondary_structure
        >>> ss = secondary_structure(polymer)
        >>> print(ss)
        '(((...)))'
    """
    from ..biochemistry import Scale

    if polymer.connections is None:
        raise ValueError(
            "Polymer has no connections. Load with skip=[] to include connections."
        )

    connections = np.asarray(polymer.connections)
    n_residues = polymer.size(Scale.RESIDUE)

    if connections.size == 0:
        return "." * n_residues

    # Get residue membership for each atom
    residue_membership = np.asarray(polymer.membership(Scale.RESIDUE))
    n_atoms = polymer.size()

    # Filter connections to only include atoms in this polymer
    # (handles case where polymer is a selection from larger structure)
    valid_mask = (connections[:, 0] < n_atoms) & (connections[:, 1] < n_atoms)
    connections = connections[valid_mask]

    if connections.size == 0:
        return "." * n_residues

    # Convert atom pairs to residue pairs
    residue_pairs = residue_membership[connections]  # (n_connections, 2)

    # Keep only inter-residue pairs (different residues)
    inter_residue_mask = residue_pairs[:, 0] != residue_pairs[:, 1]
    residue_pairs = residue_pairs[inter_residue_mask]

    if residue_pairs.size == 0:
        return "." * n_residues

    # Ensure i < j for each pair
    residue_pairs = np.sort(residue_pairs, axis=1)

    # Filter by minimum loop size (removes backbone H-bonds)
    distances = residue_pairs[:, 1] - residue_pairs[:, 0]
    residue_pairs = residue_pairs[distances > min_loop_size]

    if residue_pairs.size == 0:
        return "." * n_residues

    # Deduplicate pairs (multiple H-bonds between same residue pair)
    unique_pairs = np.unique(residue_pairs, axis=0)

    # Each residue can only be in one base pair - keep longest range pairs
    # Count how many pairs each residue participates in
    paired: dict[int, tuple[int, int]] = {}  # residue -> (partner, distance)
    for i, j in unique_pairs:
        dist = j - i
        # For position i, keep pair with largest distance
        if i not in paired or dist > paired[i][1]:
            paired[i] = (j, dist)
        # For position j, keep pair with largest distance
        if j not in paired or dist > paired[j][1]:
            paired[j] = (i, dist)

    # Reconstruct pairs ensuring each residue is in at most one pair
    final_pairs = set()
    for res, (partner, _) in paired.items():
        if res < partner:
            # Check if partner also points back to res
            if partner in paired and paired[partner][0] == res:
                final_pairs.add((res, partner))

    if not final_pairs:
        return "." * n_residues

    final_pairs = np.array(sorted(final_pairs), dtype=np.int64)

    # Convert to dot-bracket
    return pairs_to_dotbracket(final_pairs, n_residues)
