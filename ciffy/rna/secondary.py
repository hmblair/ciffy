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

import numpy as np

__all__ = ["dotbracket_to_pairs", "pairs_to_dotbracket"]

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
