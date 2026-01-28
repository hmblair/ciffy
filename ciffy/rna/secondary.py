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

# Watson-Crick hydrogen bond patterns: (base1, base2) -> [(atom1, atom2), ...]
# Each base pair type requires ALL listed H-bonds to be present.
# Atom indices from ciffy.Residue enum.
_WC_HBONDS: dict[tuple[int, int], list[tuple[int, int]]] = {}


def _init_wc_hbonds() -> None:
    """Initialize Watson-Crick hydrogen bond patterns."""
    from ..biochemistry import Residue

    A, G, C, U = Residue.A.value, Residue.G.value, Residue.C.value, Residue.U.value

    # A-U: 2 H-bonds
    _WC_HBONDS[(A, U)] = [
        (int(Residue.A.N1), int(Residue.U.N3)),  # N1--N3
        (int(Residue.A.N6), int(Residue.U.O4)),  # N6--O4
    ]
    _WC_HBONDS[(U, A)] = [(b, a) for a, b in _WC_HBONDS[(A, U)]]

    # G-C: 3 H-bonds
    _WC_HBONDS[(G, C)] = [
        (int(Residue.G.O6), int(Residue.C.N4)),  # O6--N4
        (int(Residue.G.N1), int(Residue.C.N3)),  # N1--N3
        (int(Residue.G.N2), int(Residue.C.O2)),  # N2--O2
    ]
    _WC_HBONDS[(C, G)] = [(b, a) for a, b in _WC_HBONDS[(G, C)]]

    # G-U wobble: 2 H-bonds
    _WC_HBONDS[(G, U)] = [
        (int(Residue.G.O6), int(Residue.U.N3)),  # O6--N3
        (int(Residue.G.N1), int(Residue.U.O2)),  # N1--O2
    ]
    _WC_HBONDS[(U, G)] = [(b, a) for a, b in _WC_HBONDS[(G, U)]]


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


def secondary_structure(polymer: "Polymer", min_loop_size: int = 3):
    """Extract base pairs from polymer hydrogen bond connections.

    Determines Watson-Crick base pairs (A-U, G-C, G-U wobble) from hydrogen
    bond connections. Only pairs with ALL expected hydrogen bonds present
    are included (2 for A-U/G-U, 3 for G-C), filtering out tertiary
    interactions like A-minor motifs.

    Args:
        polymer: RNA polymer with connections loaded. Must be loaded with
            ``ciffy.load(file, skip=[])`` or ``skip=["descriptions"]`` to
            include connections.
        min_loop_size: Minimum number of residues between paired positions.
            Pairs with |i - j| <= min_loop_size are filtered out. Default 3.

    Returns:
        Array of shape (n_pairs, 2) where each row [i, j] is a base pair
        with i < j. Returns empty array of shape (0, 2) if no pairs found.
        Array type matches the polymer's backend (numpy or torch).

    Raises:
        ValueError: If polymer has no connections loaded.

    Example:
        >>> import ciffy
        >>> polymer = ciffy.load("structure.cif", skip=[])
        >>> from ciffy.rna import secondary_structure
        >>> pairs = secondary_structure(polymer)
        >>> pairs
        array([[0, 6],
               [1, 5]])
    """
    from ..backend import ops
    from ..biochemistry import Scale

    # Initialize WC patterns on first call
    if not _WC_HBONDS:
        _init_wc_hbonds()

    if polymer.connections is None:
        raise ValueError(
            "Polymer has no connections. Load with skip=[] to include connections."
        )

    # Reference array for backend-aware operations
    ref = polymer._hierarchy._ref

    connections = np.asarray(polymer.connections)
    atoms = np.asarray(polymer.atoms)
    sequence = np.asarray(polymer.sequence)

    if connections.size == 0:
        return ops.empty((0, 2), like=ref, dtype='int64')

    # Get residue membership for each atom
    residue_membership = np.asarray(polymer.membership(Scale.RESIDUE))
    n_atoms = polymer.size()

    # Filter connections to only include atoms in this polymer
    valid_mask = (connections[:, 0] < n_atoms) & (connections[:, 1] < n_atoms)
    connections = connections[valid_mask]

    if connections.size == 0:
        return ops.empty((0, 2), like=ref, dtype='int64')

    # Build set of (res_i, res_j, atom_i, atom_j) for fast lookup
    # Normalize so res_i < res_j
    hbond_set: set[tuple[int, int, int, int]] = set()
    for atom1, atom2 in connections:
        res1 = residue_membership[atom1]
        res2 = residue_membership[atom2]
        if res1 == res2:
            continue
        if res1 < res2:
            hbond_set.add((res1, res2, atoms[atom1], atoms[atom2]))
        else:
            hbond_set.add((res2, res1, atoms[atom2], atoms[atom1]))

    # Find Watson-Crick base pairs with all required H-bonds
    base_pairs: list[tuple[int, int]] = []

    # Check all potential residue pairs
    checked: set[tuple[int, int]] = set()
    for res_i, res_j, _, _ in hbond_set:
        if (res_i, res_j) in checked:
            continue
        checked.add((res_i, res_j))

        # Skip if loop too small
        if res_j - res_i <= min_loop_size:
            continue

        # Get base types
        base_i = sequence[res_i]
        base_j = sequence[res_j]

        # Look up required H-bonds for this base pair type
        required = _WC_HBONDS.get((base_i, base_j))
        if required is None:
            continue

        # Check if ALL required H-bonds are present
        all_present = all(
            (res_i, res_j, atom_i, atom_j) in hbond_set
            for atom_i, atom_j in required
        )

        if all_present:
            base_pairs.append((res_i, res_j))

    if not base_pairs:
        return ops.empty((0, 2), like=ref, dtype='int64')

    # Each residue can only be in one base pair - keep longest range pairs
    paired: dict[int, tuple[int, int]] = {}  # residue -> (partner, distance)
    for i, j in base_pairs:
        dist = j - i
        if i not in paired or dist > paired[i][1]:
            paired[i] = (j, dist)
        if j not in paired or dist > paired[j][1]:
            paired[j] = (i, dist)

    # Reconstruct pairs ensuring each residue is in at most one pair
    final_pairs = set()
    for res, (partner, _) in paired.items():
        if res < partner:
            if partner in paired and paired[partner][0] == res:
                final_pairs.add((res, partner))

    if not final_pairs:
        return ops.empty((0, 2), like=ref, dtype='int64')

    # Convert to array in polymer's backend
    result = np.array(sorted(final_pairs), dtype=np.int64)
    return ops.to_backend(result, like=ref)
