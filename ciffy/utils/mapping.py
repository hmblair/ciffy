"""
Array mapping utilities.

Functions for converting between array representations and dictionary mappings,
used extensively for atom-to-column index lookups throughout ciffy.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..backend.core import Array


def atoms_to_col_map(
    atoms: Sequence[int] | "Array",
) -> dict[int, int]:
    """
    Build atom value -> column index mapping from atoms array or sequence.

    This is the canonical function for creating atom-to-column mappings,
    used for coordinate indexing throughout ciffy.

    Args:
        atoms: Sequence of atom type indices (list, tuple, NumPy array, or tensor).

    Returns:
        Dict mapping each atom type value to its column index.

    Example:
        >>> import numpy as np
        >>> atoms = np.array([2, 5, 8, 12])  # P, O5', C5', C4'
        >>> col_map = atoms_to_col_map(atoms)
        >>> col_map[5]  # O5' is at column 1
        1
    """
    from ..backend import is_torch

    # Handle torch tensors - need tolist() for proper int conversion
    if hasattr(atoms, 'tolist') and is_torch(atoms):
        return {int(a): i for i, a in enumerate(atoms.tolist())}

    # Handle numpy arrays and Python sequences (list, tuple)
    return {int(a): i for i, a in enumerate(atoms)}
