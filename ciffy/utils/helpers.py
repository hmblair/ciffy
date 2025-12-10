"""
Helper functions for ciffy.

Common utilities used throughout the codebase.
"""

from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch
    import numpy as np
    # Array type for type hints - accepts both backends
    ArrayLike = Union[np.ndarray, "torch.Tensor"]

T = TypeVar('T')


def all_equal(*values) -> bool:
    """
    Check if all values are equal.

    Args:
        *values: Values to compare.

    Returns:
        True if all values are equal (or no values provided), False otherwise.
    """
    return len(set(values)) <= 1


def filter_by_mask(items: list[T], mask: "ArrayLike") -> list[T]:
    """
    Filter a list by a boolean array mask.

    Args:
        items: List of items to filter.
        mask: Boolean array (NumPy or PyTorch) where True indicates items to keep.

    Returns:
        New list containing only items where mask is True.

    Example:
        >>> items = ['a', 'b', 'c']
        >>> mask = np.array([True, False, True])
        >>> filter_by_mask(items, mask)
        ['a', 'c']
    """
    return [item for ix, item in enumerate(items) if mask[ix]]
