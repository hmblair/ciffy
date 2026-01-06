"""
Selection module for Polymer objects.

Provides functions for filtering, iterating, and selecting molecular structures.
"""

# Filter functions
from .filters import (
    by_index,
    by_atom,
    by_residue,
    by_type,
    by_element,
)

# Iteration functions
from .iterators import (
    hetero,
    chains,
)

# Mask and specialized selection functions
from .masks import (
    mask,
    resolved,
    strip,
    backbone,
    nucleobase,
    phosphate,
    sidechain,
    heavy,
)

__all__ = [
    # Filters
    "by_index",
    "by_atom",
    "by_residue",
    "by_type",
    "by_element",
    # Iterators
    "hetero",
    "chains",
    # Masks
    "mask",
    "resolved",
    "strip",
    "backbone",
    "nucleobase",
    "phosphate",
    "sidechain",
    "heavy",
]
