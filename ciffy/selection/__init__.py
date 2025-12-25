"""
Selection module for Polymer objects.

Provides functions for filtering, iterating, and selecting molecular structures.
"""

# Filter functions
from .filters import (
    by_index,
    by_atom,
    by_residue,
    by_residue_index,
    by_type,
)

# Iteration functions
from .iterators import (
    poly,
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
)

__all__ = [
    # Filters
    "by_index",
    "by_atom",
    "by_residue",
    "by_residue_index",
    "by_type",
    # Iterators
    "poly",
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
]
