"""
Selection module for Polymer objects.

Provides functions for filtering, iterating, and selecting molecular structures.
"""

# Filter functions
from .filters import (
    by_index,
    atom_type,
    residue_type,
    molecule_type,
    element_type,
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
    "atom_type",
    "residue_type",
    "molecule_type",
    "element_type",
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
