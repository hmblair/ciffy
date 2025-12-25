"""
Utility classes and functions for ciffy.
"""

from .atom import (
    Atom,
    AtomGroup,
    build_atom_group,
)

from .enum_base import IndexEnum
from .helpers import filter_by_mask, all_equal

__all__ = [
    # Atom system (2 classes)
    "Atom",
    "AtomGroup",
    "build_atom_group",
    # Enum utilities
    "IndexEnum",
    # Helpers
    "filter_by_mask",
    "all_equal",
]
