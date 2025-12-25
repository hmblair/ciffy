"""
Utility classes and functions for ciffy.
"""

from .atom import (
    Atom,
    AtomGroup,
    build_atom_group,
)

from .helpers import filter_by_mask, all_equal

__all__ = [
    # Atom system (2 classes)
    "Atom",
    "AtomGroup",
    "build_atom_group",
    # Helpers
    "filter_by_mask",
    "all_equal",
]
