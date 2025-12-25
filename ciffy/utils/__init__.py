"""
Utility classes and functions for ciffy.
"""

from .atom import (
    Atom,
    AtomGroup,
    build_atom_group,
)

from .enum_base import (
    # Legacy enum system (to be removed after full migration)
    IndexEnum,
    PairEnum,
    ResidueType,
    ResidueMeta,
    HierarchicalEnumMeta,
    build_hierarchical_enum,
    build_atom_group as build_atom_group_legacy,
)
from .helpers import filter_by_mask, all_equal

__all__ = [
    # New atom system (2 classes only)
    "Atom",
    "AtomGroup",
    "build_atom_group",
    # Legacy enum system (to be removed after migration)
    "IndexEnum",
    "PairEnum",
    "ResidueType",
    "ResidueMeta",
    "HierarchicalEnumMeta",
    "build_hierarchical_enum",
    "build_atom_group_legacy",
    # Helpers
    "filter_by_mask",
    "all_equal",
]
