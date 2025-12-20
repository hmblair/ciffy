"""
Utility classes and functions for ciffy.
"""

from .enum_base import (
    IndexEnum,
    PairEnum,
    ResidueType,
    ResidueMeta,
    HierarchicalEnumMeta,
    build_hierarchical_enum,
    build_atom_group,
)
from .helpers import filter_by_mask, all_equal

__all__ = [
    "IndexEnum",
    "PairEnum",
    "ResidueType",
    "ResidueMeta",
    "HierarchicalEnumMeta",
    "build_hierarchical_enum",
    "build_atom_group",
    "filter_by_mask",
    "all_equal",
]
