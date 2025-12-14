"""
Core type definitions for ciffy.
"""

from .scale import Scale
from .dihedral import DihedralType

# Re-export Molecule from biochemistry for backwards compatibility
from ..biochemistry import Molecule

__all__ = [
    "Scale",
    "Molecule",
    "DihedralType",
]
