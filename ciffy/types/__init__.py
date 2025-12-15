"""
Core type definitions for ciffy.
"""

from .scale import Scale
from .dihedral import DihedralType, PROTEIN_BACKBONE, RNA_BACKBONE

# Re-export Molecule from biochemistry for backwards compatibility
from ..biochemistry import Molecule

__all__ = [
    "Scale",
    "Molecule",
    "DihedralType",
    "PROTEIN_BACKBONE",
    "RNA_BACKBONE",
]
