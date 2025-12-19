"""
Core type definitions for ciffy.
"""

from .scale import Scale
from .dihedral import (
    DihedralType, PROTEIN_BACKBONE, RNA_BACKBONE, RNA_GLYCOSIDIC,
    DIHEDRAL_ATOMS, DIHEDRAL_NAME_TO_TYPE,
)

# Re-export Molecule from biochemistry for backwards compatibility
from ..biochemistry import Molecule

__all__ = [
    "Scale",
    "Molecule",
    "DihedralType",
    "PROTEIN_BACKBONE",
    "RNA_BACKBONE",
    "RNA_GLYCOSIDIC",
    "DIHEDRAL_ATOMS",
    "DIHEDRAL_NAME_TO_TYPE",
]
