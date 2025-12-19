"""
Dihedral angle type enumeration for biomolecules.

Re-exports DihedralType from biochemistry for backwards compatibility.
The canonical definition is now auto-generated in biochemistry/_generated_dihedraltypes.py.
"""

# Re-export generated DihedralType and convenience tuples
from ..biochemistry._generated_dihedraltypes import (
    DihedralType,
    PROTEIN_BACKBONE,
    RNA_BACKBONE,
    RNA_GLYCOSIDIC,
    RNA_BACKBONE_EXTENDED,
    PROTEIN_SIDECHAIN,
    DIHEDRAL_NAME_TO_TYPE,
    DIHEDRAL_ATOMS,
    INDEX_TO_DIHEDRAL_TYPE,
    NUM_PROTEIN_BACKBONE_DIHEDRALS,
    NUM_RNA_BACKBONE_DIHEDRALS,
    MAX_DIHEDRALS_PER_RESIDUE,
)

__all__ = [
    "DihedralType",
    "PROTEIN_BACKBONE",
    "RNA_BACKBONE",
    "RNA_GLYCOSIDIC",
    "RNA_BACKBONE_EXTENDED",
    "PROTEIN_SIDECHAIN",
    "DIHEDRAL_NAME_TO_TYPE",
    "DIHEDRAL_ATOMS",
    "INDEX_TO_DIHEDRAL_TYPE",
    "NUM_PROTEIN_BACKBONE_DIHEDRALS",
    "NUM_RNA_BACKBONE_DIHEDRALS",
    "MAX_DIHEDRALS_PER_RESIDUE",
]
