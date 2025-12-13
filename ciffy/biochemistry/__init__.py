"""
Biochemistry constants and enumerations.

Defines atoms, residues, elements, and nucleotide structures.
"""

from ._generated_elements import Element, ELEMENT_NAMES
from ._generated_residues import Residue, CIF_RESIDUE_NAMES
from ._generated_atoms import (
    # RNA (CCD names)
    A, C, G, U,
    # DNA (CCD names)
    DA, DC, DG, DT,
    # Amino acids (CCD names)
    ALA, ARG, ASN, ASP, CYS,
    GLN, GLU, GLY, HIS, ILE,
    LEU, LYS, MET, PHE, PRO,
    SER, THR, TRP, TYR, VAL,
    # Combined enums
    RibonucleicAcid,
    RibonucleicAcidNoPrefix,
    DeoxyribonucleicAcid,
    ModifiedNucleotides,
    AminoAcids,
    # Reverse lookup
    ATOM_NAMES,
)
from .constants import (
    Backbone,
    Nucleobase,
    Phosphate,
    Sidechain,
)
from .linking import (
    LinkingDefinition,
    NUCLEIC_ACID_LINK,
    PEPTIDE_LINK,
    LINKING_BY_TYPE,
)

# =============================================================================
# VOCABULARY SIZES (for embedding layers)
# =============================================================================

# Number of element types (max index + 1)
NUM_ELEMENTS: int = max(e.value for e in Element) + 1

# Number of residue types
NUM_RESIDUES: int = len(Residue)

# Number of atom types (max index + 1)
NUM_ATOMS: int = max(ATOM_NAMES.keys()) + 1


__all__ = [
    # Vocabulary sizes
    "NUM_ELEMENTS",
    "NUM_RESIDUES",
    "NUM_ATOMS",
    # Reverse lookups
    "ATOM_NAMES",
    "ELEMENT_NAMES",
    "CIF_RESIDUE_NAMES",
    # Elements
    "Element",
    # Residues
    "Residue",
    # RNA nucleotides
    "A", "C", "G", "U",
    "RibonucleicAcid",
    "RibonucleicAcidNoPrefix",
    # DNA nucleotides
    "DA", "DC", "DG", "DT",
    "DeoxyribonucleicAcid",
    # Modified nucleotides
    "ModifiedNucleotides",
    # Amino acids
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "AminoAcids",
    # Constants
    "Backbone",
    "Nucleobase",
    "Phosphate",
    "Sidechain",
    # Linking
    "LinkingDefinition",
    "NUCLEIC_ACID_LINK",
    "PEPTIDE_LINK",
    "LINKING_BY_TYPE",
]
