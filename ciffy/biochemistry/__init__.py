"""
Biochemistry constants and enumerations.

Defines atoms, residues, elements, and nucleotide structures.
"""

from ._generated_elements import Element, ELEMENT_NAMES
from ._generated_residues import Residue, RESIDUE_ABBREV, RESIDUE_MOLECULE_TYPE
from ._generated_atoms import (
    # RNA (CCD names)
    A, C, G, U,
    # DNA (CCD names)
    Da, Dc, Dg, Dt,
    # Amino acids (CCD names)
    Ala, Arg, Asn, Asp, Cys,
    Gln, Glu, Gly, His, Ile,
    Leu, Lys, Met, Phe, Pro,
    Ser, Thr, Trp, Tyr, Val,
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

# =============================================================================
# VOCABULARY SIZES (for embedding layers)
# =============================================================================

# Number of element types (max index + 1)
NUM_ELEMENTS: int = max(e.value for e in Element) + 1

# Number of residue types (max index + 1)
NUM_RESIDUES: int = max(r.value for r in Residue) + 1

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
    # Elements
    "Element",
    # Residues
    "Residue",
    "RESIDUE_ABBREV",
    "RESIDUE_MOLECULE_TYPE",
    # RNA nucleotides
    "A", "C", "G", "U",
    "RibonucleicAcid",
    "RibonucleicAcidNoPrefix",
    # DNA nucleotides
    "Da", "Dc", "Dg", "Dt",
    "DeoxyribonucleicAcid",
    # Modified nucleotides
    "ModifiedNucleotides",
    # Amino acids
    "Ala", "Arg", "Asn", "Asp", "Cys",
    "Gln", "Glu", "Gly", "His", "Ile",
    "Leu", "Lys", "Met", "Phe", "Pro",
    "Ser", "Thr", "Trp", "Tyr", "Val",
    "AminoAcids",
    # Constants
    "Backbone",
    "Nucleobase",
    "Phosphate",
    "Sidechain",
]
