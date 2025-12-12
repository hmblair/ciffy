"""
Biochemistry constants and enumerations.

Defines atoms, residues, elements, and nucleotide structures.
"""

from .elements import Element, ELEMENT_NAMES
from .residues import Residue, RESIDUE_ABBREV, RESIDUE_MOLECULE_TYPE
from .nucleotides import (
    # RNA
    Adenosine,
    Cytosine,
    Guanosine,
    Uridine,
    RibonucleicAcid,
    RibonucleicAcidNoPrefix,
    # DNA
    Deoxyadenosine,
    Deoxycytidine,
    Deoxyguanosine,
    Thymidine,
    DeoxyribonucleicAcid,
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
# These are computed from the enums and include +1 for index 0 (unknown/padding)

# Number of element types (max index + 1)
NUM_ELEMENTS: int = max(e.value for e in Element) + 1

# Number of residue types (max index + 1)
NUM_RESIDUES: int = max(r.value for r in Residue) + 1

# Number of atom types - computed from definitions
# Index 0 is reserved for unknown, so max index + 1
from .definitions import ALL_RESIDUES
NUM_ATOMS: int = sum(len(r.atoms) for r in ALL_RESIDUES) + 1

# Reverse lookup: atom index -> atom name
from ._generated_atoms import ATOM_NAMES


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
    "Adenosine",
    "Cytosine",
    "Guanosine",
    "Uridine",
    "RibonucleicAcid",
    "RibonucleicAcidNoPrefix",
    # DNA nucleotides
    "Deoxyadenosine",
    "Deoxycytidine",
    "Deoxyguanosine",
    "Thymidine",
    "DeoxyribonucleicAcid",
    # Constants
    "Backbone",
    "Nucleobase",
    "Phosphate",
    "Sidechain",
]
