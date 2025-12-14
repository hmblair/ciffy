"""
Biochemistry constants and enumerations.

Defines atoms, residues, elements, and nucleotide structures.

Atom enums are accessed via Residue members:
    Residue.A.C3p.value    # atom index for C3' in adenosine
    Residue.ALA.CA.value   # atom index for CA in alanine
"""

from ._generated_elements import Element, ELEMENT_NAMES
from ._generated_residues import Residue, CIF_RESIDUE_NAMES
from ._generated_atoms import (
    # Combined enums (for bulk operations)
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
from .molecule import Molecule, molecule_type

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
    # Residues (access atoms via Residue.X.atom_name)
    "Residue",
    # Combined atom enums (for bulk operations)
    "RibonucleicAcid",
    "RibonucleicAcidNoPrefix",
    "DeoxyribonucleicAcid",
    "ModifiedNucleotides",
    "AminoAcids",
    # Atom group constants
    "Backbone",
    "Nucleobase",
    "Phosphate",
    "Sidechain",
    # Linking
    "LinkingDefinition",
    "NUCLEIC_ACID_LINK",
    "PEPTIDE_LINK",
    "LINKING_BY_TYPE",
    # Molecule types
    "Molecule",
    "molecule_type",
]
