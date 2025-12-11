"""
Biochemistry constants and enumerations.

Defines atoms, residues, elements, and nucleotide structures.
"""

from .elements import Element
from .residues import Residue, RES_ABBREV, RESIDUE_MOLECULE_TYPE, residue_to_molecule
from .nucleotides import (
    Adenosine,
    Cytosine,
    Guanosine,
    Uridine,
    RibonucleicAcid,
    RibonucleicAcidNoPrefix,
)
from .constants import (
    FRAMES,
    FRAME1,
    FRAME2,
    FRAME3,
    COARSE,
    Backbone,
    Nucleobase,
    Phosphate,
)

# =============================================================================
# VOCABULARY SIZES (for embedding layers)
# =============================================================================
# These are computed from the enums and include +1 for index 0 (unknown/padding)

# Number of element types (max index + 1)
# Elements use atomic numbers, so max is 16 (S) + 1 = 17
NUM_ELEMENTS: int = max(e.value for e in Element) + 1

# Number of residue types (max index + 1)
# Residues are 0-indexed, max is ~30
NUM_RESIDUES: int = max(r.value for r in Residue) + 1

# Number of atom types - imported from generated file
# This is the total across all residue types
from .atoms import ALL_ATOMS
NUM_ATOMS: int = sum(len(atoms) for atoms in ALL_ATOMS.values()) + 1  # +1 for unknown

# Reverse lookup: atom index -> atom name
from ._generated_atoms import ATOM_NAMES


__all__ = [
    # Vocabulary sizes
    "NUM_ELEMENTS",
    "NUM_RESIDUES",
    "NUM_ATOMS",
    # Reverse lookup
    "ATOM_NAMES",
    # Elements
    "Element",
    # Residues
    "Residue",
    "RES_ABBREV",
    "RESIDUE_MOLECULE_TYPE",
    "residue_to_molecule",
    # Nucleotides
    "Adenosine",
    "Cytosine",
    "Guanosine",
    "Uridine",
    "RibonucleicAcid",
    "RibonucleicAcidNoPrefix",
    # Constants
    "FRAMES",
    "FRAME1",
    "FRAME2",
    "FRAME3",
    "COARSE",
    "Backbone",
    "Nucleobase",
    "Phosphate",
]
