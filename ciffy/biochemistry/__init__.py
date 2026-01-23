"""
Biochemistry constants and enumerations.

Defines atoms, residues, elements, and nucleotide structures.

Residue members are AtomGroups with geometry:
    Residue.A.P             # Atom(P, 2) - IS an int
    Residue.A.ideal         # (n, 3) ideal coordinates
    Residue.A.bonds         # (n, 2) bond pairs
    Residue.ALA.CA          # Atom for CA in alanine
"""

# Core atom classes
from .atom import Atom, AtomGroup, build_atom_group, atom_to_element

# Scale enum for hierarchical levels
from .scale import Scale

from ._generated_elements import Element, ELEMENT_NAMES
from ._generated_molecule import Molecule, molecule_type

# =============================================================================
# RESIDUE SYSTEM - AtomGroup-based
# =============================================================================
from ._generated_residues import Residue

# DihedralType enum
from ._generated_dihedraltypes import (
    DihedralType,
    PROTEIN_BACKBONE,
    RNA_BACKBONE,
    RNA_GLYCOSIDIC,
    RNA_BACKBONE_EXTENDED,
    PROTEIN_SIDECHAIN,
    DIHEDRAL_NAME_TO_TYPE,
    DIHEDRAL_ATOMS,
)

# Flat atom groups (for polymer selection methods)
from .constants import (
    Backbone,
    Nucleobase,
    Phosphate,
    Sidechain,
)

# Hierarchical atom groups
from .constants import (
    Sugar,
    PhosphateGroup,
    PurineImidazole,
    PurinePyrimidine,
    PurineBase,
    PyrimidineBase,
)

# Molecule-type namespaces for structured access
from .groups import RNA, DNA, Protein

# Canonical residue lists
from .constants import CANONICAL_ALL

# Van der Waals radii
from .constants import VDW_RADII, VDW_RADII_ARRAY

from .linking import (
    LinkingDefinition,
    NUCLEIC_ACID_LINK,
    PEPTIDE_LINK,
    LINKING_BY_TYPE,
)

# =============================================================================
# VOCABULARY SIZES (for embedding layers)
# =============================================================================
# All indices start at 1, with 0 reserved for unknown/sentinel.
# Sizes are max_value + 1 to accommodate index 0.

NUM_ELEMENTS: int = max(e.value for e in Element) + 1
NUM_RESIDUES: int = max(r.value for r in Residue.all()) + 1
NUM_ATOMS: int = Atom.count()


__all__ = [
    # Core atom classes
    "Atom",
    "AtomGroup",
    "build_atom_group",
    "atom_to_element",
    # Scale enum
    "Scale",
    # Vocabulary sizes
    "NUM_ELEMENTS",
    "NUM_RESIDUES",
    "NUM_ATOMS",
    # Reverse lookups
    "ELEMENT_NAMES",
    # Elements
    "Element",
    # Residues (AtomGroup-based)
    "Residue",
    # Flat atom groups (for polymer selection)
    "Backbone",
    "Nucleobase",
    "Phosphate",
    "Sidechain",
    # Hierarchical atom groups
    "Sugar",
    "PhosphateGroup",
    "PurineImidazole",
    "PurinePyrimidine",
    "PurineBase",
    "PyrimidineBase",
    # Molecule-type namespaces
    "RNA",
    "DNA",
    "Protein",
    # Canonical residue lists
    "CANONICAL_ALL",
    # Van der Waals radii
    "VDW_RADII",
    "VDW_RADII_ARRAY",
    # Linking
    "LinkingDefinition",
    "NUCLEIC_ACID_LINK",
    "PEPTIDE_LINK",
    "LINKING_BY_TYPE",
    # Molecule types
    "Molecule",
    "molecule_type",
    # Dihedral types
    "DihedralType",
    "PROTEIN_BACKBONE",
    "RNA_BACKBONE",
    "RNA_GLYCOSIDIC",
    "RNA_BACKBONE_EXTENDED",
    "PROTEIN_SIDECHAIN",
    "DIHEDRAL_NAME_TO_TYPE",
    "DIHEDRAL_ATOMS",
]
