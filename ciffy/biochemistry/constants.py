"""
Biochemistry constants for structure analysis.

This module provides two types of atom groupings:

1. **Flat IndexEnums** (legacy): Backbone, Nucleobase, Phosphate, Sidechain
   - Prefixed names like "A_C5p", "GLY_CA"
   - Use: `Backbone.index()` for all values

2. **Hierarchical Atom Groups** (preferred): Sugar, PurineBase, PyrimidineBase, etc.
   - Nested structure: `Group.atom.residue.value`
   - Use: `Sugar.C5p.index()` for all C5' values

Hierarchical Access Patterns
----------------------------
::

    PurineBase.N1.index()  # All N1 atoms in purines (A, G, DA, DG)
    PurineBase.N1.A.value  # Just adenine N1 value
    PurineBase.index()     # All purine base atom values
    Sugar.C5p.index()      # All C5' atoms across all nucleotides

Single Source of Truth
----------------------
All values come from the generated atom enums, ensuring consistency::

    PurineBase.N1.A.value == Residue.A.N1.value  # Always True

Adding New Atom Groups
----------------------
To add a new atom group (e.g., aromatic sidechain atoms):

1. **Define residue sources** - list of (name, ResidueType) tuples::

    _AROMATIC_RESIDUES = [
        ("PHE", Residue.PHE),
        ("TYR", Residue.TYR),
        ("TRP", Residue.TRP),
        ("HIS", Residue.HIS),
    ]

2. **Define atom filter** - set of atom names to include::

    _AROMATIC_RING_NAMES = {
        'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  # PHE/TYR ring
        'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2',  # TRP indole
        'ND1', 'CE1', 'NE2',  # HIS imidazole
    }

3. **Build the group** using `build_atom_group`::

    AromaticRing = build_atom_group(
        "AromaticRing",
        _AROMATIC_RESIDUES,
        _AROMATIC_RING_NAMES
    )

4. **Export** from `__init__.py`::

    from .constants import AromaticRing

The resulting group supports::

    AromaticRing.CG.PHE.value   # Specific atom value
    AromaticRing.CG.index()     # All CG values across aromatics
    AromaticRing.index()        # All aromatic ring atom values
    polymer.by_atom(AromaticRing.index())  # Select all aromatic atoms

Finding Atom Names
------------------
To find available atom names for a residue::

    >>> from ciffy.biochemistry import Residue
    >>> [atom.name for atom in Residue.PHE]
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', ...]

Existing Groups
---------------
- **Sugar**: Ribose/deoxyribose atoms (C1'-C5', O2'-O5', hydrogens)
- **PhosphateGroup**: Phosphate atoms (P, OP1, OP2, OP3)
- **PurineBase**: Full purine nucleobase (A, G, DA, DG)
- **PurineImidazole**: 5-membered ring of purines
- **PurinePyrimidine**: 6-membered ring of purines
- **PyrimidineBase**: Pyrimidine nucleobase (C, U, DC, DT)
"""

from typing import Callable

from ..utils import IndexEnum, build_atom_group
from ._generated_residues import Residue

# Residue groupings (prefix, residue) - atoms accessed via residue.atoms
_RNA_NUCLEOTIDES = [
    ("A_", Residue.A),
    ("C_", Residue.C),
    ("G_", Residue.G),
    ("U_", Residue.U),
]

_DNA_NUCLEOTIDES = [
    ("DA_", Residue.DA),
    ("DC_", Residue.DC),
    ("DG_", Residue.DG),
    ("DT_", Residue.DT),
]

_AMINO_ACIDS = [
    ("GLY_", Residue.GLY), ("ALA_", Residue.ALA), ("VAL_", Residue.VAL), ("LEU_", Residue.LEU),
    ("ILE_", Residue.ILE), ("PRO_", Residue.PRO), ("PHE_", Residue.PHE),
    ("TRP_", Residue.TRP), ("MET_", Residue.MET), ("CYS_", Residue.CYS),
    ("SER_", Residue.SER), ("THR_", Residue.THR), ("ASN_", Residue.ASN),
    ("GLN_", Residue.GLN), ("ASP_", Residue.ASP), ("GLU_", Residue.GLU),
    ("LYS_", Residue.LYS), ("ARG_", Residue.ARG), ("HIS_", Residue.HIS), ("TYR_", Residue.TYR),
]

# Protein backbone atom names
_PROTEIN_BACKBONE_NAMES = {'N', 'CA', 'C', 'O'}


def _filter_atoms(
    residues: list[tuple[str, type]],
    predicate: Callable[[str], bool],
) -> dict[str, int]:
    """
    Filter atoms across residues using a predicate.

    Args:
        residues: List of (prefix, ResidueType) tuples.
        predicate: Function that takes an atom name and returns True to include.

    Returns:
        Dictionary mapping prefixed atom names to their indices.
    """
    result = {}
    for prefix, residue in residues:
        for name, value in residue.atoms.dict().items():
            if predicate(name):
                result[prefix + name] = value
    return result


# Nucleic acid backbone: sugar-phosphate atoms (contain 'p' or 'P')
_nucleic_backbone = lambda n: 'p' in n or 'P' in n

# Nucleobase atoms: neither 'p' nor 'P'
_nucleobase = lambda n: 'p' not in n and 'P' not in n

# Phosphate atoms: contain uppercase 'P'
_phosphate = lambda n: 'P' in n

# Protein backbone atoms
_protein_backbone = lambda n: n in _PROTEIN_BACKBONE_NAMES

# Sidechain atoms: not backbone
_sidechain = lambda n: n not in _PROTEIN_BACKBONE_NAMES and n not in {'OXT', 'H', 'H2', 'H3', 'HA', 'HA2', 'HA3'}


# Combined Backbone: RNA + DNA + Protein
Backbone = IndexEnum(
    "Backbone",
    _filter_atoms(_RNA_NUCLEOTIDES, _nucleic_backbone) |
    _filter_atoms(_DNA_NUCLEOTIDES, _nucleic_backbone) |
    _filter_atoms(_AMINO_ACIDS, _protein_backbone)
)

# Nucleobase atoms (RNA only for now)
Nucleobase = IndexEnum(
    "Nucleobase",
    _filter_atoms(_RNA_NUCLEOTIDES, _nucleobase)
)

# Phosphate atoms (RNA + DNA)
Phosphate = IndexEnum(
    "Phosphate",
    _filter_atoms(_RNA_NUCLEOTIDES, _phosphate) |
    _filter_atoms(_DNA_NUCLEOTIDES, _phosphate)
)

# Sidechain atoms (protein only)
Sidechain = IndexEnum(
    "Sidechain",
    _filter_atoms(_AMINO_ACIDS, _sidechain)
)


# =============================================================================
# Hierarchical Atom Group Classes
# =============================================================================
#
# These use HierarchicalEnumMeta to provide hierarchical access to atoms by
# chemical identity. Each class has nested IndexEnums for each atom position,
# plus standard IndexEnum-like methods (index, dict, list, revdict).
#
# Source of truth: Values come from generated atom enums (A, G, C, U, etc.)
# so PurineBase.N1.A.value == Residue.A.N1.value is always true.

# Residue groupings for hierarchical access
_PURINES = [("A", Residue.A), ("G", Residue.G), ("DA", Residue.DA), ("DG", Residue.DG)]
_PYRIMIDINES = [("C", Residue.C), ("U", Residue.U), ("DC", Residue.DC), ("DT", Residue.DT)]
_ALL_NUCLEOTIDES = _PURINES + _PYRIMIDINES


# =============================================================================
# Atom name sets by chemical identity
# =============================================================================

# Sugar (ribose/deoxyribose) - identical across all nucleotides
_SUGAR_NAMES = {
    'C1p', 'C2p', 'C3p', 'C4p', 'C5p',  # Ring carbons
    'O2p', 'O3p', 'O4p', 'O5p',          # Ring/chain oxygens
    'H1p', 'H2p', 'H3p', 'H4p', 'H5p', 'H5pp', 'HO2p', 'HO3p', 'H2pp',  # Hydrogens
}

# Phosphate group - identical across all nucleotides
_PHOSPHATE_NAMES = {'P', 'OP1', 'OP2', 'OP3', 'HOP2', 'HOP3'}

# Purine imidazole ring (5-membered)
_PURINE_IMIDAZOLE_NAMES = {'N9', 'C8', 'N7', 'C5', 'C4', 'H8'}

# Purine pyrimidine ring (6-membered, fused with imidazole)
_PURINE_PYRIMIDINE_NAMES = {
    'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'H2',  # Core ring
    'N6', 'H61', 'H62',  # Adenine-specific
    'O6', 'N2', 'H1', 'H21', 'H22',  # Guanine-specific
}

# Full purine base (union of imidazole and pyrimidine rings)
_PURINE_BASE_NAMES = _PURINE_IMIDAZOLE_NAMES | _PURINE_PYRIMIDINE_NAMES

# Pyrimidine base (single 6-membered ring)
_PYRIMIDINE_BASE_NAMES = {
    'N1', 'C2', 'O2', 'N3', 'C4', 'C5', 'C6', 'H5', 'H6',  # Core ring
    'H3',  # U-specific
    'N4', 'H41', 'H42',  # C-specific
    'O4',  # U-specific
    'C7', 'H71', 'H72', 'H73',  # T-specific (methyl)
}


# =============================================================================
# Build hierarchical atom group classes
# =============================================================================

# Sugar atoms - present in all nucleotides
Sugar = build_atom_group("Sugar", _ALL_NUCLEOTIDES, _SUGAR_NAMES)

# Phosphate atoms - present in all nucleotides
PhosphateGroup = build_atom_group("PhosphateGroup", _ALL_NUCLEOTIDES, _PHOSPHATE_NAMES)

# Purine hierarchy - A, G, DA, DG only
PurineImidazole = build_atom_group("PurineImidazole", _PURINES, _PURINE_IMIDAZOLE_NAMES)
PurinePyrimidine = build_atom_group("PurinePyrimidine", _PURINES, _PURINE_PYRIMIDINE_NAMES)
PurineBase = build_atom_group("PurineBase", _PURINES, _PURINE_BASE_NAMES)

# Pyrimidine base - C, U, DC, DT only
PyrimidineBase = build_atom_group("PyrimidineBase", _PYRIMIDINES, _PYRIMIDINE_BASE_NAMES)
