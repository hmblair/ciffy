"""
Biochemistry constants for structure analysis.

Provides atom groups for selecting atoms by chemical identity across
multiple residue types.

Two types of groups:

1. **Flat groups** (Backbone, Nucleobase, Phosphate, Sidechain):
   - Simple collections of atoms across residue types
   - Use: `Backbone.index()` for all backbone atom values
   - Use: `polymer.by_atom(Backbone.index())`

2. **Hierarchical groups** (Sugar, PurineBase, etc.):
   - Nested structure for accessing atoms by position and residue
   - Use: `Sugar.C5p.A` for adenine C5' atom
   - Use: `Sugar.C5p.index()` for all C5' values

Hierarchical Access Patterns
----------------------------
::

    PurineBase.N1.index()  # All N1 atoms in purines (A, G, DA, DG)
    PurineBase.N1.A        # Atom for adenine N1
    PurineBase.index()     # All purine base atom values
    Sugar.C5p.index()      # All C5' atoms across all nucleotides

Existing Groups
---------------
Flat groups:
- **Backbone**: Protein backbone (N, CA, C, O) + nucleic acid sugar-phosphate
- **Nucleobase**: RNA nucleobase atoms (no sugar or phosphate)
- **Phosphate**: Phosphate atoms across all nucleotides
- **Sidechain**: Protein sidechain atoms (non-backbone)

Hierarchical groups:
- **Sugar**: Ribose/deoxyribose atoms (C1'-C5', O2'-O5', hydrogens)
- **PhosphateGroup**: Phosphate atoms (P, OP1, OP2, OP3)
- **PurineBase**: Full purine nucleobase (A, G, DA, DG)
- **PurineImidazole**: 5-membered ring of purines
- **PurinePyrimidine**: 6-membered ring of purines
- **PyrimidineBase**: Pyrimidine nucleobase (C, U, DC, DT)
"""

from typing import Callable

from ..utils import Atom, AtomGroup, build_atom_group
from ._generated_residues_v2 import Residue


# =============================================================================
# Helper for building flat atom groups
# =============================================================================

def _build_flat_group(
    name: str,
    residues: list[tuple[str, AtomGroup]],
    predicate: Callable[[str], bool],
) -> AtomGroup:
    """
    Build a flat AtomGroup from residues using a predicate on atom names.

    Args:
        name: Name for the group.
        residues: List of (prefix, AtomGroup) pairs.
        predicate: Function taking atom name, returns True to include.

    Returns:
        AtomGroup with all matching atoms (prefixed names to avoid collisions).
    """
    atoms: dict[str, Atom] = {}
    for prefix, res in residues:
        for atom in res:
            if predicate(atom.name):
                key = f"{prefix}_{atom.name}"
                atoms[key] = Atom(key, int(atom), atom.local)
    return AtomGroup(name, atoms)

# =============================================================================
# Residue groupings
# =============================================================================

_RNA_NUCLEOTIDES = [
    ("A", Residue.A), ("C", Residue.C),
    ("G", Residue.G), ("U", Residue.U),
]
_DNA_NUCLEOTIDES = [
    ("DA", Residue.DA), ("DC", Residue.DC),
    ("DG", Residue.DG), ("DT", Residue.DT),
]
_PURINES = [
    ("A", Residue.A), ("G", Residue.G),
    ("DA", Residue.DA), ("DG", Residue.DG),
]
_PYRIMIDINES = [
    ("C", Residue.C), ("U", Residue.U),
    ("DC", Residue.DC), ("DT", Residue.DT),
]
_ALL_NUCLEOTIDES = _PURINES + _PYRIMIDINES

_AMINO_ACIDS = [
    ("GLY", Residue.GLY), ("ALA", Residue.ALA), ("VAL", Residue.VAL),
    ("LEU", Residue.LEU), ("ILE", Residue.ILE), ("PRO", Residue.PRO),
    ("PHE", Residue.PHE), ("TRP", Residue.TRP), ("MET", Residue.MET),
    ("CYS", Residue.CYS), ("SER", Residue.SER), ("THR", Residue.THR),
    ("ASN", Residue.ASN), ("GLN", Residue.GLN), ("ASP", Residue.ASP),
    ("GLU", Residue.GLU), ("LYS", Residue.LYS), ("ARG", Residue.ARG),
    ("HIS", Residue.HIS), ("TYR", Residue.TYR),
]


# =============================================================================
# Predicates for flat groups
# =============================================================================

# Nucleic acid backbone: sugar-phosphate atoms (contain 'p' or 'P')
_is_nucleic_backbone = lambda n: 'p' in n or 'P' in n

# Nucleobase atoms: neither 'p' nor 'P'
_is_nucleobase = lambda n: 'p' not in n and 'P' not in n

# Phosphate atoms: contain uppercase 'P'
_is_phosphate = lambda n: 'P' in n

# Protein backbone atoms
_PROTEIN_BACKBONE_NAMES = {'N', 'CA', 'C', 'O'}
_is_protein_backbone = lambda n: n in _PROTEIN_BACKBONE_NAMES
_is_sidechain = lambda n: n not in _PROTEIN_BACKBONE_NAMES


# =============================================================================
# Flat atom groups (for polymer selection methods)
# =============================================================================

# Backbone: nucleic acid sugar-phosphate + protein backbone
Backbone = AtomGroup(
    "Backbone",
    {
        **_build_flat_group("_rna", _RNA_NUCLEOTIDES, _is_nucleic_backbone)._members,
        **_build_flat_group("_dna", _DNA_NUCLEOTIDES, _is_nucleic_backbone)._members,
        **_build_flat_group("_prot", _AMINO_ACIDS, _is_protein_backbone)._members,
    }
)

# Nucleobase atoms (RNA only)
Nucleobase = _build_flat_group("Nucleobase", _RNA_NUCLEOTIDES, _is_nucleobase)

# Phosphate atoms (RNA + DNA)
Phosphate = _build_flat_group(
    "Phosphate",
    _RNA_NUCLEOTIDES + _DNA_NUCLEOTIDES,
    _is_phosphate
)

# Sidechain atoms (protein only)
Sidechain = _build_flat_group("Sidechain", _AMINO_ACIDS, _is_sidechain)


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
# Build hierarchical atom groups
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
