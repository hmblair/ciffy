"""
Code generation configuration and constants.

This module contains all constants and data definitions used during code generation:
- Element symbols and atomic numbers
- Ion identifiers
- Residue whitelist
- Molecule type definitions
- Dihedral type definitions (single source of truth)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# URL for the PDB Chemical Component Dictionary
CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


# =============================================================================
# CONSTANTS - Single source of truth for elements and ions
# =============================================================================

# Element symbol -> atomic number
ELEMENTS: dict[str, int] = {
    "H": 1, "LI": 3, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12,
    "AL": 13, "P": 15, "S": 16, "CL": 17, "K": 19, "CA": 20, "MN": 25,
    "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30, "SE": 34, "BR": 35,
    "RB": 37, "SR": 38, "MO": 42, "AG": 47, "CD": 48, "I": 53, "CS": 55,
    "BA": 56, "W": 74, "PT": 78, "AU": 79, "HG": 80, "PB": 82,
}

# Single-atom ions (used for classification and gperf generation)
IONS: set[str] = {
    "AG", "AL", "AU", "BA", "BR", "CA", "CD", "CL", "CO", "CS", "CU",
    "F", "FE", "HG", "I", "K", "LI", "MG", "MN", "NA", "NI", "PB",
    "PT", "RB", "SE", "SR", "W", "ZN",
}


# =============================================================================
# RESIDUE WHITELIST
# =============================================================================
# Only these residues will be included. Set to None to include all from CCD.

RESIDUE_WHITELIST: set[str] | None = {
    # Standard RNA nucleotides
    "A", "C", "G", "U",
    "N",    # Unknown nucleotide (ribose-phosphate backbone only)
    # Standard DNA nucleotides
    "DA", "DC", "DG", "DT",
    # Standard amino acids (20)
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",  # Unknown amino acid
    # Common modified nucleotides
    "PSU",  # Pseudouridine
    "5MU",  # 5-methyluridine
    "1MG",  # 1-methylguanosine
    "2MG",  # 2-methylguanosine
    "7MG",  # 7-methylguanosine
    "M2G",  # N2-methylguanosine
    "OMG",  # 2'-O-methylguanosine
    "OMC",  # 2'-O-methylcytidine
    "OMU",  # 2'-O-methyluridine
    "5MC",  # 5-methylcytidine
    "H2U",  # Dihydrouridine
    "4SU",  # 4-thiouridine
    "FHU",  # 5-fluorohydroxyuridine (modified uracil)
    "PPU",  # Puromycin (modified adenosine)
    "I",    # Inosine
    "2MA",  # 2-methyladenosine-5'-monophosphate (RNA)
    "6MZ",  # N6-methyladenosine-5'-monophosphate (RNA)
    # Additional modified amino acids
    "MEQ",  # N5-methylglutamine
    "MS6",  # 2-amino-4-(methylsulfanyl)butane-1-thiol
    "4D4",  # Modified arginine
    # Common modified amino acids
    "MSE",  # Selenomethionine
    "SEP",  # Phosphoserine
    "TPO",  # Phosphothreonine
    "PTR",  # Phosphotyrosine
    "CSO",  # S-hydroxycysteine
    "OCS",  # Cysteinesulfonic acid
    "HYP",  # Hydroxyproline
    "MLY",  # N-dimethyl-lysine
    # Water, ions, and common ligands
    "HOH", "MG", "K", "NA", "ZN", "ACT",
    "G7M",  # 2'-O-7-methylguanosine (modified RNA)
    "6O1",  # Evernimicin (antibiotic ligand)
    "GTP",  # Guanosine triphosphate
    "CCC",  # Cytidine-5'-monophosphate
    "GNG",  # Guanine
    "CS",   # Cesium ion
}

# =============================================================================
# MOLECULE TYPE DEFINITIONS
# =============================================================================
# Order determines integer values. This is the single source of truth.

@dataclass
class MoleculeType:
    """Definition for a molecule type."""
    name: str  # Enum name (e.g., "RNA")
    entity_poly_type: str | None  # mmCIF _entity_poly.type value, or None
    description: str  # Documentation string


# Ordered list - integer values assigned sequentially (index = value)
MOLECULE_TYPES: list[MoleculeType] = [
    # Polymer types (from _entity_poly.type)
    MoleculeType("PROTEIN", "polypeptide(L)", "Standard L-amino acid chains"),
    MoleculeType("RNA", "polyribonucleotide", "Ribonucleic acid"),
    MoleculeType("DNA", "polydeoxyribonucleotide", "Deoxyribonucleic acid"),
    MoleculeType("HYBRID", "polydeoxyribonucleotide/polyribonucleotide hybrid", "DNA/RNA hybrid"),
    MoleculeType("PROTEIN_D", "polypeptide(D)", "D-amino acid chains (rare)"),
    MoleculeType("POLYSACCHARIDE", "polysaccharide(D)", "Carbohydrates"),
    MoleculeType("PNA", "peptide nucleic acid", "Peptide nucleic acid (synthetic)"),
    MoleculeType("CYCLIC_PEPTIDE", "cyclic-pseudo-peptide", "Cyclic peptides"),
    # Non-polymer types (from _entity.type, no _entity_poly.type)
    MoleculeType("LIGAND", None, "Small molecules, cofactors, drugs"),
    MoleculeType("ION", None, "Metal ions (Mg2+, Ca2+, Zn2+, etc.)"),
    MoleculeType("WATER", None, "Water molecules (HOH)"),
    # Special
    MoleculeType("OTHER", "other", "Unclassified polymer type"),
    MoleculeType("UNKNOWN", None, "Residue type not recognized"),
]


# Build name -> index mapping for easy access
class Molecule:
    """Molecule type constants. Access via Molecule.RNA, Molecule.DNA, etc."""
    pass


for _idx, _mt in enumerate(MOLECULE_TYPES):
    setattr(Molecule, _mt.name, _idx)


# =============================================================================
# DIHEDRAL TYPE DEFINITIONS
# =============================================================================
# Single source of truth for dihedral angle types. Order determines integer values.


@dataclass
class DihedralDefinition:
    """Definition for a dihedral angle type."""

    name: str  # Enum name (e.g., "PHI", "PSI")
    atoms: tuple[str, str, str, str]  # Atom names defining the dihedral (Python naming)
    molecule_types: tuple[int, ...]  # Which Molecule type indices have this dihedral
    is_backbone: bool  # True if backbone dihedral
    is_glycosidic: bool = False  # True if glycosidic bond dihedral
    is_sidechain: bool = False  # True if sidechain chi angle


# Ordered list - integer values assigned sequentially (index = value)
# Protein backbone: PHI=0, PSI=1, OMEGA=2
# Nucleic acid backbone: ALPHA=3, BETA=4, GAMMA=5, DELTA=6, EPSILON=7, ZETA=8
# Glycosidic: CHI_PURINE=9, CHI_PYRIMIDINE=10
# Protein sidechain: CHI1=11, CHI2=12, CHI3=13, CHI4=14
DIHEDRAL_TYPES: list[DihedralDefinition] = [
    # Protein backbone (0-2)
    # PHI: C(i-1) - N(i) - CA(i) - C(i)
    DihedralDefinition("PHI", ("C", "N", "CA", "C"), (Molecule.PROTEIN,), True),
    # PSI: N(i) - CA(i) - C(i) - N(i+1)
    DihedralDefinition("PSI", ("N", "CA", "C", "N"), (Molecule.PROTEIN,), True),
    # OMEGA: CA(i) - C(i) - N(i+1) - CA(i+1)
    DihedralDefinition("OMEGA", ("CA", "C", "N", "CA"), (Molecule.PROTEIN,), True),
    # Nucleic acid backbone (3-8)
    # ALPHA: O3'(i-1) - P(i) - O5'(i) - C5'(i)
    DihedralDefinition("ALPHA", ("O3p", "P", "O5p", "C5p"), (Molecule.RNA, Molecule.DNA), True),
    # BETA: P(i) - O5'(i) - C5'(i) - C4'(i)
    DihedralDefinition("BETA", ("P", "O5p", "C5p", "C4p"), (Molecule.RNA, Molecule.DNA), True),
    # GAMMA: O5'(i) - C5'(i) - C4'(i) - C3'(i)
    DihedralDefinition("GAMMA", ("O5p", "C5p", "C4p", "C3p"), (Molecule.RNA, Molecule.DNA), True),
    # DELTA: C5'(i) - C4'(i) - C3'(i) - O3'(i)
    DihedralDefinition("DELTA", ("C5p", "C4p", "C3p", "O3p"), (Molecule.RNA, Molecule.DNA), True),
    # EPSILON: C4'(i) - C3'(i) - O3'(i) - P(i+1)
    DihedralDefinition("EPSILON", ("C4p", "C3p", "O3p", "P"), (Molecule.RNA, Molecule.DNA), True),
    # ZETA: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
    DihedralDefinition("ZETA", ("C3p", "O3p", "P", "O5p"), (Molecule.RNA, Molecule.DNA), True),
    # Glycosidic dihedrals (9-10)
    # CHI_PURINE: O4' - C1' - N9 - C4 (adenine, guanine)
    DihedralDefinition("CHI_PURINE", ("O4p", "C1p", "N9", "C4"), (Molecule.RNA, Molecule.DNA), False, True),
    # CHI_PYRIMIDINE: O4' - C1' - N1 - C2 (cytosine, uracil, thymine)
    DihedralDefinition("CHI_PYRIMIDINE", ("O4p", "C1p", "N1", "C2"), (Molecule.RNA, Molecule.DNA), False, True),
    # Protein sidechain dihedrals (11-14)
    # CHI1: N - CA - CB - XG (varies by residue)
    DihedralDefinition("CHI1", ("N", "CA", "CB", "XG"), (Molecule.PROTEIN,), False, False, True),
    # CHI2: CA - CB - CG - XD (varies by residue)
    DihedralDefinition("CHI2", ("CA", "CB", "CG", "XD"), (Molecule.PROTEIN,), False, False, True),
    # CHI3: CB - CG - CD - XE (varies by residue)
    DihedralDefinition("CHI3", ("CB", "CG", "CD", "XE"), (Molecule.PROTEIN,), False, False, True),
    # CHI4: CG - CD - XE - XZ (LYS, ARG only)
    DihedralDefinition("CHI4", ("CG", "CD", "XE", "XZ"), (Molecule.PROTEIN,), False, False, True),
]

# Derived mappings from DIHEDRAL_TYPES
DIHEDRAL_TYPE_INDEX: dict[str, int] = {
    d.name.lower(): i for i, d in enumerate(DIHEDRAL_TYPES)
}

DIHEDRAL_ATOMS: dict[str, tuple[str, str, str, str]] = {
    d.name.lower(): d.atoms for d in DIHEDRAL_TYPES
}


# =============================================================================
# BACKBONE NAME IDS - For inter-residue reference resolution in C
# =============================================================================
# These are canonical identifiers for backbone atoms that can be referenced
# across residue boundaries. Used in the C Z-matrix builder to resolve
# inter-residue refs without knowing specific atom types.

BACKBONE_NAMES: dict[str, int] = {
    # Protein backbone
    "N": 0,
    "CA": 1,
    "C": 2,
    "O": 3,
    # Nucleic acid backbone
    "P": 4,
    "OP1": 5,
    "OP2": 6,
    "O5'": 7,
    "C5'": 8,
    "C4'": 9,
    "O4'": 10,
    "C3'": 11,
    "O3'": 12,
    "C2'": 13,
    "C1'": 14,
    "O2'": 15,  # RNA only
}

# Python name -> CIF name for backbone atoms
BACKBONE_PYTHON_TO_CIF: dict[str, str] = {
    "N": "N",
    "CA": "CA",
    "C": "C",
    "O": "O",
    "P": "P",
    "OP1": "OP1",
    "OP2": "OP2",
    "O5p": "O5'",
    "C5p": "C5'",
    "C4p": "C4'",
    "O4p": "O4'",
    "C3p": "C3'",
    "O3p": "O3'",
    "C2p": "C2'",
    "C1p": "C1'",
    "O2p": "O2'",
}

NUM_BACKBONE_NAMES: int = len(BACKBONE_NAMES)
