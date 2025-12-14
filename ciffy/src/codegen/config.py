"""
Code generation configuration and constants.

This module contains all constants and data definitions used during code generation:
- Element symbols and atomic numbers
- Ion identifiers
- Residue whitelist
- Molecule type definitions
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Add package root to path so we can import from ciffy.types during build
_PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ciffy.types.dihedral import DIHEDRAL_TYPE_TO_INDEX

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

# Dihedral type index mapping - derived from ciffy.types.dihedral (single source of truth)
DIHEDRAL_TYPE_INDEX: dict[str, int] = {
    dtype.value: idx for dtype, idx in DIHEDRAL_TYPE_TO_INDEX.items()
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
