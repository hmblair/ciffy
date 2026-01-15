"""
Code generation configuration and constants.

This module contains all constants and data definitions used during code generation:
- Ion identifiers (loaded from residues.yaml)
- Residue whitelist (loaded from residues.yaml)
- Nucleotide classification (loaded from residues.yaml)
- Molecule type definitions
- Dihedral type definitions (single source of truth)

Curated biochemistry data is loaded from residues.yaml. Edit that file to
add or remove residues from code generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import yaml


class ConfigError(Exception):
    """Error loading or validating codegen configuration."""
    pass

# URL for the PDB Chemical Component Dictionary
CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"

# Path to the YAML config file
RESIDUES_YAML_PATH = Path(__file__).parent / "residues.yaml"


# =============================================================================
# MOLECULE TYPE DEFINITIONS
# =============================================================================
# Order determines integer values. This is the single source of truth.
# Defined early so YAML loader can validate against these names.


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

# Valid molecule type names for validation
VALID_MOLECULE_NAMES: set[str] = {mt.name for mt in MOLECULE_TYPES}


# Build name -> index mapping for easy access
class Molecule:
    """Molecule type constants. Access via Molecule.RNA, Molecule.DNA, etc."""
    pass


for _idx, _mt in enumerate(MOLECULE_TYPES):
    setattr(Molecule, _mt.name, _idx)


# Molecule type groups for convenience
# Used in linking and backbone detection across related types
NUCLEIC_MOLECULE_TYPES: tuple[int, ...] = (Molecule.RNA, Molecule.DNA, Molecule.HYBRID)
PROTEIN_MOLECULE_TYPES: tuple[int, ...] = (Molecule.PROTEIN, Molecule.PROTEIN_D, Molecule.CYCLIC_PEPTIDE)


# =============================================================================
# YAML CONFIG LOADER
# =============================================================================


@dataclass
class ClassificationRule:
    """A molecule classification rule from YAML config."""
    pattern: str  # Substring to match in CCD type (case-insensitive)
    molecule_type: str  # Name of Molecule enum member


class ResidueConfig(TypedDict):
    """Type definition for residues.yaml structure."""
    residue_whitelist: list[str] | None
    ions: list[str]
    purine_residues: list[str]
    pyrimidine_residues: list[str]
    molecule_classification: list[ClassificationRule]
    molecule_classification_default: str


def load_residue_config(path: Path | None = None) -> ResidueConfig:
    """
    Load curated biochemistry data from residues.yaml.

    Args:
        path: Path to YAML file. Defaults to codegen/residues.yaml.

    Returns:
        Dictionary with all configuration including classification rules.

    Raises:
        ConfigError: If the file is missing, malformed, or contains invalid values.
    """
    if path is None:
        path = RESIDUES_YAML_PATH

    # Load YAML with informative errors
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            f"This file is required for code generation. If you're developing ciffy,\n"
            f"ensure the codegen/residues.yaml file exists."
        ) from None
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Invalid YAML in configuration file: {path}\n"
            f"Error: {e}"
        ) from None

    if data is None:
        raise ConfigError(f"Configuration file is empty: {path}")

    # Parse and validate classification rules
    raw_rules = data.get('molecule_classification', [])
    classification_rules = []

    for i, rule in enumerate(raw_rules):
        # Validate rule structure
        if not isinstance(rule, dict):
            raise ConfigError(
                f"Invalid classification rule at index {i} in {path}:\n"
                f"Expected a mapping with 'pattern' and 'molecule_type' keys, got {type(rule).__name__}"
            )

        if 'pattern' not in rule:
            raise ConfigError(
                f"Classification rule at index {i} missing 'pattern' key in {path}"
            )

        if 'molecule_type' not in rule:
            raise ConfigError(
                f"Classification rule at index {i} missing 'molecule_type' key in {path}"
            )

        # Validate molecule_type is a known value
        mol_type = rule['molecule_type']
        if mol_type not in VALID_MOLECULE_NAMES:
            raise ConfigError(
                f"Invalid molecule_type '{mol_type}' in classification rule at index {i} in {path}\n"
                f"Valid values are: {', '.join(sorted(VALID_MOLECULE_NAMES))}"
            )

        classification_rules.append(
            ClassificationRule(pattern=rule['pattern'], molecule_type=mol_type)
        )

    # Validate default molecule type
    default_mol = data.get('molecule_classification_default', 'OTHER')
    if default_mol not in VALID_MOLECULE_NAMES:
        raise ConfigError(
            f"Invalid molecule_classification_default '{default_mol}' in {path}\n"
            f"Valid values are: {', '.join(sorted(VALID_MOLECULE_NAMES))}"
        )

    return ResidueConfig(
        residue_whitelist=data.get('residue_whitelist'),
        ions=data.get('ions', []),
        purine_residues=data.get('purine_residues', []),
        pyrimidine_residues=data.get('pyrimidine_residues', []),
        molecule_classification=classification_rules,
        molecule_classification_default=default_mol,
    )


# Load configuration from YAML
_config = load_residue_config()

# Single-atom ions (used for classification and gperf generation)
IONS: set[str] = set(_config['ions'])

# Residue whitelist - only these residues will be included from CCD.
# Set to None to include all residues (not recommended).
RESIDUE_WHITELIST: set[str] | None = (
    set(_config['residue_whitelist']) if _config['residue_whitelist'] else None
)

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
# SIDECHAIN CHI ANGLE DEFINITIONS
# =============================================================================
# Residue-specific chi angle definitions for amino acid sidechains.
# Maps residue name -> chi_name -> (atom1, atom2, atom3, atom4)
# All atoms are in the same residue (offset 0).
#
# These are the actual atom names from the CCD, unlike DIHEDRAL_TYPES
# which uses placeholder names (XG, XD, etc.) for the generic definitions.

SIDECHAIN_CHI_DEFS: dict[str, dict[str, tuple[str, str, str, str]]] = {
    # Standard amino acids (17 with chi angles, GLY and ALA have none)
    "ARG": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "NE"), "chi4": ("CG", "CD", "NE", "CZ")},
    "ASN": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "OD1")},
    "ASP": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "OD1")},
    "CYS": {"chi1": ("N", "CA", "CB", "SG")},
    "GLN": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "OE1")},
    "GLU": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "OE1")},
    "HIS": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "ND1")},
    "ILE": {"chi1": ("N", "CA", "CB", "CG1"), "chi2": ("CA", "CB", "CG1", "CD1")},
    "LEU": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "LYS": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "CE"), "chi4": ("CG", "CD", "CE", "NZ")},
    "MET": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "SD"),
            "chi3": ("CB", "CG", "SD", "CE")},
    "PHE": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "PRO": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD")},
    "SER": {"chi1": ("N", "CA", "CB", "OG")},
    "THR": {"chi1": ("N", "CA", "CB", "OG1")},
    "TRP": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "TYR": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "VAL": {"chi1": ("N", "CA", "CB", "CG1")},
    # Modified amino acids (inherit from parent residue)
    "MSE": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "SE"),
            "chi3": ("CB", "CG", "SE", "CE")},  # Selenomethionine (like MET)
    "SEP": {"chi1": ("N", "CA", "CB", "OG")},  # Phosphoserine (like SER)
    "TPO": {"chi1": ("N", "CA", "CB", "OG1")},  # Phosphothreonine (like THR)
    "PTR": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},  # Phosphotyrosine (like TYR)
    "CSO": {"chi1": ("N", "CA", "CB", "SG")},  # S-hydroxycysteine (like CYS)
    "OCS": {"chi1": ("N", "CA", "CB", "SG")},  # Cysteinesulfonic acid (like CYS)
    "HYP": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD")},  # Hydroxyproline (like PRO)
    "MLY": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "CE"), "chi4": ("CG", "CD", "CE", "NZ")},  # N-dimethyl-lysine (like LYS)
}


# =============================================================================
# BACKBONE NAME IDS - For inter-residue reference resolution in C
# =============================================================================
# These are canonical identifiers for backbone atoms that can be referenced
# across residue boundaries. Used to resolve inter-residue refs without
# knowing specific atom types.

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


# =============================================================================
# UNIFIED BACKBONE ATOM VALUES
# =============================================================================
# Backbone atoms share the same integer value across all residue types.
# This enables:
# 1. Robustness to modified residues with standard backbones
# 2. Simpler mental model - a backbone P is the same regardless of residue
#
# Values are 1-indexed (0 reserved for unknown). Sidechain/base atoms
# start at NUM_UNIFIED_BACKBONE + 1.

# Which atoms are considered "backbone" for each molecule type
BACKBONE_ATOMS_NUCLEIC: frozenset[str] = frozenset({
    "P", "OP1", "OP2", "OP3",  # Phosphate group (OP3 is 5' terminal)
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",  # Sugar
})

BACKBONE_ATOMS_PROTEIN: frozenset[str] = frozenset({
    "N", "CA", "C", "O",  # Peptide backbone
})

# All backbone atoms (union of nucleic and protein)
ALL_BACKBONE_ATOMS: frozenset[str] = BACKBONE_ATOMS_NUCLEIC | BACKBONE_ATOMS_PROTEIN


# =============================================================================
# TERMINAL ATOM DEFINITIONS
# =============================================================================
# Atoms present only at chain termini. Tuple of (start_atoms, end_atoms).
# Used for filtering during chain building with Residue.terminal().

TERMINAL_ATOMS: dict[int, tuple[frozenset[str], frozenset[str]]] = {
    Molecule.RNA: (frozenset({'OP3', 'HOP3'}), frozenset({'HO3p'})),
    Molecule.DNA: (frozenset({'OP3', 'HOP3'}), frozenset({'HO3p'})),
    Molecule.HYBRID: (frozenset({'OP3', 'HOP3'}), frozenset({'HO3p'})),
    Molecule.PROTEIN: (frozenset({'H2', 'H3'}), frozenset({'OXT', 'HXT'})),
    Molecule.PROTEIN_D: (frozenset({'H2', 'H3'}), frozenset({'OXT', 'HXT'})),
    Molecule.CYCLIC_PEPTIDE: (frozenset({'H2', 'H3'}), frozenset({'OXT', 'HXT'})),
}

# Note: UNIFIED_BACKBONE_VALUES is now auto-generated during codegen
# based on CCD atom order, just like non-backbone atoms.
# See codegen/__init__.py _build_indices() for the generation logic.
# The generated values are exported to ciffy/biochemistry/_generated_atoms.py


def is_backbone_atom(atom_name: str, molecule_type: int) -> bool:
    """
    Check if an atom is a backbone atom for the given molecule type.

    Args:
        atom_name: Atom name in CIF format (e.g., "P", "O3'", "CA").
        molecule_type: Molecule type index (e.g., Molecule.RNA, Molecule.PROTEIN).

    Returns:
        True if the atom is a backbone atom for this molecule type.
    """
    if molecule_type in NUCLEIC_MOLECULE_TYPES:
        return atom_name in BACKBONE_ATOMS_NUCLEIC
    elif molecule_type in PROTEIN_MOLECULE_TYPES:
        return atom_name in BACKBONE_ATOMS_PROTEIN
    return False


# =============================================================================
# NUCLEOTIDE CLASSIFICATION (Purine vs Pyrimidine)
# =============================================================================
# Used to determine which chi angle (CHI_PURINE vs CHI_PYRIMIDINE) applies.
# Purines have N9 as glycosidic nitrogen, pyrimidines have N1.
# Loaded from residues.yaml.

PURINE_RESIDUES: set[str] = set(_config['purine_residues'])

PYRIMIDINE_RESIDUES: set[str] = set(_config['pyrimidine_residues'])

# Molecule classification rules (loaded from YAML)
MOLECULE_CLASSIFICATION_RULES: list[ClassificationRule] = _config['molecule_classification']
MOLECULE_CLASSIFICATION_DEFAULT: str = _config['molecule_classification_default']
