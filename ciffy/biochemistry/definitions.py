"""
Residue and atom definitions - single source of truth.

This file defines all residue types and their atoms in a declarative format.
The code generator reads from here to produce:
- Python enums (Residue, atom classes)
- C hash tables (gperf)
- Reverse lookup tables

To add a new residue:
  1. Create a ResidueDefinition with all required fields
  2. Add it to ALL_RESIDUES (position = index)
  3. Run: python ciffy/src/codegen/generate.py
  4. Rebuild: pip install -e .
"""

from dataclasses import dataclass, field
from typing import Optional

from ..types import Molecule


@dataclass
class ResidueDefinition:
    """Complete definition of a residue type."""

    name: str
    """Enum name (e.g., 'ADE', 'DA', 'ALA')."""

    cif_names: list[str]
    """CIF file names that map to this residue (e.g., ['A'] or ['DA'])."""

    molecule_type: Molecule
    """Molecular classification (RNA, DNA, PROTEIN, etc.)."""

    abbreviation: str
    """Single-letter code (lowercase for nucleotides, uppercase for amino acids)."""

    atoms: list[str]
    """Ordered list of atom names. Order determines atom indices."""

    class_name: Optional[str] = None
    """Python class name for atom enum. Defaults to name if not specified."""

    def __post_init__(self):
        if self.class_name is None:
            self.class_name = self.name.title()


# =============================================================================
# RNA NUCLEOTIDES (indices 0-3)
# =============================================================================

ADENOSINE = ResidueDefinition(
    name="ADE",
    cif_names=["A"],
    molecule_type=Molecule.RNA,
    abbreviation="a",
    class_name="Adenosine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Nucleobase
        "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
        # Hydrogens
        "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
        "H1'", "H8", "H61", "H62", "H2",
    ],
)

CYTIDINE = ResidueDefinition(
    name="CYT",
    cif_names=["C"],
    molecule_type=Molecule.RNA,
    abbreviation="c",
    class_name="Cytosine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Nucleobase
        "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6",
        # Hydrogens
        "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
        "H1'", "H41", "H42", "H5", "H6",
    ],
)

GUANOSINE = ResidueDefinition(
    name="GUA",
    cif_names=["G"],
    molecule_type=Molecule.RNA,
    abbreviation="g",
    class_name="Guanosine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Nucleobase
        "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        # Hydrogens
        "HOP3", "HOP2", "H5''", "H5'", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
        "H1'", "H8", "H1", "H21", "H22",
    ],
)

URIDINE = ResidueDefinition(
    name="URA",
    cif_names=["U"],
    molecule_type=Molecule.RNA,
    abbreviation="u",
    class_name="Uridine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Nucleobase
        "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6",
        # Hydrogens
        "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
        "H1'", "H3", "H5", "H6",
    ],
)

# =============================================================================
# DNA NUCLEOTIDES (indices 4-7) - UNIQUE INDICES, separate from RNA
# =============================================================================

DEOXYADENOSINE = ResidueDefinition(
    name="DA",
    cif_names=["DA"],
    molecule_type=Molecule.DNA,
    abbreviation="a",
    class_name="Deoxyadenosine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone (no O2')
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Nucleobase
        "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
        # Hydrogens (H2'' instead of HO2')
        "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "H2''", "HO5'",
        "H1'", "H8", "H61", "H62", "H2",
    ],
)

DEOXYCYTIDINE = ResidueDefinition(
    name="DC",
    cif_names=["DC"],
    molecule_type=Molecule.DNA,
    abbreviation="c",
    class_name="Deoxycytidine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone (no O2')
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Nucleobase
        "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6",
        # Hydrogens (H2'' instead of HO2')
        "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "H2''", "HO5'",
        "H1'", "H41", "H42", "H5", "H6",
    ],
)

DEOXYGUANOSINE = ResidueDefinition(
    name="DG",
    cif_names=["DG"],
    molecule_type=Molecule.DNA,
    abbreviation="g",
    class_name="Deoxyguanosine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone (no O2')
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Nucleobase
        "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        # Hydrogens (H2'' instead of HO2')
        "HOP3", "HOP2", "H5''", "H5'", "H4'", "H3'", "HO3'", "H2'", "H2''", "HO5'",
        "H1'", "H8", "H1", "H21", "H22",
    ],
)

THYMIDINE = ResidueDefinition(
    name="DT",
    cif_names=["DT", "T"],
    molecule_type=Molecule.DNA,
    abbreviation="t",
    class_name="Thymidine",
    atoms=[
        # Phosphate group
        "OP3", "P", "OP1", "OP2",
        # Sugar backbone (no O2')
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Nucleobase (C7/C5M is methyl group at C5)
        "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6",
        # Hydrogens (H2'' instead of HO2', H71-H73 for methyl)
        "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "H2''", "HO5'",
        "H1'", "H3", "H71", "H72", "H73", "H6",
    ],
)

# =============================================================================
# AMINO ACIDS (indices 8-27)
# =============================================================================

ALANINE = ResidueDefinition(
    name="ALA",
    cif_names=["ALA"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="A",
    class_name="Alanine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB",
        "H", "H2", "H3", "HA", "HB1", "HB2", "HB3",
    ],
)

CYSTEINE = ResidueDefinition(
    name="CYS",
    cif_names=["CYS"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="C",
    class_name="Cysteine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "SG",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG",
    ],
)

ASPARTIC_ACID = ResidueDefinition(
    name="ASP",
    cif_names=["ASP"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="D",
    class_name="AsparticAcid",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "OD1", "OD2",
        "H", "H2", "H3", "HA", "HB2", "HB3",
    ],
)

GLUTAMIC_ACID = ResidueDefinition(
    name="GLU",
    cif_names=["GLU"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="E",
    class_name="GlutamicAcid",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD", "OE1", "OE2",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3",
    ],
)

PHENYLALANINE = ResidueDefinition(
    name="PHE",
    cif_names=["PHE"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="F",
    class_name="Phenylalanine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ",
    ],
)

GLYCINE = ResidueDefinition(
    name="GLY",
    cif_names=["GLY"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="G",
    class_name="Glycine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "H", "H2", "H3", "HA2", "HA3",
    ],
)

HISTIDINE = ResidueDefinition(
    name="HIS",
    cif_names=["HIS"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="H",
    class_name="Histidine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "ND1", "CD2", "CE1", "NE2",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2",
    ],
)

ISOLEUCINE = ResidueDefinition(
    name="ILE",
    cif_names=["ILE"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="I",
    class_name="Isoleucine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG1", "CG2", "CD1",
        "H", "H2", "H3", "HA", "HB", "HG12", "HG13", "HG21", "HG22", "HG23",
        "HD11", "HD12", "HD13",
    ],
)

LYSINE = ResidueDefinition(
    name="LYS",
    cif_names=["LYS"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="K",
    class_name="Lysine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD", "CE", "NZ",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3",
        "HE2", "HE3", "HZ1", "HZ2", "HZ3",
    ],
)

LEUCINE = ResidueDefinition(
    name="LEU",
    cif_names=["LEU"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="L",
    class_name="Leucine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD1", "CD2",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG",
        "HD11", "HD12", "HD13", "HD21", "HD22", "HD23",
    ],
)

METHIONINE = ResidueDefinition(
    name="MET",
    cif_names=["MET"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="M",
    class_name="Methionine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "SD", "CE",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3",
    ],
)

ASPARAGINE = ResidueDefinition(
    name="ASN",
    cif_names=["ASN"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="N",
    class_name="Asparagine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "OD1", "ND2",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HD21", "HD22",
    ],
)

PROLINE = ResidueDefinition(
    name="PRO",
    cif_names=["PRO"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="P",
    class_name="Proline",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD",
        "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3",
    ],
)

GLUTAMINE = ResidueDefinition(
    name="GLN",
    cif_names=["GLN"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="Q",
    class_name="Glutamine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD", "OE1", "NE2",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HE21", "HE22",
    ],
)

ARGININE = ResidueDefinition(
    name="ARG",
    cif_names=["ARG"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="R",
    class_name="Arginine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3",
        "HE", "HH11", "HH12", "HH21", "HH22",
    ],
)

SERINE = ResidueDefinition(
    name="SER",
    cif_names=["SER"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="S",
    class_name="Serine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "OG",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HG",
    ],
)

THREONINE = ResidueDefinition(
    name="THR",
    cif_names=["THR"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="T",
    class_name="Threonine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "OG1", "CG2",
        "H", "H2", "H3", "HA", "HB", "HG1", "HG21", "HG22", "HG23",
    ],
)

VALINE = ResidueDefinition(
    name="VAL",
    cif_names=["VAL"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="V",
    class_name="Valine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG1", "CG2",
        "H", "H2", "H3", "HA", "HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23",
    ],
)

TRYPTOPHAN = ResidueDefinition(
    name="TRP",
    cif_names=["TRP"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="W",
    class_name="Tryptophan",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2",
    ],
)

TYROSINE = ResidueDefinition(
    name="TYR",
    cif_names=["TYR"],
    molecule_type=Molecule.PROTEIN,
    abbreviation="Y",
    class_name="Tyrosine",
    atoms=[
        "N", "CA", "C", "O", "OXT",
        "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH",
        "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HH",
    ],
)

# =============================================================================
# NON-POLYMER (indices 28-30)
# =============================================================================

WATER = ResidueDefinition(
    name="HOH",
    cif_names=["HOH"],
    molecule_type=Molecule.WATER,
    abbreviation="~",
    class_name="Water",
    atoms=["O", "H1", "H2"],
)

MAGNESIUM = ResidueDefinition(
    name="MG",
    cif_names=["MG"],
    molecule_type=Molecule.ION,
    abbreviation="~",
    class_name="Magnesium",
    atoms=["MG"],
)

CESIUM = ResidueDefinition(
    name="CS",
    cif_names=["CS"],
    molecule_type=Molecule.ION,
    abbreviation="~",
    class_name="Cesium",
    atoms=["CS"],
)

# =============================================================================
# MODIFIED NUCLEOTIDES (indices 31+)
# =============================================================================

GUANOSINE_TRIPHOSPHATE = ResidueDefinition(
    name="GTP",
    cif_names=["GTP"],
    molecule_type=Molecule.RNA,
    abbreviation="g",
    class_name="GuanosineTriphosphate",
    atoms=[
        # Gamma phosphate group
        "PG", "O1G", "O2G", "O3G",
        # Beta phosphate group
        "O3B", "PB", "O1B", "O2B",
        # Alpha phosphate group
        "O3A", "PA", "O1A", "O2A",
        # Sugar backbone
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Nucleobase (guanine)
        "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        # Hydrogens
        "HOG2", "HOG3", "HOB2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "HO2'",
        "H1'", "H8", "H1", "H21", "H22",
    ],
)

CYTIDINE_TRIPHOSPHATE = ResidueDefinition(
    name="CCC",
    cif_names=["CCC"],
    molecule_type=Molecule.RNA,
    abbreviation="c",
    class_name="CytidineTriphosphate",
    atoms=[
        # Gamma phosphate group
        "PC", "O1C", "O2C",
        # Standard phosphate
        "P", "OP1", "OP2", "OP3",
        # Sugar backbone
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        # Nucleobase (cytosine)
        "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6",
        # Hydrogens
        "HOC2", "HOP2", "HOP3", "H5'", "H5''", "H4'", "H3'", "H2'", "H1'",
        "H41", "H42", "H5", "H6",
    ],
)

DEOXYGUANOSINE_2PRIME = ResidueDefinition(
    name="GNG",
    cif_names=["GNG"],
    molecule_type=Molecule.DNA,
    abbreviation="g",
    class_name="Deoxyguanosine2Prime",
    atoms=[
        # Phosphate group
        "P", "OP1", "OP2", "OP3",
        # Sugar backbone (no O2')
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        # Nucleobase (guanine)
        "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        # Hydrogens
        "HOP2", "HOP3", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "H2''",
        "H1'", "H8", "H1", "H21", "H22",
    ],
)


# =============================================================================
# ALL RESIDUES - Order determines index
# =============================================================================

ALL_RESIDUES: list[ResidueDefinition] = [
    # RNA nucleotides (indices 0-3)
    ADENOSINE,      # 0
    CYTIDINE,       # 1
    GUANOSINE,      # 2
    URIDINE,        # 3
    # DNA nucleotides (indices 4-7) - UNIQUE INDICES
    DEOXYADENOSINE, # 4
    DEOXYCYTIDINE,  # 5
    DEOXYGUANOSINE, # 6
    THYMIDINE,      # 7
    # Amino acids (indices 8-27)
    ALANINE,        # 8
    CYSTEINE,       # 9
    ASPARTIC_ACID,  # 10
    GLUTAMIC_ACID,  # 11
    PHENYLALANINE,  # 12
    GLYCINE,        # 13
    HISTIDINE,      # 14
    ISOLEUCINE,     # 15
    LYSINE,         # 16
    LEUCINE,        # 17
    METHIONINE,     # 18
    ASPARAGINE,     # 19
    PROLINE,        # 20
    GLUTAMINE,      # 21
    ARGININE,       # 22
    SERINE,         # 23
    THREONINE,      # 24
    VALINE,         # 25
    TRYPTOPHAN,     # 26
    TYROSINE,       # 27
    # Non-polymer (indices 28-30)
    WATER,          # 28
    MAGNESIUM,      # 29
    CESIUM,         # 30
    # Modified nucleotides (indices 31+)
    GUANOSINE_TRIPHOSPHATE,  # 31
    CYTIDINE_TRIPHOSPHATE,   # 32
    DEOXYGUANOSINE_2PRIME,   # 33
]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_definitions() -> None:
    """
    Validate all residue definitions.

    Raises:
        ValueError: If any validation check fails.
    """
    seen_names: set[str] = set()
    seen_cif_names: dict[str, str] = {}

    for idx, res in enumerate(ALL_RESIDUES):
        # Check for duplicate residue names
        if res.name in seen_names:
            raise ValueError(f"Duplicate residue name: {res.name}")
        seen_names.add(res.name)

        # Check for duplicate CIF names
        for cif_name in res.cif_names:
            if cif_name in seen_cif_names:
                raise ValueError(
                    f"Duplicate CIF name '{cif_name}' used by both "
                    f"'{seen_cif_names[cif_name]}' and '{res.name}'"
                )
            seen_cif_names[cif_name] = res.name

        # Check for duplicate atoms within residue
        seen_atoms: set[str] = set()
        for atom in res.atoms:
            if atom in seen_atoms:
                raise ValueError(
                    f"Duplicate atom '{atom}' in residue '{res.name}'"
                )
            seen_atoms.add(atom)

        # Check required fields
        if not res.atoms:
            raise ValueError(f"Residue '{res.name}' has no atoms defined")
        if not res.cif_names:
            raise ValueError(f"Residue '{res.name}' has no CIF names defined")


# Run validation on import (catches errors early)
validate_definitions()
