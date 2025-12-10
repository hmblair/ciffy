"""
Atom definitions for nucleotides and amino acids.

This file defines the atoms for each residue type as simple lists.
The actual IndexEnum classes with integer indices are auto-generated
by `ciffy/src/codegen/generate.py` into `_generated_atoms.py`.

To add a new atom:
  1. Add it to the appropriate list below
  2. Run: python ciffy/src/codegen/generate.py
  3. Rebuild: pip install -e .

The order of atoms within each list determines their index, but you
don't need to manage indices manually - they're assigned automatically.
"""

# =============================================================================
# NUCLEOTIDES
# =============================================================================

# Adenosine (A) - purine nucleoside
ADENOSINE_ATOMS = [
    # Phosphate group
    "OP3", "P", "OP1", "OP2",
    # Sugar backbone
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    # Nucleobase
    "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
    # Hydrogens
    "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
    "H1'", "H8", "H61", "H62", "H2",
]

# Cytosine (C) - pyrimidine nucleoside
CYTOSINE_ATOMS = [
    # Phosphate group
    "OP3", "P", "OP1", "OP2",
    # Sugar backbone
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    # Nucleobase
    "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6",
    # Hydrogens
    "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
    "H1'", "H41", "H42", "H5", "H6",
]

# Guanosine (G) - purine nucleoside
GUANOSINE_ATOMS = [
    # Phosphate group
    "OP3", "P", "OP1", "OP2",
    # Sugar backbone
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    # Nucleobase
    "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
    # Hydrogens
    "HOP3", "HOP2", "H5''", "H5'", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
    "H1'", "H8", "H1", "H21", "H22",
]

# Uridine (U) - pyrimidine nucleoside
URIDINE_ATOMS = [
    # Phosphate group
    "OP3", "P", "OP1", "OP2",
    # Sugar backbone
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    # Nucleobase
    "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6",
    # Hydrogens
    "HOP3", "HOP2", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "HO2'", "HO5'",
    "H1'", "H3", "H5", "H6",
]

# Guanosine-5'-triphosphate (GTP)
GTP_ATOMS = [
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
]

# Cytidine-5'-triphosphate (CCC) - 3' terminal modification
CCC_ATOMS = [
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
]

# 2'-Deoxyguanosine (GNG)
GNG_ATOMS = [
    # Phosphate group
    "P", "OP1", "OP2", "OP3",
    # Sugar backbone (no O2')
    "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
    # Nucleobase (guanine)
    "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
    # Hydrogens
    "HOP2", "HOP3", "H5'", "H5''", "H4'", "H3'", "HO3'", "H2'", "H2''",
    "H1'", "H8", "H1", "H21", "H22",
]

# =============================================================================
# AMINO ACIDS
# =============================================================================

# Glycine (GLY) - simplest amino acid, no side chain
GLYCINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Hydrogens
    "H", "H2", "H3", "HA2", "HA3",
]

# Alanine (ALA)
ALANINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB1", "HB2", "HB3",
]

# Valine (VAL)
VALINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG1", "CG2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23",
]

# Leucine (LEU)
LEUCINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "CD1", "CD2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG",
    "HD11", "HD12", "HD13", "HD21", "HD22", "HD23",
]

# Isoleucine (ILE)
ISOLEUCINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG1", "CG2", "CD1",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB", "HG12", "HG13", "HG21", "HG22", "HG23",
    "HD11", "HD12", "HD13",
]

# Proline (PRO)
PROLINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain (cyclic)
    "CB", "CG", "CD",
    # Hydrogens (no H on ring N)
    "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3",
]

# Phenylalanine (PHE)
PHENYLALANINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain (benzyl)
    "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ",
]

# Tryptophan (TRP)
TRYPTOPHAN_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain (indole)
    "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2",
]

# Methionine (MET)
METHIONINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "SD", "CE",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3",
]

# Cysteine (CYS)
CYSTEINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "SG",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG",
]

# Serine (SER)
SERINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "OG",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG",
]

# Threonine (THR)
THREONINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "OG1", "CG2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB", "HG1", "HG21", "HG22", "HG23",
]

# Asparagine (ASN)
ASPARAGINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "OD1", "ND2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HD21", "HD22",
]

# Glutamine (GLN)
GLUTAMINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "CD", "OE1", "NE2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HE21", "HE22",
]

# Aspartic acid (ASP)
ASPARTIC_ACID_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "OD1", "OD2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3",
]

# Glutamic acid (GLU)
GLUTAMIC_ACID_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "CD", "OE1", "OE2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3",
]

# Lysine (LYS)
LYSINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "CD", "CE", "NZ",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3",
    "HE2", "HE3", "HZ1", "HZ2", "HZ3",
]

# Arginine (ARG)
ARGININE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain
    "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3",
    "HE", "HH11", "HH12", "HH21", "HH22",
]

# Histidine (HIS)
HISTIDINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain (imidazole)
    "CB", "CG", "ND1", "CD2", "CE1", "NE2",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2",
]

# Tyrosine (TYR)
TYROSINE_ATOMS = [
    # Backbone
    "N", "CA", "C", "O", "OXT",
    # Side chain (phenol)
    "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH",
    # Hydrogens
    "H", "H2", "H3", "HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HH",
]


# =============================================================================
# RESIDUE DEFINITIONS
# Maps 3-letter codes to atom lists for code generation
# =============================================================================

NUCLEOTIDE_ATOMS = {
    "A": ADENOSINE_ATOMS,
    "C": CYTOSINE_ATOMS,
    "G": GUANOSINE_ATOMS,
    "U": URIDINE_ATOMS,
    "GTP": GTP_ATOMS,
    "CCC": CCC_ATOMS,
    "GNG": GNG_ATOMS,
}

AMINO_ACID_ATOMS = {
    "GLY": GLYCINE_ATOMS,
    "ALA": ALANINE_ATOMS,
    "VAL": VALINE_ATOMS,
    "LEU": LEUCINE_ATOMS,
    "ILE": ISOLEUCINE_ATOMS,
    "PRO": PROLINE_ATOMS,
    "PHE": PHENYLALANINE_ATOMS,
    "TRP": TRYPTOPHAN_ATOMS,
    "MET": METHIONINE_ATOMS,
    "CYS": CYSTEINE_ATOMS,
    "SER": SERINE_ATOMS,
    "THR": THREONINE_ATOMS,
    "ASN": ASPARAGINE_ATOMS,
    "GLN": GLUTAMINE_ATOMS,
    "ASP": ASPARTIC_ACID_ATOMS,
    "GLU": GLUTAMIC_ACID_ATOMS,
    "LYS": LYSINE_ATOMS,
    "ARG": ARGININE_ATOMS,
    "HIS": HISTIDINE_ATOMS,
    "TYR": TYROSINE_ATOMS,
}

# All residue atoms combined
ALL_ATOMS = {**NUCLEOTIDE_ATOMS, **AMINO_ACID_ATOMS}
