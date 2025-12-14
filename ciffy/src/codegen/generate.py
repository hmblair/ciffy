#!/usr/bin/env python3
"""
Auto-generate hash lookup tables and Python enums from the PDB Chemical Component Dictionary.

Reads the CCD file directly and generates:
  - hash/*.gperf (forward lookups)
  - hash/*.c (gperf output)
  - hash/reverse.h (reverse lookups for CIF writing)
  - biochemistry/_generated_atoms.py (Python atom enums)
  - biochemistry/_generated_residues.py (Python Residue enum + mappings)

Usage:
  python generate.py [ccd_path] [--gperf-path /path/to/gperf] [--skip-gperf]

If ccd_path is not provided, the CCD will be auto-downloaded to ~/.cache/ciffy/.
This script is called automatically during build via setup.py.
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import shutil
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

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

# Dihedral angle type mapping (ordered list, index = integer key)
# Must match DihedralType enum in ciffy/types/dihedral.py
DIHEDRAL_TYPES: list[str] = [
    "phi", "psi", "omega",  # Protein backbone (0-2)
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",  # Nucleic acid backbone (3-8)
    "chi_purine", "chi_pyrimidine",  # Glycosidic (9-10)
]

# Map dihedral name -> integer index
DIHEDRAL_TYPE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(DIHEDRAL_TYPES)}


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
# RESIDUE DEFINITION
# =============================================================================

@dataclass
class ResidueDefinition:
    """Residue definition parsed from CCD."""
    name: str  # Enum name (e.g., "A", "DA", "ALA")
    cif_names: list[str]  # CIF file names that map to this residue
    molecule_type: int  # Index into MOLECULE_TYPES
    abbreviation: str  # Single-letter code
    atoms: list[str]  # Ordered list of atom names
    ideal_coords: dict[str, tuple[float, float, float]]  # Atom name -> (x, y, z)
    bonds: list[tuple[str, str]]  # List of (atom1, atom2) bonded pairs
    class_name: str = ""  # Python class name

    def __post_init__(self):
        if not self.class_name:
            self.class_name = _to_class_name(self.name)


# =============================================================================
# NAME CONVERSION UTILITIES
# =============================================================================

def _clean_atom_name(name: str) -> str:
    """Remove outer double quotes from CIF atom names."""
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1]
    return name


def _sanitize_identifier(name: str) -> str:
    """
    Apply common substitutions to make a string a valid Python identifier.

    Replacements:
        ' -> p  (apostrophe, e.g., O3' -> O3p)
        ` -> p  (backtick)
        " -> "" (remove quotes)
        * -> s  (star, e.g., HN* -> HNs)

    Does NOT handle leading digits (caller should check).
    """
    return name.replace("'", "p").replace("`", "p").replace('"', "").replace("*", "s")


def _to_enum_name(comp_id: str) -> str:
    """
    Convert CCD component ID to valid Python enum name (UPPERCASE).

    Examples:
        "A" -> "A"
        "5MU" -> "X5MU"
        "ALA" -> "ALA"
    """
    name = _sanitize_identifier(comp_id).replace("-", "_").replace("+", "PLUS")
    if name[0].isdigit():
        name = "X" + name
    return name.upper()


def _to_class_name(comp_id: str) -> str:
    """
    Convert CCD component ID to Python class name (UPPERCASE).

    Uses uppercase to match biochemistry convention where residue codes
    are always uppercase (e.g., ALA, CCC, PSU).

    Examples:
        "A" -> "A"
        "5MU" -> "X5MU"
        "ALA" -> "ALA"
    """
    name = _sanitize_identifier(comp_id).replace("-", "_").replace("+", "PLUS")
    if name[0].isdigit():
        name = "X" + name
    return name.upper()


def _to_python_name(cif_name: str) -> str:
    """
    Convert CIF atom name to valid Python identifier.

    Examples:
        "O3'" -> "O3p"
        "HN*" -> "HNs"
        "1H2" -> "X1H2"
    """
    name = _sanitize_identifier(cif_name)
    if name and name[0].isdigit():
        name = "X" + name
    return name


def _compute_dihedral_patterns(res: ResidueDefinition) -> dict[int, list[int]]:
    """
    Compute dihedral angle patterns for a residue.

    Returns a dictionary mapping dihedral type index (integer) to a list
    of 4 local atom indices. Uses -1 for atoms in adjacent residues.

    Args:
        res: Residue definition with atom names and molecule type.

    Returns:
        Dict mapping DIHEDRAL_TYPE_INDEX value -> [idx1, idx2, idx3, idx4]
        where indices are local positions in res.atoms, or -1 for inter-residue atoms.
    """
    # Build mapping: atom_name (Python format) -> local index
    name_to_local: dict[str, int] = {}
    for i, atom_name in enumerate(res.atoms):
        py_name = _to_python_name(atom_name)
        name_to_local[py_name] = i

    # Select dihedral definitions based on molecule type
    if res.molecule_type == Molecule.PROTEIN:
        dihedral_defs = {
            "phi": ("C", "N", "CA", "C"),
            "psi": ("N", "CA", "C", "N"),
            "omega": ("CA", "C", "N", "CA"),
        }
    elif res.molecule_type in (Molecule.RNA, Molecule.DNA, Molecule.HYBRID):
        dihedral_defs = {
            "alpha": ("O3p", "P", "O5p", "C5p"),
            "beta": ("P", "O5p", "C5p", "C4p"),
            "gamma": ("O5p", "C5p", "C4p", "C3p"),
            "delta": ("C5p", "C4p", "C3p", "O3p"),
            "epsilon": ("C4p", "C3p", "O3p", "P"),
            "zeta": ("C3p", "O3p", "P", "O5p"),
            "chi_purine": ("O4p", "C1p", "N9", "C4"),
            "chi_pyrimidine": ("O4p", "C1p", "N1", "C2"),
        }
    else:
        return {}

    patterns = {}
    for dihedral_name, (a1, a2, a3, a4) in dihedral_defs.items():
        # Map to local indices (-1 if not in this residue)
        idx1 = name_to_local.get(a1, -1)
        idx2 = name_to_local.get(a2, -1)
        idx3 = name_to_local.get(a3, -1)
        idx4 = name_to_local.get(a4, -1)

        # Only include if at least 2 atoms are in this residue
        if [idx1, idx2, idx3, idx4].count(-1) <= 2:
            type_idx = DIHEDRAL_TYPE_INDEX[dihedral_name]
            patterns[type_idx] = [idx1, idx2, idx3, idx4]

    return patterns


# =============================================================================
# CCD PARSING
# =============================================================================

def _determine_molecule_type(comp_type: str, name: str, comp_id: str) -> int:
    """Determine Molecule type index from CCD type string."""
    t = comp_type.upper()

    # Polymer types
    if 'RNA' in t:
        return Molecule.RNA
    if 'DNA' in t:
        return Molecule.DNA
    if 'D-PEPTIDE' in t:
        return Molecule.PROTEIN_D
    if 'PEPTIDE' in t:
        return Molecule.PROTEIN

    # Non-polymer types
    if 'NON-POLYMER' in t:
        if comp_id == "HOH" or name.upper() == "WATER":
            return Molecule.WATER
        if comp_id in IONS:
            return Molecule.ION
        return Molecule.LIGAND

    return Molecule.OTHER


def _get_abbreviation(one_letter: str, comp_type: str) -> str:
    """Get single-letter abbreviation (lowercase for nucleotides)."""
    if one_letter and one_letter != '?':
        t = comp_type.upper()
        if 'RNA' in t or 'DNA' in t:
            return one_letter.lower()
        return one_letter.upper()
    return '~'


def parse_ccd(filepath: str, whitelist: set[str] | None = None) -> Iterator[ResidueDefinition]:
    """Parse the CCD file and yield residue definitions.

    Args:
        filepath: Path to components.cif
        whitelist: If provided, only yield components in this set.

    Yields:
        ResidueDefinition for each component (skips obsolete).
    """
    # State for current component
    comp_id = ""
    name = ""
    comp_type = ""
    status = ""
    one_letter = ""
    atoms: list[str] = []
    ideal_coords: dict[str, tuple[float, float, float]] = {}
    bonds: list[tuple[str, str]] = []
    in_atom_loop = False
    in_bond_loop = False
    # Column indices for atom loop parsing
    atom_columns: list[str] = []
    atom_id_col = -1
    x_ideal_col = -1
    y_ideal_col = -1
    z_ideal_col = -1
    # Column indices for bond loop parsing
    bond_columns: list[str] = []
    bond_atom1_col = -1
    bond_atom2_col = -1

    def make_residue() -> ResidueDefinition | None:
        """Create ResidueDefinition from current state if valid."""
        if not comp_id or status == "OBS":
            return None
        if whitelist is not None and comp_id not in whitelist:
            return None
        return ResidueDefinition(
            name=_to_enum_name(comp_id),
            cif_names=[comp_id],
            molecule_type=_determine_molecule_type(comp_type, name, comp_id),
            abbreviation=_get_abbreviation(one_letter, comp_type),
            atoms=atoms.copy(),
            ideal_coords=ideal_coords.copy(),
            bonds=bonds.copy(),
        )

    def _parse_float(s: str) -> float | None:
        """Parse a float, returning None for missing values."""
        if s == '?' or s == '.':
            return None
        try:
            return float(s)
        except ValueError:
            return None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # New component
            if line.startswith('data_'):
                if res := make_residue():
                    yield res
                # Reset state
                comp_id = line[5:]
                name = ""
                comp_type = ""
                status = ""
                one_letter = ""
                atoms = []
                ideal_coords = {}
                bonds = []
                in_atom_loop = False
                in_bond_loop = False
                atom_columns = []
                atom_id_col = -1
                x_ideal_col = -1
                y_ideal_col = -1
                z_ideal_col = -1
                bond_columns = []
                bond_atom1_col = -1
                bond_atom2_col = -1
                continue

            if not comp_id:
                continue

            # Parse _chem_comp fields
            if line.startswith('_chem_comp.id '):
                comp_id = line.split()[-1].strip()
            elif line.startswith('_chem_comp.name '):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    name = parts[1].strip().strip('"')
            elif line.startswith('_chem_comp.type '):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    comp_type = parts[1].strip().strip('"')
            elif line.startswith('_chem_comp.pdbx_release_status '):
                status = line.split()[-1].strip()
            elif line.startswith('_chem_comp.one_letter_code '):
                val = line.split()[-1].strip()
                if val != '?':
                    one_letter = val

            # Detect loop start - reset loop states
            elif line.startswith('loop_'):
                in_atom_loop = False
                in_bond_loop = False
                atom_columns = []
                bond_columns = []
            elif line.startswith('_chem_comp_atom.'):
                col_name = line.strip().split()[0]  # e.g., "_chem_comp_atom.atom_id"
                field = col_name.split('.')[-1]  # e.g., "atom_id"
                parts = line.split()

                # Check for single-value format (e.g., "_chem_comp_atom.atom_id MG")
                if len(parts) >= 2:
                    value = parts[-1]
                    if field == 'atom_id':
                        atom_id = _clean_atom_name(value)
                        if atom_id not in atoms:
                            atoms.append(atom_id)
                            # Single-value format: store coords later when we see them
                    elif field == 'pdbx_model_Cartn_x_ideal' and atoms:
                        try:
                            _single_x = float(value)
                            ideal_coords.setdefault(atoms[-1], [None, None, None])[0] = _single_x
                        except ValueError:
                            pass
                    elif field == 'pdbx_model_Cartn_y_ideal' and atoms:
                        try:
                            _single_y = float(value)
                            ideal_coords.setdefault(atoms[-1], [None, None, None])[1] = _single_y
                        except ValueError:
                            pass
                    elif field == 'pdbx_model_Cartn_z_ideal' and atoms:
                        try:
                            _single_z = float(value)
                            coord = ideal_coords.get(atoms[-1], [None, None, None])
                            coord[2] = float(value)
                            if all(c is not None for c in coord):
                                ideal_coords[atoms[-1]] = tuple(coord)
                        except ValueError:
                            pass
                else:
                    # Loop header format - track column position
                    atom_columns.append(field)
                    col_idx = len(atom_columns) - 1
                    if field == 'atom_id':
                        atom_id_col = col_idx
                    elif field == 'pdbx_model_Cartn_x_ideal':
                        x_ideal_col = col_idx
                    elif field == 'pdbx_model_Cartn_y_ideal':
                        y_ideal_col = col_idx
                    elif field == 'pdbx_model_Cartn_z_ideal':
                        z_ideal_col = col_idx
                    in_atom_loop = True
            elif in_atom_loop and line.startswith('_'):
                pass
            elif in_atom_loop and line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2 and parts[0] == comp_id:
                    # Parse atom_id
                    if atom_id_col >= 0 and atom_id_col < len(parts):
                        atom_id = _clean_atom_name(parts[atom_id_col])
                        if atom_id not in atoms:
                            atoms.append(atom_id)
                        # Parse ideal coordinates if available
                        if (x_ideal_col >= 0 and y_ideal_col >= 0 and z_ideal_col >= 0 and
                            x_ideal_col < len(parts) and y_ideal_col < len(parts) and z_ideal_col < len(parts)):
                            x = _parse_float(parts[x_ideal_col])
                            y = _parse_float(parts[y_ideal_col])
                            z = _parse_float(parts[z_ideal_col])
                            if x is not None and y is not None and z is not None:
                                ideal_coords[atom_id] = (x, y, z)

            # Detect bond definitions
            elif line.startswith('_chem_comp_bond.'):
                col_name = line.strip().split()[0]
                field = col_name.split('.')[-1]
                parts = line.split()

                if len(parts) == 1:
                    # Loop header format - track column position
                    bond_columns.append(field)
                    col_idx = len(bond_columns) - 1
                    if field == 'atom_id_1':
                        bond_atom1_col = col_idx
                    elif field == 'atom_id_2':
                        bond_atom2_col = col_idx
                    in_bond_loop = True
            elif in_bond_loop and line.startswith('_'):
                pass
            elif in_bond_loop and line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3 and parts[0] == comp_id:
                    # Parse bond atom pair
                    if (bond_atom1_col >= 0 and bond_atom2_col >= 0 and
                        bond_atom1_col < len(parts) and bond_atom2_col < len(parts)):
                        atom1 = _clean_atom_name(parts[bond_atom1_col])
                        atom2 = _clean_atom_name(parts[bond_atom2_col])
                        bonds.append((atom1, atom2))

            elif line.startswith('#'):
                in_atom_loop = False
                in_bond_loop = False

    # Yield last component
    if res := make_residue():
        yield res


def load_residues_from_ccd(
    ccd_path: str,
    whitelist: set[str] | None = RESIDUE_WHITELIST
) -> list[ResidueDefinition]:
    """Load and sort residue definitions from CCD."""
    print(f"Parsing CCD: {ccd_path}")
    if whitelist:
        print(f"  Using whitelist with {len(whitelist)} entries")

    components = list(parse_ccd(ccd_path, whitelist))

    # Group by molecule type and sort each group
    groups: dict[int, list[ResidueDefinition]] = {}
    for comp in components:
        groups.setdefault(comp.molecule_type, []).append(comp)

    for mol_type in groups:
        groups[mol_type].sort(key=lambda c: c.name)

    # Combine in canonical order
    order = [Molecule.RNA, Molecule.DNA, Molecule.PROTEIN, Molecule.PROTEIN_D,
             Molecule.WATER, Molecule.ION, Molecule.LIGAND, Molecule.OTHER]
    all_residues = []
    for mol_type in order:
        all_residues.extend(groups.get(mol_type, []))

    # Print summary
    print(f"  RNA: {len(groups.get(Molecule.RNA, []))}")
    print(f"  DNA: {len(groups.get(Molecule.DNA, []))}")
    print(f"  L-peptides: {len(groups.get(Molecule.PROTEIN, []))}")
    print(f"  D-peptides: {len(groups.get(Molecule.PROTEIN_D, []))}")
    print(f"  Water: {len(groups.get(Molecule.WATER, []))}, "
          f"Ions: {len(groups.get(Molecule.ION, []))}, "
          f"Ligands: {len(groups.get(Molecule.LIGAND, []))}")
    print(f"  Total: {len(all_residues)} residues")

    return all_residues


# =============================================================================
# GPERF GENERATION
# =============================================================================

def _gperf_header(lookup_name: str, hash_name: str, prefix: str) -> str:
    """Generate standard gperf file header."""
    return f"""%define lookup-function-name {lookup_name}
%define hash-function-name {hash_name}
%define constants-prefix {prefix}
%struct-type
%{{
#include "../codegen/lookup.h"
%}}
struct _LOOKUP;
%%
"""


def find_gperf() -> str:
    """Find gperf executable (requires version 3.1+)."""
    candidates = [
        "/opt/homebrew/bin/gperf",
        "/usr/local/bin/gperf",
        shutil.which("gperf"),
        "/usr/bin/gperf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise RuntimeError(
        "gperf not found. Install with: brew install gperf (macOS) "
        "or apt install gperf (Linux)"
    )


def run_gperf(gperf_path: str, hash_dir: Path) -> None:
    """Run gperf to generate .c files from .gperf files."""
    for name in ["element", "residue", "atom", "molecule", "entity", "ion"]:
        input_file = hash_dir / f"{name}.gperf"
        output_file = hash_dir / f"{name}.c"

        result = subprocess.run(
            [gperf_path, str(input_file)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"gperf failed for {input_file}: {result.stderr}")

        output_file.write_text(result.stdout)

    print("Generated: hash/*.c")


def generate_gperf_files(
    hash_dir: Path,
    atom_index: dict[tuple[str, str], int],
    cif_to_residue: dict[str, int],
    residue_index: dict[str, int],
    all_residues: list[ResidueDefinition],
) -> None:
    """Generate all .gperf files."""

    # atom.gperf
    content = _gperf_header("_lookup_atom", "_hash_atom", "ATOM")
    for (residue, atom), idx in sorted(atom_index.items(), key=lambda x: x[1]):
        content += f"{residue}_{atom}, {idx}\n"
    (hash_dir / "atom.gperf").write_text(content)

    # residue.gperf
    content = _gperf_header("_lookup_residue", "_hash_residue", "RESIDUE")
    added: set[str] = set()
    for cif_name, idx in sorted(cif_to_residue.items(), key=lambda x: x[1]):
        if cif_name not in added:
            content += f"{cif_name}, {idx}\n"
            added.add(cif_name)
    for res in all_residues:
        if res.name not in added:
            content += f"{res.name}, {residue_index[res.name]}\n"
            added.add(res.name)
    (hash_dir / "residue.gperf").write_text(content)

    # element.gperf
    content = _gperf_header("_lookup_element", "_hash_element", "ELEMENT")
    for symbol, atomic_num in sorted(ELEMENTS.items(), key=lambda x: x[1]):
        content += f"{symbol}, {atomic_num}\n"
    (hash_dir / "element.gperf").write_text(content)

    # molecule.gperf
    content = _gperf_header("_lookup_molecule", "_hash_molecule", "MOLECULE")
    for idx, mt in enumerate(MOLECULE_TYPES):
        if mt.entity_poly_type:
            name = mt.entity_poly_type
            if ' ' in name or '(' in name or '/' in name:
                content += f'"{name}", {idx}\n'
            else:
                content += f"{name}, {idx}\n"
    content += f'"polysaccharide(L)", {Molecule.POLYSACCHARIDE}\n'
    (hash_dir / "molecule.gperf").write_text(content)

    # entity.gperf - maps _entity.type to Molecule indices
    content = _gperf_header("_lookup_entity", "_hash_entity", "ENTITY")
    content += f"polymer, {Molecule.UNKNOWN}\n"
    content += f"non-polymer, {Molecule.LIGAND}\n"
    content += f"water, {Molecule.WATER}\n"
    content += f"branched, {Molecule.POLYSACCHARIDE}\n"
    content += f"macrolide, {Molecule.LIGAND}\n"
    (hash_dir / "entity.gperf").write_text(content)

    # ion.gperf
    content = _gperf_header("_lookup_ion", "_hash_ion", "ION")
    for ion in sorted(IONS):
        content += f"{ion}, {Molecule.ION}\n"
    (hash_dir / "ion.gperf").write_text(content)

    print("Generated: hash/*.gperf")


# =============================================================================
# REVERSE HEADER GENERATION
# =============================================================================

def generate_reverse_header(
    hash_dir: Path,
    atom_index: dict[tuple[str, str], int],
    residue_to_cif: dict[int, str],
) -> None:
    """Generate reverse.h for CIF writing."""

    # Build reverse mappings
    atoms = {idx: (res, atom) for (res, atom), idx in atom_index.items()}
    elements_reverse = {v: k for k, v in ELEMENTS.items()}
    molecule_types = {i: mt.entity_poly_type for i, mt in enumerate(MOLECULE_TYPES)
                      if mt.entity_poly_type}

    atom_max = max(atoms.keys()) + 1
    residue_max = max(residue_to_cif.keys()) + 1
    element_max = max(ELEMENTS.values()) + 1
    molecule_max = len(MOLECULE_TYPES)

    lines = [
        '#ifndef _CIFFY_REVERSE_H',
        '#define _CIFFY_REVERSE_H',
        '',
        '/**',
        ' * @file reverse.h',
        ' * @brief Reverse lookup tables for CIF writing.',
        ' * AUTO-GENERATED by generate.py - DO NOT EDIT MANUALLY.',
        ' */',
        '',
        '#include <stddef.h>',
        '#include "../log.h"',
        '',
        '#define UNKNOWN_INDEX    (-1)',
        '#define UNKNOWN_ELEMENT  "X"',
        '#define UNKNOWN_RESIDUE  "UNK"',
        '#define UNKNOWN_ATOM     "X"',
        '',
        '/* ELEMENT REVERSE LOOKUP */',
        f'#define ELEMENT_MAX {element_max}',
        '',
        'static const char *ELEMENT_NAMES[ELEMENT_MAX] = {',
    ]

    for i in range(element_max):
        name = elements_reverse.get(i)
        val = f'"{name}"' if name else "NULL"
        lines.append(f'    [{i}] = {val},')

    lines.extend([
        '};',
        '',
        'static inline const char *element_name(int idx) {',
        '    if (idx < 0 || idx >= ELEMENT_MAX || ELEMENT_NAMES[idx] == NULL) {',
        '        LOG_WARNING("Unknown element index %d", idx);',
        '        return UNKNOWN_ELEMENT;',
        '    }',
        '    return ELEMENT_NAMES[idx];',
        '}',
        '',
        '/* RESIDUE REVERSE LOOKUP */',
        f'#define RESIDUE_MAX {residue_max}',
        '',
        'static const char *RESIDUE_NAMES[RESIDUE_MAX] = {',
    ])

    for i in range(residue_max):
        name = residue_to_cif.get(i)
        val = f'"{name}"' if name else "NULL"
        lines.append(f'    [{i}] = {val},')

    lines.extend([
        '};',
        '',
        'static inline const char *residue_name(int idx) {',
        '    if (idx < 0 || idx >= RESIDUE_MAX || RESIDUE_NAMES[idx] == NULL) {',
        '        LOG_WARNING("Unknown residue index %d", idx);',
        '        return UNKNOWN_RESIDUE;',
        '    }',
        '    return RESIDUE_NAMES[idx];',
        '}',
        '',
        '/* ATOM REVERSE LOOKUP */',
        'typedef struct {',
        '    const char *res;',
        '    const char *atom;',
        '} AtomInfo;',
        '',
        f'#define ATOM_MAX {atom_max}',
        '',
        'static const AtomInfo ATOM_INFO[ATOM_MAX] = {',
    ])

    for i in range(atom_max):
        if i in atoms:
            res, atom = atoms[i]
            lines.append(f'    [{i}] = {{"{res}", "{atom}"}},')
        else:
            lines.append(f'    [{i}] = {{NULL, NULL}},')

    lines.extend([
        '};',
        '',
        'static inline const AtomInfo *atom_info(int idx) {',
        '    static const AtomInfo UNKNOWN = {UNKNOWN_RESIDUE, UNKNOWN_ATOM};',
        '    if (idx < 0 || idx >= ATOM_MAX || ATOM_INFO[idx].atom == NULL) {',
        '        LOG_WARNING("Unknown atom index %d", idx);',
        '        return &UNKNOWN;',
        '    }',
        '    return &ATOM_INFO[idx];',
        '}',
        '',
        '/* MOLECULE TYPE REVERSE LOOKUP */',
        f'#define MOLECULE_MAX {molecule_max}',
        '',
        'static const char *MOLECULE_TYPE_NAMES[MOLECULE_MAX] = {',
    ])

    for i in range(molecule_max):
        name = molecule_types.get(i)
        val = f'"{name}"' if name else "NULL"
        lines.append(f'    [{i}] = {val},')

    lines.extend([
        '};',
        '',
        'static inline const char *molecule_type_name(int idx) {',
        '    if (idx < 0 || idx >= MOLECULE_MAX || MOLECULE_TYPE_NAMES[idx] == NULL) {',
        '        return "other";',
        '    }',
        '    return MOLECULE_TYPE_NAMES[idx];',
        '}',
        '',
        '#endif /* _CIFFY_REVERSE_H */',
        '',
    ])

    (hash_dir / "reverse.h").write_text('\n'.join(lines))
    print("Generated: hash/reverse.h")


# =============================================================================
# PYTHON CODE GENERATION
# =============================================================================

def generate_python_molecule(types_dir: Path) -> None:
    """Generate Python Molecule enum from MOLECULE_TYPES."""

    lines = [
        '"""',
        'Molecule type enumeration.',
        '',
        'Based on PDB/mmCIF entity types from wwPDB:',
        '- _entity.type: polymer, non-polymer, water, branched',
        '- _entity_poly.type: polypeptide(L/D), polyribonucleotide, etc.',
        '',
        'See: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html',
        '',
        'AUTO-GENERATED by ciffy/src/codegen/generate.py - DO NOT EDIT MANUALLY.',
        '"""',
        '',
        'from enum import Enum',
        '',
        '',
        'class Molecule(Enum):',
        '    """',
        '    Types of molecules that can appear in a structure.',
        '',
        '    Used to classify chains by their molecular type, enabling',
        '    filtering and type-specific operations.',
        '    """',
        '',
    ]

    for idx, mt in enumerate(MOLECULE_TYPES):
        lines.append(f"    {mt.name} = {idx}  # {mt.description}")

    lines.extend([
        '',
        '',
        'def molecule_type(value: int) -> Molecule:',
        '    """',
        '    Convert an integer value to the corresponding Molecule type.',
        '',
        '    Args:',
        '        value: Integer representing molecule type.',
        '',
        '    Returns:',
        '        The corresponding Molecule enum value.',
        '',
        '    Raises:',
        '        ValueError: If value doesn\'t correspond to a known molecule type.',
        '    """',
        '    try:',
        '        return Molecule(value)',
        '    except ValueError as e:',
        '        raise ValueError(f"Unknown molecule type value: {value}") from e',
        '',
    ])

    (types_dir / "molecule.py").write_text('\n'.join(lines))
    print("Generated: types/molecule.py")


def generate_python_elements(biochem_dir: Path) -> None:
    """Generate Python Element enum from ELEMENTS dict."""

    lines = [
        '"""',
        'Chemical element definitions.',
        '',
        'AUTO-GENERATED by ciffy/src/codegen/generate.py - DO NOT EDIT MANUALLY.',
        '"""',
        '',
        'from ..utils import IndexEnum',
        '',
        '',
        'class Element(IndexEnum):',
        '    """',
        '    Chemical elements with their atomic numbers.',
        '',
        '    Values correspond to atomic numbers for common biological elements.',
        '    """',
        '',
    ]

    # Group elements by category for readability
    organic = ["H", "C", "N", "O", "P", "S"]
    halogens = ["F", "CL", "BR", "I"]
    alkali = ["LI", "NA", "K", "RB", "CS"]
    alkaline = ["MG", "CA", "SR", "BA"]
    transition = ["MN", "FE", "CO", "NI", "CU", "ZN", "MO", "AG", "CD", "W", "PT", "AU", "HG"]
    other = ["AL", "SE", "PB"]

    def write_group(name: str, symbols: list[str]) -> None:
        lines.append(f"    # {name}")
        for sym in symbols:
            if sym in ELEMENTS:
                lines.append(f"    {sym} = {ELEMENTS[sym]}")
        lines.append("")

    write_group("Common organic elements", organic)
    write_group("Halogens", halogens)
    write_group("Alkali metals", alkali)
    write_group("Alkaline earth metals", alkaline)
    write_group("Transition metals", transition)
    write_group("Other elements", other)

    lines.extend([
        '',
        '# Pre-computed reverse lookup: atomic number -> element name',
        'ELEMENT_NAMES: dict[int, str] = {e.value: e.name for e in Element}',
        '',
    ])

    (biochem_dir / "_generated_elements.py").write_text('\n'.join(lines))
    print("Generated: biochemistry/_generated_elements.py")


def generate_python_atoms(
    biochem_dir: Path,
    atom_index: dict[tuple[str, str], int],
    all_residues: list[ResidueDefinition],
) -> None:
    """Generate Python atom enum file."""

    # Build per-residue atom dicts
    residue_atoms: dict[str, dict[str, int]] = {}
    for (cif_name, atom), idx in atom_index.items():
        residue_atoms.setdefault(cif_name, {})[_to_python_name(atom)] = idx

    # Group residues by type
    by_type: dict[int, list[ResidueDefinition]] = {}
    for res in all_residues:
        by_type.setdefault(res.molecule_type, []).append(res)

    lines = [
        '"""',
        'Auto-generated atom enum definitions.',
        'DO NOT EDIT - Generated by ciffy/src/codegen/generate.py from CCD.',
        '"""',
        '',
        'import numpy as np',
        '',
        'from ..utils import IndexEnum, PairEnum',
        '',
        '',
    ]

    # Generate per-residue classes
    sections = [
        ("RNA", Molecule.RNA),
        ("DNA", Molecule.DNA),
        ("PROTEIN", Molecule.PROTEIN),
        ("WATER", Molecule.WATER),
        ("ION", Molecule.ION),
        ("LIGAND", Molecule.LIGAND),
    ]

    for section_name, mol_type in sections:
        residues = by_type.get(mol_type, [])
        if not residues:
            continue

        lines.append(f"# {'=' * 77}")
        lines.append(f"# {section_name}")
        lines.append(f"# {'=' * 77}")
        lines.append('')

        for res in residues:
            cif = res.cif_names[0]
            atoms = residue_atoms.get(cif, {})
            if not atoms:
                continue

            lines.append(f"class {res.class_name}(IndexEnum):")
            lines.append(f'    """{res.class_name} ({cif}) atom indices."""')
            for py_name, idx in atoms.items():
                lines.append(f"    {py_name} = {idx}")
            lines.append('')

            # Add ideal coordinates as class attribute
            # Note: Use array index, not enum value (e.g., A.ideal[0] for first atom)
            if res.ideal_coords and res.atoms:
                coords = []
                for atom in res.atoms:
                    if atom in res.ideal_coords:
                        x, y, z = res.ideal_coords[atom]
                        coords.append(f"[{x:.3f}, {y:.3f}, {z:.3f}]")
                    else:
                        coords.append("[0.0, 0.0, 0.0]")
                if coords:
                    lines.append(f"{res.class_name}.ideal = np.array([")
                    for i, coord in enumerate(coords):
                        comma = "," if i < len(coords) - 1 else ""
                        lines.append(f"    {coord}{comma}  # {res.atoms[i]}")
                    lines.append("], dtype=np.float32)")
                    lines.append('')

            # Add bonds as PairEnum
            if res.bonds:
                lines.append(f"{res.class_name}.bonds = PairEnum([")
                for atom1, atom2 in res.bonds:
                    py_atom1 = _to_python_name(atom1)
                    py_atom2 = _to_python_name(atom2)
                    lines.append(f"    ({res.class_name}.{py_atom1}, {res.class_name}.{py_atom2}),")
                lines.append("])")
                lines.append('')

                # Add bond_indices as numpy array
                lines.append(f"{res.class_name}.bond_indices = np.array([")
                for atom1, atom2 in res.bonds:
                    py_atom1 = _to_python_name(atom1)
                    py_atom2 = _to_python_name(atom2)
                    idx1 = atoms[py_atom1]
                    idx2 = atoms[py_atom2]
                    lines.append(f"    [{idx1}, {idx2}],")
                lines.append("], dtype=np.int32)")
                lines.append('')

            # Add dihedral_patterns as dict of numpy arrays
            dihedral_patterns = _compute_dihedral_patterns(res)
            if dihedral_patterns:
                lines.append(f"{res.class_name}.dihedral_patterns = {{")
                for type_idx, indices in sorted(dihedral_patterns.items()):
                    lines.append(f"    {type_idx}: np.array({indices}, dtype=np.int32),")
                lines.append("}")
                lines.append('')

            lines.append('')

    # Combined enums
    lines.append(f"# {'=' * 77}")
    lines.append("# COMBINED ENUMS")
    lines.append(f"# {'=' * 77}")
    lines.append('')

    rna = by_type.get(Molecule.RNA, [])
    dna = by_type.get(Molecule.DNA, [])
    protein = by_type.get(Molecule.PROTEIN, [])

    rna_bases = [r for r in rna if r.cif_names[0] in ("A", "C", "G", "U")]
    if rna_bases:
        lines.append("RibonucleicAcid = IndexEnum(")
        lines.append("    'RibonucleicAcid',")
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in rna_bases]
        lines.append("    " + " |\n    ".join(parts))
        lines.append(")")
        lines.append('')
        lines.append("RibonucleicAcidNoPrefix = IndexEnum(")
        lines.append("    'RibonucleicAcid',")
        parts = [f'{r.class_name}.dict()' for r in rna_bases]
        lines.append("    " + " |\n    ".join(parts))
        lines.append(")")
        lines.append('')

    dna_bases = [r for r in dna if r.cif_names[0] in ("DA", "DC", "DG", "DT")]
    if dna_bases:
        lines.append("DeoxyribonucleicAcid = IndexEnum(")
        lines.append("    'DeoxyribonucleicAcid',")
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in dna_bases]
        lines.append("    " + " |\n    ".join(parts))
        lines.append(")")
        lines.append('')

    modified = [r for r in all_residues
                if r.molecule_type in (Molecule.RNA, Molecule.DNA)
                and r.cif_names[0] not in ("A", "C", "G", "U", "DA", "DC", "DG", "DT")]
    if modified:
        lines.append("ModifiedNucleotides = IndexEnum(")
        lines.append("    'ModifiedNucleotides',")
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in modified]
        lines.append("    " + " |\n    ".join(parts))
        lines.append(")")
        lines.append('')

    if protein:
        lines.append("AminoAcids = IndexEnum(")
        lines.append("    'AminoAcids',")
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in protein]
        lines.append("    " + " |\n    ".join(parts))
        lines.append(")")
        lines.append('')

    # Reverse lookup
    lines.append(f"# {'=' * 77}")
    lines.append("# REVERSE LOOKUP")
    lines.append(f"# {'=' * 77}")
    lines.append('')
    lines.append("ATOM_NAMES: dict[int, str] = {")
    for (res, atom), idx in sorted(atom_index.items(), key=lambda x: x[1]):
        lines.append(f'    {idx}: "{atom}",')
    lines.append("}")
    lines.append('')

    (biochem_dir / "_generated_atoms.py").write_text('\n'.join(lines))
    print("Generated: biochemistry/_generated_atoms.py")


def generate_python_residues(
    biochem_dir: Path,
    all_residues: list[ResidueDefinition],
) -> None:
    """Generate Python Residue class with ResidueType instances."""

    # Determine which residues have atom classes (non-empty atoms list)
    residues_with_atoms = [res for res in all_residues if res.atoms]
    atom_class_names = [res.class_name for res in residues_with_atoms]

    lines = [
        '"""',
        'Auto-generated residue definitions.',
        'DO NOT EDIT - Generated by ciffy/src/codegen/generate.py from CCD.',
        '"""',
        '',
        'from ..utils import ResidueType, ResidueMeta',
        'from ..types import Molecule',
    ]

    # Import atom classes (only those that exist)
    if atom_class_names:
        lines.append('from ._generated_atoms import (')
        for name in atom_class_names:
            lines.append(f'    {name},')
        lines.append(')')

    lines.append('')
    lines.append('')
    lines.append('class Residue(metaclass=ResidueMeta):')
    lines.append('    """')
    lines.append('    Consolidated residue definitions with nested atom access.')
    lines.append('')
    lines.append('    Usage:')
    lines.append('        Residue.A.value         # → residue index')
    lines.append('        Residue.A.C3p.value     # → atom index')
    lines.append('        Residue.A.molecule_type # → Molecule.RNA')
    lines.append('        Residue.A.bonds         # → bond list')
    lines.append('')
    lines.append('        list(Residue)           # → iterate all residues')
    lines.append('        Residue(0)              # → lookup by index')
    lines.append('        Residue["ALA"]          # → lookup by name')
    lines.append('    """')
    lines.append('')

    # Build set of residue names with atom classes
    has_atoms = {res.class_name for res in residues_with_atoms}

    for idx, res in enumerate(all_residues):
        mol_name = MOLECULE_TYPES[res.molecule_type].name
        atom_class = res.class_name if res.class_name in has_atoms else "None"
        lines.append(
            f'    {res.name} = ResidueType('
            f'"{res.name}", {idx}, {atom_class}, '
            f'Molecule.{mol_name}, "{res.abbreviation}")'
        )

    lines.append('')
    lines.append('    _members: dict[str, ResidueType] = {}')
    lines.append('    _by_index: dict[int, ResidueType] = {}')
    lines.append('')
    lines.append('')

    # Initialization function
    lines.append('def _init_residue_class() -> None:')
    lines.append('    """Populate lookup tables from class attributes."""')
    lines.append('    for name in dir(Residue):')
    lines.append('        if name.startswith("_"):')
    lines.append('            continue')
    lines.append('        attr = getattr(Residue, name)')
    lines.append('        if isinstance(attr, ResidueType):')
    lines.append('            Residue._members[name] = attr')
    lines.append('            Residue._by_index[attr.value] = attr')
    lines.append('')
    lines.append('')
    lines.append('_init_residue_class()')
    lines.append('')
    lines.append('')

    # Keep CIF name mappings (needed for C extension compatibility)
    lines.append("# CIF name -> Residue index (for C extension)")
    lines.append("CIF_RESIDUE_NAMES: dict[str, int] = {")
    for idx, res in enumerate(all_residues):
        for cif in res.cif_names:
            lines.append(f'    "{cif}": {idx},')
    lines.append("}")
    lines.append('')

    lines.append("# Residue index -> CIF name (for C extension)")
    lines.append("RESIDUE_CIF_NAMES: dict[int, str] = {")
    for idx, res in enumerate(all_residues):
        lines.append(f'    {idx}: "{res.cif_names[0]}",')
    lines.append("}")
    lines.append('')

    (biochem_dir / "_generated_residues.py").write_text('\n'.join(lines))
    print("Generated: biochemistry/_generated_residues.py")


# =============================================================================
# MAIN GENERATION ENTRY POINT
# =============================================================================

def generate_all(ccd_path: str) -> tuple[Path, dict[tuple[str, str], int]]:
    """Generate all lookup tables and Python enums from CCD."""

    all_residues = load_residues_from_ccd(ccd_path)

    # Validate - check for duplicate CIF names
    seen_cif: dict[str, str] = {}
    for res in all_residues:
        for cif_name in res.cif_names:
            if cif_name in seen_cif:
                raise ValueError(
                    f"Duplicate CIF name '{cif_name}' in {res.name} and {seen_cif[cif_name]}"
                )
            seen_cif[cif_name] = res.name

    # Output directories
    script_dir = Path(__file__).parent
    hash_dir = script_dir.parent / "hash"
    biochem_dir = script_dir.parent.parent / "biochemistry"
    types_dir = script_dir.parent.parent / "types"
    hash_dir.mkdir(exist_ok=True)

    # Build derived mappings
    residue_index = {res.name: idx for idx, res in enumerate(all_residues)}
    cif_to_residue = {cif: idx for idx, res in enumerate(all_residues) for cif in res.cif_names}
    residue_to_cif = {idx: res.cif_names[0] for idx, res in enumerate(all_residues)}

    # Assign atom indices (1-indexed, 0 reserved for unknown)
    atom_index: dict[tuple[str, str], int] = {}
    current_idx = 1
    for res in all_residues:
        primary_cif = res.cif_names[0]
        for atom in res.atoms:
            key = (primary_cif, atom)
            if key not in atom_index:
                atom_index[key] = current_idx
                current_idx += 1

    # Add aliases
    for res in all_residues:
        primary_cif = res.cif_names[0]
        for alias in res.cif_names[1:]:
            for atom in res.atoms:
                primary_key = (primary_cif, atom)
                alias_key = (alias, atom)
                if alias_key not in atom_index:
                    atom_index[alias_key] = atom_index[primary_key]

    print(f"Assigned {current_idx - 1} unique atoms, {len(atom_index)} total entries")

    # Generate all files
    generate_gperf_files(hash_dir, atom_index, cif_to_residue, residue_index, all_residues)
    generate_reverse_header(hash_dir, atom_index, residue_to_cif)
    generate_python_molecule(types_dir)
    generate_python_elements(biochem_dir)
    generate_python_atoms(biochem_dir, atom_index, all_residues)
    generate_python_residues(biochem_dir, all_residues)

    return hash_dir, atom_index


# =============================================================================
# CCD DOWNLOAD
# =============================================================================

def download_ccd(dest_path: Path) -> bool:
    """Download and decompress the CCD file."""
    print(f"Downloading CCD from {CCD_URL}...")
    gz_path = dest_path.with_suffix(".cif.gz")

    try:
        urllib.request.urlretrieve(CCD_URL, gz_path)
        print("Decompressing CCD...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        print(f"CCD downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download CCD: {e}")
        if gz_path.exists():
            gz_path.unlink()
        return False


def get_ccd_path() -> Path:
    """Get path to CCD file, downloading if necessary."""
    # Check environment variable first
    env_path = os.environ.get("CIFFY_CCD_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Use centralized cache location
    cache_dir = Path.home() / ".cache" / "ciffy"
    ccd_path = cache_dir / "components.cif"

    if ccd_path.exists():
        return ccd_path

    # Download to cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    if download_ccd(ccd_path):
        return ccd_path

    raise FileNotFoundError(
        f"CCD file not found and download failed. "
        f"Set CIFFY_CCD_PATH or download manually from {CCD_URL}"
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hash tables from PDB Chemical Component Dictionary"
    )
    parser.add_argument(
        "ccd_path",
        nargs="?",
        help="Path to components.cif file (auto-downloaded if not provided)"
    )
    parser.add_argument("--gperf-path", help="Path to gperf executable")
    parser.add_argument("--skip-gperf", action="store_true", help="Skip running gperf")
    args = parser.parse_args()

    # Get CCD path (auto-download if not provided)
    ccd_path = Path(args.ccd_path) if args.ccd_path else get_ccd_path()

    hash_dir, _ = generate_all(str(ccd_path))

    if not args.skip_gperf:
        gperf_path = args.gperf_path or find_gperf()
        run_gperf(gperf_path, hash_dir)

    print("Generation complete!")


if __name__ == "__main__":
    main()
