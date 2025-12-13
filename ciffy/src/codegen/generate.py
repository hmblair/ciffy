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
    class_name: str = ""  # Python class name

    def __post_init__(self):
        if not self.class_name:
            self.class_name = _to_class_name(self.name)


# =============================================================================
# NAME CONVERSION UTILITIES
# =============================================================================

def _clean_atom_name(name: str) -> str:
    """Remove outer double quotes, keep internal single quotes."""
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1]
    return name


def _to_enum_name(comp_id: str) -> str:
    """Convert comp_id to valid Python enum name (uppercase)."""
    name = comp_id.replace("'", "p").replace('"', "").replace("-", "_").replace("+", "plus")
    if name[0].isdigit():
        name = "X" + name
    return name.upper()


def _to_class_name(comp_id: str) -> str:
    """Convert comp_id to readable class name (TitleCase)."""
    base = comp_id.replace("'", "p").replace('"', "").replace("-", "").replace("+", "Plus")
    if base[0].isdigit():
        base = "X" + base
    return base.title()


def _to_python_name(cif_name: str) -> str:
    """Convert CIF atom name to valid Python identifier."""
    name = cif_name.replace("'", "p").replace("`", "p").replace('"', "").replace("*", "s")
    if name and name[0].isdigit():
        name = "X" + name
    return name


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
    in_atom_loop = False

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
        )

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
                in_atom_loop = False
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

            # Detect atom definitions
            elif line.startswith('loop_'):
                in_atom_loop = False
            elif line.startswith('_chem_comp_atom.atom_id '):
                # Single-value format: _chem_comp_atom.atom_id   MG
                parts = line.split()
                if len(parts) >= 2:
                    atom_id = _clean_atom_name(parts[1])
                    if atom_id not in atoms:
                        atoms.append(atom_id)
            elif line.startswith('_chem_comp_atom.'):
                in_atom_loop = True
            elif in_atom_loop and line.startswith('_'):
                pass
            elif in_atom_loop and line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2 and parts[0] == comp_id:
                    atom_id = _clean_atom_name(parts[1])
                    if atom_id not in atoms:
                        atoms.append(atom_id)
            elif line.startswith('#'):
                in_atom_loop = False

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
        'from ..utils import IndexEnum',
        '',
        '',
    ]

    # Generate per-residue classes
    sections = [
        ("RNA", Molecule.RNA),
        ("DNA", Molecule.DNA),
        ("PROTEIN", Molecule.PROTEIN),
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
    """Generate Python Residue enum and mappings."""

    lines = [
        '"""',
        'Auto-generated residue definitions.',
        'DO NOT EDIT - Generated by ciffy/src/codegen/generate.py from CCD.',
        '"""',
        '',
        'from ..utils import IndexEnum',
        'from ..types import Molecule',
        '',
        '',
        'class Residue(IndexEnum):',
        '    """Residue types with unique integer indices."""',
        '',
    ]

    for idx, res in enumerate(all_residues):
        lines.append(f"    {res.name} = {idx}")

    lines.append('')
    lines.append('')
    lines.append("# Residue index -> Molecule type")
    lines.append("RESIDUE_MOLECULE_TYPE: dict[int, Molecule] = {")
    for idx, res in enumerate(all_residues):
        mol_name = MOLECULE_TYPES[res.molecule_type].name
        lines.append(f"    {idx}: Molecule.{mol_name},")
    lines.append("}")
    lines.append('')

    lines.append("# CIF name -> Residue index")
    lines.append("CIF_RESIDUE_NAMES: dict[str, int] = {")
    for idx, res in enumerate(all_residues):
        for cif in res.cif_names:
            lines.append(f'    "{cif}": {idx},')
    lines.append("}")
    lines.append('')

    lines.append("# Residue index -> CIF name")
    lines.append("RESIDUE_CIF_NAMES: dict[int, str] = {")
    for idx, res in enumerate(all_residues):
        lines.append(f'    {idx}: "{res.cif_names[0]}",')
    lines.append("}")
    lines.append('')

    lines.append("# Residue index -> single-letter abbreviation")
    lines.append("RESIDUE_ABBREV: dict[int, str] = {")
    for idx, res in enumerate(all_residues):
        lines.append(f'    {idx}: "{res.abbreviation}",')
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

    # Assign atom indices (0 reserved for unknown)
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
