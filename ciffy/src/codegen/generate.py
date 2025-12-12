#!/usr/bin/env python3
"""
Auto-generate hash lookup tables and Python enums.

Reads from ciffy/biochemistry/definitions.py (single source of truth) and generates:
  - hash/*.gperf (forward lookups)
  - hash/*.c (gperf output)
  - hash/reverse.h (reverse lookups for CIF writing)
  - biochemistry/_generated_atoms.py (Python atom enums)
  - biochemistry/_generated_residues.py (Python Residue enum + mappings)

Usage:
  python generate.py [--gperf-path /path/to/gperf] [--skip-gperf]

This script is called automatically during build via setup.py.
"""

import argparse
import subprocess
import shutil
from pathlib import Path


def find_gperf():
    """Find gperf executable (requires version 3.1+ for constants-prefix)."""
    candidates = [
        "/opt/homebrew/bin/gperf",  # macOS Homebrew ARM
        "/usr/local/bin/gperf",      # macOS Homebrew Intel
        shutil.which("gperf"),       # System PATH
        "/usr/bin/gperf",            # Linux fallback
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise RuntimeError(
        "gperf not found. Install with: brew install gperf (macOS) "
        "or apt install gperf (Linux)"
    )


def to_python_name(cif_name: str) -> str:
    """Convert CIF atom name to valid Python identifier.

    Examples:
        "C5'" -> "C5p"
        "H5''" -> "H5pp"
        "CA" -> "CA"
    """
    return cif_name.replace("'", "p")


def generate_all():
    """Generate all lookup tables and Python enums from definitions."""
    from ciffy.biochemistry.definitions import ALL_RESIDUES, validate_definitions
    from ciffy.biochemistry.elements import Element
    from ciffy.biochemistry.molecule_types import (
        ENTITY_POLY_TYPES, ENTITY_TYPES, ION_COMP_IDS
    )
    from ciffy.types import Molecule

    # Validate definitions before generating
    validate_definitions()

    # Output directories
    hash_dir = Path(__file__).parent.parent / "hash"
    biochem_dir = Path(__file__).parent.parent.parent / "biochemistry"
    hash_dir.mkdir(exist_ok=True)

    # === Build derived data from definitions ===

    # Build residue index mapping
    residue_index = {}  # residue_name -> index
    cif_to_residue = {}  # cif_name -> residue_index
    residue_to_cif = {}  # residue_index -> first cif_name (for writing)
    residue_to_molecule = {}  # residue_index -> molecule_type
    residue_abbreviations = {}  # residue_name -> abbreviation

    for idx, res in enumerate(ALL_RESIDUES):
        residue_index[res.name] = idx
        residue_to_molecule[idx] = res.molecule_type
        residue_abbreviations[res.name] = res.abbreviation

        # First CIF name is the canonical one for writing
        residue_to_cif[idx] = res.cif_names[0]

        # All CIF names map to this residue
        for cif_name in res.cif_names:
            cif_to_residue[cif_name] = idx

    print(f"Loaded {len(ALL_RESIDUES)} residue definitions")

    # === Assign indices to all atoms ===
    # Index 0 is reserved for "unknown"
    atom_index = {}  # (cif_residue_name, atom_name) -> index
    current_idx = 1

    for res in ALL_RESIDUES:
        # Use the first CIF name as the residue prefix in atom hash
        cif_name = res.cif_names[0]
        for atom in res.atoms:
            key = (cif_name, atom)
            if key not in atom_index:
                atom_index[key] = current_idx
                current_idx += 1

    print(f"Assigned indices to {len(atom_index)} atoms (1-{current_idx - 1})")

    # === Generate atom.gperf ===
    atom_gperf = """%define lookup-function-name _lookup_atom
%define hash-function-name _hash_atom
%define constants-prefix ATOM
%struct-type
%{
#include "../codegen/lookup.h"
%}
struct _LOOKUP;
%%
"""
    for (residue, atom), idx in sorted(atom_index.items(), key=lambda x: x[1]):
        atom_gperf += f"{residue}_{atom}, {idx}\n"

    with open(hash_dir / "atom.gperf", "w") as f:
        f.write(atom_gperf)

    # === Generate residue.gperf ===
    residue_gperf = """%define lookup-function-name _lookup_residue
%define hash-function-name _hash_residue
%define constants-prefix RESIDUE
%struct-type
%{
#include "../codegen/lookup.h"
%}
struct _LOOKUP;
%%
"""
    # Add all CIF names that map to residue indices
    added_names = set()
    for cif_name, idx in sorted(cif_to_residue.items(), key=lambda x: x[1]):
        if cif_name not in added_names:
            residue_gperf += f"{cif_name}, {idx}\n"
            added_names.add(cif_name)

    # Also add enum names for round-trip compatibility
    for res in ALL_RESIDUES:
        if res.name not in added_names:
            residue_gperf += f"{res.name}, {residue_index[res.name]}\n"
            added_names.add(res.name)

    with open(hash_dir / "residue.gperf", "w") as f:
        f.write(residue_gperf)

    # === Generate element.gperf ===
    element_gperf = """%define lookup-function-name _lookup_element
%define hash-function-name _hash_element
%define constants-prefix ELEMENT
%struct-type
%{
#include "../codegen/lookup.h"
%}
struct _LOOKUP;
%%
"""
    for member in Element:
        element_gperf += f"{member.name}, {member.value}\n"

    with open(hash_dir / "element.gperf", "w") as f:
        f.write(element_gperf)

    # === Generate molecule.gperf ===
    molecule_gperf = """%define lookup-function-name _lookup_molecule
%define hash-function-name _hash_molecule
%define constants-prefix MOLECULE
%struct-type
%{
#include "../codegen/lookup.h"
%}
struct _LOOKUP;
%%
"""
    for type_str, mol_value in ENTITY_POLY_TYPES.items():
        if "(" in type_str or "/" in type_str or " " in type_str:
            molecule_gperf += f'"{type_str}", {mol_value}\n'
        else:
            molecule_gperf += f"{type_str}, {mol_value}\n"

    with open(hash_dir / "molecule.gperf", "w") as f:
        f.write(molecule_gperf)

    # === Generate entity.gperf ===
    entity_gperf = """%define lookup-function-name _lookup_entity
%define hash-function-name _hash_entity
%define constants-prefix ENTITY
%struct-type
%{
#include "../codegen/lookup.h"
%}
struct _LOOKUP;
%%
"""
    for type_str, mol_value in ENTITY_TYPES.items():
        entity_gperf += f"{type_str}, {mol_value}\n"

    with open(hash_dir / "entity.gperf", "w") as f:
        f.write(entity_gperf)

    # === Generate ion.gperf ===
    ion_gperf = """%define lookup-function-name _lookup_ion
%define hash-function-name _hash_ion
%define constants-prefix ION
%struct-type
%{
#include "../codegen/lookup.h"
%}
struct _LOOKUP;
%%
"""
    for comp_id in sorted(ION_COMP_IDS):
        ion_gperf += f"{comp_id}, {Molecule.ION.value}\n"

    with open(hash_dir / "ion.gperf", "w") as f:
        f.write(ion_gperf)

    print("Generated: hash/*.gperf (atom, residue, element, molecule, entity, ion)")

    # === Generate reverse.h ===
    generate_reverse_header(
        hash_dir, atom_index, residue_to_cif, Element, Molecule
    )

    # === Generate Python enums ===
    generate_python_enums(biochem_dir, atom_index, ALL_RESIDUES)
    generate_residue_module(biochem_dir, ALL_RESIDUES, residue_to_molecule)

    return atom_index


def generate_reverse_header(hash_dir, atom_index, residue_to_cif, Element, Molecule):
    """Generate reverse.h for CIF writing."""
    from ciffy.biochemistry.molecule_types import ENTITY_POLY_TYPES

    # Collect atoms info
    atoms = {}  # idx -> (residue, atom_name)
    for (residue, atom), idx in atom_index.items():
        atoms[idx] = (residue, atom)

    # Collect elements
    elements = {member.value: member.name for member in Element}

    # Molecule type CIF strings
    molecule_types = {
        Molecule.RNA.value: "polyribonucleotide",
        Molecule.DNA.value: "polydeoxyribonucleotide",
        Molecule.HYBRID.value: "polydeoxyribonucleotide/polyribonucleotide hybrid",
        Molecule.PROTEIN.value: "polypeptide(L)",
        Molecule.PROTEIN_D.value: "polypeptide(D)",
        Molecule.CYCLIC_PEPTIDE.value: "cyclic-pseudo-peptide",
        Molecule.POLYSACCHARIDE.value: "polysaccharide(D)",
        Molecule.PNA.value: "peptide nucleic acid",
        Molecule.OTHER.value: "other",
    }

    # Find max indices
    atom_max = max(atoms.keys()) + 1
    residue_max = max(residue_to_cif.keys()) + 1
    element_max = max(elements.keys()) + 1
    molecule_max = max(molecule_types.keys()) + 1

    header = f'''#ifndef _CIFFY_REVERSE_H
#define _CIFFY_REVERSE_H

/**
 * @file reverse.h
 * @brief Reverse lookup tables for CIF writing.
 *
 * Maps integer indices back to their string representations.
 * AUTO-GENERATED by generate.py - DO NOT EDIT MANUALLY.
 */

#include <stddef.h>

/* ============================================================================
 * ELEMENT REVERSE LOOKUP
 * ============================================================================ */

#define ELEMENT_MAX {element_max}

static const char *ELEMENT_NAMES[ELEMENT_MAX] = {{
'''
    for i in range(element_max):
        if i in elements:
            header += f'    [{i}] = "{elements[i]}",\n'
        else:
            header += f'    [{i}] = NULL,\n'

    header += '''};

static inline const char *element_name(int idx) {
    if (idx < 0 || idx >= ELEMENT_MAX || ELEMENT_NAMES[idx] == NULL) {
        return "X";
    }
    return ELEMENT_NAMES[idx];
}

/* ============================================================================
 * RESIDUE REVERSE LOOKUP
 * ============================================================================ */

'''
    header += f'#define RESIDUE_MAX {residue_max}\n\n'
    header += 'static const char *RESIDUE_NAMES[RESIDUE_MAX] = {\n'

    for i in range(residue_max):
        if i in residue_to_cif:
            header += f'    [{i}] = "{residue_to_cif[i]}",\n'
        else:
            header += f'    [{i}] = NULL,\n'

    header += '''};

static inline const char *residue_name(int idx) {
    if (idx < 0 || idx >= RESIDUE_MAX || RESIDUE_NAMES[idx] == NULL) {
        return "UNK";
    }
    return RESIDUE_NAMES[idx];
}

/* ============================================================================
 * ATOM REVERSE LOOKUP
 * ============================================================================ */

typedef struct {
    const char *res;
    const char *atom;
} AtomInfo;

'''
    header += f'#define ATOM_MAX {atom_max}\n\n'
    header += 'static const AtomInfo ATOM_INFO[ATOM_MAX] = {\n'

    for i in range(atom_max):
        if i in atoms:
            res, atom = atoms[i]
            header += f'    [{i}] = {{"{res}", "{atom}"}},\n'
        else:
            header += f'    [{i}] = {{NULL, NULL}},\n'

    header += '''};

static inline const AtomInfo *atom_info(int idx) {
    static const AtomInfo UNKNOWN = {"UNK", "X"};
    if (idx < 0 || idx >= ATOM_MAX || ATOM_INFO[idx].atom == NULL) {
        return &UNKNOWN;
    }
    return &ATOM_INFO[idx];
}

/* ============================================================================
 * MOLECULE TYPE REVERSE LOOKUP
 * ============================================================================ */

'''
    header += f'#define MOLECULE_MAX {molecule_max}\n\n'
    header += 'static const char *MOLECULE_TYPE_NAMES[MOLECULE_MAX] = {\n'

    for i in range(molecule_max):
        if i in molecule_types:
            header += f'    [{i}] = "{molecule_types[i]}",\n'
        else:
            header += f'    [{i}] = NULL,\n'

    header += '''};

static inline const char *molecule_type_name(int idx) {
    if (idx < 0 || idx >= MOLECULE_MAX || MOLECULE_TYPE_NAMES[idx] == NULL) {
        return "other";
    }
    return MOLECULE_TYPE_NAMES[idx];
}

#endif /* _CIFFY_REVERSE_H */
'''

    with open(hash_dir / "reverse.h", "w") as f:
        f.write(header)

    print("Generated: hash/reverse.h")


def generate_python_enums(biochem_dir, atom_index, all_residues):
    """Generate Python atom enum file with auto-assigned indices."""
    from ciffy.types import Molecule

    # Build per-residue atom dicts using CIF name as key
    residue_atoms = {}  # cif_name -> {python_name: index}
    for (cif_name, atom), idx in atom_index.items():
        if cif_name not in residue_atoms:
            residue_atoms[cif_name] = {}
        python_name = to_python_name(atom)
        residue_atoms[cif_name][python_name] = idx

    # Group residues by type
    rna_residues = [r for r in all_residues if r.molecule_type == Molecule.RNA]
    dna_residues = [r for r in all_residues if r.molecule_type == Molecule.DNA]
    protein_residues = [r for r in all_residues if r.molecule_type == Molecule.PROTEIN]

    code = '''"""
Auto-generated atom enum definitions.

DO NOT EDIT MANUALLY - Generated by ciffy/src/codegen/generate.py

To modify atoms, edit ciffy/biochemistry/definitions.py and run:
    python ciffy/src/codegen/generate.py
"""

from ..utils import IndexEnum


'''

    # Generate individual atom classes
    for section_name, residues in [
        ("RNA NUCLEOTIDES", rna_residues),
        ("DNA NUCLEOTIDES", dna_residues),
        ("AMINO ACIDS", protein_residues),
    ]:
        if not residues:
            continue

        code += "# " + "=" * 77 + "\n"
        code += f"# {section_name}\n"
        code += "# " + "=" * 77 + "\n\n"

        for res in residues:
            cif_name = res.cif_names[0]
            if cif_name not in residue_atoms:
                continue  # Skip residues with no atoms (water, ions)

            atoms = residue_atoms[cif_name]
            code += f"class {res.class_name}(IndexEnum):\n"
            code += f'    """{res.class_name} ({cif_name}) atom indices."""\n'
            for python_name, idx in atoms.items():
                code += f"    {python_name} = {idx}\n"
            code += "\n\n"

    # Generate combined enums
    code += "# " + "=" * 77 + "\n"
    code += "# COMBINED ENUMS\n"
    code += "# " + "=" * 77 + "\n\n"

    # RibonucleicAcid (RNA only - 4 bases)
    rna_bases = [r for r in rna_residues if r.cif_names[0] in ("A", "C", "G", "U")]
    if rna_bases:
        code += "RibonucleicAcid = IndexEnum(\n"
        code += '    "RibonucleicAcid",\n'
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in rna_bases]
        code += "    " + " |\n    ".join(parts) + "\n"
        code += ")\n\n"

        code += "RibonucleicAcidNoPrefix = IndexEnum(\n"
        code += '    "RibonucleicAcid",\n'
        parts = [f'{r.class_name}.dict()' for r in rna_bases]
        code += "    " + " |\n    ".join(parts) + "\n"
        code += ")\n\n"

    # DeoxyribonucleicAcid (DNA only - 4 bases)
    dna_bases = [r for r in dna_residues if r.cif_names[0] in ("DA", "DC", "DG", "DT")]
    if dna_bases:
        code += "DeoxyribonucleicAcid = IndexEnum(\n"
        code += '    "DeoxyribonucleicAcid",\n'
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in dna_bases]
        code += "    " + " |\n    ".join(parts) + "\n"
        code += ")\n\n"

    # ModifiedNucleotides
    modified = [r for r in all_residues
                if r.molecule_type in (Molecule.RNA, Molecule.DNA)
                and r.cif_names[0] not in ("A", "C", "G", "U", "DA", "DC", "DG", "DT")]
    if modified:
        code += "ModifiedNucleotides = IndexEnum(\n"
        code += '    "ModifiedNucleotides",\n'
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in modified]
        code += "    " + " |\n    ".join(parts) + "\n"
        code += ")\n\n"

    # AminoAcids
    if protein_residues:
        code += "AminoAcids = IndexEnum(\n"
        code += '    "AminoAcids",\n'
        parts = [f'{r.class_name}.dict("{r.cif_names[0]}_")' for r in protein_residues]
        # Format nicely with line breaks
        formatted_parts = []
        line = "    "
        for part in parts:
            if len(line) + len(part) > 76:
                formatted_parts.append(line.rstrip(" |"))
                line = "    " + part + " |"
            else:
                line += part + " | "
        formatted_parts.append(line.rstrip(" |"))
        code += " |\n".join(formatted_parts) + "\n"
        code += ")\n\n"

    # Generate reverse lookup dict
    code += "# " + "=" * 77 + "\n"
    code += "# REVERSE LOOKUP\n"
    code += "# " + "=" * 77 + "\n\n"
    code += "# Maps atom index -> atom name (for all residue types)\n"
    code += "ATOM_NAMES: dict[int, str] = {\n"

    for (residue, atom), idx in sorted(atom_index.items(), key=lambda x: x[1]):
        code += f'    {idx}: "{atom}",\n'

    code += "}\n"

    with open(biochem_dir / "_generated_atoms.py", "w") as f:
        f.write(code)

    print("Generated: biochemistry/_generated_atoms.py")


def generate_residue_module(biochem_dir, all_residues, residue_to_molecule):
    """Generate Python Residue enum and mapping dicts."""

    code = '''"""
Auto-generated residue definitions.

DO NOT EDIT MANUALLY - Generated by ciffy/src/codegen/generate.py

To modify residues, edit ciffy/biochemistry/definitions.py and run:
    python ciffy/src/codegen/generate.py
"""

from ..utils import IndexEnum
from ..types import Molecule


'''

    # Generate Residue enum
    code += "class Residue(IndexEnum):\n"
    code += '    """\n'
    code += '    Residue types with unique integer indices.\n'
    code += '    \n'
    code += '    Includes nucleotides (RNA and DNA), amino acids, water, and ions.\n'
    code += '    """\n\n'

    for idx, res in enumerate(all_residues):
        code += f"    {res.name} = {idx}  # {res.class_name}\n"

    code += "\n\n"

    # Generate RESIDUE_MOLECULE_TYPE mapping
    code += "# Mapping from residue index to molecule type\n"
    code += "RESIDUE_MOLECULE_TYPE: dict[int, Molecule] = {\n"
    for idx, res in enumerate(all_residues):
        code += f"    {idx}: Molecule.{res.molecule_type.name},  # {res.name}\n"
    code += "}\n\n"

    # Generate CIF_RESIDUE_NAMES (CIF input names -> index)
    code += "# CIF residue names -> Residue index (for parsing)\n"
    code += "CIF_RESIDUE_NAMES: dict[str, int] = {\n"
    for idx, res in enumerate(all_residues):
        for cif_name in res.cif_names:
            code += f'    "{cif_name}": {idx},  # {res.name}\n'
    code += "}\n\n"

    # Generate RESIDUE_CIF_NAMES (index -> CIF output name)
    code += "# Residue index -> CIF output name (for writing)\n"
    code += "RESIDUE_CIF_NAMES: dict[int, str] = {\n"
    for idx, res in enumerate(all_residues):
        # Use first CIF name as canonical output
        code += f'    {idx}: "{res.cif_names[0]}",  # {res.name}\n'
    code += "}\n\n"

    # Generate RESIDUE_ABBREV (residue index -> single-letter abbreviation)
    code += "# Residue index -> single-letter abbreviation\n"
    code += "RESIDUE_ABBREV: dict[int, str] = {\n"
    for idx, res in enumerate(all_residues):
        code += f'    {idx}: "{res.abbreviation}",  # {res.name}\n'
    code += "}\n"

    with open(biochem_dir / "_generated_residues.py", "w") as f:
        f.write(code)

    print("Generated: biochemistry/_generated_residues.py")


def run_gperf(gperf_path):
    """Run gperf to generate .c files from .gperf files."""
    hash_dir = Path(__file__).parent.parent / "hash"

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

        with open(output_file, "w") as f:
            f.write(result.stdout)

    print("Generated: hash/*.c (atom, residue, element, molecule, entity, ion)")


def main():
    parser = argparse.ArgumentParser(description="Generate hash lookup tables")
    parser.add_argument("--gperf-path", help="Path to gperf executable")
    parser.add_argument(
        "--skip-gperf", action="store_true",
        help="Skip running gperf (only generate .gperf files)"
    )
    args = parser.parse_args()

    # Generate everything from definitions
    generate_all()

    # Run gperf
    if not args.skip_gperf:
        gperf_path = args.gperf_path or find_gperf()
        run_gperf(gperf_path)

    print("Hash table generation complete!")


if __name__ == "__main__":
    main()
