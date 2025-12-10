#!/usr/bin/env python3
"""
Auto-generate hash lookup tables for CIF parsing and writing.

Generates:
  - atom.gperf, residue.gperf, element.gperf (forward lookups)
  - hash/atom.c, hash/residue.c, hash/element.c (gperf output)
  - hash/reverse.h (reverse lookups for writing)

Usage:
  python generate.py [--gperf-path /path/to/gperf]

This script is called automatically during build via setup.py.
"""

import argparse
import subprocess
import shutil
from pathlib import Path


def find_gperf():
    """Find gperf executable (requires version 3.1+ for constants-prefix)."""
    # Check Homebrew paths first (they have newer versions)
    candidates = [
        "/opt/homebrew/bin/gperf",  # macOS Homebrew ARM
        "/usr/local/bin/gperf",      # macOS Homebrew Intel
        shutil.which("gperf"),       # System PATH
        "/usr/bin/gperf",            # Linux fallback
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise RuntimeError("gperf not found. Install with: brew install gperf (macOS) or apt install gperf (Linux)")


def generate_gperf_files():
    """Generate .gperf source files from Python enums."""
    from ciffy.biochemistry.nucleotides import (
        Adenosine, Cytosine, Guanosine, Uridine,
        GuanosineTriphosphate, CytidineTriphosphate, Deoxyguanosine,
    )
    from ciffy.biochemistry.residues import Residue
    from ciffy.biochemistry.elements import Element

    # === atom.gperf ===
    atom_str = """%define lookup-function-name _lookup_atom
%define hash-function-name _hash_atom
%define constants-prefix ATOM
%struct-type
%{
#include "lookup.h"
%}
struct _LOOKUP;
%%
"""
    # Helper to convert Python name to CIF name
    def to_cif_name(name):
        if name.endswith('pp'):
            return name[:-2] + "''"
        elif name.endswith('p'):
            return name[:-1] + "'"
        return name

    # Standard nucleotides
    for prefix, cls in [("A", Adenosine), ("C", Cytosine), ("G", Guanosine), ("U", Uridine)]:
        for member in cls:
            cif_name = to_cif_name(member.name)
            atom_str += f"{prefix}_{cif_name}, {member.value}\n"

    # Modified nucleotides
    for prefix, cls in [("GTP", GuanosineTriphosphate), ("CCC", CytidineTriphosphate), ("GNG", Deoxyguanosine)]:
        for member in cls:
            cif_name = to_cif_name(member.name)
            atom_str += f"{prefix}_{cif_name}, {member.value}\n"

    with open("atom.gperf", "w") as f:
        f.write(atom_str)

    # === residue.gperf ===
    residue_str = """%define lookup-function-name _lookup_residue
%define hash-function-name _hash_residue
%define constants-prefix RESIDUE
%struct-type
%{
#include "lookup.h"
%}
struct _LOOKUP;
%%
"""
    for member in Residue:
        residue_str += f"{member.name}, {member.value}\n"

    with open("residue.gperf", "w") as f:
        f.write(residue_str)

    # === element.gperf (static, rarely changes) ===
    element_str = """%define lookup-function-name _lookup_element
%define hash-function-name _hash_element
%define constants-prefix ELEMENT
%struct-type
%{
#include "lookup.h"
%}
struct _LOOKUP;
%%
"""
    for member in Element:
        element_str += f"{member.name}, {member.value}\n"

    with open("element.gperf", "w") as f:
        f.write(element_str)

    print("Generated: atom.gperf, residue.gperf, element.gperf")


def generate_reverse_header():
    """Generate reverse.h for CIF writing."""
    from ciffy.biochemistry.nucleotides import (
        Adenosine, Cytosine, Guanosine, Uridine,
        GuanosineTriphosphate, CytidineTriphosphate, Deoxyguanosine,
    )
    from ciffy.biochemistry.residues import Residue
    from ciffy.biochemistry.elements import Element

    # Helper to convert Python name to CIF name
    def to_cif_name(name):
        if name.endswith('pp'):
            return name[:-2] + "''"
        elif name.endswith('p'):
            return name[:-1] + "'"
        return name

    # Collect all atoms with their info
    atoms = {}  # idx -> (residue_prefix, atom_name)
    for prefix, cls in [
        ("A", Adenosine), ("C", Cytosine), ("G", Guanosine), ("U", Uridine),
        ("GTP", GuanosineTriphosphate), ("CCC", CytidineTriphosphate), ("GNG", Deoxyguanosine),
    ]:
        for member in cls:
            cif_name = to_cif_name(member.name)
            atoms[member.value] = (prefix, cif_name)

    # Collect residues
    residues = {}  # idx -> name
    for member in Residue:
        residues[member.value] = member.name

    # Collect elements
    elements = {}  # idx -> symbol
    for member in Element:
        elements[member.value] = member.name

    # Find max indices
    atom_max = max(atoms.keys()) + 1
    residue_max = max(residues.keys()) + 1
    element_max = max(elements.keys()) + 1

    # Generate header
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
        if i in residues:
            header += f'    [{i}] = "{residues[i]}",\n'
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

#endif /* _CIFFY_REVERSE_H */
'''

    with open("../hash/reverse.h", "w") as f:
        f.write(header)

    print("Generated: hash/reverse.h")


def run_gperf(gperf_path):
    """Run gperf to generate .c files from .gperf files."""
    hash_dir = Path("../hash")
    hash_dir.mkdir(exist_ok=True)

    for name in ["element", "residue", "atom"]:
        input_file = f"{name}.gperf"
        output_file = hash_dir / f"{name}.c"

        result = subprocess.run(
            [gperf_path, input_file],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"gperf failed for {input_file}: {result.stderr}")

        with open(output_file, "w") as f:
            f.write(result.stdout)

    print("Generated: hash/atom.c, hash/residue.c, hash/element.c")


def main():
    parser = argparse.ArgumentParser(description="Generate hash lookup tables")
    parser.add_argument("--gperf-path", help="Path to gperf executable")
    parser.add_argument("--skip-gperf", action="store_true", help="Skip running gperf (only generate .gperf files)")
    args = parser.parse_args()

    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)

    # Generate .gperf files
    generate_gperf_files()

    # Generate reverse.h
    generate_reverse_header()

    # Run gperf
    if not args.skip_gperf:
        gperf_path = args.gperf_path or find_gperf()
        run_gperf(gperf_path)

    print("Hash table generation complete!")


if __name__ == "__main__":
    main()
