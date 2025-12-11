#!/usr/bin/env python3
"""
Auto-generate hash lookup tables for CIF parsing and writing.

Generates:
  - hash/atom.gperf, hash/residue.gperf, hash/element.gperf (forward lookups)
  - hash/atom.c, hash/residue.c, hash/element.c (gperf output)
  - hash/reverse.h (reverse lookups for writing)
  - biochemistry/_generated_atoms.py (Python enums with auto-assigned indices)

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


def to_python_name(cif_name: str) -> str:
    """Convert CIF atom name to valid Python identifier.

    Examples:
        "C5'" -> "C5p"
        "H5''" -> "H5pp"
        "CA" -> "CA"
    """
    return cif_name.replace("'", "p")


def generate_all():
    """Generate all lookup tables and Python enums."""
    from ciffy.biochemistry.atoms import ALL_ATOMS
    from ciffy.biochemistry.residues import Residue
    from ciffy.biochemistry.elements import Element
    from ciffy.biochemistry.molecule_types import ENTITY_POLY_TYPES

    # Output directories
    hash_dir = Path(__file__).parent.parent / "hash"
    biochem_dir = Path(__file__).parent.parent.parent / "biochemistry"
    hash_dir.mkdir(exist_ok=True)

    # === Assign indices to all atoms ===
    # Index 0 is reserved for "unknown"
    atom_index = {}  # (residue, atom_name) -> index
    current_idx = 1

    for residue, atoms in ALL_ATOMS.items():
        for atom in atoms:
            key = (residue, atom)
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
    for member in Residue:
        residue_gperf += f"{member.name}, {member.value}\n"

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
        # Quote strings with special characters
        if "(" in type_str or "/" in type_str or " " in type_str:
            molecule_gperf += f'"{type_str}", {mol_value}\n'
        else:
            molecule_gperf += f"{type_str}, {mol_value}\n"

    with open(hash_dir / "molecule.gperf", "w") as f:
        f.write(molecule_gperf)

    print("Generated: hash/atom.gperf, hash/residue.gperf, hash/element.gperf, hash/molecule.gperf")

    # === Generate reverse.h ===
    generate_reverse_header(hash_dir, atom_index, Residue, Element)

    # === Generate Python enums ===
    generate_python_enums(biochem_dir, atom_index)

    return atom_index


def generate_reverse_header(hash_dir, atom_index, Residue, Element):
    """Generate reverse.h for CIF writing."""
    from ciffy.biochemistry.molecule_types import ENTITY_POLY_TYPES

    # Collect atoms info
    atoms = {}  # idx -> (residue, atom_name)
    for (residue, atom), idx in atom_index.items():
        atoms[idx] = (residue, atom)

    # Collect residues
    residues = {member.value: member.name for member in Residue}

    # Collect elements
    elements = {member.value: member.name for member in Element}

    # Collect molecule types (reverse: value -> CIF string)
    # Use preferred canonical CIF strings for each molecule type
    from ciffy.types import Molecule
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
    residue_max = max(residues.keys()) + 1
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


def generate_python_enums(biochem_dir, atom_index):
    """Generate Python enum file with auto-assigned indices."""
    from ciffy.biochemistry.atoms import (
        NUCLEOTIDE_ATOMS, AMINO_ACID_ATOMS,
    )

    # Build per-residue atom dicts
    residue_atoms = {}  # residue -> {python_name: index}
    for (residue, atom), idx in atom_index.items():
        if residue not in residue_atoms:
            residue_atoms[residue] = {}
        python_name = to_python_name(atom)
        residue_atoms[residue][python_name] = idx

    # Class name mapping
    class_names = {
        # Nucleotides
        "A": "Adenosine",
        "C": "Cytosine",
        "G": "Guanosine",
        "U": "Uridine",
        "GTP": "GuanosineTriphosphate",
        "CCC": "CytidineTriphosphate",
        "GNG": "Deoxyguanosine",
        # Amino acids
        "GLY": "Glycine",
        "ALA": "Alanine",
        "VAL": "Valine",
        "LEU": "Leucine",
        "ILE": "Isoleucine",
        "PRO": "Proline",
        "PHE": "Phenylalanine",
        "TRP": "Tryptophan",
        "MET": "Methionine",
        "CYS": "Cysteine",
        "SER": "Serine",
        "THR": "Threonine",
        "ASN": "Asparagine",
        "GLN": "Glutamine",
        "ASP": "AsparticAcid",
        "GLU": "GlutamicAcid",
        "LYS": "Lysine",
        "ARG": "Arginine",
        "HIS": "Histidine",
        "TYR": "Tyrosine",
    }

    code = '''"""
Auto-generated atom enum definitions.

DO NOT EDIT MANUALLY - Generated by ciffy/src/codegen/generate.py

To modify atoms, edit ciffy/biochemistry/atoms.py and run:
    python ciffy/src/codegen/generate.py
"""

from ..utils import IndexEnum


'''

    # Generate nucleotide classes
    code += "# " + "=" * 77 + "\n"
    code += "# NUCLEOTIDES\n"
    code += "# " + "=" * 77 + "\n\n"

    for residue in NUCLEOTIDE_ATOMS.keys():
        class_name = class_names[residue]
        atoms = residue_atoms[residue]
        code += f"class {class_name}(IndexEnum):\n"
        code += f'    """{class_name} ({residue}) atom indices."""\n'
        for python_name, idx in atoms.items():
            code += f"    {python_name} = {idx}\n"
        code += "\n\n"

    # Generate amino acid classes
    code += "# " + "=" * 77 + "\n"
    code += "# AMINO ACIDS\n"
    code += "# " + "=" * 77 + "\n\n"

    for residue in AMINO_ACID_ATOMS.keys():
        class_name = class_names[residue]
        atoms = residue_atoms[residue]
        code += f"class {class_name}(IndexEnum):\n"
        code += f'    """{class_name} ({residue}) atom indices."""\n'
        for python_name, idx in atoms.items():
            code += f"    {python_name} = {idx}\n"
        code += "\n\n"

    # Generate combined enums
    code += "# " + "=" * 77 + "\n"
    code += "# COMBINED ENUMS\n"
    code += "# " + "=" * 77 + "\n\n"

    # RibonucleicAcid
    code += "RibonucleicAcid = IndexEnum(\n"
    code += '    "RibonucleicAcid",\n'
    code += '    Adenosine.dict("A_") | Cytosine.dict("C_") |\n'
    code += '    Guanosine.dict("G_") | Uridine.dict("U_")\n'
    code += ")\n\n"

    # RibonucleicAcidNoPrefix
    code += "RibonucleicAcidNoPrefix = IndexEnum(\n"
    code += '    "RibonucleicAcid",\n'
    code += '    Adenosine.dict() | Cytosine.dict() |\n'
    code += '    Guanosine.dict() | Uridine.dict()\n'
    code += ")\n\n"

    # ModifiedNucleotides
    code += "ModifiedNucleotides = IndexEnum(\n"
    code += '    "ModifiedNucleotides",\n'
    code += '    GuanosineTriphosphate.dict("GTP_") |\n'
    code += '    CytidineTriphosphate.dict("CCC_") |\n'
    code += '    Deoxyguanosine.dict("GNG_")\n'
    code += ")\n\n"

    # AminoAcids
    code += "AminoAcids = IndexEnum(\n"
    code += '    "AminoAcids",\n'
    code += '    Glycine.dict("GLY_") | Alanine.dict("ALA_") |\n'
    code += '    Valine.dict("VAL_") | Leucine.dict("LEU_") |\n'
    code += '    Isoleucine.dict("ILE_") | Proline.dict("PRO_") |\n'
    code += '    Phenylalanine.dict("PHE_") | Tryptophan.dict("TRP_") |\n'
    code += '    Methionine.dict("MET_") | Cysteine.dict("CYS_") |\n'
    code += '    Serine.dict("SER_") | Threonine.dict("THR_") |\n'
    code += '    Asparagine.dict("ASN_") | Glutamine.dict("GLN_") |\n'
    code += '    AsparticAcid.dict("ASP_") | GlutamicAcid.dict("GLU_") |\n'
    code += '    Lysine.dict("LYS_") | Arginine.dict("ARG_") |\n'
    code += '    Histidine.dict("HIS_") | Tyrosine.dict("TYR_")\n'
    code += ")\n\n"

    # Generate reverse lookup dict (index -> atom_name)
    code += "# " + "=" * 77 + "\n"
    code += "# REVERSE LOOKUP\n"
    code += "# " + "=" * 77 + "\n\n"
    code += "# Maps atom index -> atom name (for all residue types)\n"
    code += "ATOM_NAMES: dict[int, str] = {\n"

    # Sort by index for readable output
    for (residue, atom), idx in sorted(atom_index.items(), key=lambda x: x[1]):
        code += f"    {idx}: \"{atom}\",\n"

    code += "}\n"

    with open(biochem_dir / "_generated_atoms.py", "w") as f:
        f.write(code)

    print("Generated: biochemistry/_generated_atoms.py")


def run_gperf(gperf_path):
    """Run gperf to generate .c files from .gperf files."""
    hash_dir = Path(__file__).parent.parent / "hash"

    for name in ["element", "residue", "atom", "molecule"]:
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

    print("Generated: hash/atom.c, hash/residue.c, hash/element.c, hash/molecule.c")


def main():
    parser = argparse.ArgumentParser(description="Generate hash lookup tables")
    parser.add_argument("--gperf-path", help="Path to gperf executable")
    parser.add_argument("--skip-gperf", action="store_true", help="Skip running gperf (only generate .gperf files)")
    args = parser.parse_args()

    # Generate .gperf files, reverse.h, and Python enums
    generate_all()

    # Run gperf
    if not args.skip_gperf:
        gperf_path = args.gperf_path or find_gperf()
        run_gperf(gperf_path)

    print("Hash table generation complete!")


if __name__ == "__main__":
    main()
