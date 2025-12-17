"""
C code generation for hash tables and headers.

Generates:
- GPERF files for hash-based lookups
- reverse.h for CIF writing
- bond_patterns.h for bond graph building
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .config import ELEMENTS, IONS, MOLECULE_TYPES, Molecule, BACKBONE_NAMES, NUM_BACKBONE_NAMES
from .residue import ResidueDefinition

# Minimum gperf version required for %define constants-prefix
GPERF_MIN_VERSION = (3, 1)


def _check_gperf_version(gperf_path: str) -> tuple[int, int] | None:
    """
    Check gperf version.

    Returns:
        Tuple of (major, minor) version, or None if version cannot be determined.
    """
    try:
        result = subprocess.run(
            [gperf_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse version from output like "GNU gperf 3.1"
            match = re.search(r"gperf\s+(\d+)\.(\d+)", result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _gperf_header(lookup_name: str, hash_name: str, prefix: str) -> str:
    """Generate standard gperf file header."""
    return f"""%define lookup-function-name {lookup_name}
%define hash-function-name {hash_name}
%define constants-prefix {prefix}
%struct-type
%{{
#include "../lookup.h"
%}}
struct _LOOKUP;
%%
"""


def find_gperf() -> str:
    """
    Find gperf executable and validate version.

    Requires gperf 3.1+ for %define constants-prefix support.

    Returns:
        Path to valid gperf executable.

    Raises:
        RuntimeError: If gperf is not found or version is too old.
    """
    candidates = [
        "/opt/homebrew/bin/gperf",
        "/usr/local/bin/gperf",
        shutil.which("gperf"),
        "/usr/bin/gperf",
    ]

    found_path = None
    found_version = None

    for path in candidates:
        if path and Path(path).exists():
            version = _check_gperf_version(path)
            if version is not None and version >= GPERF_MIN_VERSION:
                return path
            # Track what we found for error message
            if found_path is None:
                found_path = path
                found_version = version

    if found_path is not None:
        version_str = f"{found_version[0]}.{found_version[1]}" if found_version else "unknown"
        min_str = f"{GPERF_MIN_VERSION[0]}.{GPERF_MIN_VERSION[1]}"
        raise RuntimeError(
            f"gperf version {version_str} found at {found_path}, but version {min_str}+ is required.\n"
            f"The %define constants-prefix feature requires gperf 3.1 or later.\n"
            f"Upgrade with: brew install gperf (macOS) or apt install gperf (Linux)"
        )

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


def generate_bond_patterns_header(
    internal_dir: Path,
    all_residues: list[ResidueDefinition],
    atom_index: dict[tuple[str, str], int],
) -> None:
    """Generate bond_patterns.h with static bond arrays for C bond graph building."""

    n_residues = len(all_residues)

    # Molecule type indices derived from MOLECULE_TYPES (single source of truth)
    mol_idx = {mt.name: idx for idx, mt in enumerate(MOLECULE_TYPES)}

    lines = [
        '#ifndef _CIFFY_BOND_PATTERNS_H',
        '#define _CIFFY_BOND_PATTERNS_H',
        '',
        '/**',
        ' * @file bond_patterns.h',
        ' * @brief Static bond patterns for each residue type.',
        ' * AUTO-GENERATED by generate.py - DO NOT EDIT MANUALLY.',
        ' *',
        ' * Each residue type has an array of bond pairs stored as',
        ' * (atom_value_1, atom_value_2) pairs. These are the atom enum',
        ' * values, not local indices.',
        ' */',
        '',
        '#include <stdint.h>',
        '',
        f'#define NUM_RESIDUE_TYPES {n_residues}',
        '',
        '/* Molecule type constants */',
        f'#define MOL_PROTEIN {mol_idx["PROTEIN"]}',
        f'#define MOL_RNA {mol_idx["RNA"]}',
        f'#define MOL_DNA {mol_idx["DNA"]}',
        f'#define MOL_HYBRID {mol_idx["HYBRID"]}',
        f'#define MOL_PROTEIN_D {mol_idx["PROTEIN_D"]}',
        f'#define MOL_CYCLIC_PEPTIDE {mol_idx["CYCLIC_PEPTIDE"]}',
        '',
    ]

    # Generate bond arrays for each residue
    bond_array_names = []
    bond_counts = []
    mol_types = []
    linking_prev = []  # Atom value for prev residue's linking atom (O3' or C)
    linking_next = []  # Atom value for next residue's linking atom (P or N)

    # Molecule type groups for linking
    nucleic_types = (mol_idx["RNA"], mol_idx["DNA"], mol_idx["HYBRID"])
    protein_types = (mol_idx["PROTEIN"], mol_idx["PROTEIN_D"], mol_idx["CYCLIC_PEPTIDE"])

    for res_idx, res in enumerate(all_residues):
        primary_cif = res.cif_names[0]
        array_name = f'BONDS_{res.class_name}'
        mol_types.append(res.molecule_type)

        # Determine linking atoms based on molecule type
        if res.molecule_type in nucleic_types:
            # Nucleic acids: O3' -> P linkage
            prev_atom = atom_index.get((primary_cif, "O3'"), 0)
            next_atom = atom_index.get((primary_cif, "P"), 0)
        elif res.molecule_type in protein_types:
            # Proteins: C -> N linkage
            prev_atom = atom_index.get((primary_cif, "C"), 0)
            next_atom = atom_index.get((primary_cif, "N"), 0)
        else:
            # No linking
            prev_atom = 0
            next_atom = 0
        linking_prev.append(prev_atom)
        linking_next.append(next_atom)

        if res.bonds:
            # Get atom indices for each bond pair
            bond_pairs = []
            for atom1, atom2 in res.bonds:
                idx1 = atom_index.get((primary_cif, atom1), 0)
                idx2 = atom_index.get((primary_cif, atom2), 0)
                if idx1 > 0 and idx2 > 0:
                    bond_pairs.append((idx1, idx2))

            if bond_pairs:
                # Output as flat array: {idx1, idx2, idx1, idx2, ...}
                lines.append(f'static const int32_t {array_name}[] = {{')
                for i, (idx1, idx2) in enumerate(bond_pairs):
                    comma = ',' if i < len(bond_pairs) - 1 else ''
                    lines.append(f'    {idx1}, {idx2}{comma}')
                lines.append('};')
                lines.append(f'#define {array_name}_COUNT {len(bond_pairs)}')
                lines.append('')
                bond_array_names.append(array_name)
                bond_counts.append(len(bond_pairs))
            else:
                bond_array_names.append('NULL')
                bond_counts.append(0)
        else:
            bond_array_names.append('NULL')
            bond_counts.append(0)

    # Generate lookup tables
    lines.append('/* Lookup tables indexed by residue type */')
    lines.append(f'static const int32_t* RESIDUE_BONDS[NUM_RESIDUE_TYPES] = {{')
    for i, name in enumerate(bond_array_names):
        comma = ',' if i < len(bond_array_names) - 1 else ''
        lines.append(f'    {name}{comma}')
    lines.append('};')
    lines.append('')

    lines.append(f'static const int RESIDUE_BOND_COUNTS[NUM_RESIDUE_TYPES] = {{')
    for i, count in enumerate(bond_counts):
        comma = ',' if i < len(bond_counts) - 1 else ''
        lines.append(f'    {count}{comma}')
    lines.append('};')
    lines.append('')

    # Molecule type per residue
    lines.append('/* Molecule type for each residue type */')
    lines.append(f'static const int8_t RESIDUE_MOLECULE_TYPE[NUM_RESIDUE_TYPES] = {{')
    for i, mt in enumerate(mol_types):
        comma = ',' if i < len(mol_types) - 1 else ''
        lines.append(f'    {mt}{comma}')
    lines.append('};')
    lines.append('')

    # Linking atoms (0 = not present)
    lines.append('/* Linking atom values (0 = not present/not applicable) */')
    lines.append(f'static const int32_t RESIDUE_LINKING_PREV[NUM_RESIDUE_TYPES] = {{')
    for i, val in enumerate(linking_prev):
        comma = ',' if i < len(linking_prev) - 1 else ''
        lines.append(f'    {val}{comma}')
    lines.append('};')
    lines.append('')

    lines.append(f'static const int32_t RESIDUE_LINKING_NEXT[NUM_RESIDUE_TYPES] = {{')
    for i, val in enumerate(linking_next):
        comma = ',' if i < len(linking_next) - 1 else ''
        lines.append(f'    {val}{comma}')
    lines.append('};')
    lines.append('')

    lines.append('#endif /* _CIFFY_BOND_PATTERNS_H */')
    lines.append('')

    (internal_dir / "bond_patterns.h").write_text('\n'.join(lines))
    print("Generated: internal/bond_patterns.h")


def generate_canonical_refs_header(
    internal_dir: Path,
    all_residues: list[ResidueDefinition],
    atom_index: dict[tuple[str, str], int],
    atom_canonical_refs: np.ndarray,
    atom_has_canonical_refs: np.ndarray,
    atom_dihedral_type: np.ndarray,
    atom_dihedral_refs: np.ndarray,
    residue_backbone_atoms: np.ndarray,
) -> None:
    """
    Generate canonical_refs.h with lookup tables for C Z-matrix construction.

    This header enables single-pass Z-matrix construction in C without Python
    post-processing. It contains:
    - ATOM_CANONICAL_REFS: Pre-defined reference atoms for each atom type
    - ATOM_HAS_CANONICAL_REFS: Whether each atom type has canonical refs
    - ATOM_DIHEDRAL_TYPE: Which dihedral (if any) each atom owns
    - ATOM_DIHEDRAL_REFS: Reference atoms for dihedral owners (offset, local_idx)
    - RESIDUE_BACKBONE_ATOMS: Backbone atom types for each residue type
    - RESIDUE_CANONICAL_ATOMS: Canonical atom ordering per residue type
    - RESIDUE_ATOM_COUNTS: Number of atoms per residue type

    Args:
        internal_dir: Output directory for header file.
        all_residues: List of all residue definitions.
        atom_index: Dict mapping (cif_name, atom_name) -> global atom index.
        atom_canonical_refs: (num_atoms, 6) array from compute_canonical_zmatrix_refs.
        atom_has_canonical_refs: (num_atoms,) bool array.
        atom_dihedral_type: (num_atoms,) int8 array from compute_atom_dihedral_ownership.
        atom_dihedral_refs: (num_atoms, 3, 2) int8 array from compute_atom_dihedral_ownership.
        residue_backbone_atoms: (num_residues, num_backbone_names) int16 array.
    """
    num_atoms = len(atom_canonical_refs)
    num_residues = len(all_residues)

    lines = [
        '#ifndef _CIFFY_CANONICAL_REFS_H',
        '#define _CIFFY_CANONICAL_REFS_H',
        '',
        '/**',
        ' * @file canonical_refs.h',
        ' * @brief Canonical Z-matrix reference tables for single-pass construction.',
        ' * AUTO-GENERATED by generate.py - DO NOT EDIT MANUALLY.',
        ' *',
        ' * Each atom type has pre-defined reference atoms for Z-matrix construction.',
        ' * For intra-residue refs (offset=0), the ref value is the atom TYPE.',
        ' * For inter-residue refs (offset!=0), the ref value is a BACKBONE NAME ID',
        ' * which must be resolved via RESIDUE_BACKBONE_ATOMS.',
        ' */',
        '',
        '#include <stdint.h>',
        '',
        f'#define NUM_ATOM_TYPES {num_atoms}',
        f'#define NUM_RESIDUE_TYPES {num_residues}',
        f'#define NUM_BACKBONE_NAMES {NUM_BACKBONE_NAMES}',
        '',
        '/* Backbone name IDs for inter-residue reference resolution */',
    ]

    # Add backbone name defines
    for name, idx in sorted(BACKBONE_NAMES.items(), key=lambda x: x[1]):
        # Convert CIF name to C-safe identifier
        c_name = name.replace("'", "P").replace("-", "_")
        lines.append(f'#define BACKBONE_{c_name} {idx}')

    lines.extend([
        '',
        '/**',
        ' * Canonical refs for each atom type.',
        ' * Columns: [dist_ref, ang_ref, dih_ref, dist_off, ang_off, dih_off]',
        ' * If offset == 0: ref value is atom TYPE (global index)',
        ' * If offset != 0: ref value is backbone NAME ID',
        ' * Value of -1 means no reference.',
        ' */',
        f'static const int16_t ATOM_CANONICAL_REFS[NUM_ATOM_TYPES][6] = {{',
    ])

    # Output ATOM_CANONICAL_REFS
    for i in range(num_atoms):
        vals = atom_canonical_refs[i]
        line = f'    [{i}] = {{{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}, {vals[4]}, {vals[5]}}},'
        lines.append(line)

    lines.extend([
        '};',
        '',
        '/* Which atoms have canonical refs defined (1 = yes, 0 = no) */',
        f'static const int8_t ATOM_HAS_CANONICAL_REFS[NUM_ATOM_TYPES] = {{',
    ])

    # Output ATOM_HAS_CANONICAL_REFS (8 values per line for compactness)
    for i in range(0, num_atoms, 16):
        chunk = atom_has_canonical_refs[i:i+16]
        vals = ', '.join(str(int(v)) for v in chunk)
        lines.append(f'    {vals},')

    lines.extend([
        '};',
        '',
        '/**',
        ' * Dihedral type ownership: -1 if not a named dihedral owner.',
        ' * Values correspond to DihedralType enum indices.',
        ' */',
        f'static const int8_t ATOM_DIHEDRAL_TYPE[NUM_ATOM_TYPES] = {{',
    ])

    # Output ATOM_DIHEDRAL_TYPE (16 values per line for compactness)
    for i in range(0, num_atoms, 16):
        chunk = atom_dihedral_type[i:i+16]
        vals = ', '.join(str(int(v)) for v in chunk)
        lines.append(f'    {vals},')

    lines.extend([
        '};',
        '',
        '/**',
        ' * Backbone name ID -> atom type for each residue type.',
        ' * Enables inter-residue ref resolution across different residue types.',
        ' * Value of -1 means that backbone atom does not exist in the residue.',
        ' */',
        f'static const int16_t RESIDUE_BACKBONE_ATOMS[NUM_RESIDUE_TYPES][NUM_BACKBONE_NAMES] = {{',
    ])

    # Output RESIDUE_BACKBONE_ATOMS
    for res_idx, res in enumerate(all_residues):
        vals = residue_backbone_atoms[res_idx]
        vals_str = ', '.join(str(int(v)) for v in vals)
        lines.append(f'    [{res_idx}] = {{{vals_str}}},  /* {res.name} */')

    lines.extend([
        '};',
        '',
    ])

    # Generate ATOM_DIHEDRAL_REFS
    # Shape: (num_atoms, 3, 2) -> flatten to (num_atoms, 6)
    # Each row: [dih_offset, dih_local_idx, ang_offset, ang_local_idx, dist_offset, dist_local_idx]
    lines.extend([
        '/**',
        ' * Dihedral reference atoms for dihedral owners.',
        ' * Each row: [dih_offset, dih_idx, ang_offset, ang_idx, dist_offset, dist_idx]',
        ' * where offset is residue offset (-1/0/+1) and idx is local atom index.',
        ' * Only meaningful where ATOM_DIHEDRAL_TYPE[atom] >= 0.',
        ' */',
        f'static const int8_t ATOM_DIHEDRAL_REFS[NUM_ATOM_TYPES][6] = {{',
    ])

    for i in range(num_atoms):
        refs = atom_dihedral_refs[i]  # Shape (3, 2)
        # Flatten: [dih, ang, dist] each with [offset, idx]
        flat = [refs[0, 0], refs[0, 1], refs[1, 0], refs[1, 1], refs[2, 0], refs[2, 1]]
        vals = ', '.join(str(int(v)) for v in flat)
        lines.append(f'    [{i}] = {{{vals}}},')

    lines.extend([
        '};',
        '',
    ])

    # Compute canonical atom ordering per residue
    # Build (residue_type, local_idx) -> global_atom_type mapping
    max_atoms_per_res = max(len(res.atoms) for res in all_residues) if all_residues else 0
    residue_canonical_atoms = np.zeros((num_residues, max_atoms_per_res), dtype=np.int16)
    residue_atom_counts = np.zeros(num_residues, dtype=np.int16)  # int16 for residues with >127 atoms

    for res_idx, res in enumerate(all_residues):
        primary_cif = res.cif_names[0]
        residue_atom_counts[res_idx] = len(res.atoms)
        for local_idx, atom_name in enumerate(res.atoms):
            key = (primary_cif, atom_name)
            if key in atom_index:
                residue_canonical_atoms[res_idx, local_idx] = atom_index[key]

    lines.extend([
        '/**',
        ' * Canonical atom ordering per residue type.',
        ' * Maps (residue_type, local_idx) -> global atom type.',
        ' * Used to resolve dihedral references by local index.',
        ' */',
        f'#define MAX_ATOMS_PER_RESIDUE {max_atoms_per_res}',
        '',
        f'static const int16_t RESIDUE_CANONICAL_ATOMS[NUM_RESIDUE_TYPES][MAX_ATOMS_PER_RESIDUE] = {{',
    ])

    for res_idx, res in enumerate(all_residues):
        vals = residue_canonical_atoms[res_idx]
        # Only output non-zero values up to atom count
        count = int(residue_atom_counts[res_idx])
        vals_str = ', '.join(str(int(v)) for v in vals[:count])
        if count < max_atoms_per_res:
            # Pad with zeros for remaining slots
            vals_str += ', ' + ', '.join(['0'] * (max_atoms_per_res - count))
        lines.append(f'    [{res_idx}] = {{{vals_str}}},  /* {res.name} */')

    lines.extend([
        '};',
        '',
        '/* Number of atoms per residue type */',
        f'static const int16_t RESIDUE_ATOM_COUNTS[NUM_RESIDUE_TYPES] = {{',
    ])

    # Output counts (16 per line for compactness)
    for i in range(0, num_residues, 16):
        chunk = residue_atom_counts[i:i+16]
        vals = ', '.join(str(int(v)) for v in chunk)
        lines.append(f'    {vals},')

    lines.extend([
        '};',
        '',
        '#endif /* _CIFFY_CANONICAL_REFS_H */',
        '',
    ])

    (internal_dir / "canonical_refs.h").write_text('\n'.join(lines))
    print("Generated: internal/canonical_refs.h")
