"""
Code generation package for ciffy.

This package generates:
- C hash tables (via gperf)
- Reverse lookup headers
- Bond pattern headers
- Python enums for molecules, elements, atoms, and residues

Main entry point: generate_all(ccd_path)
CLI entry point: cli.main()
"""

from __future__ import annotations

from pathlib import Path

from .config import MOLECULE_TYPES, ELEMENTS, IONS, RESIDUE_WHITELIST
from .ccd import load_residues_from_ccd
from .c_codegen import (
    generate_gperf_files,
    generate_reverse_header,
    generate_bond_patterns_header,
    generate_canonical_refs_header,
)
from .residue import (
    compute_canonical_zmatrix_refs,
    compute_atom_dihedral_ownership,
    compute_residue_backbone_atoms,
)
from .python_codegen import (
    generate_python_molecule,
    generate_python_elements,
    generate_python_atoms,
    generate_python_residues,
    generate_dihedral_arrays,
    generate_zmatrix_arrays,
    generate_python_dihedraltypes,
)


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

    # Output directories (codegen is at project root, outputs go into ciffy package)
    project_root = Path(__file__).parent.parent
    ciffy_dir = project_root / "ciffy"
    hash_dir = ciffy_dir / "src" / "hash"
    internal_dir = ciffy_dir / "src" / "internal"
    biochem_dir = ciffy_dir / "biochemistry"
    types_dir = ciffy_dir / "types"
    hash_dir.mkdir(exist_ok=True)
    internal_dir.mkdir(exist_ok=True)

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

    # Compute arrays needed for multiple generators
    atom_dihedral_type, atom_dihedral_refs = compute_atom_dihedral_ownership(all_residues, atom_index)
    atom_canonical_refs, atom_has_canonical_refs = compute_canonical_zmatrix_refs(
        all_residues, atom_index
    )
    residue_backbone_atoms = compute_residue_backbone_atoms(all_residues, atom_index)

    # Generate all files
    generate_gperf_files(hash_dir, atom_index, cif_to_residue, residue_index, all_residues)
    generate_reverse_header(hash_dir, atom_index, residue_to_cif)
    generate_bond_patterns_header(internal_dir, all_residues, atom_index)
    generate_canonical_refs_header(
        internal_dir,
        all_residues,
        atom_index,
        atom_canonical_refs,
        atom_has_canonical_refs,
        atom_dihedral_type,
        atom_dihedral_refs,
        residue_backbone_atoms,
    )
    generate_python_molecule(biochem_dir)
    generate_python_elements(biochem_dir)
    generate_python_atoms(biochem_dir, atom_index, all_residues)
    generate_python_residues(biochem_dir, all_residues)
    generate_dihedral_arrays(biochem_dir, all_residues, atom_index)
    generate_zmatrix_arrays(biochem_dir, all_residues, atom_index)
    generate_python_dihedraltypes(biochem_dir)

    return hash_dir, atom_index


__all__ = [
    "generate_all",
    "MOLECULE_TYPES",
    "ELEMENTS",
    "IONS",
    "RESIDUE_WHITELIST",
]
