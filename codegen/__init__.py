"""
Code generation package for ciffy.

This package generates:
- C hash tables (via gperf)
- Reverse lookup headers
- Bond pattern headers
- Python enums for molecules, elements, atoms, and residues

Main entry point: generate_all(ccd_path)
CLI entry point: cli.main()

The generation process has four phases:
1. Load: Read elements from PubChem and residues from CCD
2. Validate: Check for duplicates and compare against authoritative sources
3. Index: Build atom/residue indices and compute derived arrays
4. Generate: Write C headers and Python modules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import MOLECULE_TYPES, IONS, RESIDUE_WHITELIST
from .ccd import load_residues_from_ccd
from .elements import load_elements
from .c_codegen import (
    generate_gperf_files,
    generate_reverse_header,
    generate_bond_patterns_header,
    generate_canonical_refs_header,
)
from .residue import (
    ResidueDefinition,
    compute_atom_dihedral_ownership,
    compute_residue_backbone_atoms,
)
from .python_codegen import (
    generate_python_molecule,
    generate_python_elements,
    generate_dihedral_arrays,
    generate_python_dihedraltypes,
    generate_python_atoms,
    generate_python_residues,
)
from .validation import run_all_validations


# =============================================================================
# DATA STRUCTURES FOR GENERATION PIPELINE
# =============================================================================


@dataclass
class LoadedData:
    """Data loaded from external sources (Phase 1)."""
    elements: dict[str, int]
    residues: list[ResidueDefinition]


@dataclass
class IndexedData:
    """Indices and mappings built from loaded data (Phase 3)."""
    residue_index: dict[str, int]  # residue name -> index
    cif_to_residue: dict[str, int]  # CIF name -> residue index
    residue_to_cif: dict[int, str]  # residue index -> primary CIF name
    atom_index: dict[tuple[str, str], int]  # (cif_name, atom_name) -> atom index
    atom_dihedral_type: np.ndarray  # atom index -> dihedral type
    atom_dihedral_refs: np.ndarray  # atom index -> reference atoms
    residue_backbone_atoms: np.ndarray  # residue index -> backbone atom indices


@dataclass
class OutputPaths:
    """Output directory paths for generated files."""
    hash_dir: Path
    internal_dir: Path
    biochem_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> 'OutputPaths':
        """Create output paths relative to project root."""
        ciffy_dir = project_root / "ciffy"
        paths = cls(
            hash_dir=ciffy_dir / "src" / "hash",
            internal_dir=ciffy_dir / "src" / "internal",
            biochem_dir=ciffy_dir / "biochemistry",
        )
        paths.hash_dir.mkdir(exist_ok=True)
        paths.internal_dir.mkdir(exist_ok=True)
        return paths


# =============================================================================
# PHASE 1: LOAD DATA
# =============================================================================


def _load_data(ccd_path: str) -> LoadedData:
    """
    Phase 1: Load data from external sources.

    Loads elements from PubChem (cached locally) and residue definitions
    from the Chemical Component Dictionary.
    """
    elements = load_elements()
    print(f"Loaded {len(elements)} elements from PubChem")

    residues = load_residues_from_ccd(ccd_path)
    return LoadedData(elements=elements, residues=residues)


# =============================================================================
# PHASE 2: VALIDATE DATA
# =============================================================================


def _validate_data(data: LoadedData) -> None:
    """
    Phase 2: Validate loaded data.

    Checks for duplicate CIF names and compares backbone dihedrals
    against authoritative MonomerLibrary source.
    """
    # Check for duplicate CIF names
    seen_cif: dict[str, str] = {}
    for res in data.residues:
        for cif_name in res.cif_names:
            if cif_name in seen_cif:
                raise ValueError(
                    f"Duplicate CIF name '{cif_name}' in {res.name} and {seen_cif[cif_name]}"
                )
            seen_cif[cif_name] = res.name

    # Run validation checks (compares against authoritative sources)
    run_all_validations(data.residues)


# =============================================================================
# PHASE 3: BUILD INDICES
# =============================================================================


def _build_indices(data: LoadedData) -> IndexedData:
    """
    Phase 3: Build indices and derived arrays.

    Creates lookup tables for residues and atoms, and computes
    dihedral ownership and backbone atom arrays.

    Atom indexing uses a two-phase algorithm:
    1. Backbone atoms get unified values (1-17) shared across all residue types
    2. Sidechain/base atoms get unique values (18+) per residue type

    This enables robustness to modified residues with standard backbones.
    """
    from .config import (
        UNIFIED_BACKBONE_VALUES,
        NUM_UNIFIED_BACKBONE,
        is_backbone_atom,
    )

    residues = data.residues

    # Build residue mappings
    residue_index = {res.name: idx for idx, res in enumerate(residues)}
    cif_to_residue = {
        cif: idx for idx, res in enumerate(residues) for cif in res.cif_names
    }
    residue_to_cif = {idx: res.cif_names[0] for idx, res in enumerate(residues)}

    # Build atom index using two-phase algorithm
    # Phase A: Backbone atoms get unified values (shared across residues)
    # Phase B: Sidechain/base atoms get unique values (per residue)
    atom_index: dict[tuple[str, str], int] = {}

    # Phase A: Assign backbone atoms their unified values
    backbone_count = 0
    for res in residues:
        primary_cif = res.cif_names[0]
        for atom in res.atoms:
            if is_backbone_atom(atom, res.molecule_type):
                key = (primary_cif, atom)
                if key not in atom_index:
                    atom_index[key] = UNIFIED_BACKBONE_VALUES[atom]
                    backbone_count += 1

    # Phase B: Assign sidechain/base atoms unique values starting after backbone
    current_idx = NUM_UNIFIED_BACKBONE + 1
    sidechain_count = 0
    for res in residues:
        primary_cif = res.cif_names[0]
        for atom in res.atoms:
            key = (primary_cif, atom)
            if key not in atom_index:  # Not a backbone atom
                atom_index[key] = current_idx
                current_idx += 1
                sidechain_count += 1

    # Add aliases pointing to same indices
    for res in residues:
        primary_cif = res.cif_names[0]
        for alias in res.cif_names[1:]:
            for atom in res.atoms:
                primary_key = (primary_cif, atom)
                alias_key = (alias, atom)
                if alias_key not in atom_index:
                    atom_index[alias_key] = atom_index[primary_key]

    print(f"Assigned {NUM_UNIFIED_BACKBONE} unified backbone + {sidechain_count} sidechain atoms")
    print(f"Total: {len(atom_index)} entries (backbone atoms shared across residues)")

    # Compute derived arrays
    atom_dihedral_type, atom_dihedral_refs = compute_atom_dihedral_ownership(
        residues, atom_index
    )
    residue_backbone_atoms = compute_residue_backbone_atoms(residues, atom_index)

    return IndexedData(
        residue_index=residue_index,
        cif_to_residue=cif_to_residue,
        residue_to_cif=residue_to_cif,
        atom_index=atom_index,
        atom_dihedral_type=atom_dihedral_type,
        atom_dihedral_refs=atom_dihedral_refs,
        residue_backbone_atoms=residue_backbone_atoms,
    )


# =============================================================================
# PHASE 4: GENERATE FILES
# =============================================================================


def _generate_files(
    data: LoadedData,
    indices: IndexedData,
    paths: OutputPaths,
) -> None:
    """
    Phase 4: Generate all output files.

    Writes C headers (gperf hash tables, reverse lookups, bond patterns)
    and Python modules (enums, atoms, residues).
    """
    # C code generation
    generate_gperf_files(
        paths.hash_dir,
        indices.atom_index,
        indices.cif_to_residue,
        indices.residue_index,
        data.residues,
        data.elements,
    )
    generate_reverse_header(
        paths.hash_dir,
        indices.atom_index,
        indices.residue_to_cif,
        data.elements,
    )
    generate_bond_patterns_header(paths.internal_dir, data.residues, indices.atom_index)
    generate_canonical_refs_header(
        paths.internal_dir,
        data.residues,
        indices.atom_index,
        indices.atom_dihedral_type,
        indices.atom_dihedral_refs,
        indices.residue_backbone_atoms,
    )

    # Python code generation
    generate_python_molecule(paths.biochem_dir)
    generate_python_elements(paths.biochem_dir, data.elements)
    generate_dihedral_arrays(paths.biochem_dir, data.residues, indices.atom_index)
    generate_python_dihedraltypes(paths.biochem_dir)
    generate_python_atoms(paths.biochem_dir, indices.atom_index, data.residues)
    generate_python_residues(paths.biochem_dir, data.residues)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def generate_all(ccd_path: str) -> tuple[Path, dict[tuple[str, str], int]]:
    """
    Generate all lookup tables and Python enums from CCD.

    This is the main entry point for code generation. It runs four phases:
    1. Load: Read elements from PubChem and residues from CCD
    2. Validate: Check for duplicates and compare against authoritative sources
    3. Index: Build atom/residue indices and compute derived arrays
    4. Generate: Write C headers and Python modules

    Args:
        ccd_path: Path to the Chemical Component Dictionary file.

    Returns:
        Tuple of (hash_dir, atom_index) for compatibility with existing code.
    """
    # Phase 1: Load data
    data = _load_data(ccd_path)

    # Phase 2: Validate
    _validate_data(data)

    # Phase 3: Build indices
    indices = _build_indices(data)

    # Phase 4: Generate files
    project_root = Path(__file__).parent.parent
    paths = OutputPaths.from_project_root(project_root)
    _generate_files(data, indices, paths)

    return paths.hash_dir, indices.atom_index


__all__ = [
    "generate_all",
    "MOLECULE_TYPES",
    "IONS",
    "RESIDUE_WHITELIST",
    "load_elements",
]
