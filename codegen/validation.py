"""
Validation functions for code generation.

Validates that generated data is consistent with authoritative sources
and that all internal references are valid. These checks catch:

1. **Backbone dihedral mismatches**: Compares DIHEDRAL_TYPES in config.py
   against MonomerLibrary (the authoritative source for mmCIF geometry).
   Discrepancies indicate our definitions may be outdated.

2. **Broken bond references**: Bonds that reference atoms not present in
   the residue's atom list, indicating CCD parsing issues.

3. **Broken torsion references**: Torsion angle definitions referencing
   missing atoms, which would cause runtime failures.

Validation issues are reported as warnings but don't halt code generation,
allowing incremental fixes. Run `python -m codegen.cli` and check output
for any warnings.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from .config import DIHEDRAL_TYPES, BACKBONE_PYTHON_TO_CIF
from .monlib import load_backbone_dihedrals, convert_to_residue_offset
from .names import sanitize_identifier

if TYPE_CHECKING:
    from .residue import ResidueDefinition


def validate_backbone_dihedrals() -> list[str]:
    """
    Validate that DIHEDRAL_TYPES matches MonomerLibrary definitions.

    Compares the hardcoded backbone dihedral definitions in config.py
    against the authoritative MonomerLibrary source.

    Returns:
        List of warning messages for any discrepancies.
    """
    issues: list[str] = []

    try:
        monlib = load_backbone_dihedrals()
    except Exception as e:
        issues.append(f"Could not load MonomerLibrary: {e}")
        return issues

    # Map dihedral name -> (our atoms, monlib atoms)
    comparisons = {
        "PHI": (None, monlib.phi),
        "PSI": (None, monlib.psi),
        "OMEGA": (None, monlib.omega),
        "ALPHA": (None, monlib.alpha),
        "BETA": (None, monlib.beta),
        "GAMMA": (None, monlib.gamma),
        "DELTA": (None, monlib.delta),
        "EPSILON": (None, monlib.epsilon),
        "ZETA": (None, monlib.zeta),
    }

    # Get our definitions
    for ddef in DIHEDRAL_TYPES:
        if ddef.name in comparisons:
            comparisons[ddef.name] = (ddef.atoms, comparisons[ddef.name][1])

    # Compare each dihedral
    for name, (our_atoms, monlib_atoms) in comparisons.items():
        if our_atoms is None:
            continue  # Not a backbone dihedral we define

        if monlib_atoms is None:
            issues.append(f"{name}: Not found in MonomerLibrary")
            continue

        # Convert monlib atoms to our format for comparison
        # MonomerLibrary: ((atom_name, comp_id), ...) where comp_id is 1 or 2
        # Ours: (python_atom_name, ...) like ("C", "N", "CA", "C")
        monlib_converted = convert_to_residue_offset(monlib_atoms)
        monlib_atom_names = tuple(
            sanitize_identifier(atom_name) for atom_name, _ in monlib_converted
        )

        # Compare atom names (ignoring offsets for now, just checking names match)
        our_atom_names = tuple(our_atoms)

        if our_atom_names != monlib_atom_names:
            issues.append(
                f"{name}: Atom names differ - "
                f"config.py={our_atom_names}, MonomerLibrary={monlib_atom_names}"
            )

    if not issues:
        print("Backbone dihedral validation passed (matches MonomerLibrary)")

    return issues


def validate_bond_atoms(
    all_residues: list["ResidueDefinition"],
) -> list[str]:
    """
    Validate that all bond references point to valid atoms.

    Args:
        all_residues: List of residue definitions.

    Returns:
        List of warning messages for invalid bonds.
    """
    issues: list[str] = []

    for res in all_residues:
        atom_set = set(res.atoms)
        for atom1, atom2 in res.bonds:
            if atom1 not in atom_set:
                issues.append(f"{res.name}: Bond references missing atom '{atom1}'")
            if atom2 not in atom_set:
                issues.append(f"{res.name}: Bond references missing atom '{atom2}'")

    if not issues:
        print(f"Bond validation passed ({len(all_residues)} residues checked)")

    return issues


def validate_torsion_atoms(
    all_residues: list["ResidueDefinition"],
) -> list[str]:
    """
    Validate that all torsion angle references point to valid atoms.

    Args:
        all_residues: List of residue definitions.

    Returns:
        List of warning messages for invalid torsions.
    """
    issues: list[str] = []

    for res in all_residues:
        if not res.torsions:
            continue

        atom_set = set(res.atoms)
        for torsion_name, atoms in res.torsions.items():
            for atom_name in atoms:
                if atom_name not in atom_set:
                    issues.append(
                        f"{res.name}: Torsion '{torsion_name}' references missing atom '{atom_name}'"
                    )

    if not issues:
        print("Torsion validation passed")

    return issues


def run_all_validations(
    all_residues: list["ResidueDefinition"],
) -> None:
    """
    Run all validation checks and report issues.

    Args:
        all_residues: List of residue definitions.

    Raises:
        No exceptions - issues are reported as warnings.
    """
    all_issues: list[str] = []

    # Validate backbone dihedrals against MonomerLibrary
    all_issues.extend(validate_backbone_dihedrals())

    # Validate bond atom references
    all_issues.extend(validate_bond_atoms(all_residues))

    # Validate torsion atom references
    all_issues.extend(validate_torsion_atoms(all_residues))

    # Report all issues
    if all_issues:
        print(f"\nValidation found {len(all_issues)} issue(s):")
        for issue in all_issues:
            warnings.warn(issue, stacklevel=2)
