"""
Residue definition for code generation.

This module provides the ResidueDefinition dataclass which holds parsed
residue data from the CCD (Chemical Component Dictionary).
"""

from __future__ import annotations

from dataclasses import dataclass

from .names import to_class_name


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
    torsions: dict[str, tuple[str, str, str, str]] | None = None  # Torsion name -> (a1,a2,a3,a4)
    parent_comp_id: str = ""  # Parent residue ID for modified residues (e.g., "A" for 1MA)
    class_name: str = ""  # Python class name

    def __post_init__(self):
        if not self.class_name:
            self.class_name = to_class_name(self.name)
