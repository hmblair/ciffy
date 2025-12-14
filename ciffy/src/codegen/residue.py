"""
Residue definition dataclass for code generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .names import to_class_name, to_python_name
from .config import DIHEDRAL_TYPE_INDEX, Molecule


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
            self.class_name = to_class_name(self.name)


def compute_dihedral_patterns(res: ResidueDefinition) -> dict[int, list[int]]:
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
        py_name = to_python_name(atom_name)
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
