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


def compute_dihedral_patterns(res: ResidueDefinition) -> dict[int, list[tuple[int, int]]]:
    """
    Compute dihedral angle patterns for a residue.

    Returns a dictionary mapping dihedral type index (integer) to a list
    of 4 (residue_offset, local_atom_index) tuples.

    Args:
        res: Residue definition with atom names and molecule type.

    Returns:
        Dict mapping DIHEDRAL_TYPE_INDEX value -> [(offset1, idx1), ..., (offset4, idx4)]
        where offset is the relative residue offset (0=current, -1=previous, +1=next)
        and idx is the local atom index within that residue.
    """
    # Build mapping: atom_name (Python format) -> local index
    name_to_local: dict[str, int] = {}
    for i, atom_name in enumerate(res.atoms):
        py_name = to_python_name(atom_name)
        name_to_local[py_name] = i

    # Select dihedral definitions based on molecule type
    # Format: (atom_name, residue_offset) where offset is relative residue
    if res.molecule_type == Molecule.PROTEIN:
        dihedral_defs = {
            # phi: C(i-1) - N(i) - CA(i) - C(i)
            "phi": (("C", -1), ("N", 0), ("CA", 0), ("C", 0)),
            # psi: N(i) - CA(i) - C(i) - N(i+1)
            "psi": (("N", 0), ("CA", 0), ("C", 0), ("N", 1)),
            # omega: CA(i) - C(i) - N(i+1) - CA(i+1)
            "omega": (("CA", 0), ("C", 0), ("N", 1), ("CA", 1)),
        }
    elif res.molecule_type in (Molecule.RNA, Molecule.DNA, Molecule.HYBRID):
        dihedral_defs = {
            # alpha: O3'(i-1) - P(i) - O5'(i) - C5'(i)
            "alpha": (("O3p", -1), ("P", 0), ("O5p", 0), ("C5p", 0)),
            # beta: P(i) - O5'(i) - C5'(i) - C4'(i) - all in current residue
            "beta": (("P", 0), ("O5p", 0), ("C5p", 0), ("C4p", 0)),
            # gamma: O5'(i) - C5'(i) - C4'(i) - C3'(i) - all in current residue
            "gamma": (("O5p", 0), ("C5p", 0), ("C4p", 0), ("C3p", 0)),
            # delta: C5'(i) - C4'(i) - C3'(i) - O3'(i) - all in current residue
            "delta": (("C5p", 0), ("C4p", 0), ("C3p", 0), ("O3p", 0)),
            # epsilon: C4'(i) - C3'(i) - O3'(i) - P(i+1)
            "epsilon": (("C4p", 0), ("C3p", 0), ("O3p", 0), ("P", 1)),
            # zeta: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
            "zeta": (("C3p", 0), ("O3p", 0), ("P", 1), ("O5p", 1)),
            # chi for purines: O4' - C1' - N9 - C4 - all in current residue
            "chi_purine": (("O4p", 0), ("C1p", 0), ("N9", 0), ("C4", 0)),
            # chi for pyrimidines: O4' - C1' - N1 - C2 - all in current residue
            "chi_pyrimidine": (("O4p", 0), ("C1p", 0), ("N1", 0), ("C2", 0)),
        }
    else:
        return {}

    patterns = {}
    for dihedral_name, atom_defs in dihedral_defs.items():
        # Build pattern with (offset, local_idx) tuples
        pattern = []
        valid = True

        for atom_name, offset in atom_defs:
            local_idx = name_to_local.get(atom_name, -1)
            if local_idx == -1 and offset == 0:
                # Atom should be in current residue but isn't found
                valid = False
                break
            pattern.append((offset, local_idx))

        if valid and len(pattern) == 4:
            type_idx = DIHEDRAL_TYPE_INDEX[dihedral_name]
            patterns[type_idx] = pattern

    return patterns
