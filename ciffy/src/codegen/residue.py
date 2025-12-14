"""
Residue definition dataclass for code generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    #
    # For inter-residue dihedrals where the 4th atom (owner) has offset +1,
    # we add an inverted pattern from the "receiving" residue's perspective:
    # - Original: [A(o1), B(o2), C(o3), D(+1)] means the NEXT residue's D atom owns it
    # - Inverted: [A(o1-1), B(o2-1), C(o3-1), D(0)] - from next residue's view
    #
    # This way, when building the Z-matrix for the "next" residue, we can capture
    # the dihedral angle that spans from the previous residue.
    if res.molecule_type == Molecule.PROTEIN:
        dihedral_defs = {
            # phi: C(i-1) - N(i) - CA(i) - C(i) - owner C is at offset 0
            "phi": (("C", -1), ("N", 0), ("CA", 0), ("C", 0)),
            # psi: N(i-1) - CA(i-1) - C(i-1) - N(i) - inverted from next residue's view
            # Original was: N(i) - CA(i) - C(i) - N(i+1) with owner at +1
            "psi": (("N", -1), ("CA", -1), ("C", -1), ("N", 0)),
            # omega: CA(i-1) - C(i-1) - N(i) - CA(i) - inverted from next residue's view
            # Original was: CA(i) - C(i) - N(i+1) - CA(i+1) with owner at +1
            "omega": (("CA", -1), ("C", -1), ("N", 0), ("CA", 0)),
        }
    elif res.molecule_type in (Molecule.RNA, Molecule.DNA, Molecule.HYBRID):
        dihedral_defs = {
            # alpha: O3'(i-1) - P(i) - O5'(i) - C5'(i) - owner C5' is at offset 0
            "alpha": (("O3p", -1), ("P", 0), ("O5p", 0), ("C5p", 0)),
            # beta: P(i) - O5'(i) - C5'(i) - C4'(i) - owner C4' is at offset 0
            "beta": (("P", 0), ("O5p", 0), ("C5p", 0), ("C4p", 0)),
            # gamma: O5'(i) - C5'(i) - C4'(i) - C3'(i) - owner C3' is at offset 0
            "gamma": (("O5p", 0), ("C5p", 0), ("C4p", 0), ("C3p", 0)),
            # delta: C5'(i) - C4'(i) - C3'(i) - O3'(i) - owner O3' is at offset 0
            "delta": (("C5p", 0), ("C4p", 0), ("C3p", 0), ("O3p", 0)),
            # epsilon: C4'(i-1) - C3'(i-1) - O3'(i-1) - P(i) - inverted
            # Original was: C4'(i) - C3'(i) - O3'(i) - P(i+1) with owner at +1
            "epsilon": (("C4p", -1), ("C3p", -1), ("O3p", -1), ("P", 0)),
            # zeta: C3'(i-1) - O3'(i-1) - P(i) - O5'(i) - inverted
            # Original was: C3'(i) - O3'(i) - P(i+1) - O5'(i+1) with owner at +1
            "zeta": (("C3p", -1), ("O3p", -1), ("P", 0), ("O5p", 0)),
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


def compute_atom_dihedral_ownership(
    all_residues: list[ResidueDefinition],
    atom_index: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build global arrays mapping atom enum values to dihedral ownership.

    For each residue's dihedral patterns, the 4th atom (D) in pattern [A, B, C, D]
    "owns" that dihedral. This function builds arrays indexed by atom enum value
    that specify:
    1. Which dihedral type (if any) each atom owns
    2. The reference atoms [A, B, C] for Z-matrix construction

    Args:
        all_residues: List of all residue definitions.
        atom_index: Dict mapping (cif_name, atom_name) -> global atom index.

    Returns:
        ATOM_DIHEDRAL_TYPE: (num_atoms,) int8 array
            Maps atom enum value -> dihedral type index, or -1 if not a dihedral owner.
        ATOM_DIHEDRAL_REFS: (num_atoms, 3, 2) int8 array
            Maps atom enum value -> [[dih_offset, dih_idx], [ang_offset, ang_idx], [dist_offset, dist_idx]]
            where offset is residue offset (-1/0/+1) and idx is local atom index.
            Only meaningful where ATOM_DIHEDRAL_TYPE >= 0.
    """
    # Find max atom index
    num_atoms = max(atom_index.values()) + 1

    # Initialize arrays
    atom_dihedral_type = np.full(num_atoms, -1, dtype=np.int8)
    atom_dihedral_refs = np.zeros((num_atoms, 3, 2), dtype=np.int8)

    for res in all_residues:
        if not res.atoms:
            continue

        primary_cif = res.cif_names[0]

        # Get dihedral patterns for this residue
        dihedral_patterns = compute_dihedral_patterns(res)

        for dtype_idx, pattern in dihedral_patterns.items():
            # pattern is [(offset1, idx1), (offset2, idx2), (offset3, idx3), (offset4, idx4)]
            # The 4th atom (pattern[3]) owns the dihedral
            owner_offset, owner_local_idx = pattern[3]

            # We can only assign ownership if the owner is in the current residue
            if owner_offset != 0:
                continue

            # Get global atom index for the owner
            owner_atom_name = res.atoms[owner_local_idx]
            owner_key = (primary_cif, owner_atom_name)
            if owner_key not in atom_index:
                continue

            global_atom_idx = atom_index[owner_key]

            # Store dihedral type
            atom_dihedral_type[global_atom_idx] = dtype_idx

            # Store references [A, B, C] (first 3 atoms of pattern)
            # For Z-matrix: dih_ref=A, ang_ref=B, dist_ref=C
            for i in range(3):
                offset, local_idx = pattern[i]
                atom_dihedral_refs[global_atom_idx, i, 0] = offset
                atom_dihedral_refs[global_atom_idx, i, 1] = local_idx

    return atom_dihedral_type, atom_dihedral_refs
