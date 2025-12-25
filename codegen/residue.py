"""
Residue definition dataclass for code generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .names import to_class_name, to_python_name
from .config import (
    DIHEDRAL_TYPE_INDEX, Molecule,
    PURINE_RESIDUES, PYRIMIDINE_RESIDUES,
)


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
    class_name: str = ""  # Python class name

    def __post_init__(self):
        if not self.class_name:
            self.class_name = to_class_name(self.name)


# =============================================================================
# SIDECHAIN DIHEDRAL DEFINITIONS
# =============================================================================
# Chi definitions for each amino acid residue.
# Format: chi_name -> (atom1, atom2, atom3, atom4)
# All atoms are in the same residue (offset 0).

SIDECHAIN_CHI_DEFS: dict[str, dict[str, tuple[str, str, str, str]]] = {
    # CHI1: N-CA-CB-XG
    "ARG": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "NE"), "chi4": ("CG", "CD", "NE", "CZ")},
    "ASN": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "OD1")},
    "ASP": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "OD1")},
    "CYS": {"chi1": ("N", "CA", "CB", "SG")},
    "GLN": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "OE1")},
    "GLU": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "OE1")},
    "HIS": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "ND1")},
    "ILE": {"chi1": ("N", "CA", "CB", "CG1"), "chi2": ("CA", "CB", "CG1", "CD1")},
    "LEU": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "LYS": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "CE"), "chi4": ("CG", "CD", "CE", "NZ")},
    "MET": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "SD"),
            "chi3": ("CB", "CG", "SD", "CE")},
    "PHE": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "PRO": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD")},
    "SER": {"chi1": ("N", "CA", "CB", "OG")},
    "THR": {"chi1": ("N", "CA", "CB", "OG1")},
    "TRP": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "TYR": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},
    "VAL": {"chi1": ("N", "CA", "CB", "CG1")},
    # Modified amino acids
    "MSE": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "SE"),
            "chi3": ("CB", "CG", "SE", "CE")},  # Selenomethionine (like MET)
    "SEP": {"chi1": ("N", "CA", "CB", "OG")},  # Phosphoserine (like SER)
    "TPO": {"chi1": ("N", "CA", "CB", "OG1")},  # Phosphothreonine (like THR)
    "PTR": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD1")},  # Phosphotyrosine (like TYR)
    "CSO": {"chi1": ("N", "CA", "CB", "SG")},  # S-hydroxycysteine (like CYS)
    "OCS": {"chi1": ("N", "CA", "CB", "SG")},  # Cysteinesulfonic acid (like CYS)
    "HYP": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD")},  # Hydroxyproline (like PRO)
    "MLY": {"chi1": ("N", "CA", "CB", "CG"), "chi2": ("CA", "CB", "CG", "CD"),
            "chi3": ("CB", "CG", "CD", "CE"), "chi4": ("CG", "CD", "CE", "NZ")},  # N-dimethyl-lysine (like LYS)
}


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
        # Add sidechain chi dihedrals if this residue has them
        # All sidechain dihedrals are intra-residue (offset 0)
        # First try CCD-parsed torsions, then fall back to hard-coded definitions
        chi_source = None
        if res.torsions:
            # Use CCD-parsed chi angles (preferred)
            chi_source = {k: v for k, v in res.torsions.items() if k.startswith("chi")}
        if not chi_source and res.name in SIDECHAIN_CHI_DEFS:
            # Fall back to hard-coded definitions
            chi_source = SIDECHAIN_CHI_DEFS[res.name]
        if chi_source:
            for chi_name, atoms in chi_source.items():
                # Convert (a1, a2, a3, a4) to ((a1, 0), (a2, 0), (a3, 0), (a4, 0))
                dihedral_defs[chi_name] = tuple((atom, 0) for atom in atoms)
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
        }
        # Add chi based on residue type - only ONE chi pattern per residue!
        # Purines (A, G, etc.) use CHI_PURINE: O4' - C1' - N9 - C4
        # Pyrimidines (C, U, T, etc.) use CHI_PYRIMIDINE: O4' - C1' - N1 - C2
        if res.name in PURINE_RESIDUES:
            dihedral_defs["chi_purine"] = (("O4p", 0), ("C1p", 0), ("N9", 0), ("C4", 0))
        elif res.name in PYRIMIDINE_RESIDUES:
            dihedral_defs["chi_pyrimidine"] = (("O4p", 0), ("C1p", 0), ("N1", 0), ("C2", 0))
        # If neither (unknown nucleotide), skip chi - backbone only
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


def compute_canonical_zmatrix_refs(
    all_residues: list[ResidueDefinition],
    atom_index: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute canonical Z-matrix reference atoms for each atom type.

    Returns:
        atom_canonical_refs: (num_atoms, 6) array with reference info
        atom_has_canonical_refs: (num_atoms,) bool array
    """
    # Find max atom index
    num_atoms = max(atom_index.values()) + 1

    # Initialize empty arrays (stub - not yet implemented)
    atom_canonical_refs = np.zeros((num_atoms, 6), dtype=np.int16)
    atom_has_canonical_refs = np.zeros(num_atoms, dtype=bool)

    return atom_canonical_refs, atom_has_canonical_refs


def compute_residue_backbone_atoms(
    all_residues: list[ResidueDefinition],
    atom_index: dict[tuple[str, str], int],
) -> np.ndarray:
    """
    Compute backbone atom types for each residue type.

    Returns:
        (num_residues, max_backbone_atoms) array with backbone atom indices
    """
    from .config import Molecule

    num_residues = len(all_residues)
    max_backbone = 6  # Conservative max (N, CA, C, O for protein; P, O5', C5', C4', C3', O3' for nucleic)

    backbone_atoms = np.full((num_residues, max_backbone), -1, dtype=np.int16)

    for res_idx, res in enumerate(all_residues):
        if not res.atoms:
            continue

        primary_cif = res.cif_names[0]

        # Define backbone atoms by molecule type
        if res.molecule_type == Molecule.PROTEIN:
            backbone_names = ['N', 'CA', 'C', 'O']
        elif res.molecule_type in (Molecule.RNA, Molecule.DNA, Molecule.HYBRID):
            backbone_names = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]
        else:
            continue

        for i, atom_name in enumerate(backbone_names):
            if i >= max_backbone:
                break
            key = (primary_cif, atom_name)
            if key in atom_index:
                backbone_atoms[res_idx, i] = atom_index[key]

    return backbone_atoms

