"""
Residue definition dataclass for code generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .names import to_class_name, to_python_name
from .config import DIHEDRAL_TYPE_INDEX, Molecule, BACKBONE_NAMES, BACKBONE_PYTHON_TO_CIF


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
        if res.name in SIDECHAIN_CHI_DEFS:
            for chi_name, atoms in SIDECHAIN_CHI_DEFS[res.name].items():
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


# =============================================================================
# Canonical Z-Matrix References
# =============================================================================

# Canonical reference patterns for each atom.
# Format: (dist_ref, ang_ref, dih_ref, dist_off, ang_off, dih_off)
# where *_ref is atom name (Python format) and *_off is residue offset (-1, 0, +1)
# None means no reference (for first atoms in chain)

# RNA/DNA backbone atoms - common to all nucleotides
_NUCLEOTIDE_BACKBONE_REFS: dict[str, tuple] = {
    # Terminal phosphate (5' end only)
    "OP3": (None, None, None, 0, 0, 0),  # Root atom
    # Phosphate group
    "P": ("O3p", "C3p", "C4p", -1, -1, -1),  # EPSILON (inverted)
    "OP1": ("P", "O3p", "C3p", 0, -1, -1),
    "OP2": ("P", "OP1", "O3p", 0, 0, -1),
    # Backbone chain
    "O5p": ("P", "OP1", "O3p", 0, 0, -1),  # ZETA (inverted)
    "C5p": ("O5p", "P", "O3p", 0, 0, -1),  # ALPHA owner
    "C4p": ("C5p", "O5p", "P", 0, 0, 0),  # BETA owner
    "C3p": ("C4p", "C5p", "O5p", 0, 0, 0),  # GAMMA owner
    "O3p": ("C3p", "C4p", "C5p", 0, 0, 0),  # DELTA owner
    # Sugar ring branches
    "O4p": ("C4p", "C3p", "C5p", 0, 0, 0),
    "C1p": ("O4p", "C4p", "C3p", 0, 0, 0),
    "C2p": ("C1p", "O4p", "C4p", 0, 0, 0),
    "O2p": ("C2p", "C1p", "O4p", 0, 0, 0),  # 2'-OH (RNA only)
}

# Purine base atoms (A, G, I, etc.) - ring-preserving dihedrals
_PURINE_BASE_REFS: dict[str, tuple] = {
    # Glycosidic connection
    "N9": ("C1p", "O4p", "C4p", 0, 0, 0),
    # CHI_PURINE: O4'-C1'-N9-C4
    "C4": ("N9", "C1p", "O4p", 0, 0, 0),  # CHI_PURINE owner
    # 5-membered ring (imidazole)
    "C8": ("N9", "C1p", "O4p", 0, 0, 0),  # Same refs as C4, different position
    "N7": ("C8", "N9", "C4", 0, 0, 0),  # 5-ring: C4-N9-C8-N7
    "C5": ("N7", "C8", "N9", 0, 0, 0),  # 5-ring: N9-C8-N7-C5
    # 6-membered ring (pyrimidine) - fused with 5-ring via C4-C5
    "C6": ("C5", "C4", "N9", 0, 0, 0),  # Fusion: N9-C4-C5-C6
    "N1": ("C6", "C5", "C4", 0, 0, 0),  # 6-ring: C4-C5-C6-N1
    "C2": ("N1", "C6", "C5", 0, 0, 0),  # 6-ring: C5-C6-N1-C2
    "N3": ("C2", "N1", "C6", 0, 0, 0),  # 6-ring: C6-N1-C2-N3
    # Substituents
    "N6": ("C6", "C5", "C4", 0, 0, 0),  # Amino (adenine)
    "O6": ("C6", "C5", "C4", 0, 0, 0),  # Carbonyl (guanine)
    "N2": ("C2", "N1", "C6", 0, 0, 0),  # Amino (guanine)
}

# Pyrimidine base atoms (U, C, T) - ring-preserving dihedrals
_PYRIMIDINE_BASE_REFS: dict[str, tuple] = {
    # Glycosidic connection
    "N1": ("C1p", "O4p", "C4p", 0, 0, 0),
    # CHI_PYRIMIDINE: O4'-C1'-N1-C2
    "C2": ("N1", "C1p", "O4p", 0, 0, 0),  # CHI_PYRIMIDINE owner
    # 6-membered ring
    "N3": ("C2", "N1", "C1p", 0, 0, 0),  # Uses sugar ref for early placement
    "C4": ("N3", "C2", "N1", 0, 0, 0),  # 6-ring: N1-C2-N3-C4
    "C5": ("C4", "N3", "C2", 0, 0, 0),  # 6-ring: C2-N3-C4-C5
    "C6": ("C5", "C4", "N3", 0, 0, 0),  # 6-ring: N3-C4-C5-C6
    # Substituents
    "O2": ("C2", "N1", "C1p", 0, 0, 0),  # Carbonyl (U, C)
    "O4": ("C4", "N3", "C2", 0, 0, 0),  # Carbonyl (U, T)
    "N4": ("C4", "N3", "C2", 0, 0, 0),  # Amino (C)
    "C7": ("C5", "C4", "N3", 0, 0, 0),  # Methyl (T)
}

# Hydrogen atoms for nucleotides - reference the heavy atom they're attached to
_NUCLEOTIDE_HYDROGEN_REFS: dict[str, tuple] = {
    # Phosphate hydrogens
    "HOP3": ("OP3", "P", "O5p", 0, 0, 0),
    "HOP2": ("OP2", "P", "OP1", 0, 0, 0),
    # Sugar hydrogens
    "H5p": ("C5p", "O5p", "P", 0, 0, 0),
    "H5pp": ("C5p", "C4p", "O5p", 0, 0, 0),
    "H4p": ("C4p", "C5p", "C3p", 0, 0, 0),
    "H3p": ("C3p", "C4p", "C2p", 0, 0, 0),
    "HO3p": ("O3p", "C3p", "C4p", 0, 0, 0),
    "H2p": ("C2p", "C1p", "C3p", 0, 0, 0),
    "HO2p": ("O2p", "C2p", "C1p", 0, 0, 0),
    "H1p": ("C1p", "O4p", "C2p", 0, 0, 0),
    # Base hydrogens (purines)
    "H8": ("C8", "N9", "N7", 0, 0, 0),
    "H2": ("C2", "N1", "N3", 0, 0, 0),
    "H61": ("N6", "C6", "C5", 0, 0, 0),
    "H62": ("N6", "C6", "N1", 0, 0, 0),
    "H1": ("N1", "C6", "C2", 0, 0, 0),  # Guanine N1-H
    "H21": ("N2", "C2", "N1", 0, 0, 0),
    "H22": ("N2", "C2", "N3", 0, 0, 0),
    # Base hydrogens (pyrimidines)
    "H3": ("N3", "C2", "C4", 0, 0, 0),
    "H5": ("C5", "C4", "C6", 0, 0, 0),
    "H6": ("C6", "C5", "N1", 0, 0, 0),
    "H41": ("N4", "C4", "N3", 0, 0, 0),
    "H42": ("N4", "C4", "C5", 0, 0, 0),
}

# Protein backbone atoms
_PROTEIN_BACKBONE_REFS: dict[str, tuple] = {
    # PSI (inverted): N(-1)-CA(-1)-C(-1)-N
    "N": ("C", "CA", "N", -1, -1, -1),
    # OMEGA (inverted): CA(-1)-C(-1)-N-CA
    "CA": ("N", "C", "CA", 0, -1, -1),
    # PHI: C(-1)-N-CA-C
    "C": ("CA", "N", "C", 0, 0, -1),
    # Backbone carbonyl
    "O": ("C", "CA", "N", 0, 0, 0),
    # Terminal atoms
    "OXT": ("C", "CA", "O", 0, 0, 0),
    # Backbone H
    "H": ("N", "C", "CA", 0, -1, -1),
    "HA": ("CA", "N", "C", 0, 0, 0),
}


def _get_canonical_refs_for_residue(
    res: ResidueDefinition,
) -> dict[str, tuple]:
    """
    Get canonical Z-matrix references for all atoms in a residue.

    Returns dict mapping atom name (Python format) -> (dist_ref, ang_ref, dih_ref,
    dist_off, ang_off, dih_off) where refs are atom names and offs are residue offsets.
    """
    refs = {}

    if res.molecule_type in (Molecule.RNA, Molecule.DNA, Molecule.HYBRID):
        # Start with backbone
        refs.update(_NUCLEOTIDE_BACKBONE_REFS)

        # Determine if purine or pyrimidine based on atoms present
        atom_set = {to_python_name(a) for a in res.atoms}
        if "N9" in atom_set:
            # Purine (A, G, I, etc.)
            refs.update(_PURINE_BASE_REFS)
        elif "N1" in atom_set and "C1p" in atom_set:
            # Pyrimidine (U, C, T)
            refs.update(_PYRIMIDINE_BASE_REFS)

        # Add hydrogen refs
        refs.update(_NUCLEOTIDE_HYDROGEN_REFS)

    elif res.molecule_type == Molecule.PROTEIN:
        refs.update(_PROTEIN_BACKBONE_REFS)
        # Sidechain atoms would need to be added per-residue
        # For now, leave them to fall back to BFS

    return refs


def _resolve_ref_value(
    ref_name: str | None,
    ref_offset: int,
    name_to_global: dict[str, int],
) -> int:
    """
    Resolve a reference atom name to either an atom type or backbone name ID.

    Args:
        ref_name: Python-format atom name (e.g., "C3p", "CA")
        ref_offset: Residue offset (0 = same residue, -1 = previous, +1 = next)
        name_to_global: Mapping of atom names to global atom indices for current residue

    Returns:
        For offset == 0: Global atom type (index)
        For offset != 0: Backbone name ID
        Returns -1 if reference cannot be resolved
    """
    if ref_name is None:
        return -1

    if ref_offset == 0:
        # Intra-residue: return atom TYPE
        return name_to_global.get(ref_name, -1)
    else:
        # Inter-residue: return backbone NAME ID
        # Convert Python name to CIF name, then look up backbone ID
        cif_name = BACKBONE_PYTHON_TO_CIF.get(ref_name)
        if cif_name is None:
            return -1
        return BACKBONE_NAMES.get(cif_name, -1)


def compute_canonical_zmatrix_refs(
    all_residues: list[ResidueDefinition],
    atom_index: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build global arrays for canonical Z-matrix construction.

    For each atom type, defines the canonical (dist_ref, ang_ref, dih_ref)
    with residue offsets. This allows deterministic Z-matrix construction
    without BFS.

    Key insight: For inter-residue refs (offset != 0), we store backbone NAME IDs
    instead of atom types. This allows the C code to resolve the actual atom type
    at runtime by looking up RESIDUE_BACKBONE_ATOMS[target_res_type][backbone_name_id].

    Args:
        all_residues: List of all residue definitions.
        atom_index: Dict mapping (cif_name, atom_name) -> global atom index.

    Returns:
        ATOM_CANONICAL_REFS: (num_atoms, 6) int16 array
            For each atom type: [dist_ref, ang_ref, dih_ref, dist_off, ang_off, dih_off]
            where:
            - If offset == 0: ref is global atom TYPE (index)
            - If offset != 0: ref is backbone NAME ID
            - -1 means no reference
        ATOM_HAS_CANONICAL_REFS: (num_atoms,) bool array
            True if this atom type has canonical refs defined.
    """
    num_atoms = max(atom_index.values()) + 1

    # Initialize arrays
    # [dist_ref, ang_ref, dih_ref, dist_off, ang_off, dih_off]
    atom_canonical_refs = np.full((num_atoms, 6), -1, dtype=np.int16)
    atom_has_canonical_refs = np.zeros(num_atoms, dtype=bool)

    for res in all_residues:
        if not res.atoms:
            continue

        primary_cif = res.cif_names[0]

        # Get canonical refs for this residue type
        refs_dict = _get_canonical_refs_for_residue(res)

        # Build name -> global index mapping for this residue
        name_to_global = {}
        for atom_name in res.atoms:
            py_name = to_python_name(atom_name)
            key = (primary_cif, atom_name)
            if key in atom_index:
                name_to_global[py_name] = atom_index[key]

        # Populate arrays
        for atom_name in res.atoms:
            py_name = to_python_name(atom_name)
            if py_name not in refs_dict:
                continue

            key = (primary_cif, atom_name)
            if key not in atom_index:
                continue

            global_idx = atom_index[key]
            dist_ref, ang_ref, dih_ref, dist_off, ang_off, dih_off = refs_dict[py_name]

            # Resolve refs - atom types for intra-residue, backbone IDs for inter-residue
            dist_val = _resolve_ref_value(dist_ref, dist_off, name_to_global)
            ang_val = _resolve_ref_value(ang_ref, ang_off, name_to_global)
            dih_val = _resolve_ref_value(dih_ref, dih_off, name_to_global)

            atom_canonical_refs[global_idx] = [
                dist_val, ang_val, dih_val,
                dist_off, ang_off, dih_off
            ]
            atom_has_canonical_refs[global_idx] = True

    return atom_canonical_refs, atom_has_canonical_refs


def compute_residue_backbone_atoms(
    all_residues: list[ResidueDefinition],
    atom_index: dict[tuple[str, str], int],
) -> np.ndarray:
    """
    Build lookup table: residue type -> backbone name ID -> atom type.

    This enables inter-residue reference resolution in C. When we need to find
    atom "C" in a previous residue of type GLY, we look up:
    RESIDUE_BACKBONE_ATOMS[GLY_idx][BACKBONE_C] -> atom type for GLY.C

    Args:
        all_residues: List of all residue definitions.
        atom_index: Dict mapping (cif_name, atom_name) -> global atom index.

    Returns:
        (num_residues, num_backbone_names) int16 array
        Value is atom type, or -1 if that backbone atom doesn't exist in residue.
    """
    from .config import NUM_BACKBONE_NAMES

    num_residues = len(all_residues)
    residue_backbone_atoms = np.full(
        (num_residues, NUM_BACKBONE_NAMES), -1, dtype=np.int16
    )

    for res_idx, res in enumerate(all_residues):
        if not res.atoms:
            continue

        primary_cif = res.cif_names[0]

        # Map backbone atoms
        for cif_name, backbone_id in BACKBONE_NAMES.items():
            key = (primary_cif, cif_name)
            if key in atom_index:
                residue_backbone_atoms[res_idx, backbone_id] = atom_index[key]

    return residue_backbone_atoms
