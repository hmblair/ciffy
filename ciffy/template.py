"""
Template polymer generation from sequences.

Generates Polymer objects with correct atom types, elements, and residue
sequences but zero coordinates - useful for generative modeling.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Sequence

import numpy as np

from .polymer import Polymer
from .types import Scale, Molecule
from .biochemistry._generated_residues import Residue, RESIDUE_ABBREV, RESIDUE_MOLECULE_TYPE
from .biochemistry.linking import LINKING_BY_TYPE
from .biochemistry._generated_atoms import (
    # RNA nucleotides
    A, C, G, U,
    # DNA nucleotides
    Da, Dc, Dg, Dt,
    # Amino acids
    Ala, Arg, Asn, Asp, Cys,
    Gln, Glu, Gly, His, Ile,
    Leu, Lys, Met, Phe, Pro,
    Ser, Thr, Trp, Tyr, Val,
)


# =============================================================================
# ELEMENT LOOKUP
# =============================================================================

# First character of atom name -> atomic number
_ELEMENT_MAP: dict[str, int] = {
    'H': 1,   # Hydrogen
    'C': 6,   # Carbon
    'N': 7,   # Nitrogen
    'O': 8,   # Oxygen
    'P': 15,  # Phosphorus
    'S': 16,  # Sulfur
}


# =============================================================================
# TERMINAL ATOM DEFINITIONS
# =============================================================================

# Nucleic acid terminal atoms (atom names as they appear in enum, with p for ')
# 5'-terminal only: OP3 and its hydrogen
_NA_5_TERMINAL_ATOMS = frozenset({'OP3', 'HOP3'})
# 3'-terminal only: hydroxyl hydrogen on O3'
_NA_3_TERMINAL_ATOMS = frozenset({'HO3p'})

# Protein terminal atoms
# N-terminal only: extra ammonium hydrogens (NH3+ vs NH in peptide bond)
_PROTEIN_N_TERMINAL_ATOMS = frozenset({'H2', 'H3'})
# C-terminal only: second carboxyl oxygen (COO- vs C=O in peptide bond)
_PROTEIN_C_TERMINAL_ATOMS = frozenset({'OXT'})


# =============================================================================
# RESIDUE -> ATOM ENUM MAPPING
# =============================================================================

# Maps Residue enum members to their atom enum classes
# Only includes residues that can be generated from sequences
RESIDUE_ATOMS: dict[Residue, type] = {
    # RNA
    Residue.A: A,
    Residue.C: C,
    Residue.G: G,
    Residue.U: U,
    # DNA
    Residue.DA: Da,
    Residue.DC: Dc,
    Residue.DG: Dg,
    Residue.DT: Dt,
    # Protein
    Residue.ALA: Ala,
    Residue.ARG: Arg,
    Residue.ASN: Asn,
    Residue.ASP: Asp,
    Residue.CYS: Cys,
    Residue.GLN: Gln,
    Residue.GLU: Glu,
    Residue.GLY: Gly,
    Residue.HIS: His,
    Residue.ILE: Ile,
    Residue.LEU: Leu,
    Residue.LYS: Lys,
    Residue.MET: Met,
    Residue.PHE: Phe,
    Residue.PRO: Pro,
    Residue.SER: Ser,
    Residue.THR: Thr,
    Residue.TRP: Trp,
    Residue.TYR: Tyr,
    Residue.VAL: Val,
}


# =============================================================================
# SEQUENCE CHARACTER MAPPINGS (built from generated data)
# =============================================================================

def _build_sequence_maps() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """
    Build sequence character -> residue index mappings from generated data.

    Returns:
        Tuple of (RNA_MAP, DNA_MAP, AMINO_ACID_MAP).
    """
    rna_map: dict[str, int] = {}
    dna_map: dict[str, int] = {}
    amino_acid_map: dict[str, int] = {}

    for residue in RESIDUE_ATOMS:
        idx = residue.value
        abbrev = RESIDUE_ABBREV[idx]
        mol_type = RESIDUE_MOLECULE_TYPE[idx]

        if mol_type == Molecule.RNA:
            rna_map[abbrev] = idx
        elif mol_type == Molecule.DNA:
            dna_map[abbrev] = idx
        elif mol_type == Molecule.PROTEIN:
            amino_acid_map[abbrev] = idx

    return rna_map, dna_map, amino_acid_map


RNA_MAP, DNA_MAP, AMINO_ACID_MAP = _build_sequence_maps()

# Characters that look like nucleotides (for ambiguity warning)
_NUCLEOTIDE_CHARS = frozenset('ACGUT')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _atom_name_to_element(name: str) -> int:
    """Convert atom name to atomic number based on first character."""
    return _ELEMENT_MAP.get(name[0].upper(), 0)


def _generate_chain_name(index: int) -> str:
    """
    Generate chain name for a given index.

    Args:
        index: 0-based chain index.

    Returns:
        Chain name: A-Z for 0-25, AA-AZ for 26-51, BA-BZ for 52-77, etc.
    """
    if index < 26:
        return chr(ord('A') + index)
    prefix = chr(ord('A') + (index // 26) - 1)
    suffix = chr(ord('A') + (index % 26))
    return f"{prefix}{suffix}"


@lru_cache(maxsize=32)
def _expand_residue_full(residue_idx: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], np.ndarray]:
    """
    Get atom indices, element indices, atom names, and ideal coordinates for a residue type.

    Results are cached since the same residue type always expands identically.

    Args:
        residue_idx: Residue index (from Residue enum value).

    Returns:
        Tuple of (atom_indices, element_indices, atom_names, ideal_coords).
        First three are tuples for hashability, ideal_coords is (N, 3) array.

    Raises:
        ValueError: If residue_idx has no atom definitions.
    """
    try:
        residue = Residue(residue_idx)
    except ValueError:
        raise ValueError(f"Invalid residue index: {residue_idx}")

    if residue not in RESIDUE_ATOMS:
        raise ValueError(f"No atom definitions for residue {residue.name}")

    atom_enum = RESIDUE_ATOMS[residue]
    atom_indices = []
    element_indices = []
    atom_names = []

    for member in atom_enum:
        atom_indices.append(member.value)
        atom_names.append(member.name)
        atom_name_display = member.name.replace('p', "'")  # C5p -> C5'
        element_indices.append(_atom_name_to_element(atom_name_display))

    # Get ideal coordinates from the atom enum class
    ideal_coords = atom_enum.ideal.copy()

    return tuple(atom_indices), tuple(element_indices), tuple(atom_names), ideal_coords


def _filter_atoms_by_position(
    atom_indices: tuple[int, ...],
    element_indices: tuple[int, ...],
    atom_names: tuple[str, ...],
    ideal_coords: np.ndarray,
    is_nucleic_acid: bool,
    is_first: bool,
    is_last: bool,
) -> tuple[list[int], list[int], list[np.ndarray]]:
    """
    Filter atoms based on residue position in chain.

    Terminal atoms are only included for terminal residues:
    - 5'/N-terminal atoms: only for first residue
    - 3'/C-terminal atoms: only for last residue
    - Internal residues: exclude all terminal atoms

    Args:
        atom_indices: Full atom index tuple from _expand_residue_full.
        element_indices: Full element index tuple from _expand_residue_full.
        atom_names: Atom names (enum member names) from _expand_residue_full.
        ideal_coords: Ideal coordinates array (N, 3) from _expand_residue_full.
        is_nucleic_acid: True for RNA/DNA, False for protein.
        is_first: True if this is the first residue in the chain.
        is_last: True if this is the last residue in the chain.

    Returns:
        Tuple of (filtered_atom_indices, filtered_element_indices, filtered_coords) as lists.
    """
    if is_nucleic_acid:
        start_terminal = _NA_5_TERMINAL_ATOMS
        end_terminal = _NA_3_TERMINAL_ATOMS
    else:
        start_terminal = _PROTEIN_N_TERMINAL_ATOMS
        end_terminal = _PROTEIN_C_TERMINAL_ATOMS

    filtered_atoms = []
    filtered_elements = []
    filtered_coords = []

    for i, (atom_idx, elem_idx, name) in enumerate(zip(atom_indices, element_indices, atom_names)):
        # Check if this is a terminal-only atom
        is_start_terminal = name in start_terminal
        is_end_terminal = name in end_terminal

        # Include atom if:
        # - It's not a terminal atom, OR
        # - It's a start-terminal atom AND we're at the start, OR
        # - It's an end-terminal atom AND we're at the end
        include = True
        if is_start_terminal and not is_first:
            include = False
        if is_end_terminal and not is_last:
            include = False

        if include:
            filtered_atoms.append(atom_idx)
            filtered_elements.append(elem_idx)
            filtered_coords.append(ideal_coords[i])

    return filtered_atoms, filtered_elements, filtered_coords


def _detect_molecule_type(sequence: str) -> tuple[dict[str, int], str]:
    """
    Detect molecule type from sequence and return appropriate mapping.

    Args:
        sequence: Single-letter sequence (already validated as single-case).

    Returns:
        Tuple of (character_to_index_map, molecule_type_name).

    Raises:
        ValueError: If sequence contains both 'u' and 't'.
    """
    if sequence[0].islower():
        # Nucleic acid
        has_u = 'u' in sequence
        has_t = 't' in sequence

        if has_u and has_t:
            raise ValueError(
                "Sequence contains both 'u' (RNA) and 't' (DNA). "
                "Use 'u' for RNA or 't' for DNA, not both."
            )
        if has_t:
            return DNA_MAP, "DNA"
        return RNA_MAP, "RNA"

    # Protein - warn if looks like nucleotides
    if set(sequence).issubset(_NUCLEOTIDE_CHARS):
        warnings.warn(
            f"Sequence '{sequence}' contains only nucleotide characters "
            "but is uppercase. Did you mean lowercase for RNA/DNA? "
            "Treating as protein.",
            UserWarning,
            stacklevel=4,
        )
    return AMINO_ACID_MAP, "protein"


def _parse_sequence(sequence: str) -> list[int]:
    """
    Parse sequence string to residue indices.

    Args:
        sequence: Single-letter sequence.
            - Lowercase with 'u': RNA (acgu)
            - Lowercase with 't': DNA (acgt)
            - Lowercase with only a/c/g: RNA (default)
            - Uppercase: Protein (ACDEFGHIKLMNPQRSTVWY)

    Returns:
        List of residue indices.

    Raises:
        ValueError: If sequence is empty, mixed case, or contains invalid chars.
    """
    if not sequence:
        raise ValueError("Empty sequence")

    has_lower = any(c.islower() for c in sequence)
    has_upper = any(c.isupper() for c in sequence)

    if has_lower and has_upper:
        raise ValueError(
            "Mixed case not supported. Use lowercase for nucleic acids "
            "(acgu for RNA, acgt for DNA) or uppercase for protein."
        )

    mapping, mol_type = _detect_molecule_type(sequence)

    residue_indices = []
    for i, char in enumerate(sequence):
        if char not in mapping:
            valid = ', '.join(sorted(mapping.keys()))
            raise ValueError(
                f"Unknown {mol_type} residue '{char}' at position {i}. "
                f"Valid characters: {valid}"
            )
        residue_indices.append(mapping[char])

    return residue_indices


def _find_atom_index(atom_names: tuple[str, ...], target: str) -> int | None:
    """Find index of atom by name in atom_names tuple."""
    try:
        return atom_names.index(target)
    except ValueError:
        return None


def _process_chain(sequence: str) -> tuple[list[int], list[int], list[int], list[int], list[np.ndarray]]:
    """
    Process a single chain sequence into atom/element/residue/coordinate data.

    Handles terminal atoms correctly:
    - 5'/N-terminal atoms only on first residue
    - 3'/C-terminal atoms only on last residue

    Positions residues with correct bond lengths between them.

    Args:
        sequence: Single-letter sequence for one chain.

    Returns:
        Tuple of (atom_indices, element_indices, atoms_per_residue, residue_indices, coords).
    """
    residue_indices = _parse_sequence(sequence)
    n_residues = len(residue_indices)

    # Determine if nucleic acid or protein based on first residue
    first_mol_type = RESIDUE_MOLECULE_TYPE.get(residue_indices[0])
    is_nucleic_acid = first_mol_type in (Molecule.RNA, Molecule.DNA)

    # Get linking definition for this molecule type
    link_def = LINKING_BY_TYPE.get(first_mol_type)

    # Direction for chain extension (along positive X axis for linear)
    extension_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    all_atoms: list[int] = []
    all_elements: list[int] = []
    all_coords: list[np.ndarray] = []
    atoms_per_res: list[int] = []

    # Track position of previous residue's "end" linking atom
    prev_link_pos: np.ndarray | None = None

    for i, res_idx in enumerate(residue_indices):
        is_first = (i == 0)
        is_last = (i == n_residues - 1)

        # Get full atom expansion (cached)
        atom_indices, element_indices, atom_names, ideal_coords = _expand_residue_full(res_idx)

        # Make a mutable copy of coordinates for positioning
        positioned_coords = ideal_coords.copy()

        # Position this residue relative to previous one
        if link_def is not None and prev_link_pos is not None:
            # Find current residue's "start" linking atom (P for NA, N for protein)
            curr_link_idx = _find_atom_index(atom_names, link_def.next_atom)
            if curr_link_idx is not None:
                # Target position: prev_link_pos + bond_length in extension direction
                target_pos = prev_link_pos + extension_dir * link_def.bond_length
                # Compute offset to move curr_link_atom to target
                offset = target_pos - positioned_coords[curr_link_idx]
                positioned_coords = positioned_coords + offset

        # Update prev_link_pos for next residue
        if link_def is not None:
            prev_link_idx = _find_atom_index(atom_names, link_def.prev_atom)
            if prev_link_idx is not None:
                prev_link_pos = positioned_coords[prev_link_idx].copy()

        # Filter based on position
        filtered_atoms, filtered_elements, filtered_coords = _filter_atoms_by_position(
            atom_indices, element_indices, atom_names, positioned_coords,
            is_nucleic_acid, is_first, is_last
        )

        all_atoms.extend(filtered_atoms)
        all_elements.extend(filtered_elements)
        all_coords.extend(filtered_coords)
        atoms_per_res.append(len(filtered_atoms))

    return all_atoms, all_elements, atoms_per_res, residue_indices, all_coords


# =============================================================================
# PUBLIC API
# =============================================================================

def from_sequence(
    sequence: str | Sequence[str],
    backend: str = "numpy",
    id: str = "template",
) -> Polymer:
    """
    Generate a template Polymer from a sequence string or list of sequences.

    Creates a Polymer with correct atom types, elements, and residue sequence
    but zero coordinates. Useful for generative modeling where coordinates
    are generated separately.

    Args:
        sequence: Single-letter sequence string, or list of strings for multi-chain.
            - Lowercase with 'u': RNA (acgu)
            - Lowercase with 't': DNA (acgt)
            - Lowercase with only a/c/g: RNA (default)
            - Uppercase: Protein (ACDEFGHIKLMNPQRSTVWY)
            - List creates multiple chains: ['acgu', 'acgt']
        backend: Array backend, either "numpy" or "torch".
        id: PDB identifier for the polymer.

    Returns:
        Polymer with zero coordinates but correct:
        - atoms: Global atom type indices
        - elements: Atomic numbers (H=1, C=6, N=7, O=8, P=15, S=16)
        - sequence: Residue type indices (matching Residue enum)
        - sizes: Atoms per residue/chain/molecule

    Raises:
        ValueError: If sequence is empty, mixed case, contains both 'u' and 't',
            or contains invalid characters.

    Examples:
        >>> rna = from_sequence("acgu")
        >>> rna.size()  # Total atoms
        148
        >>> rna.size(Scale.RESIDUE)  # Number of residues
        4

        >>> dna = from_sequence("acgt")
        >>> dna.size(Scale.RESIDUE)
        4

        >>> protein = from_sequence("MGKLF")
        >>> protein.size(Scale.RESIDUE)
        5

        >>> multi = from_sequence(["acgu", "acgu"])  # Two RNA chains
        >>> multi.size(Scale.CHAIN)
        2
    """
    # Normalize input
    sequences = [sequence] if isinstance(sequence, str) else list(sequence)
    if not sequences:
        raise ValueError("Empty sequence list")

    # Accumulate data across all chains
    all_atoms: list[int] = []
    all_elements: list[int] = []
    all_coords: list[np.ndarray] = []
    all_atoms_per_res: list[int] = []
    all_residue_indices: list[int] = []
    atoms_per_chain: list[int] = []
    residues_per_chain: list[int] = []
    chain_names: list[str] = []

    for chain_idx, seq in enumerate(sequences):
        atoms, elements, atoms_per_res, residues, coords = _process_chain(seq)

        all_atoms.extend(atoms)
        all_elements.extend(elements)
        all_coords.extend(coords)
        all_atoms_per_res.extend(atoms_per_res)
        all_residue_indices.extend(residues)
        atoms_per_chain.append(len(atoms))
        residues_per_chain.append(len(residues))
        chain_names.append(_generate_chain_name(chain_idx))

    # Build arrays
    n_atoms = len(all_atoms)

    polymer = Polymer(
        coordinates=np.array(all_coords, dtype=np.float32),
        atoms=np.array(all_atoms, dtype=np.int64),
        elements=np.array(all_elements, dtype=np.int64),
        sequence=np.array(all_residue_indices, dtype=np.int64),
        sizes={
            Scale.RESIDUE: np.array(all_atoms_per_res, dtype=np.int64),
            Scale.CHAIN: np.array(atoms_per_chain, dtype=np.int64),
            Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
        },
        id=id,
        names=chain_names,
        strands=chain_names,
        lengths=np.array(residues_per_chain, dtype=np.int64),
        polymer_count=n_atoms,
    )

    return polymer.torch() if backend == "torch" else polymer
