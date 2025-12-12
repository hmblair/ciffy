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
from .biochemistry.residues import Residue, RESIDUE_ABBREV, RESIDUE_MOLECULE_TYPE
from .biochemistry._generated_atoms import (
    # RNA nucleotides
    Adenosine,
    Cytosine,
    Guanosine,
    Uridine,
    # DNA nucleotides
    Deoxyadenosine,
    Deoxycytidine,
    Deoxyguanosine,
    Thymidine,
    # Amino acids
    Alanine,
    Arginine,
    Asparagine,
    AsparticAcid,
    Cysteine,
    Glutamine,
    GlutamicAcid,
    Glycine,
    Histidine,
    Isoleucine,
    Leucine,
    Lysine,
    Methionine,
    Phenylalanine,
    Proline,
    Serine,
    Threonine,
    Tryptophan,
    Tyrosine,
    Valine,
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
# RESIDUE -> ATOM ENUM MAPPING
# =============================================================================

# Maps Residue enum members to their atom enum classes
# Only includes residues that can be generated from sequences
RESIDUE_ATOMS: dict[Residue, type] = {
    # RNA (indices 0-3)
    Residue.ADE: Adenosine,
    Residue.CYT: Cytosine,
    Residue.GUA: Guanosine,
    Residue.URA: Uridine,
    # DNA (indices 4-7)
    Residue.DA: Deoxyadenosine,
    Residue.DC: Deoxycytidine,
    Residue.DG: Deoxyguanosine,
    Residue.DT: Thymidine,
    # Protein (indices 8-27)
    Residue.ALA: Alanine,
    Residue.CYS: Cysteine,
    Residue.ASP: AsparticAcid,
    Residue.GLU: GlutamicAcid,
    Residue.PHE: Phenylalanine,
    Residue.GLY: Glycine,
    Residue.HIS: Histidine,
    Residue.ILE: Isoleucine,
    Residue.LYS: Lysine,
    Residue.LEU: Leucine,
    Residue.MET: Methionine,
    Residue.ASN: Asparagine,
    Residue.PRO: Proline,
    Residue.GLN: Glutamine,
    Residue.ARG: Arginine,
    Residue.SER: Serine,
    Residue.THR: Threonine,
    Residue.VAL: Valine,
    Residue.TRP: Tryptophan,
    Residue.TYR: Tyrosine,
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
def _expand_residue(residue_idx: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Get atom indices and element indices for a residue type.

    Results are cached since the same residue type always expands identically.

    Args:
        residue_idx: Residue index (from Residue enum value).

    Returns:
        Tuple of (atom_indices, element_indices) as tuples for hashability.

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

    for member in atom_enum:
        atom_indices.append(member.value)
        atom_name = member.name.replace('p', "'")  # C5p -> C5'
        element_indices.append(_atom_name_to_element(atom_name))

    return tuple(atom_indices), tuple(element_indices)


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


def _process_chain(sequence: str) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Process a single chain sequence into atom/element/residue data.

    Args:
        sequence: Single-letter sequence for one chain.

    Returns:
        Tuple of (atom_indices, element_indices, atoms_per_residue, residue_indices).
    """
    residue_indices = _parse_sequence(sequence)

    all_atoms: list[int] = []
    all_elements: list[int] = []
    atoms_per_res: list[int] = []

    for res_idx in residue_indices:
        atom_indices, element_indices = _expand_residue(res_idx)
        all_atoms.extend(atom_indices)
        all_elements.extend(element_indices)
        atoms_per_res.append(len(atom_indices))

    return all_atoms, all_elements, atoms_per_res, residue_indices


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
    all_atoms_per_res: list[int] = []
    all_residue_indices: list[int] = []
    atoms_per_chain: list[int] = []
    residues_per_chain: list[int] = []
    chain_names: list[str] = []

    for chain_idx, seq in enumerate(sequences):
        atoms, elements, atoms_per_res, residues = _process_chain(seq)

        all_atoms.extend(atoms)
        all_elements.extend(elements)
        all_atoms_per_res.extend(atoms_per_res)
        all_residue_indices.extend(residues)
        atoms_per_chain.append(len(atoms))
        residues_per_chain.append(len(residues))
        chain_names.append(_generate_chain_name(chain_idx))

    # Build arrays
    n_atoms = len(all_atoms)

    polymer = Polymer(
        coordinates=np.zeros((n_atoms, 3), dtype=np.float32),
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
