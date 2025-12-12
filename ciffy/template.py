"""
Template polymer generation from sequences.

Generates Polymer objects with correct atom types, elements, and residue
sequences but zero coordinates - useful for generative modeling.
"""

from __future__ import annotations
import warnings
import numpy as np

from .polymer import Polymer
from .types import Scale
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
    Glycine,
    Alanine,
    Valine,
    Leucine,
    Isoleucine,
    Proline,
    Phenylalanine,
    Tryptophan,
    Methionine,
    Cysteine,
    Serine,
    Threonine,
    Asparagine,
    Glutamine,
    AsparticAcid,
    GlutamicAcid,
    Lysine,
    Arginine,
    Histidine,
    Tyrosine,
)


# =============================================================================
# MAPPINGS (indices match Residue enum in _generated_residues.py)
# =============================================================================

# Single-letter code to Residue index (lowercase = RNA)
RNA_MAP: dict[str, int] = {
    'a': 0,  # Adenosine (ADE)
    'c': 1,  # Cytosine (CYT)
    'g': 2,  # Guanosine (GUA)
    'u': 3,  # Uridine (URA)
}

# Single-letter code to Residue index (lowercase = DNA)
DNA_MAP: dict[str, int] = {
    'a': 4,  # Deoxyadenosine (DA)
    'c': 5,  # Deoxycytidine (DC)
    'g': 6,  # Deoxyguanosine (DG)
    't': 7,  # Thymidine (DT)
}

# Single-letter code to Residue index (uppercase = protein)
AMINO_ACID_MAP: dict[str, int] = {
    'A': 8,   # Alanine (ALA)
    'C': 9,   # Cysteine (CYS)
    'D': 10,  # Aspartic acid (ASP)
    'E': 11,  # Glutamic acid (GLU)
    'F': 12,  # Phenylalanine (PHE)
    'G': 13,  # Glycine (GLY)
    'H': 14,  # Histidine (HIS)
    'I': 15,  # Isoleucine (ILE)
    'K': 16,  # Lysine (LYS)
    'L': 17,  # Leucine (LEU)
    'M': 18,  # Methionine (MET)
    'N': 19,  # Asparagine (ASN)
    'P': 20,  # Proline (PRO)
    'Q': 21,  # Glutamine (GLN)
    'R': 22,  # Arginine (ARG)
    'S': 23,  # Serine (SER)
    'T': 24,  # Threonine (THR)
    'V': 25,  # Valine (VAL)
    'W': 26,  # Tryptophan (TRP)
    'Y': 27,  # Tyrosine (TYR)
}

# Residue index to atom enum class
RESIDUE_ATOMS: dict[int, type] = {
    # RNA nucleotides
    0: Adenosine,
    1: Cytosine,
    2: Guanosine,
    3: Uridine,
    # DNA nucleotides
    4: Deoxyadenosine,
    5: Deoxycytidine,
    6: Deoxyguanosine,
    7: Thymidine,
    # Amino acids
    8: Alanine,
    9: Cysteine,
    10: AsparticAcid,
    11: GlutamicAcid,
    12: Phenylalanine,
    13: Glycine,
    14: Histidine,
    15: Isoleucine,
    16: Lysine,
    17: Leucine,
    18: Methionine,
    19: Asparagine,
    20: Proline,
    21: Glutamine,
    22: Arginine,
    23: Serine,
    24: Threonine,
    25: Valine,
    26: Tryptophan,
    27: Tyrosine,
}

# Characters that look like nucleotides (for warning)
NUCLEOTIDE_CHARS = set('ACGUT')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _atom_name_to_element(name: str) -> int:
    """
    Convert atom name to element index (atomic number).

    Args:
        name: Atom name (e.g., "CA", "N", "O5'", "P").

    Returns:
        Element index (atomic number): H=1, C=6, N=7, O=8, P=15, S=16.
    """
    first = name[0].upper()
    if first == 'C':
        return 6   # Carbon
    if first == 'N':
        return 7   # Nitrogen
    if first == 'O':
        return 8   # Oxygen
    if first == 'P':
        return 15  # Phosphorus
    if first == 'S':
        return 16  # Sulfur
    if first == 'H':
        return 1   # Hydrogen
    return 0  # Unknown


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

    # Check for mixed case
    has_lower = any(c.islower() for c in sequence)
    has_upper = any(c.isupper() for c in sequence)

    if has_lower and has_upper:
        raise ValueError(
            "Mixed case not supported. Use lowercase for nucleic acids "
            "(acgu for RNA, acgt for DNA) or uppercase for protein."
        )

    # Determine molecule type and mapping
    if has_lower:
        # Nucleic acid - determine RNA vs DNA
        has_u = 'u' in sequence
        has_t = 't' in sequence

        if has_u and has_t:
            raise ValueError(
                "Sequence contains both 'u' (RNA) and 't' (DNA). "
                "Use 'u' for RNA or 't' for DNA, not both."
            )
        elif has_t:
            mapping = DNA_MAP
            mol_type = "DNA"
        else:
            # Has 'u' or only a/c/g - treat as RNA
            mapping = RNA_MAP
            mol_type = "RNA"
    else:
        # Protein sequence - but check for nucleotide-like sequences
        if set(sequence.upper()).issubset(NUCLEOTIDE_CHARS):
            warnings.warn(
                f"Sequence '{sequence}' contains only nucleotide characters "
                "but is uppercase. Did you mean lowercase for RNA/DNA? "
                "Treating as protein.",
                UserWarning,
                stacklevel=3
            )
        mapping = AMINO_ACID_MAP
        mol_type = "protein"

    # Parse sequence
    residue_indices = []
    for i, char in enumerate(sequence):
        if char not in mapping:
            raise ValueError(
                f"Unknown {mol_type} residue '{char}' at position {i}. "
                f"Valid characters: {', '.join(sorted(mapping.keys()))}"
            )
        residue_indices.append(mapping[char])

    return residue_indices


def _expand_residue_atoms(residue_idx: int) -> tuple[list[int], list[int]]:
    """
    Get atom indices and element indices for a residue.

    Args:
        residue_idx: Residue index from Residue enum.

    Returns:
        Tuple of (atom_indices, element_indices).
    """
    if residue_idx not in RESIDUE_ATOMS:
        raise ValueError(f"No atom definitions for residue index {residue_idx}")

    atom_enum = RESIDUE_ATOMS[residue_idx]

    atom_indices = []
    element_indices = []

    for member in atom_enum:
        atom_indices.append(member.value)
        # Get atom name and derive element
        atom_name = member.name.replace('p', "'")  # Convert C5p back to C5'
        element_indices.append(_atom_name_to_element(atom_name))

    return atom_indices, element_indices


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def from_sequence(
    sequence: str | list[str],
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
    # Normalize to list of sequences
    if isinstance(sequence, str):
        sequences = [sequence]
    else:
        sequences = list(sequence)

    if len(sequences) == 0:
        raise ValueError("Empty sequence list")

    # Process each chain
    all_atoms: list[int] = []
    all_elements: list[int] = []
    atoms_per_res: list[int] = []
    atoms_per_chain: list[int] = []
    residues_per_chain: list[int] = []
    all_residue_indices: list[int] = []
    chain_names: list[str] = []

    for chain_idx, seq in enumerate(sequences):
        # Parse sequence to residue indices
        residue_indices = _parse_sequence(seq)

        chain_atoms = 0
        for res_idx in residue_indices:
            atom_indices, element_indices = _expand_residue_atoms(res_idx)
            all_atoms.extend(atom_indices)
            all_elements.extend(element_indices)
            atoms_per_res.append(len(atom_indices))
            chain_atoms += len(atom_indices)

        all_residue_indices.extend(residue_indices)
        atoms_per_chain.append(chain_atoms)
        residues_per_chain.append(len(residue_indices))

        # Generate chain name (A, B, C, ... Z, AA, AB, ...)
        if chain_idx < 26:
            chain_names.append(chr(ord('A') + chain_idx))
        else:
            chain_names.append(f"A{chr(ord('A') + chain_idx - 26)}")

    # Build arrays
    n_atoms = len(all_atoms)

    coordinates = np.zeros((n_atoms, 3), dtype=np.float32)
    atoms = np.array(all_atoms, dtype=np.int64)
    elements = np.array(all_elements, dtype=np.int64)
    sequence_arr = np.array(all_residue_indices, dtype=np.int64)

    sizes = {
        Scale.RESIDUE: np.array(atoms_per_res, dtype=np.int64),
        Scale.CHAIN: np.array(atoms_per_chain, dtype=np.int64),
        Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
    }

    # Create Polymer
    polymer = Polymer(
        coordinates=coordinates,
        atoms=atoms,
        elements=elements,
        sequence=sequence_arr,
        sizes=sizes,
        id=id,
        names=chain_names,
        strands=chain_names,
        lengths=np.array(residues_per_chain, dtype=np.int64),
        polymer_count=n_atoms,  # All atoms are polymer atoms
    )

    # Convert backend if needed
    if backend == "torch":
        return polymer.torch()
    return polymer
