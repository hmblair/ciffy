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
# MAPPINGS
# =============================================================================

# Single-letter code to Residue index (lowercase = RNA)
NUCLEOTIDE_MAP: dict[str, int] = {
    'a': 0,  # Adenosine
    'c': 1,  # Cytosine
    'g': 2,  # Guanosine
    'u': 3,  # Uridine
    # 't': 4 - DNA not supported in v1 (Thymidine atoms not defined)
}

# Single-letter code to Residue index (uppercase = protein)
AMINO_ACID_MAP: dict[str, int] = {
    'A': 5,   # Alanine
    'C': 6,   # Cysteine
    'D': 7,   # Aspartic acid
    'E': 8,   # Glutamic acid
    'F': 9,   # Phenylalanine
    'G': 10,  # Glycine
    'H': 11,  # Histidine
    'I': 12,  # Isoleucine
    'K': 13,  # Lysine
    'L': 14,  # Leucine
    'M': 15,  # Methionine
    'N': 16,  # Asparagine
    'P': 17,  # Proline
    'Q': 18,  # Glutamine
    'R': 19,  # Arginine
    'S': 20,  # Serine
    'T': 21,  # Threonine
    'V': 22,  # Valine
    'W': 23,  # Tryptophan
    'Y': 24,  # Tyrosine
}

# Residue index to atom enum class
RESIDUE_ATOMS: dict[int, type] = {
    # RNA nucleotides
    0: Adenosine,
    1: Cytosine,
    2: Guanosine,
    3: Uridine,
    # Amino acids
    5: Alanine,
    6: Cysteine,
    7: AsparticAcid,
    8: GlutamicAcid,
    9: Phenylalanine,
    10: Glycine,
    11: Histidine,
    12: Isoleucine,
    13: Lysine,
    14: Leucine,
    15: Methionine,
    16: Asparagine,
    17: Proline,
    18: Glutamine,
    19: Arginine,
    20: Serine,
    21: Threonine,
    22: Valine,
    23: Tryptophan,
    24: Tyrosine,
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
        sequence: Single-letter sequence (lowercase=RNA, uppercase=protein).

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
            "Mixed case not supported. Use lowercase for RNA (acgu) "
            "or uppercase for protein (ACDEFGHIKLMNPQRSTVWY)."
        )

    # Determine molecule type and mapping
    if has_lower:
        # RNA sequence
        mapping = NUCLEOTIDE_MAP
        mol_type = "RNA"
    else:
        # Protein sequence - but check for nucleotide-like sequences
        if set(sequence.upper()).issubset(NUCLEOTIDE_CHARS):
            warnings.warn(
                f"Sequence '{sequence}' contains only nucleotide characters "
                "but is uppercase. Did you mean lowercase for RNA? "
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
            if char.lower() == 't':
                raise ValueError(
                    f"DNA (thymine 't') not supported in v1. "
                    f"Only RNA (acgu) and protein sequences are supported."
                )
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
    sequence: str,
    backend: str = "numpy",
    id: str = "template",
) -> Polymer:
    """
    Generate a template Polymer from a sequence string.

    Creates a Polymer with correct atom types, elements, and residue sequence
    but zero coordinates. Useful for generative modeling where coordinates
    are generated separately.

    Args:
        sequence: Single-letter sequence string.
            - Lowercase for RNA: 'acgu'
            - Uppercase for protein: 'ACDEFGHIKLMNPQRSTVWY'
            - DNA ('t') not supported in v1.
        backend: Array backend, either "numpy" or "torch".
        id: PDB identifier for the polymer.

    Returns:
        Polymer with zero coordinates but correct:
        - atoms: Global atom type indices
        - elements: Atomic numbers (H=1, C=6, N=7, O=8, P=15, S=16)
        - sequence: Residue type indices
        - sizes: Atoms per residue/chain/molecule

    Raises:
        ValueError: If sequence is empty, mixed case, or contains invalid chars.

    Examples:
        >>> rna = from_sequence("acgu")
        >>> rna.size()  # Total atoms
        148
        >>> rna.size(Scale.RESIDUE)  # Number of residues
        4

        >>> protein = from_sequence("MGKLF")
        >>> protein.size(Scale.RESIDUE)
        5
    """
    # 1. Parse sequence to residue indices
    residue_indices = _parse_sequence(sequence)

    # 2. Expand atoms for each residue
    all_atoms: list[int] = []
    all_elements: list[int] = []
    atoms_per_res: list[int] = []

    for res_idx in residue_indices:
        atom_indices, element_indices = _expand_residue_atoms(res_idx)
        all_atoms.extend(atom_indices)
        all_elements.extend(element_indices)
        atoms_per_res.append(len(atom_indices))

    # 3. Build arrays
    n_atoms = len(all_atoms)
    n_residues = len(residue_indices)

    coordinates = np.zeros((n_atoms, 3), dtype=np.float32)
    atoms = np.array(all_atoms, dtype=np.int64)
    elements = np.array(all_elements, dtype=np.int64)
    sequence_arr = np.array(residue_indices, dtype=np.int64)

    sizes = {
        Scale.RESIDUE: np.array(atoms_per_res, dtype=np.int64),
        Scale.CHAIN: np.array([n_atoms], dtype=np.int64),
        Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
    }

    # 4. Create Polymer
    polymer = Polymer(
        coordinates=coordinates,
        atoms=atoms,
        elements=elements,
        sequence=sequence_arr,
        sizes=sizes,
        id=id,
        names=["A"],  # Single chain named "A"
        strands=["A"],
        lengths=np.array([n_residues], dtype=np.int64),
        polymer_count=n_atoms,  # All atoms are polymer atoms
    )

    # 5. Convert backend if needed
    if backend == "torch":
        return polymer.torch()
    return polymer
