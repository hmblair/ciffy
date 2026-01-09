"""
Template polymer generation from sequences.

Generates Polymer templates with correct atom types, elements, and residue
sequences, but without coordinates. Use Polymer.copy(coordinates=...) to add
predicted coordinates.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .polymer import Polymer, Field
from ..biochemistry import Scale, Molecule, Residue, atom_to_element

if TYPE_CHECKING:
    from ..biochemistry.atom import AtomGroup


# =============================================================================
# SEQUENCE CHARACTER MAPPINGS
# =============================================================================

def _build_sequence_maps() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """
    Build sequence character -> residue index mappings from generated data.

    Returns:
        Tuple of (RNA_MAP, DNA_MAP, AMINO_ACID_MAP).
    """
    canonical_rna = {'A', 'C', 'G', 'U', 'I', 'N'}
    canonical_dna = {'DA', 'DC', 'DG', 'DT'}
    canonical_protein = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
        'SER', 'THR', 'TRP', 'TYR', 'VAL',
    }

    rna_map: dict[str, int] = {}
    dna_map: dict[str, int] = {}
    amino_acid_map: dict[str, int] = {}

    for residue in Residue.all():
        if len(residue) == 0:
            continue

        name = residue.name
        idx = residue.value
        abbrev = residue.abbrev

        if name in canonical_rna:
            rna_map[abbrev] = idx
        elif name in canonical_dna:
            dna_map[abbrev] = idx
        elif name in canonical_protein:
            amino_acid_map[abbrev] = idx

    return rna_map, dna_map, amino_acid_map


RNA_MAP, DNA_MAP, AMINO_ACID_MAP = _build_sequence_maps()

_NUCLEOTIDE_CHARS = frozenset('ACGUT')


# =============================================================================
# SEQUENCE PARSING
# =============================================================================

def _detect_molecule_type(sequence: str) -> tuple[dict[str, int], Molecule, str]:
    """
    Detect molecule type from sequence.

    Returns:
        Tuple of (char_to_index_map, Molecule enum, type_name_for_errors).

    Raises:
        ValueError: If sequence is empty.
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    if sequence[0].islower():
        has_u = 'u' in sequence
        has_t = 't' in sequence

        if has_u and has_t:
            raise ValueError(
                "Sequence contains both 'u' (RNA) and 't' (DNA). "
                "Use 'u' for RNA or 't' for DNA, not both."
            )
        if has_t:
            return DNA_MAP, Molecule.DNA, "DNA"
        return RNA_MAP, Molecule.RNA, "RNA"

    if set(sequence).issubset(_NUCLEOTIDE_CHARS):
        warnings.warn(
            f"Sequence '{sequence}' contains only nucleotide characters "
            "but is uppercase. Did you mean lowercase for RNA/DNA? "
            "Treating as protein.",
            UserWarning,
            stacklevel=4,
        )
    return AMINO_ACID_MAP, Molecule.PROTEIN, "protein"


def _parse_sequence(sequence: str) -> tuple[list[int], Molecule]:
    """
    Parse sequence string to residue indices and molecule type.

    Returns:
        Tuple of (residue_indices, molecule_type).
    """
    if not sequence:
        return [], Molecule.RNA  # Default, won't be used for empty

    has_lower = any(c.islower() for c in sequence)
    has_upper = any(c.isupper() for c in sequence)

    if has_lower and has_upper:
        raise ValueError(
            "Mixed case not supported. Use lowercase for nucleic acids "
            "(acgu for RNA, acgt for DNA) or uppercase for protein."
        )

    mapping, mol_type, type_name = _detect_molecule_type(sequence)

    residue_indices = []
    for i, char in enumerate(sequence):
        if char not in mapping:
            valid = ', '.join(sorted(mapping.keys()))
            raise ValueError(
                f"Unknown {type_name} residue '{char}' at position {i}. "
                f"Valid characters: {valid}"
            )
        residue_indices.append(mapping[char])

    return residue_indices, mol_type


def _generate_chain_name(index: int) -> str:
    """Generate chain name for a given index (A-Z, AA-AZ, etc.)."""
    if index < 26:
        return chr(ord('A') + index)
    prefix = chr(ord('A') + (index // 26) - 1)
    suffix = chr(ord('A') + (index % 26))
    return f"{prefix}{suffix}"


# =============================================================================
# PUBLIC API
# =============================================================================

def template(
    sequence: str | Sequence[str],
    backend: str = "numpy",
    id: str = "template",
    atoms: dict[int, Sequence[int]] | None = None,
) -> Polymer:
    """
    Generate a template Polymer from a sequence string or list of sequences.

    Creates a Polymer template with correct atom types, elements, and residue
    sequence, but without coordinates. Use Polymer.copy(coordinates=...) to add
    predicted coordinates.

    Args:
        sequence: Single-letter sequence string, or list of strings for multi-chain.
            - Lowercase with 'u': RNA (acgu)
            - Lowercase with 't': DNA (acgt)
            - Lowercase with only a/c/g: RNA (default)
            - Uppercase: Protein (ACDEFGHIKLMNPQRSTVWY)
            - List creates multiple chains: ['acgu', 'acgt']
            - Empty strings are filtered out; "" returns empty polymer
        backend: Array backend, either "numpy" or "torch".
        id: PDB identifier for the polymer.
        atoms: Optional dict mapping residue type (int) to atom values to include.

    Returns:
        Polymer template (coordinates=None). Use copy(coordinates=...) to add coords.

    Examples:
        >>> template = template("acgu")
        >>> template.size(Scale.RESIDUE)
        4
        >>> template.coordinates  # None - template has no coordinates
        >>> polymer = template.copy(coordinates=predicted_coords)

        >>> protein = template("MGKLF")
        >>> protein.size(Scale.RESIDUE)
        5

        >>> multi = template(["acgu", "acgu"])
        >>> multi.size(Scale.CHAIN)
        2
    """
    sequences = [sequence] if isinstance(sequence, str) else list(sequence)
    sequences = [s for s in sequences if s]

    if not sequences:
        empty = Polymer(pdb_id=id)
        return empty.torch() if backend == "torch" else empty

    # Build polymer using append() for each residue
    polymer = Polymer(pdb_id=id)

    for chain_idx, seq in enumerate(sequences):
        residue_indices, mol_type = _parse_sequence(seq)
        chain_name = _generate_chain_name(chain_idx)
        n_residues = len(residue_indices)

        for i, res_idx in enumerate(residue_indices):
            residue = Residue.from_index(res_idx)
            atom_group = residue.terminal(start=(i == 0), end=(i == n_residues - 1))

            if atoms is not None and res_idx in atoms:
                atom_group = atom_group.subset(set(atoms[res_idx]))

            polymer = polymer.append(atom_group, residue=residue, name=chain_name)

    return polymer.torch() if backend == "torch" else polymer


def from_extract(
    coords: np.ndarray,
    atoms: list[int],
    residue: "AtomGroup",
    backend: str = "numpy",
    id: str = "extracted",
) -> Polymer:
    """
    Convert extracted coordinates back to a Polymer.

    Takes the output of `extract()` and creates a Polymer that can be
    saved as a CIF file. Each row in coords becomes a separate residue.

    Args:
        coords: Coordinate array of shape (n_residues, n_atoms, 3).
        atoms: List of atom type indices.
        residue: The residue AtomGroup that was extracted.
        backend: Array backend, either "numpy" or "torch".
        id: PDB identifier for the polymer.

    Returns:
        Polymer with the extracted coordinates.

    Example:
        >>> from ciffy import load
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.operations import extract
        >>> from ciffy import from_extract
        >>>
        >>> poly = load("structure.cif")
        >>> coords, atoms = extract(poly, Residue.A, align=True, scale=True)
        >>> result = from_extract(coords, atoms, Residue.A)
        >>> result.write("output.cif")
    """
    from ..backend import to_numpy as _to_numpy

    coords_np = np.asarray(_to_numpy(coords))
    n_residues, n_atoms, _ = coords_np.shape

    if len(atoms) != n_atoms:
        raise ValueError(
            f"Mismatch: coords has {n_atoms} atoms per residue, "
            f"but atoms list has {len(atoms)} entries"
        )

    # Build atom -> element mapping
    atom_enum = residue.atoms
    idx_to_member = {m.value: m for m in atom_enum}

    element_indices = []
    for atom_idx in atoms:
        if atom_idx not in idx_to_member:
            raise ValueError(
                f"Atom index {atom_idx} not found in {residue.name} atom enum"
            )
        element_indices.append(atom_to_element(idx_to_member[atom_idx]))

    # Flatten coordinates
    flat_coords = coords_np.reshape(-1, 3).astype(np.float32)

    # Build arrays
    all_atoms = np.tile(atoms, n_residues).astype(np.int64)
    all_elements = np.tile(element_indices, n_residues).astype(np.int64)
    sequence = np.full(n_residues, residue.value, dtype=np.int64)
    atoms_per_res = np.full(n_residues, n_atoms, dtype=np.int64)

    total_atoms = n_residues * n_atoms

    # Create sizes and lengths arrays
    sizes = {
        Scale.RESIDUE: atoms_per_res,
        Scale.CHAIN: np.array([total_atoms], dtype=np.int64),
        Scale.MOLECULE: np.array([total_atoms], dtype=np.int64),
    }
    lengths = np.array([n_residues], dtype=np.int64)

    # Create hierarchy
    from .hierarchy import _Hierarchy
    hierarchy = _Hierarchy.from_sizes_and_lengths(
        sizes=sizes,
        lengths=lengths,
        ref=flat_coords,
    )

    polymer = Polymer(
        hierarchy,
        coordinates=Field(flat_coords, Scale.ATOM),
        atoms=Field(all_atoms, Scale.ATOM),
        elements=Field(all_elements, Scale.ATOM),
        sequence=Field(sequence, Scale.RESIDUE),
        pdb_id=id,
        names=["A"],
    )

    return polymer.torch() if backend == "torch" else polymer
