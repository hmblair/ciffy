"""
Template polymer generation from sequences.

Generates Polymer objects with correct atom types, elements, and residue
sequences using ideal CCD coordinates.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .polymer import Polymer
from .builder import ChainBuilder, expand_residue
from ..biochemistry import Scale, Molecule, Residue, atom_to_element
from ..utils import atoms_to_col_map

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
    """
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
# CHAIN BUILDING
# =============================================================================

def _build_chain(
    sequence: str,
    atom_filter: dict[int, Sequence[int]] | None = None,
) -> dict:
    """
    Build arrays for a single chain from sequence.

    Args:
        sequence: Single-letter sequence string.
        atom_filter: Optional dict mapping residue type to allowed atom values.

    Returns:
        Dict with coordinates, atoms, elements, sequence, sizes.
    """
    residue_indices, mol_type = _parse_sequence(sequence)

    if not residue_indices:
        return {
            'coordinates': np.empty((0, 3), dtype=np.float32),
            'atoms': np.empty(0, dtype=np.int64),
            'elements': np.empty(0, dtype=np.int64),
            'sequence': np.empty(0, dtype=np.int64),
            'atoms_per_residue': [],
            'residue_indices': [],
        }

    # Use ChainBuilder for positioning and terminal filtering
    builder = ChainBuilder(mol_type, filter_terminal=True)

    for res_idx in residue_indices:
        residue = Residue.from_index(res_idx)
        builder.append(residue)

    # Build arrays
    result = builder.build()

    # Apply custom atom filter if provided
    if atom_filter is not None:
        result = _apply_atom_filter(result, builder, atom_filter)

    # Add residue_indices for multi-chain aggregation
    result['residue_indices'] = residue_indices
    result['atoms_per_residue'] = result['sizes'][Scale.RESIDUE].tolist()

    return result


def _apply_atom_filter(
    arrays: dict,
    builder: ChainBuilder,
    atom_filter: dict[int, Sequence[int]],
) -> dict:
    """Apply custom atom filter to built arrays."""
    # Rebuild from residue data with filtering
    all_coords = []
    all_atoms = []
    all_elements = []
    atoms_per_residue = []

    coord_offset = 0
    for res_data in builder._residues:
        res_idx = res_data.residue.value
        n_atoms = res_data.n_atoms

        if res_idx in atom_filter:
            allowed = set(atom_filter[res_idx])
            # Filter this residue's atoms
            filtered_atoms = []
            filtered_elements = []
            filtered_indices = []

            for i, (atom, elem) in enumerate(zip(res_data.atoms, res_data.elements)):
                if atom in allowed:
                    filtered_atoms.append(atom)
                    filtered_elements.append(elem)
                    filtered_indices.append(i)

            if filtered_indices:
                coords_slice = arrays['coordinates'][coord_offset:coord_offset + n_atoms]
                all_coords.append(coords_slice[filtered_indices])
                all_atoms.extend(filtered_atoms)
                all_elements.extend(filtered_elements)
                atoms_per_residue.append(len(filtered_atoms))
        else:
            # Keep all atoms for this residue
            all_coords.append(arrays['coordinates'][coord_offset:coord_offset + n_atoms])
            all_atoms.extend(res_data.atoms)
            all_elements.extend(res_data.elements)
            atoms_per_residue.append(n_atoms)

        coord_offset += n_atoms

    if all_coords:
        coords = np.concatenate(all_coords, axis=0)
    else:
        coords = np.empty((0, 3), dtype=np.float32)

    n_atoms = len(all_atoms)

    return {
        'coordinates': coords,
        'atoms': np.array(all_atoms, dtype=np.int64),
        'elements': np.array(all_elements, dtype=np.int64),
        'sequence': arrays['sequence'],
        'sizes': {
            Scale.RESIDUE: np.array(atoms_per_residue, dtype=np.int64),
            Scale.CHAIN: np.array([n_atoms], dtype=np.int64),
            Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
        },
    }


# =============================================================================
# PUBLIC API
# =============================================================================

def from_sequence(
    sequence: str | Sequence[str],
    backend: str = "numpy",
    id: str = "template",
    atoms: dict[int, Sequence[int]] | None = None,
) -> Polymer:
    """
    Generate a template Polymer from a sequence string or list of sequences.

    Creates a Polymer with correct atom types, elements, and residue sequence
    using ideal CCD coordinates. Useful for generative modeling where coordinates
    are generated separately.

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
        Polymer with ideal CCD coordinates.

    Examples:
        >>> rna = from_sequence("acgu")
        >>> rna.size(Scale.RESIDUE)
        4

        >>> protein = from_sequence("MGKLF")
        >>> protein.size(Scale.RESIDUE)
        5

        >>> multi = from_sequence(["acgu", "acgu"])
        >>> multi.size(Scale.CHAIN)
        2
    """
    sequences = [sequence] if isinstance(sequence, str) else list(sequence)
    sequences = [s for s in sequences if s]

    if not sequences:
        return Polymer.create_empty(pdb_id=id, backend=backend)

    # Build each chain
    all_coords = []
    all_atoms = []
    all_elements = []
    all_atoms_per_res = []
    all_residue_indices = []
    atoms_per_chain = []
    residues_per_chain = []
    chain_names = []

    for chain_idx, seq in enumerate(sequences):
        chain_data = _build_chain(seq, atom_filter=atoms)

        all_coords.append(chain_data['coordinates'])
        all_atoms.append(chain_data['atoms'])
        all_elements.append(chain_data['elements'])
        all_atoms_per_res.extend(chain_data['atoms_per_residue'])
        all_residue_indices.extend(chain_data['residue_indices'])
        atoms_per_chain.append(len(chain_data['atoms']))
        residues_per_chain.append(len(chain_data['residue_indices']))
        chain_names.append(_generate_chain_name(chain_idx))

    # Concatenate arrays
    coords = np.concatenate(all_coords, axis=0) if all_coords else np.empty((0, 3), dtype=np.float32)
    atoms_arr = np.concatenate(all_atoms) if all_atoms else np.empty(0, dtype=np.int64)
    elements_arr = np.concatenate(all_elements) if all_elements else np.empty(0, dtype=np.int64)

    n_atoms = len(atoms_arr)

    polymer = Polymer(
        coordinates=coords,
        atoms=atoms_arr,
        elements=elements_arr,
        sequence=np.array(all_residue_indices, dtype=np.int64),
        sizes={
            Scale.RESIDUE: np.array(all_atoms_per_res, dtype=np.int64),
            Scale.CHAIN: np.array(atoms_per_chain, dtype=np.int64),
            Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
        },
        pdb_id=id,
        names=chain_names,
        strands=chain_names,
        lengths=np.array(residues_per_chain, dtype=np.int64),
        polymer_count=n_atoms,
    )

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

    polymer = Polymer(
        coordinates=flat_coords,
        atoms=all_atoms,
        elements=all_elements,
        sequence=sequence,
        sizes={
            Scale.RESIDUE: atoms_per_res,
            Scale.CHAIN: np.array([total_atoms], dtype=np.int64),
            Scale.MOLECULE: np.array([total_atoms], dtype=np.int64),
        },
        pdb_id=id,
        names=["A"],
        strands=["A"],
        lengths=np.array([n_residues], dtype=np.int64),
        polymer_count=total_atoms,
    )

    return polymer.torch() if backend == "torch" else polymer
