"""
Template polymer generation from sequences.

Generates Polymer objects with correct atom types, elements, and residue
sequences but zero coordinates - useful for generative modeling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .polymer import Polymer
from .types import Scale, Molecule
from .biochemistry._generated_residues import Residue
from .biochemistry.linking import LINKING_BY_TYPE, LinkingDefinition, NUCLEIC_ACID_LINK, PEPTIDE_LINK


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ResidueExpansion:
    """Expanded atom data for a single residue type."""
    atom_indices: tuple[int, ...]
    element_indices: tuple[int, ...]
    atom_names: tuple[str, ...]
    ideal_coords: np.ndarray  # (N, 3)


@dataclass
class ChainData:
    """Processed data for a single chain."""
    atom_indices: list[int]
    element_indices: list[int]
    atoms_per_residue: list[int]
    residue_indices: list[int]
    coords: list[np.ndarray]


@dataclass(frozen=True)
class MoleculeTypeConfig:
    """
    Configuration for a molecule type in sequence generation.

    Bundles all molecule-specific parameters needed for template generation:
    linking info, terminal atoms, and sequence character mapping.
    """
    linking: LinkingDefinition | None
    start_terminal_atoms: frozenset[str]  # 5'/N-terminal only
    end_terminal_atoms: frozenset[str]    # 3'/C-terminal only
    sequence_map: dict[str, int]          # char -> residue index


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
# SEQUENCE CHARACTER MAPPINGS (built from generated data)
# =============================================================================

def _build_sequence_maps() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """
    Build sequence character -> residue index mappings from generated data.

    Only includes canonical residues for sequence generation.

    Returns:
        Tuple of (RNA_MAP, DNA_MAP, AMINO_ACID_MAP).
    """
    # Canonical residue names for sequence generation
    canonical_rna = {'A', 'C', 'G', 'U', 'I', 'N'}  # I=inosine, N=unknown
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

    for residue in Residue:
        # Only include canonical residues with atom definitions
        if residue.atoms is None:
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

# Characters that look like nucleotides (for ambiguity warning)
_NUCLEOTIDE_CHARS = frozenset('ACGUT')


# =============================================================================
# MOLECULE TYPE CONFIGURATIONS
# =============================================================================

_MOLECULE_CONFIGS: dict[Molecule, MoleculeTypeConfig] = {
    Molecule.RNA: MoleculeTypeConfig(
        linking=NUCLEIC_ACID_LINK,
        start_terminal_atoms=frozenset({'OP3', 'HOP3'}),
        end_terminal_atoms=frozenset({'HO3p'}),
        sequence_map=RNA_MAP,
    ),
    Molecule.DNA: MoleculeTypeConfig(
        linking=NUCLEIC_ACID_LINK,
        start_terminal_atoms=frozenset({'OP3', 'HOP3'}),
        end_terminal_atoms=frozenset({'HO3p'}),
        sequence_map=DNA_MAP,
    ),
    Molecule.PROTEIN: MoleculeTypeConfig(
        linking=PEPTIDE_LINK,
        start_terminal_atoms=frozenset({'H2', 'H3'}),
        end_terminal_atoms=frozenset({'OXT', 'HXT'}),
        sequence_map=AMINO_ACID_MAP,
    ),
}


def _get_molecule_config(residue_idx: int) -> MoleculeTypeConfig:
    """Get config for a residue's molecule type."""
    mol_type = Residue(residue_idx).molecule_type
    if mol_type not in _MOLECULE_CONFIGS:
        raise ValueError(f"Unsupported molecule type: {mol_type.name}")
    return _MOLECULE_CONFIGS[mol_type]


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
def _expand_residue_cached(residue_idx: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], np.ndarray]:
    """
    Internal cached expansion - returns tuple for hashability.

    The numpy array is cached; callers MUST copy if mutating.
    The _expand_residue() wrapper handles copying for safe mutation.
    """
    try:
        residue = Residue(residue_idx)
    except ValueError:
        raise ValueError(f"Invalid residue index: {residue_idx}")

    if residue.atoms is None:
        raise ValueError(f"No atom definitions for residue {residue.name}")

    atom_indices = []
    element_indices = []
    atom_names = []

    for member in residue.atoms:
        atom_indices.append(member.value)
        atom_names.append(member.name)
        atom_name_display = member.name.replace('p', "'")  # C5p -> C5'
        element_indices.append(_atom_name_to_element(atom_name_display))

    # Get ideal coordinates from the residue (delegates to atom enum)
    # No copy here - _expand_residue() copies when needed
    ideal_coords = residue.ideal

    return tuple(atom_indices), tuple(element_indices), tuple(atom_names), ideal_coords


def _expand_residue(residue_idx: int) -> ResidueExpansion:
    """
    Get atom data for a residue type.

    Results are cached since the same residue type always expands identically.
    Coordinates are copied to allow safe mutation by callers.

    Args:
        residue_idx: Residue index (from Residue enum value).

    Returns:
        ResidueExpansion with atom indices, elements, names, and ideal coordinates.

    Raises:
        ValueError: If residue_idx has no atom definitions.
    """
    atoms, elements, names, coords = _expand_residue_cached(residue_idx)
    return ResidueExpansion(atoms, elements, names, coords.copy())


def _filter_atoms_by_position(
    expansion: ResidueExpansion,
    coords: np.ndarray,
    config: MoleculeTypeConfig,
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
        expansion: Residue expansion with atom data.
        coords: Positioned coordinates (may differ from expansion.ideal_coords).
        config: Molecule type configuration with terminal atom definitions.
        is_first: True if this is the first residue in the chain.
        is_last: True if this is the last residue in the chain.

    Returns:
        Tuple of (filtered_atom_indices, filtered_element_indices, filtered_coords) as lists.
    """
    filtered_atoms = []
    filtered_elements = []
    filtered_coords = []

    for i, (atom_idx, elem_idx, name) in enumerate(zip(
        expansion.atom_indices, expansion.element_indices, expansion.atom_names
    )):
        # Check if this is a terminal-only atom
        is_start_terminal = name in config.start_terminal_atoms
        is_end_terminal = name in config.end_terminal_atoms

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
            filtered_coords.append(coords[i])

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
        List of residue indices (empty list for empty sequence).

    Raises:
        ValueError: If sequence is mixed case or contains invalid chars.
    """
    if not sequence:
        return []

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


# Direction for chain extension (along positive X axis for linear templates)
_EXTENSION_DIR = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _position_residue(
    expansion: ResidueExpansion,
    residue_idx: int,
    prev_link_pos: np.ndarray | None,
    config: MoleculeTypeConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Position residue coordinates relative to previous residue.

    Args:
        expansion: Residue expansion with ideal coordinates.
        residue_idx: Residue index (for error messages).
        prev_link_pos: Position of previous residue's linking atom, or None for first.
        config: Molecule type configuration with linking definition.

    Returns:
        Tuple of (positioned_coords, new_link_pos for next residue).
        new_link_pos is None if no linking definition.

    Raises:
        ValueError: If linking atoms are not found in the residue.
    """
    # expansion.ideal_coords is already a copy from _expand_residue()
    coords = expansion.ideal_coords
    link_def = config.linking

    # Position relative to previous residue
    if link_def is not None and prev_link_pos is not None:
        curr_link_idx = _find_atom_index(expansion.atom_names, link_def.next_atom)
        if curr_link_idx is None:
            res_name = Residue(residue_idx).name
            raise ValueError(
                f"Linking atom '{link_def.next_atom}' not found in residue {res_name}. "
                f"Available atoms: {list(expansion.atom_names)[:10]}..."
            )
        target_pos = prev_link_pos + _EXTENSION_DIR * link_def.bond_length
        offset = target_pos - coords[curr_link_idx]
        coords = coords + offset

    # Compute new link position for next residue
    new_link_pos = None
    if link_def is not None:
        prev_link_idx = _find_atom_index(expansion.atom_names, link_def.prev_atom)
        if prev_link_idx is None:
            res_name = Residue(residue_idx).name
            raise ValueError(
                f"Linking atom '{link_def.prev_atom}' not found in residue {res_name}. "
                f"Available atoms: {list(expansion.atom_names)[:10]}..."
            )
        new_link_pos = coords[prev_link_idx].copy()

    return coords, new_link_pos


def _process_chain(sequence: str) -> ChainData:
    """
    Process a single chain sequence into atom/element/residue/coordinate data.

    Handles terminal atoms correctly:
    - 5'/N-terminal atoms only on first residue
    - 3'/C-terminal atoms only on last residue

    Positions residues with correct bond lengths between them.

    Args:
        sequence: Single-letter sequence for one chain.

    Returns:
        ChainData with all atom and coordinate information.
    """
    residue_indices = _parse_sequence(sequence)
    n_residues = len(residue_indices)
    config = _get_molecule_config(residue_indices[0])

    all_atoms: list[int] = []
    all_elements: list[int] = []
    all_coords: list[np.ndarray] = []
    atoms_per_res: list[int] = []
    prev_link_pos: np.ndarray | None = None

    for i, res_idx in enumerate(residue_indices):
        is_first = (i == 0)
        is_last = (i == n_residues - 1)

        # Get residue data and position it
        expansion = _expand_residue(res_idx)
        positioned_coords, prev_link_pos = _position_residue(
            expansion, res_idx, prev_link_pos, config
        )

        # Filter terminal atoms based on position
        filtered_atoms, filtered_elements, filtered_coords = _filter_atoms_by_position(
            expansion, positioned_coords, config, is_first, is_last
        )

        all_atoms.extend(filtered_atoms)
        all_elements.extend(filtered_elements)
        all_coords.extend(filtered_coords)
        atoms_per_res.append(len(filtered_atoms))

    return ChainData(
        atom_indices=all_atoms,
        element_indices=all_elements,
        atoms_per_residue=atoms_per_res,
        residue_indices=residue_indices,
        coords=all_coords,
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def from_sequence(
    sequence: str | Sequence[str],
    backend: str = "numpy",
    id: str = "template",
    sample_dihedrals: bool = False,
    seed: int | None = None,
) -> Polymer:
    """
    Generate a template Polymer from a sequence string or list of sequences.

    Creates a Polymer with correct atom types, elements, and residue sequence
    using ideal CCD coordinates. Useful for generative modeling where coordinates
    are generated separately, or for generating realistic conformations.

    Args:
        sequence: Single-letter sequence string, or list of strings for multi-chain.
            - Lowercase with 'u': RNA (acgu)
            - Lowercase with 't': DNA (acgt)
            - Lowercase with only a/c/g: RNA (default)
            - Uppercase: Protein (ACDEFGHIKLMNPQRSTVWY)
            - List creates multiple chains: ['acgu', 'acgt']
            - Empty strings are filtered out; "" returns empty polymer with 0 chains
        backend: Array backend, either "numpy" or "torch".
        id: PDB identifier for the polymer.
        sample_dihedrals: If True, randomize backbone dihedrals using empirical
            Ramachandran distributions fitted to PDB data. Only affects proteins.
        seed: Random seed for reproducible dihedral sampling. Only used when
            sample_dihedrals=True.

    Returns:
        Polymer with:
        - atoms: Global atom type indices
        - elements: Atomic numbers (H=1, C=6, N=7, O=8, P=15, S=16)
        - sequence: Residue type indices (matching Residue enum)
        - sizes: Atoms per residue/chain/molecule
        - coordinates: Ideal CCD coordinates (or randomized if sample_dihedrals=True)
        Empty sequence returns an empty Polymer with 0 atoms and 0 chains.

    Raises:
        ValueError: If sequence is mixed case, contains both 'u' and 't',
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

        >>> # Generate protein with random backbone conformations
        >>> protein = from_sequence("MGKLF", sample_dihedrals=True, seed=42)
    """
    # Normalize input and filter out empty sequences
    sequences = [sequence] if isinstance(sequence, str) else list(sequence)
    sequences = [s for s in sequences if s]  # Remove empty strings

    # Handle empty polymer (0 chains)
    if not sequences:
        return Polymer.create_empty(id=id, backend=backend)

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
        chain_data = _process_chain(seq)

        all_atoms.extend(chain_data.atom_indices)
        all_elements.extend(chain_data.element_indices)
        all_coords.extend(chain_data.coords)
        all_atoms_per_res.extend(chain_data.atoms_per_residue)
        all_residue_indices.extend(chain_data.residue_indices)
        atoms_per_chain.append(len(chain_data.atom_indices))
        residues_per_chain.append(len(chain_data.residue_indices))
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

    # Apply backbone dihedral sampling if requested (proteins only)
    if sample_dihedrals:
        from .sampling.backbone import randomize_backbone
        polymer = randomize_backbone(polymer, seed=seed)

    return polymer.torch() if backend == "torch" else polymer


def from_extract(
    coords: np.ndarray,
    atoms: list[int],
    residue: "ResidueType",
    backend: str = "numpy",
    id: str = "extracted",
) -> Polymer:
    """
    Convert extracted coordinates back to a Polymer.

    Takes the output of `extract()` and creates a Polymer that can be
    saved as a CIF file. Each row in coords becomes a separate residue
    (in a single chain).

    Args:
        coords: Coordinate array of shape (n_residues, n_atoms, 3) from extract().
        atoms: List of atom type indices from extract().
        residue: The ResidueType that was extracted (e.g., Residue.A).
        backend: Array backend, either "numpy" or "torch".
        id: PDB identifier for the polymer.

    Returns:
        Polymer with the extracted coordinates and correct atom metadata.

    Example:
        >>> from ciffy import load
        >>> from ciffy.biochemistry import Residue
        >>> from ciffy.operations import extract
        >>> from ciffy.template import from_extract
        >>>
        >>> poly = load("structure.cif")
        >>> coords, atoms = extract(poly, Residue.A, align=True, scale=True)
        >>>
        >>> # Process coords (e.g., run through a model)
        >>> new_coords = model(coords)
        >>>
        >>> # Convert back to Polymer for saving
        >>> result = from_extract(new_coords, atoms, Residue.A)
        >>> result.write("output.cif")
    """
    from .backend import to_numpy as _to_numpy

    # Handle torch input
    coords_np = np.asarray(_to_numpy(coords))

    n_residues, n_atoms, _ = coords_np.shape

    if len(atoms) != n_atoms:
        raise ValueError(
            f"Mismatch: coords has {n_atoms} atoms per residue, "
            f"but atoms list has {len(atoms)} entries"
        )

    # Build atom name -> member lookup for this residue type
    atom_enum = residue.atoms
    idx_to_member = {m.value: m for m in atom_enum}

    # Get element indices for each atom
    element_indices = []
    for atom_idx in atoms:
        if atom_idx not in idx_to_member:
            raise ValueError(
                f"Atom index {atom_idx} not found in {residue.name} atom enum"
            )
        atom_name = idx_to_member[atom_idx].name
        element_indices.append(_atom_name_to_element(atom_name))

    # Flatten coordinates: (n_residues, n_atoms, 3) -> (n_residues * n_atoms, 3)
    flat_coords = coords_np.reshape(-1, 3).astype(np.float32)

    # Build arrays for all residues
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
        id=id,
        names=["A"],
        strands=["A"],
        lengths=np.array([n_residues], dtype=np.int64),
        polymer_count=total_atoms,
    )

    return polymer.torch() if backend == "torch" else polymer
