"""
Unified chain building for polymer construction.

Provides ChainBuilder class used by:
- from_sequence() for template generation
- Polymer.extend() for appending residues
- Autoregressive models for generation
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ..biochemistry import Scale, Molecule, Residue, atom_to_element
from ..biochemistry.linking import LINKING_BY_TYPE, LinkingDefinition, NUCLEIC_ACID_LINK, PEPTIDE_LINK
from ..backend import Array, is_torch
from ..utils import atoms_to_col_map

if TYPE_CHECKING:
    from ..biochemistry.atom import AtomGroup


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ResidueData:
    """
    A residue with positioned coordinates and atom data.

    This is the fundamental unit that ChainBuilder accumulates.
    """
    coords: Array                  # (n_atoms, 3) positioned coordinates
    atoms: tuple[int, ...]         # atom type indices
    elements: tuple[int, ...]      # element indices
    residue: Residue               # residue type
    atom_to_col: dict[int, int]    # atom value -> column index

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this residue."""
        return len(self.atoms)


@dataclass(frozen=True)
class MoleculeConfig:
    """
    Configuration for a molecule type.

    Bundles molecule-specific parameters for chain building.
    """
    linking: LinkingDefinition | None
    start_terminal_atoms: frozenset[str]  # 5'/N-terminal only
    end_terminal_atoms: frozenset[str]    # 3'/C-terminal only


@dataclass
class FrameIndices:
    """
    Pre-resolved frame column indices for a residue type.

    Uses arrays with -1 sentinel for missing perp_ref, following ciffy's
    philosophy of arrays over Python data types. This enables efficient
    frame computation without Python attribute lookups.

    Attributes:
        prev_cols: shape (3,) int32 array [origin, z_ref, perp_ref] for outgoing frame.
            -1 indicates missing perp_ref.
        prev_z_toward: If True, Z points from z_ref toward origin.
        next_cols: shape (3,) int32 array [origin, z_ref, perp_ref] for incoming frame.
            -1 indicates missing perp_ref.
        next_z_toward: If True, Z points from z_ref toward origin.
    """
    prev_cols: np.ndarray   # shape (3,), dtype=int32, -1 = no perp_ref
    prev_z_toward: bool
    next_cols: np.ndarray   # shape (3,), dtype=int32, -1 = no perp_ref
    next_z_toward: bool


# =============================================================================
# MOLECULE CONFIGURATIONS
# =============================================================================

_MOLECULE_CONFIGS: dict[Molecule, MoleculeConfig] = {
    Molecule.RNA: MoleculeConfig(
        linking=NUCLEIC_ACID_LINK,
        start_terminal_atoms=frozenset({'OP3', 'HOP3'}),
        end_terminal_atoms=frozenset({'HO3p'}),
    ),
    Molecule.DNA: MoleculeConfig(
        linking=NUCLEIC_ACID_LINK,
        start_terminal_atoms=frozenset({'OP3', 'HOP3'}),
        end_terminal_atoms=frozenset({'HO3p'}),
    ),
    Molecule.PROTEIN: MoleculeConfig(
        linking=PEPTIDE_LINK,
        start_terminal_atoms=frozenset({'H2', 'H3'}),
        end_terminal_atoms=frozenset({'OXT', 'HXT'}),
    ),
}


def get_molecule_config(mol_type: Molecule) -> MoleculeConfig:
    """Get configuration for a molecule type."""
    if mol_type not in _MOLECULE_CONFIGS:
        raise ValueError(f"Unsupported molecule type: {mol_type.name}")
    return _MOLECULE_CONFIGS[mol_type]


# =============================================================================
# RESIDUE EXPANSION (cached ideal coordinates and atom data)
# =============================================================================

@lru_cache(maxsize=64)
def _expand_residue_cached(residue_idx: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], np.ndarray]:
    """
    Get cached atom data for a residue type.

    Returns tuple of (atom_indices, element_indices, atom_names, ideal_coords).
    The numpy array is cached; callers should copy if mutating.
    """
    try:
        residue = Residue.from_index(residue_idx)
    except (ValueError, KeyError):
        raise ValueError(f"Invalid residue index: {residue_idx}")

    if residue.atoms is None:
        raise ValueError(f"No atom definitions for residue {residue.name}")

    atom_indices = []
    element_indices = []
    atom_names = []

    for member in residue.atoms:
        atom_indices.append(member.value)
        atom_names.append(member.name)
        element_indices.append(atom_to_element(member))

    return tuple(atom_indices), tuple(element_indices), tuple(atom_names), residue.ideal


def expand_residue(
    residue: Residue,
    start_terminal: bool = True,
    end_terminal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get atom data for a residue type with optional terminal filtering.

    Args:
        residue: Residue type.
        start_terminal: If True, include 5'/N-terminal atoms (OP3, HOP3 for RNA).
        end_terminal: If True, include 3'/C-terminal atoms (HO3' for RNA).

    Returns:
        Tuple of (atoms, elements, coords) as numpy arrays.

    Example:
        >>> # Internal residue (no terminal atoms)
        >>> atoms, elements, coords = expand_residue(Residue.A, start_terminal=False, end_terminal=False)
        >>> # First residue (5' terminal only)
        >>> atoms, elements, coords = expand_residue(Residue.A, start_terminal=True, end_terminal=False)
        >>> # Last residue (3' terminal only)
        >>> atoms, elements, coords = expand_residue(Residue.U, start_terminal=False, end_terminal=True)
    """
    atom_indices, element_indices, atom_names, coords = _expand_residue_cached(residue.value)

    # Check if filtering is needed
    if start_terminal and end_terminal:
        # No filtering - return all atoms as arrays
        return (
            np.array(atom_indices, dtype=np.int64),
            np.array(element_indices, dtype=np.int64),
            coords.copy(),
        )

    # Get molecule config for terminal atoms
    mol_type = residue.molecule_type
    if mol_type not in _MOLECULE_CONFIGS:
        # Unknown molecule type - return all atoms
        return (
            np.array(atom_indices, dtype=np.int64),
            np.array(element_indices, dtype=np.int64),
            coords.copy(),
        )

    config = _MOLECULE_CONFIGS[mol_type]

    # Build set of atoms to exclude
    exclude: set[str] = set()
    if not start_terminal:
        exclude.update(config.start_terminal_atoms)
    if not end_terminal:
        exclude.update(config.end_terminal_atoms)

    if not exclude:
        # Nothing to exclude
        return (
            np.array(atom_indices, dtype=np.int64),
            np.array(element_indices, dtype=np.int64),
            coords.copy(),
        )

    # Filter atoms
    keep_indices = [i for i, name in enumerate(atom_names) if name not in exclude]

    return (
        np.array([atom_indices[i] for i in keep_indices], dtype=np.int64),
        np.array([element_indices[i] for i in keep_indices], dtype=np.int64),
        coords[keep_indices].copy(),
    )


def linear_extend_transform(
    prev_coords: Array,
    prev_atoms: Array,
    prev_residue: Residue,
    next_atoms: Array,
    next_residue: Residue,
) -> np.ndarray:
    """
    Calculate SE(3) transform for linear chain extension.

    Computes the transform that positions the next residue along the backbone
    axis with proper spacing, maintaining the same orientation as the previous
    residue.

    Args:
        prev_coords: (n_atoms, 3) coordinates of previous residue.
        prev_atoms: Atom type indices of previous residue.
        prev_residue: Previous residue type.
        next_atoms: Atom type indices of next residue.
        next_residue: Next residue type.

    Returns:
        (6,) SE(3) transform [axis-angle (3), translation (3)] as numpy array.
        The axis-angle is [0, 0, 0] (no rotation), and translation is
        [0, 0, spacing] where spacing is the backbone span + bond length.

    Example:
        >>> atoms1, elements1, coords1 = expand_residue(Residue.A)
        >>> atoms2, elements2, coords2 = expand_residue(Residue.C, start_terminal=False)
        >>> transform = linear_extend_transform(coords1, atoms1, Residue.A, atoms2, Residue.C)
        >>> # Use with Polymer.extend() or position_residue_fast()
    """
    from ..biochemistry.linking import LINKING_BY_TYPE

    link_def = LINKING_BY_TYPE.get(prev_residue.molecule_type)

    if link_def is None:
        # No linking definition - use default spacing
        spacing = 6.0
    else:
        # Build atom_to_col mappings
        prev_atom_to_col = atoms_to_col_map(tuple(int(a) for a in prev_atoms))
        next_atom_to_col = atoms_to_col_map(tuple(int(a) for a in next_atoms))

        # Get linking atoms
        prev_link_atom = getattr(prev_residue, link_def.prev_atom)  # e.g., O3' for RNA
        next_link_atom = getattr(next_residue, link_def.next_atom)  # e.g., P for RNA

        # Get P position of previous residue to calculate backbone span
        prev_p_atom = getattr(prev_residue, link_def.next_atom)  # P atom

        if prev_link_atom.value in prev_atom_to_col and prev_p_atom.value in prev_atom_to_col:
            prev_link_pos = prev_coords[prev_atom_to_col[prev_link_atom.value]]
            prev_p_pos = prev_coords[prev_atom_to_col[prev_p_atom.value]]
            # Backbone span is distance from P to O3' plus bond length
            backbone_span = float(np.linalg.norm(prev_link_pos - prev_p_pos))
            spacing = backbone_span + link_def.bond_length
        else:
            # Missing atoms - use default spacing
            spacing = 6.0

    # Return identity rotation + Z-axis translation
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, spacing], dtype=np.float32)


# =============================================================================
# FRAME RESOLUTION (cached for fast positioning)
# =============================================================================

@lru_cache(maxsize=256)
def _resolve_frame_indices(residue_idx: int, atom_indices: tuple[int, ...]) -> FrameIndices:
    """
    Resolve frame column indices for a (residue_type, atom_subset) pair.

    This is cached so that repeated positioning of the same residue type
    with the same atom subset doesn't require repeated lookups.

    Args:
        residue_idx: Residue enum value (int).
        atom_indices: Tuple of atom type values in the residue's coordinate array.

    Returns:
        FrameIndices with pre-resolved column indices for both incoming and
        outgoing frames.

    Raises:
        ValueError: If required linking atoms are missing from atom_indices.
    """
    residue = Residue.from_index(residue_idx)
    atom_to_col = atoms_to_col_map(atom_indices)
    link_def = LINKING_BY_TYPE.get(residue.molecule_type)

    if link_def is None:
        # Non-polymer residue (ligand, etc.) - no linking frames
        return FrameIndices(
            prev_cols=np.array([-1, -1, -1], dtype=np.int32),
            prev_z_toward=True,
            next_cols=np.array([-1, -1, -1], dtype=np.int32),
            next_z_toward=True,
        )

    # Resolve outgoing (prev) frame
    prev_tuple = link_def.prev_frame.resolve(residue, atom_to_col)
    prev_cols = np.array([
        prev_tuple[0],
        prev_tuple[1],
        prev_tuple[2] if prev_tuple[2] is not None else -1,
    ], dtype=np.int32)

    # Resolve incoming (next) frame
    next_tuple = link_def.next_frame.resolve(residue, atom_to_col)
    next_cols = np.array([
        next_tuple[0],
        next_tuple[1],
        next_tuple[2] if next_tuple[2] is not None else -1,
    ], dtype=np.int32)

    return FrameIndices(
        prev_cols=prev_cols,
        prev_z_toward=link_def.prev_frame.z_toward_origin,
        next_cols=next_cols,
        next_z_toward=link_def.next_frame.z_toward_origin,
    )


# =============================================================================
# CHAIN ASSEMBLY (standalone function for generative models)
# =============================================================================

def assemble_chain(
    residue_coords: list[Array],
    transforms: list[Array],
    residues: list[Residue],
    atom_subsets: list[tuple[int, ...]],
) -> Array:
    """
    Assemble a chain from pre-decoded residue coordinates and transforms.

    This is the unified assembly function for all generative models (flow models,
    autoregressive models, etc.). It handles frame resolution with caching and
    uses the fast positioning path.

    Args:
        residue_coords: List of (n_atoms, 3) coordinate arrays per residue.
            These are the decoded/predicted coordinates in canonical frame.
        transforms: List of (6,) SE(3) transforms per residue [axis-angle, translation].
            The transform for residue i positions it relative to residue i-1.
        residues: List of Residue enum values.
        atom_subsets: List of atom index tuples for each residue's coordinate array.
            Must match the atom ordering in residue_coords.

    Returns:
        (N, 3) concatenated positioned coordinates for the entire chain.

    Example:
        >>> # In PolymerFlowModel.decode()
        >>> residue_coords, transforms = [], []
        >>> for i, res_type in enumerate(sequence):
        ...     coords, transform = model.decode(latents[i])
        ...     residue_coords.append(coords)
        ...     transforms.append(transform)
        >>> positioned = assemble_chain(residue_coords, transforms, residues, atom_subsets)
    """
    from ..geometry import position_residue_fast

    if len(residue_coords) == 0:
        return np.empty((0, 3), dtype=np.float32)

    all_coords = []
    prev_coords = None
    prev_frame: FrameIndices | None = None
    prev_transform = None

    for i, (coords, transform, residue, atoms) in enumerate(
        zip(residue_coords, transforms, residues, atom_subsets)
    ):
        # Get cached frame indices for this residue type + atom subset
        frame = _resolve_frame_indices(residue.value, atoms)

        if i == 0:
            # First residue - no positioning needed, place at origin
            positioned = coords
        else:
            # Position relative to previous residue
            positioned = position_residue_fast(
                prev_coords,
                coords,
                prev_transform,
                prev_frame.prev_cols,
                prev_frame.prev_z_toward,
                frame.next_cols,
                frame.next_z_toward,
            )

        all_coords.append(positioned)

        # Store for next iteration
        prev_coords = positioned
        prev_frame = frame
        prev_transform = transform

    # Concatenate all positioned coordinates
    if is_torch(all_coords[0]):
        import torch
        return torch.cat(all_coords, dim=0)
    return np.concatenate(all_coords, axis=0)


# =============================================================================
# CHAIN BUILDER
# =============================================================================

class ChainBuilder:
    """
    Accumulates residues into a chain with proper positioning.

    Handles:
    - Positioning residues relative to previous using linking frames
    - Optional terminal atom filtering (for template generation)
    - Building final arrays for Polymer construction

    Example:
        >>> builder = ChainBuilder(Molecule.RNA)
        >>> builder.append(Residue.A)
        >>> builder.append(Residue.C)
        >>> builder.append(Residue.G)
        >>> builder.append(Residue.U)
        >>> arrays = builder.build()
        >>> polymer = Polymer(**arrays, pdb_id="test", ...)
    """

    __slots__ = (
        'mol_type', 'config', 'filter_terminal',
        '_residues', '_prev_coords', '_prev_atoms', '_prev_residue',
    )

    def __init__(
        self,
        mol_type: Molecule,
        filter_terminal: bool = True,
    ):
        """
        Initialize a chain builder.

        Args:
            mol_type: Molecule type (RNA, DNA, PROTEIN).
            filter_terminal: If True, filter terminal-only atoms based on
                position in chain. Set to False for Polymer.extend() where
                the existing polymer already has proper terminal atoms.
        """
        self.mol_type = mol_type
        self.config = get_molecule_config(mol_type)
        self.filter_terminal = filter_terminal

        self._residues: list[ResidueData] = []
        self._prev_coords: Array | None = None
        self._prev_atoms: tuple[int, ...] | None = None
        self._prev_residue: Residue | None = None

    def set_previous(
        self,
        coords: Array,
        atoms: tuple[int, ...] | list[int],
        residue: Residue,
    ) -> None:
        """
        Set previous residue state for positioning.

        Use this when extending an existing polymer - the first appended
        residue will be positioned relative to this state.

        Args:
            coords: (n_atoms, 3) coordinates of previous residue.
            atoms: Tuple of atom type values in the coordinate array.
            residue: Previous residue type.
        """
        self._prev_coords = coords
        self._prev_atoms = tuple(atoms) if not isinstance(atoms, tuple) else atoms
        self._prev_residue = residue

    def append(
        self,
        residue: Residue,
        coords: Array | None = None,
        transform: Array | None = None,
    ) -> ResidueData:
        """
        Position and append a residue to the chain.

        Args:
            residue: Residue type to append.
            coords: Optional custom coordinates. If None, uses ideal coordinates.
            transform: Optional (6,) SE(3) transform [axis-angle, translation].
                If None, uses linear extension along Z-axis.

        Returns:
            The positioned ResidueData that was appended.

        Raises:
            ValueError: If required linking atoms are missing.
        """
        from ..geometry import position_residue_fast

        # Determine terminal filtering
        is_first = len(self._residues) == 0 and self._prev_coords is None

        # Get atom data with terminal filtering
        # Note: end_terminal=False here; it's handled in build() for the last residue
        if self.filter_terminal:
            atom_arr, element_arr, ideal_coords = expand_residue(
                residue, start_terminal=is_first, end_terminal=False
            )
        else:
            atom_arr, element_arr, ideal_coords = expand_residue(residue)

        # Convert to tuples for compatibility with existing code
        atom_indices = tuple(int(a) for a in atom_arr)
        element_indices = tuple(int(e) for e in element_arr)

        # Use provided coords or ideal
        if coords is None:
            coords = ideal_coords
        else:
            # Ensure we have a copy if numpy
            if not is_torch(coords):
                coords = np.asarray(coords).copy()

        atom_to_col = atoms_to_col_map(atom_indices)

        # Position relative to previous residue
        if self._prev_coords is not None:
            # Validate linking atoms
            link_def = self.config.linking
            if link_def is not None:
                # Get atom names for validation
                residue_atoms = residue.atoms
                atom_names = tuple(
                    next((m.name for m in residue_atoms if m.value == a), '?')
                    for a in atom_indices
                )
                self._validate_linking_atoms(
                    atom_indices, atom_names, residue, link_def, "next"
                )

            # Compute transform if not provided
            if transform is None:
                transform = linear_extend_transform(
                    self._prev_coords,
                    np.array(self._prev_atoms, dtype=np.int64),
                    self._prev_residue,
                    np.array(atom_indices, dtype=np.int64),
                    residue,
                )

            # Position using frame-based approach
            prev_frame = _resolve_frame_indices(self._prev_residue.value, self._prev_atoms)
            next_frame = _resolve_frame_indices(residue.value, atom_indices)
            coords = position_residue_fast(
                self._prev_coords,
                coords,
                transform,
                prev_frame.prev_cols,
                prev_frame.prev_z_toward,
                next_frame.next_cols,
                next_frame.next_z_toward,
            )

        # Create residue data
        residue_data = ResidueData(
            coords=coords,
            atoms=atom_indices,
            elements=element_indices,
            residue=residue,
            atom_to_col=atom_to_col,
        )

        # Update state for next residue
        self._residues.append(residue_data)
        self._prev_coords = coords
        self._prev_atoms = atom_indices
        self._prev_residue = residue

        # Validate outgoing frame atoms for next positioning
        if self.config.linking is not None:
            # Get atom names by looking up each atom value
            names = tuple(residue.atoms[a].name for a in atom_indices)
            self._validate_linking_atoms(
                atom_indices, names, residue, self.config.linking, "prev"
            )

        return residue_data

    def _validate_linking_atoms(
        self,
        atoms: tuple[int, ...],
        names: tuple[str, ...],
        residue: Residue,
        link_def: LinkingDefinition,
        which: str,
    ) -> None:
        """Validate that required linking atoms are present."""
        available = set(atoms)
        missing = link_def.validate_atoms(residue, available, which=which)

        if missing:
            frame_name = "incoming" if which == "next" else "outgoing"
            raise ValueError(
                f"Cannot compute {frame_name} frame for {residue.name}: "
                f"missing atoms {missing}. "
                f"Available: {list(names)[:10]}..."
            )

    def __len__(self) -> int:
        """Number of residues appended."""
        return len(self._residues)

    @property
    def empty(self) -> bool:
        """True if no residues have been appended."""
        return len(self._residues) == 0

    def build(self) -> dict:
        """
        Build arrays for Polymer construction.

        If filter_terminal is True, the last residue is reprocessed to include
        its terminal atoms.

        Returns:
            Dict with keys: coordinates, atoms, elements, sequence, sizes
        """
        if self.empty:
            return {
                'coordinates': np.empty((0, 3), dtype=np.float32),
                'atoms': np.empty(0, dtype=np.int64),
                'elements': np.empty(0, dtype=np.int64),
                'sequence': np.empty(0, dtype=np.int64),
                'sizes': {
                    Scale.RESIDUE: np.empty(0, dtype=np.int64),
                    Scale.CHAIN: np.empty(0, dtype=np.int64),
                    Scale.MOLECULE: np.array([0], dtype=np.int64),
                },
            }

        # If filtering terminal atoms, reprocess last residue to include end-terminal atoms
        residues = list(self._residues)
        if self.filter_terminal and len(residues) > 0:
            last = residues[-1]
            is_first = len(residues) == 1

            # Re-expand with end-terminal atoms included
            atom_arr, element_arr, _ = expand_residue(
                last.residue, start_terminal=is_first, end_terminal=True
            )
            atom_indices = tuple(int(a) for a in atom_arr)
            element_indices = tuple(int(e) for e in element_arr)

            # Get positioned coordinates for the expanded atom set
            positioned_coords = self._get_full_coords_for_last()

            residues[-1] = ResidueData(
                coords=positioned_coords,
                atoms=atom_indices,
                elements=element_indices,
                residue=last.residue,
                atom_to_col=atoms_to_col_map(atom_indices),
            )

        # Accumulate arrays
        all_coords = []
        all_atoms = []
        all_elements = []
        all_sequence = []
        atoms_per_residue = []

        for res_data in residues:
            if is_torch(res_data.coords):
                all_coords.append(res_data.coords.detach().cpu().numpy())
            else:
                all_coords.append(np.asarray(res_data.coords))
            all_atoms.extend(res_data.atoms)
            all_elements.extend(res_data.elements)
            all_sequence.append(res_data.residue.value)
            atoms_per_residue.append(res_data.n_atoms)

        # Build final arrays
        coords = np.concatenate(all_coords, axis=0).astype(np.float32)
        atoms = np.array(all_atoms, dtype=np.int64)
        elements = np.array(all_elements, dtype=np.int64)
        sequence = np.array(all_sequence, dtype=np.int64)

        n_atoms = len(atoms)

        return {
            'coordinates': coords,
            'atoms': atoms,
            'elements': elements,
            'sequence': sequence,
            'sizes': {
                Scale.RESIDUE: np.array(atoms_per_residue, dtype=np.int64),
                Scale.CHAIN: np.array([n_atoms], dtype=np.int64),
                Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
            },
        }

    def _get_full_coords_for_last(self) -> Array:
        """Get full coordinates (with end-terminal atoms) for the last residue."""
        is_first = len(self._residues) == 1

        # Get full atom set with end-terminal atoms
        atom_arr, _, ideal = expand_residue(
            self._residues[-1].residue,
            start_terminal=is_first,
            end_terminal=True,
        )
        atom_indices = tuple(int(a) for a in atom_arr)

        if len(self._residues) < 2:
            # First (and last) residue - use ideal coords
            return ideal

        # Re-position using second-to-last residue's state
        from ..geometry import position_residue_fast

        last = self._residues[-1]
        second_last = self._residues[-2]

        # Compute linear extension transform
        transform = linear_extend_transform(
            second_last.coords,
            np.array(second_last.atoms, dtype=np.int64),
            second_last.residue,
            atom_arr,
            last.residue,
        )

        # Position using frame-based approach
        prev_frame = _resolve_frame_indices(second_last.residue.value, second_last.atoms)
        next_frame = _resolve_frame_indices(last.residue.value, atom_indices)

        return position_residue_fast(
            second_last.coords,
            ideal,
            transform,
            prev_frame.prev_cols,
            prev_frame.prev_z_toward,
            next_frame.next_cols,
            next_frame.next_z_toward,
        )
