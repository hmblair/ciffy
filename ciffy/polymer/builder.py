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


def expand_residue(residue: Residue) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], np.ndarray]:
    """
    Get atom data for a residue type.

    Args:
        residue: Residue type.

    Returns:
        Tuple of (atom_indices, element_indices, atom_names, ideal_coords).
        Coordinates are copied for safe mutation.
    """
    atoms, elements, names, coords = _expand_residue_cached(residue.value)
    return atoms, elements, names, coords.copy()


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
        '_residues', '_prev_coords', '_prev_atom_to_col', '_prev_residue',
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
        self._prev_atom_to_col: dict[int, int] | None = None
        self._prev_residue: Residue | None = None

    def set_previous(
        self,
        coords: Array,
        atom_to_col: dict[int, int],
        residue: Residue,
    ) -> None:
        """
        Set previous residue state for positioning.

        Use this when extending an existing polymer - the first appended
        residue will be positioned relative to this state.

        Args:
            coords: (n_atoms, 3) coordinates of previous residue.
            atom_to_col: Atom value -> column index mapping.
            residue: Previous residue type.
        """
        self._prev_coords = coords
        self._prev_atom_to_col = atom_to_col
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
        from ..geometry import position_residue

        # Get atom data for this residue
        atom_indices, element_indices, atom_names, ideal_coords = expand_residue(residue)

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
                self._validate_linking_atoms(
                    atom_indices, atom_names, residue, link_def, "next"
                )

            coords = position_residue(
                prev_coords=self._prev_coords,
                next_coords=coords,
                prev_atom_to_col=self._prev_atom_to_col,
                next_atom_to_col=atom_to_col,
                prev_residue=self._prev_residue,
                next_residue=residue,
                transform=transform,
            )

        # Filter terminal atoms if enabled
        if self.filter_terminal:
            is_first = len(self._residues) == 0 and self._prev_coords is None
            # Note: is_last is handled in finalize() by refiltering the last residue
            atom_indices, element_indices, coords = self._filter_atoms(
                atom_indices, element_indices, atom_names, coords,
                is_first=is_first, is_last=False,
            )
            atom_to_col = atoms_to_col_map(atom_indices)

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
        self._prev_atom_to_col = atom_to_col
        self._prev_residue = residue

        # Validate outgoing frame atoms for next positioning
        if self.config.linking is not None:
            # Get atom names by looking up each atom value
            names = tuple(residue.atoms[a].name for a in atom_indices)
            self._validate_linking_atoms(
                atom_indices, names, residue, self.config.linking, "prev"
            )

        return residue_data

    def _filter_atoms(
        self,
        atoms: tuple[int, ...],
        elements: tuple[int, ...],
        names: tuple[str, ...],
        coords: Array,
        is_first: bool,
        is_last: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...], Array]:
        """Filter atoms based on position in chain."""
        filtered_atoms = []
        filtered_elements = []
        filtered_indices = []

        for i, (atom, elem, name) in enumerate(zip(atoms, elements, names)):
            is_start_terminal = name in self.config.start_terminal_atoms
            is_end_terminal = name in self.config.end_terminal_atoms

            include = True
            if is_start_terminal and not is_first:
                include = False
            if is_end_terminal and not is_last:
                include = False

            if include:
                filtered_atoms.append(atom)
                filtered_elements.append(elem)
                filtered_indices.append(i)

        # Slice coordinates
        if is_torch(coords):
            import torch
            filtered_coords = coords[torch.tensor(filtered_indices)]
        else:
            filtered_coords = coords[filtered_indices]

        return tuple(filtered_atoms), tuple(filtered_elements), filtered_coords

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
            # Re-expand to get all atoms including terminal
            all_atoms, all_elements, all_names, _ = expand_residue(last.residue)

            is_first = len(residues) == 1
            filtered_atoms, filtered_elements, filtered_coords = self._filter_atoms(
                all_atoms, all_elements, all_names,
                # Need to get unfiltered coords - use the positioned coords from prev state
                self._get_full_coords_for_last(),
                is_first=is_first, is_last=True,
            )

            residues[-1] = ResidueData(
                coords=filtered_coords,
                atoms=filtered_atoms,
                elements=filtered_elements,
                residue=last.residue,
                atom_to_col=atoms_to_col_map(filtered_atoms),
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
        """Get full (unfiltered) coordinates for the last residue."""
        # The _prev_coords has the positioned but potentially filtered coords
        # We need to reposition using full coords
        if len(self._residues) < 2:
            # First (and last) residue - use ideal coords
            _, _, _, ideal = expand_residue(self._residues[-1].residue)
            return ideal

        # Re-position using second-to-last residue's state
        from ..geometry import position_residue

        last = self._residues[-1]
        second_last = self._residues[-2]

        _, _, _, ideal = expand_residue(last.residue)
        atom_to_col = atoms_to_col_map(tuple(a.value for a in last.residue.atoms))

        return position_residue(
            prev_coords=second_last.coords,
            next_coords=ideal,
            prev_atom_to_col=second_last.atom_to_col,
            next_atom_to_col=atom_to_col,
            prev_residue=second_last.residue,
            next_residue=last.residue,
            transform=None,
        )
