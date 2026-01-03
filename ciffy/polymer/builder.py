"""
Chain building utilities for polymer construction and generative models.

This module provides functions for:

- **Residue expansion**: Get atom data with terminal filtering (expand_residue)
- **Linear extension**: Calculate SE(3) transform for linear chain extension (linear_extend_transform)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..biochemistry import Molecule, Residue, atom_to_element
from ..biochemistry.linking import LinkingDefinition, NUCLEIC_ACID_LINK, PEPTIDE_LINK
from ..backend import Array
from ..utils import atoms_to_col_map


# =============================================================================
# DATA CLASSES
# =============================================================================

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
        # Build atom_to_col mapping for previous residue
        prev_atom_to_col = atoms_to_col_map(tuple(int(a) for a in prev_atoms))

        # Get linking atom on previous residue
        prev_link_atom = getattr(prev_residue, link_def.prev_atom)  # e.g., O3' for RNA

        # Get P position of previous residue to calculate backbone span
        prev_p_atom = getattr(prev_residue, link_def.next_atom)  # P atom (start of current residue)

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

