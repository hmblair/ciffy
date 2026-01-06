"""
Selection filters for Polymer objects.

Functions for selecting atoms, residues, and chains based on various criteria.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..polymer import Polymer

from ..backend import Array, ops
from ..biochemistry import Scale, Molecule


def by_index(polymer: Polymer, ix: Array | int) -> Polymer:
    """
    Select chains by index.

    Args:
        polymer: Source polymer.
        ix: Chain index or indices to select.

    Returns:
        New Polymer with selected chains.

    Raises:
        IndexError: If any index is out of range.
    """
    return polymer.select(ix, Scale.CHAIN)


def atom_type(polymer: Polymer, atom: Array | int) -> Polymer:
    """
    Select atoms by atom type index.

    Args:
        polymer: Source polymer.
        atom: Atom type index or indices.

    Returns:
        New Polymer with matching atoms.
    """
    atom = ops.convert_backend(atom, polymer.atoms)
    mask = (polymer.atoms[:, None] == atom).any(1)
    return polymer[mask]


def residue_type(polymer: Polymer, residue: Array | int) -> Polymer:
    """
    Select residues by residue type index.

    Args:
        polymer: Source polymer.
        residue: Residue type index or indices (from Residue enum).

    Returns:
        New Polymer with matching residues.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> adenosines = residue_type(polymer, Residue.A)
        >>> purines = residue_type(polymer, [Residue.A, Residue.G])
    """
    # Extract .value from Residue enums (which are AtomGroups)
    if hasattr(residue, 'value') and hasattr(residue, 'atoms'):
        # Single Residue enum
        residue = residue.value
    elif isinstance(residue, (list, tuple)):
        # List of Residue enums or values
        residue = [r.value if hasattr(r, 'value') and hasattr(r, 'atoms') else r for r in residue]

    residue = ops.convert_backend(residue, polymer.sequence)
    res_mask = (polymer.sequence[:, None] == residue).any(1)
    atom_mask = polymer.expand(res_mask, Scale.RESIDUE, Scale.ATOM)
    return polymer[atom_mask]


def molecule_type(polymer: Polymer, mol: Molecule) -> Polymer:
    """
    Select chains by molecule type.

    Args:
        polymer: Source polymer.
        mol: Molecule type to select.

    Returns:
        New Polymer with chains of that type.

    Raises:
        ValueError: If molecule_types is not available on the polymer.
    """
    if polymer.molecule_types is None:
        raise ValueError("Cannot filter by type: molecule_types not available on this polymer")
    ix = ops.nonzero_1d(polymer.molecule_types == mol.value)
    return polymer.select(ix, Scale.CHAIN)


def element_type(polymer: Polymer, element: Array | int) -> Polymer:
    """
    Select atoms by element index.

    Args:
        polymer: Source polymer.
        element: Element index or indices (from Element enum).

    Returns:
        New Polymer with matching atoms.

    Example:
        >>> from ciffy.biochemistry import Element
        >>> carbons = element_type(polymer, Element.C)
        >>> organic = element_type(polymer, [Element.C, Element.N, Element.O])
    """
    element = ops.convert_backend(element, polymer.elements)
    mask = (polymer.elements[:, None] == element).any(1)
    return polymer[mask]
