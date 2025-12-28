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


def by_atom(polymer: Polymer, name: Array | int) -> Polymer:
    """
    Select atoms by atom type index.

    Args:
        polymer: Source polymer.
        name: Atom type index or indices.

    Returns:
        New Polymer with matching atoms.
    """
    name = ops.convert_backend(name, polymer.atoms)
    mask = (polymer.atoms[:, None] == name).any(1)
    return polymer[mask]


def by_residue(polymer: Polymer, res: Array | int) -> Polymer:
    """
    Select residues by residue type index.

    Args:
        polymer: Source polymer.
        res: Residue type index or indices (from Residue enum).

    Returns:
        New Polymer with matching residues.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> adenosines = by_residue(polymer, Residue.A)
        >>> purines = by_residue(polymer, [Residue.A, Residue.G])
    """
    res = ops.convert_backend(res, polymer.sequence)
    res_mask = (polymer.sequence[:, None] == res).any(1)
    atom_mask = polymer.expand(res_mask, Scale.RESIDUE, Scale.ATOM)
    return polymer[atom_mask]


def by_type(polymer: Polymer, mol: Molecule) -> Polymer:
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
