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
    from ..polymer import Polymer
    from ..utils import filter_by_mask

    if isinstance(ix, int):
        ix = ops.array([ix], like=polymer.coordinates)

    # Validate indices
    max_chain = polymer.size(Scale.CHAIN)
    ix_list = ix.tolist() if hasattr(ix, 'tolist') else list(ix)
    for j in ix_list:
        if j < 0 or j >= max_chain:
            raise IndexError(
                f"Chain index {j} out of range for Polymer with {max_chain} chains"
            )

    atm_ix = polymer.mask(ix, Scale.CHAIN, Scale.ATOM)
    res_ix = polymer.mask(ix, Scale.CHAIN, Scale.RESIDUE)

    coordinates = polymer.coordinates[atm_ix]
    atoms = polymer.atoms[atm_ix]
    elements = polymer.elements[atm_ix]
    lengths = polymer.lengths[ix]

    sizes = {
        Scale.RESIDUE: polymer._sizes[Scale.RESIDUE][res_ix],
        Scale.CHAIN: polymer._sizes[Scale.CHAIN][ix],
        Scale.MOLECULE: ops.array([len(coordinates)], like=polymer.coordinates),
    }

    sequence = polymer.sequence[res_ix]
    names = [polymer.names[j] for j in ix]
    strands = [polymer.strands[j] for j in ix]

    # Calculate new polymer_count from residue sizes
    new_polymer_count = sizes[Scale.RESIDUE].sum().item()

    # Preserve molecule types if available
    mol_types = polymer._molecule_types[ix] if polymer._molecule_types is not None else None

    # Slice bfactors by atom mask
    bfactors = polymer._bfactors[atm_ix] if polymer._bfactors is not None else None

    return Polymer(
        coordinates, atoms, elements, sequence, sizes,
        polymer.pdb_id, names, strands, lengths, new_polymer_count,
        mol_types,
        bfactors=bfactors,
        resolution=polymer._resolution,
    )


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


def by_residue_index(polymer: Polymer, ix: Array | int) -> Polymer:
    """
    Select residues by positional index.

    Unlike by_residue() which selects by residue TYPE (e.g., all adenines),
    this method selects by positional INDEX (e.g., residue 0, 1, 2...).

    Args:
        polymer: Source polymer.
        ix: Residue index or indices (0-indexed position in polymer).

    Returns:
        New Polymer with selected residues.

    Raises:
        IndexError: If any index is out of range.

    Example:
        >>> first = by_residue_index(polymer, 0)
        >>> subset = by_residue_index(polymer, [0, 2, 4])
    """
    if isinstance(ix, int):
        ix = ops.array([ix], like=polymer.coordinates)

    # Validate indices
    max_res = polymer.size(Scale.RESIDUE)
    ix_list = ix.tolist() if hasattr(ix, 'tolist') else list(ix)
    for j in ix_list:
        if j < 0 or j >= max_res:
            raise IndexError(
                f"Residue index {j} out of range for Polymer with {max_res} residues"
            )

    atom_mask = polymer.mask(ix, Scale.RESIDUE, Scale.ATOM)
    return polymer[atom_mask]


def by_type(polymer: Polymer, mol: Molecule) -> Polymer:
    """
    Select chains by molecule type.

    Args:
        polymer: Source polymer.
        mol: Molecule type to select.

    Returns:
        New Polymer with chains of that type.
    """
    ix = ops.nonzero_1d(polymer.molecule_type == mol.value)
    return by_index(polymer, ix)
