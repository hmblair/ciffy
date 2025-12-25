"""
Iteration and partitioning functions for Polymer objects.

Functions for iterating over chains and partitioning polymers.
"""

from __future__ import annotations
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..polymer import Polymer

from ..backend import ops
from ..biochemistry import Scale, Molecule
from ..operations.reduction import Reduction


def poly(polymer: Polymer) -> Polymer:
    """
    Return polymer portion only (excludes HETATM/non-polymer atoms).

    The returned Polymer has valid residue information and can be used
    with residue-scale operations like reduce(scale=Scale.RESIDUE).

    Args:
        polymer: Source polymer.

    Returns:
        New Polymer with only polymer atoms, or the input polymer if no HETATM atoms.

    Example:
        >>> p = load("file.cif")
        >>> rna = poly(p)  # Get polymer only
        >>> rna.reduce(features, Scale.RESIDUE)  # Works correctly
    """
    from ..polymer import Polymer
    from ..utils import filter_by_mask

    if polymer.nonpoly == 0:
        return polymer

    # Slice to polymer atoms only
    coordinates = polymer.coordinates[:polymer.polymer_count]
    atoms = polymer.atoms[:polymer.polymer_count]
    elements = polymer.elements[:polymer.polymer_count]

    # Keep only chains that have residues (polymer chains)
    chain_mask = polymer.lengths > 0
    lengths = polymer.lengths[chain_mask]
    names = filter_by_mask(polymer.names, chain_mask)
    strands = filter_by_mask(polymer.strands, chain_mask)

    # Calculate chain sizes from residue sizes
    chn_sizes = polymer.rreduce(polymer._sizes[Scale.RESIDUE], Scale.CHAIN, Reduction.SUM)
    chn_sizes = chn_sizes[chain_mask]

    sizes = {
        Scale.RESIDUE: polymer._sizes[Scale.RESIDUE],  # Unchanged
        Scale.CHAIN: chn_sizes,
        Scale.MOLECULE: ops.array([polymer.polymer_count], like=polymer.coordinates),
    }

    # Filter molecule types if available
    mol_types = polymer._molecule_types[chain_mask] if polymer._molecule_types is not None else None

    return Polymer(
        coordinates, atoms, elements, polymer.sequence, sizes,
        polymer.pdb_id, names, strands, lengths, polymer.polymer_count,
        mol_types,
    )


def hetero(polymer: Polymer) -> Polymer:
    """
    Return non-polymer atoms only (HETATM: water, ions, ligands).

    Warning:
        The returned Polymer has no valid residue information.
        Residue-scale operations like reduce(scale=Scale.RESIDUE)
        will return empty results.

    Args:
        polymer: Source polymer.

    Returns:
        New Polymer with only HETATM atoms. If there are no HETATM atoms,
        returns a Polymer with 0 atoms.

    Example:
        >>> p = load("file.cif")
        >>> ligands = hetero(p)  # Get waters/ions/ligands
        >>> if not ligands.empty():
        ...     ligands.center(Scale.ATOM)  # Works on atom scale
    """
    return polymer[polymer.polymer_count:]


def chains(
    polymer: Polymer,
    mol: Molecule | None = None,
) -> Generator[Polymer, None, None]:
    """
    Iterate over chains, optionally filtered by type.

    Args:
        polymer: Source polymer.
        mol: Optional molecule type filter.

    Yields:
        Individual chain Polymers.
    """
    from .filters import by_index

    for ix in range(polymer.size(Scale.CHAIN)):
        chain = by_index(polymer, ix)
        if mol is None or chain.istype(mol):
            yield chain
