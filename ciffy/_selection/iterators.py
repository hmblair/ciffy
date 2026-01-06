"""
Iteration and partitioning functions for Polymer objects.

Functions for iterating over chains and partitioning polymers.
"""

from __future__ import annotations
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..hetero import HeteroAtoms

from ..backend import ops
from ..biochemistry import Scale


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
    # Polymer now only contains polymer atoms (HETATM is separate)
    return polymer


def hetero(polymer: Polymer) -> "HeteroAtoms":
    """
    Return non-polymer atoms only (HETATM: water, ions, ligands).

    Returns a lightweight HeteroAtoms container with only atom-level data.
    Unlike Polymer, HeteroAtoms has no residue or chain hierarchy.

    Args:
        polymer: Source polymer.

    Returns:
        HeteroAtoms container with HETATM atoms. If there are no HETATM atoms,
        returns an empty HeteroAtoms.

    Example:
        >>> p = load("file.cif")
        >>> hetero_atoms = hetero(p)  # Get waters/ions/ligands
        >>> if not hetero_atoms.empty():
        ...     waters = hetero_atoms.by_element(8)  # Oxygen atoms
    """
    from ..hetero import HeteroAtoms

    # Return stored HETATM data if available
    if polymer._hetero is not None:
        return polymer._hetero

    # No HETATM data - return empty container
    return HeteroAtoms.create_empty(polymer.pdb_id, polymer.backend)


def chains(polymer: Polymer) -> Generator[Polymer, None, None]:
    """
    Iterate over chains.

    To filter by molecule type, use `polymer.by_type(mol).chains()`.

    Args:
        polymer: Source polymer.

    Yields:
        Individual chain Polymers.
    """
    from .filters import by_index

    for ix in range(polymer.size(Scale.CHAIN)):
        yield by_index(polymer, ix)
