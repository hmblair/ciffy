"""
Lightweight container for non-polymer atoms (HETATM).

This module provides the HeteroAtoms class for storing and manipulating
non-polymer atoms such as water molecules, ions, and ligands.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch, to_numpy
from ..backend import ops
from ..biochemistry import Scale

if TYPE_CHECKING:
    import torch

from .hierarchy import _Hierarchy
from .base import AtomContainer, Field


class HeteroAtoms(AtomContainer):
    """
    Lightweight container for non-polymer atoms (HETATM).

    Stores coordinates and atom information for water molecules, ions,
    and ligands. Unlike Polymer, HeteroAtoms has no residue hierarchy -
    it only has ATOM and CHAIN scales.

    The `chains` field stores per-atom chain membership (which chain each
    atom belongs to), enabling chain-level operations.

    Attributes:
        coordinates: (H, 3) array of atom positions.
        elements: (H,) array of element indices.
        chains: (H,) array of chain indices (per-atom chain membership).
        bfactors: Optional (H,) array of B-factors.
        pdb_id: Molecule identifier (inherited from AtomContainer).

    Example:
        >>> p = load("file.cif")
        >>> hetero = p.hetero()
        >>> if not hetero.empty():
        ...     waters = hetero.element_type(8)  # Oxygen atoms (likely water)
        ...     print(f"Found {waters.size()} oxygen atoms")
    """

    # HeteroAtoms does not support RESIDUE scale
    _allowed_scales = {Scale.ATOM, Scale.CHAIN, Scale.MOLECULE}

    def _init_from_kwargs(self, kwargs: dict) -> None:
        """Handle any extra kwargs (none expected for HeteroAtoms)."""
        if kwargs:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # HeteroAtoms-specific Selection Methods
    # ─────────────────────────────────────────────────────────────────────────

    def element_type(self, element: int | Array) -> "HeteroAtoms":
        """
        Filter by element type.

        Args:
            element: Element index or array of element indices to select.

        Returns:
            New HeteroAtoms with only atoms of the specified element(s).

        Example:
            >>> hetero.element_type(8)   # Oxygen atoms
            >>> hetero.element_type(11)  # Sodium ions
        """
        # Handle single int vs array
        if isinstance(element, (int, np.integer)):
            mask = self.elements == element
        else:
            element = ops.convert_backend(element, self.elements)
            mask = (self.elements[:, None] == element).any(1)
        return self[mask]

    def chain(self, chain_idx: int) -> "HeteroAtoms":
        """
        Select atoms belonging to a specific chain.

        Args:
            chain_idx: Chain index to select.

        Returns:
            New HeteroAtoms with only atoms from the specified chain.

        Example:
            >>> hetero_chain0 = hetero.chain(0)
        """
        mask = self.chains == chain_idx
        return self[mask]

    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_arrays(
        cls,
        coordinates: Array,
        elements: Array,
        chains: Array,
        bfactors: Array | None = None,
        pdb_id: str = "",
    ) -> "HeteroAtoms":
        """
        Create HeteroAtoms from raw arrays.

        This is the main factory method for creating HeteroAtoms objects.
        It computes the hierarchy from the per-atom chain assignments.

        Args:
            coordinates: (H, 3) array of atom positions.
            elements: (H,) array of element indices.
            chains: (H,) array of per-atom chain indices.
            bfactors: Optional (H,) array of B-factors.
            pdb_id: Molecule identifier.

        Returns:
            HeteroAtoms object.
        """
        # Compute atoms_per_chain from chains array
        # chains is per-atom, so we need to count atoms per unique chain
        if len(chains) == 0:
            atoms_per_chain = ops.zeros(0, like=coordinates, dtype='int64')
        else:
            # Get max chain index and validate bounds
            chains_np = np.asarray(chains)
            if chains_np.min() < 0:
                raise ValueError(
                    f"Negative chain index found: {chains_np.min()}. "
                    "Chain indices must be non-negative."
                )
            max_chain = int(chains_np.max()) + 1
            # Count atoms per chain using numpy (then convert back)
            counts = np.bincount(chains_np, minlength=max_chain)
            atoms_per_chain = ops.array(counts, like=coordinates, dtype='int64')

        hierarchy = _Hierarchy.from_atoms_per_chain(atoms_per_chain, ref=coordinates)

        kwargs = {
            'coordinates': Field(coordinates, Scale.ATOM),
            'elements': Field(elements, Scale.ATOM),
            'chains': Field(chains, Scale.ATOM),
            'pdb_id': pdb_id,
        }
        if bfactors is not None:
            kwargs['bfactors'] = Field(bfactors, Scale.ATOM)

        return cls(hierarchy, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """String representation."""
        if self.empty():
            return f"HeteroAtoms({self.pdb_id}, empty)"
        return f"HeteroAtoms({self.pdb_id}, {self.size()} atoms, {self.backend})"
