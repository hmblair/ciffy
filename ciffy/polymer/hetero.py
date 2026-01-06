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
        ...     waters = hetero.by_element(8)  # Oxygen atoms (likely water)
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

    def _clone(self, **overrides) -> "HeteroAtoms":
        """
        Create a copy of this HeteroAtoms with optional field overrides.

        Args:
            **overrides: Field values to override.

        Returns:
            New HeteroAtoms with the specified overrides applied.
        """
        # Extract hierarchy
        hierarchy = overrides.pop('hierarchy', object.__getattribute__(self, '_hierarchy'))

        # Extract field metadata (for dynamic fields from slicing/conversion)
        field_meta = overrides.pop('_field_meta', None)

        # Create new instance bypassing __init__ for efficiency
        hetero = object.__new__(HeteroAtoms)
        object.__setattr__(hetero, '_hierarchy', hierarchy)

        # Reconstruct Fields
        current_fields = self._get_fields()
        for name, field in current_fields.items():
            # Get scale from override metadata or original field
            if field_meta and name in field_meta:
                scale = field_meta[name]
            else:
                scale = field.scale

            # Get data from override or original
            if name in overrides:
                data = overrides.pop(name)
            else:
                data = field.data

            # Only set attribute if data exists
            if data is not None:
                new_field = Field(data, scale)
                object.__setattr__(hetero, name, new_field)

        # Copy Metadata descriptors
        for name, desc in self._get_metadata().items():
            if name in overrides:
                value = overrides.pop(name)
            else:
                value = getattr(self, desc.private_name, None)
            setattr(hetero, desc.private_name, value)

        return hetero

    # ─────────────────────────────────────────────────────────────────────────
    # HeteroAtoms-specific Selection Methods
    # ─────────────────────────────────────────────────────────────────────────

    def by_element(self, element: int | Array) -> "HeteroAtoms":
        """
        Filter by element type.

        Args:
            element: Element index or array of element indices to select.

        Returns:
            New HeteroAtoms with only atoms of the specified element(s).

        Example:
            >>> hetero.by_element(8)   # Oxygen atoms
            >>> hetero.by_element(11)  # Sodium ions
        """
        # Handle single int vs array
        if isinstance(element, (int, np.integer)):
            mask = self.elements == element
        else:
            element = ops.convert_backend(element, self.elements)
            mask = (self.elements[:, None] == element).any(1)
        return self[mask]

    def by_chain(self, chain_idx: int) -> "HeteroAtoms":
        """
        Select atoms belonging to a specific chain.

        Args:
            chain_idx: Chain index to select.

        Returns:
            New HeteroAtoms with only atoms from the specified chain.

        Example:
            >>> hetero_chain0 = hetero.by_chain(0)
        """
        mask = self.chains == chain_idx
        return self[mask]

    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def create_empty(cls, pdb_id: str = "empty", backend: str = "numpy") -> "HeteroAtoms":
        """
        Create an empty HeteroAtoms container.

        Args:
            pdb_id: Molecule identifier.
            backend: Array backend ('numpy' or 'torch').

        Returns:
            HeteroAtoms with 0 atoms.
        """
        if backend == "numpy":
            coords = np.empty((0, 3), dtype=np.float32)
            elements = np.empty((0,), dtype=np.int64)
            chains = np.empty((0,), dtype=np.int64)
        else:
            import torch
            coords = torch.empty((0, 3), dtype=torch.float32)
            elements = torch.empty((0,), dtype=torch.int64)
            chains = torch.empty((0,), dtype=torch.int64)

        # Compute atoms_per_chain (empty array since no chains)
        if backend == "numpy":
            atoms_per_chain = np.empty((0,), dtype=np.int64)
        else:
            atoms_per_chain = torch.empty((0,), dtype=torch.int64)

        hierarchy = _Hierarchy.from_atoms_per_chain(atoms_per_chain, ref=coords)

        return cls(
            hierarchy,
            coordinates=Field(coords, Scale.ATOM),
            elements=Field(elements, Scale.ATOM),
            chains=Field(chains, Scale.ATOM),
            pdb_id=pdb_id,
        )

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
            # Get max chain index
            chains_np = np.asarray(chains)
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
