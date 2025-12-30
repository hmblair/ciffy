"""
Lightweight container for non-polymer atoms (HETATM).

This module provides the HeteroAtoms class for storing and manipulating
non-polymer atoms such as water molecules, ions, and ligands.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .backend import (
    Array,
    get_backend,
    get_device,
    is_numpy,
    is_torch,
    to_numpy,
    to_torch,
    size as arr_size,
)
from .backend import ops

if TYPE_CHECKING:
    pass


@dataclass
class HeteroAtoms:
    """
    Lightweight container for non-polymer atoms (HETATM).

    Stores coordinates and atom information for water molecules, ions,
    and ligands. Unlike Polymer, HeteroAtoms has no residue or chain
    hierarchy - it's simply a collection of atoms.

    Attributes:
        coordinates: (H, 3) array of atom positions.
        atoms: (H,) array of atom type indices.
        elements: (H,) array of element indices.
        bfactors: Optional (H,) array of B-factors.
        pdb_id: Molecule identifier.

    Example:
        >>> p = load("file.cif")
        >>> hetero = p.hetero()
        >>> if not hetero.empty():
        ...     waters = hetero.by_element(8)  # Oxygen atoms (likely water)
        ...     print(f"Found {waters.size()} oxygen atoms")
    """

    coordinates: Array
    atoms: Array
    elements: Array
    bfactors: Array | None
    pdb_id: str

    def size(self) -> int:
        """
        Get the number of atoms.

        Returns:
            Number of atoms in this container.
        """
        return arr_size(self.coordinates, 0)

    def empty(self) -> bool:
        """
        Check if there are no atoms.

        Returns:
            True if there are no atoms, False otherwise.
        """
        return self.size() == 0

    @property
    def backend(self) -> str:
        """
        Get the array backend type.

        Returns:
            'numpy' if arrays are NumPy, 'torch' if PyTorch tensors.
        """
        return get_backend(self.coordinates).value

    @property
    def device(self) -> str | None:
        """
        Get the device of the arrays.

        Returns:
            Device string (e.g., 'cpu', 'cuda:0') for PyTorch tensors,
            None for NumPy arrays.
        """
        return get_device(self.coordinates)

    def numpy(self) -> HeteroAtoms:
        """
        Convert all arrays to NumPy.

        Returns:
            New HeteroAtoms with NumPy arrays. If already NumPy, returns self.
        """
        if is_numpy(self.coordinates):
            return self

        return HeteroAtoms(
            coordinates=to_numpy(self.coordinates),
            atoms=to_numpy(self.atoms),
            elements=to_numpy(self.elements),
            bfactors=to_numpy(self.bfactors) if self.bfactors is not None else None,
            pdb_id=self.pdb_id,
        )

    def torch(self) -> HeteroAtoms:
        """
        Convert all arrays to PyTorch tensors.

        Returns:
            New HeteroAtoms with PyTorch tensors. If already PyTorch, returns self.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if is_torch(self.coordinates):
            return self

        return HeteroAtoms(
            coordinates=to_torch(self.coordinates).float(),
            atoms=to_torch(self.atoms).long(),
            elements=to_torch(self.elements).long(),
            bfactors=to_torch(self.bfactors).float() if self.bfactors is not None else None,
            pdb_id=self.pdb_id,
        )

    def to(
        self,
        device: "str | torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ) -> HeteroAtoms:
        """
        Move tensors to device and/or convert dtype (torch backend only).

        Args:
            device: Target device (e.g., 'cuda', 'cpu').
            dtype: Target dtype for float tensors only.

        Returns:
            New HeteroAtoms with tensors on the specified device/dtype.

        Raises:
            ValueError: If called on NumPy backend.
        """
        if not is_torch(self.coordinates):
            raise ValueError("to() is only supported for torch backend. "
                           "Use hetero.torch().to(...) to convert first.")

        if device is None and dtype is None:
            return self

        # For coordinates (float), apply both device and dtype
        coords = self.coordinates
        if device is not None:
            coords = coords.to(device)
        if dtype is not None:
            coords = coords.to(dtype)

        # For integer tensors, only apply device
        def move_int(t):
            return t.to(device) if device is not None else t

        # For float tensors, apply device and dtype
        def move_float(t):
            if t is None:
                return None
            result = t
            if device is not None:
                result = result.to(device)
            if dtype is not None:
                result = result.to(dtype)
            return result

        return HeteroAtoms(
            coordinates=coords,
            atoms=move_int(self.atoms),
            elements=move_int(self.elements),
            bfactors=move_float(self.bfactors),
            pdb_id=self.pdb_id,
        )

    def cpu(self) -> HeteroAtoms:
        """Move to CPU (torch backend only)."""
        return self.to(device="cpu")

    def cuda(self) -> HeteroAtoms:
        """Move to CUDA (torch backend only)."""
        return self.to(device="cuda")

    def __getitem__(self, mask: Array) -> HeteroAtoms:
        """
        Select atoms by boolean mask or indices.

        Args:
            mask: Boolean mask or integer indices.

        Returns:
            New HeteroAtoms with selected atoms.

        Example:
            >>> hetero = polymer.hetero()
            >>> subset = hetero[hetero.elements == 8]  # Oxygen atoms
        """
        return HeteroAtoms(
            coordinates=self.coordinates[mask],
            atoms=self.atoms[mask],
            elements=self.elements[mask],
            bfactors=self.bfactors[mask] if self.bfactors is not None else None,
            pdb_id=self.pdb_id,
        )

    def by_element(self, element: int | Array) -> HeteroAtoms:
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
        import numpy as np

        # Handle single int vs array
        if isinstance(element, (int, np.integer)):
            mask = self.elements == element
        else:
            element = ops.convert_backend(element, self.elements)
            mask = (self.elements[:, None] == element).any(1)
        return self[mask]

    @classmethod
    def create_empty(cls, pdb_id: str = "empty", backend: str = "numpy") -> HeteroAtoms:
        """
        Create an empty HeteroAtoms container.

        Args:
            pdb_id: Molecule identifier.
            backend: Array backend ('numpy' or 'torch').

        Returns:
            HeteroAtoms with 0 atoms.
        """
        if backend == "numpy":
            return cls(
                coordinates=np.empty((0, 3), dtype=np.float32),
                atoms=np.empty((0,), dtype=np.int64),
                elements=np.empty((0,), dtype=np.int64),
                bfactors=None,
                pdb_id=pdb_id,
            )
        else:
            import torch
            return cls(
                coordinates=torch.empty((0, 3), dtype=torch.float32),
                atoms=torch.empty((0,), dtype=torch.int64),
                elements=torch.empty((0,), dtype=torch.int64),
                bfactors=None,
                pdb_id=pdb_id,
            )

    def __repr__(self) -> str:
        """String representation."""
        if self.empty():
            return f"HeteroAtoms({self.pdb_id}, empty)"
        return f"HeteroAtoms({self.pdb_id}, {self.size()} atoms, {self.backend})"

    def __len__(self) -> int:
        """Return number of atoms."""
        return self.size()
