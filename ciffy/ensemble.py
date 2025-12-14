"""
Conformational ensemble of a single residue type.

Provides a convenient wrapper around extracted coordinates for statistical
learning on molecular conformations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .backend import Array, is_torch, to_numpy

if TYPE_CHECKING:
    from .polymer import Polymer
    from .utils.enum_base import ResidueType


@dataclass
class Ensemble:
    """
    A conformational ensemble of a single residue type.

    Bundles coordinates, atom indices, and residue type together for
    convenient manipulation and conversion back to Polymer format.

    Attributes:
        coords: Dense coordinate array of shape (n_instances, n_atoms, 3).
        atoms: List of atom type indices in canonical order.
        residue: The ResidueType for this ensemble.

    Examples:
        Extract an ensemble from a structure::

            from ciffy import load
            from ciffy.biochemistry import Residue
            from ciffy.ensemble import Ensemble

            poly = load("ribosome.cif")
            ens = Ensemble.from_polymer(poly, Residue.A, align=True, scale=True)
            print(ens.coords.shape)  # (n_adenosines, n_atoms, 3)

        Process and convert back to Polymer::

            # Run through a model
            ens.coords = model(ens.coords)

            # Save as CIF
            ens.to_polymer().write("output.cif")

        Access atom names::

            print(ens.atom_names)  # ['P', 'OP1', 'OP2', 'O5p', ...]
    """

    coords: Array
    atoms: list[int]
    residue: "ResidueType"

    def __len__(self) -> int:
        """Number of residue instances in the ensemble."""
        return self.coords.shape[0]

    def __getitem__(self, idx: int | slice | list) -> "Ensemble":
        """
        Select conformations by index.

        Args:
            idx: Integer index, slice, or list of indices.

        Returns:
            New Ensemble with selected conformations.

        Examples:
            >>> ens[0]           # Single conformation
            >>> ens[:10]         # First 10
            >>> ens[[0, 5, 10]]  # Specific indices
        """
        coords = self.coords[idx]
        # Ensure 3D shape even for single index
        if coords.ndim == 2:
            coords = coords[None, ...]
        return Ensemble(coords=coords, atoms=self.atoms, residue=self.residue)

    def with_coords(self, coords: Array) -> "Ensemble":
        """
        Create a new Ensemble with different coordinates.

        Keeps the same atoms and residue type, but replaces coordinates.
        Useful for visualizing sampled or generated conformations.

        Args:
            coords: New coordinates. Can be:
                - (n_atoms, 3) for a single conformation
                - (n, n_atoms, 3) for multiple conformations

        Returns:
            New Ensemble with the given coordinates.

        Example:
            >>> # Sample a new conformation from a model
            >>> new_coords = model.sample()  # (22, 3)
            >>> sampled = ens.with_coords(new_coords)
            >>> sampled.to_polymer().write("sampled.cif")
        """
        # Handle torch tensors
        if is_torch(coords):
            shape = tuple(coords.shape)
        else:
            shape = coords.shape

        # Ensure 3D
        if len(shape) == 2:
            coords = coords[None, ...]

        # Validate atom count
        if coords.shape[1] != self.n_atoms:
            raise ValueError(
                f"Expected {self.n_atoms} atoms, got {coords.shape[1]}"
            )

        return Ensemble(coords=coords, atoms=self.atoms, residue=self.residue)

    @property
    def n_atoms(self) -> int:
        """Number of atoms per residue."""
        return len(self.atoms)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the coordinate array (n_instances, n_atoms, 3)."""
        return self.coords.shape

    @property
    def atom_names(self) -> list[str]:
        """List of atom names in column order."""
        idx_to_member = {m.value: m for m in self.residue.atoms}
        return [idx_to_member[a].name for a in self.atoms]

    @classmethod
    def from_polymer(
        cls,
        poly: "Polymer",
        residue: "ResidueType",
        atoms: list | None = None,
        center: bool = False,
        align: bool = False,
        scale: bool = False,
    ) -> "Ensemble":
        """
        Extract an ensemble from a Polymer structure.

        Args:
            poly: Polymer structure to extract from.
            residue: Residue type to extract (e.g., Residue.A).
            atoms: Optional list of specific atoms to include.
            center: If True, center each residue at the origin.
            align: If True, align each residue to principal axes.
            scale: If True, scale each residue to unit standard deviation.

        Returns:
            Ensemble containing all instances of the residue type.
        """
        from .operations.extract import extract

        coords, atom_indices = extract(
            poly, residue, atoms=atoms, center=center, align=align, scale=scale
        )
        return cls(coords=coords, atoms=atom_indices, residue=residue)

    def to_polymer(self, id: str = "ensemble", backend: str = "numpy") -> "Polymer":
        """
        Convert the ensemble back to a Polymer.

        Each instance becomes a separate residue in a single chain.

        Args:
            id: PDB identifier for the output polymer.
            backend: Array backend ("numpy" or "torch").

        Returns:
            Polymer with the ensemble coordinates.
        """
        from .template import from_extract

        return from_extract(
            self.coords, self.atoms, self.residue, backend=backend, id=id
        )

    def numpy(self) -> "Ensemble":
        """Return ensemble with NumPy coordinates."""
        return Ensemble(coords=to_numpy(self.coords), atoms=self.atoms, residue=self.residue)

    def torch(self, device=None) -> "Ensemble":
        """Return ensemble with PyTorch coordinates."""
        import torch

        if is_torch(self.coords):
            coords = self.coords
        else:
            coords = torch.from_numpy(np.asarray(self.coords))

        if device is not None:
            coords = coords.to(device)

        return Ensemble(coords=coords, atoms=self.atoms, residue=self.residue)

    def __repr__(self) -> str:
        n_inst, n_atoms, _ = self.coords.shape
        backend = "torch" if is_torch(self.coords) else "numpy"
        return (
            f"Ensemble({self.residue.name}, "
            f"n={n_inst}, atoms={n_atoms}, backend={backend})"
        )
