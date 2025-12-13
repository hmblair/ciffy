"""
Internal coordinate representation of molecular structures.

Provides the InternalPolymer class that stores molecular geometry using
internal coordinates (bond lengths, bond angles, dihedral angles) rather
than Cartesian XYZ positions.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict
from copy import copy

import numpy as np

from ..backend import Array, is_torch, to_numpy, to_torch

if TYPE_CHECKING:
    from ..polymer import Polymer
    from .graph import ZMatrixEntry


class InternalPolymer:
    """
    Molecular structure represented in internal coordinates (Z-matrix).

    Stores bond lengths, bond angles, and dihedral angles rather than
    Cartesian XYZ positions. Enables efficient conformational sampling
    and provides differentiable conversion to Cartesian coordinates.

    Attributes:
        distances: (N,) array of bond lengths in Angstroms.
        angles: (N,) array of bond angles in radians.
        dihedrals: (N,) array of dihedral angles in radians.

    Note:
        First atom has no internal coordinates (placed at origin).
        Second atom has only distance (placed along +X axis).
        Third atom has distance and angle (placed in XY plane).
        All subsequent atoms have all three coordinates.

    Example:
        >>> polymer = ciffy.load("structure.cif", backend="torch")
        >>> internal = polymer.to_internal()
        >>>
        >>> # Access backbone dihedrals
        >>> phi = internal.phi    # Protein phi angles
        >>> psi = internal.psi    # Protein psi angles
        >>>
        >>> # Modify and convert back
        >>> internal.dihedrals[100] = 1.5  # Set dihedral to ~86 degrees
        >>> modified = internal.to_cartesian()
    """

    __slots__ = (
        '_distances', '_angles', '_dihedrals',
        '_zmatrix', '_source', '_dihedral_indices',
        '_orphan_atoms', '_orphan_coords'
    )

    def __init__(
        self,
        distances: Array,
        angles: Array,
        dihedrals: Array,
        zmatrix: list["ZMatrixEntry"],
        source_polymer: "Polymer",
        orphan_atoms: Optional[list[int]] = None,
        orphan_coords: Optional[Array] = None,
    ) -> None:
        """
        Initialize from internal coordinate arrays.

        Args:
            distances: Bond lengths (N,). First value is 0 (placeholder).
            angles: Bond angles in radians (N,). First two values are 0.
            dihedrals: Dihedral angles in radians (N,). First three values are 0.
            zmatrix: Z-matrix entries defining the reference frame for each atom.
            source_polymer: Original Polymer for metadata access.
            orphan_atoms: Indices of atoms with no bonds (not in Z-matrix).
            orphan_coords: Original coordinates for orphan atoms.
        """
        self._distances = distances
        self._angles = angles
        self._dihedrals = dihedrals
        self._zmatrix = zmatrix
        self._source = source_polymer
        self._dihedral_indices: Optional[Dict[str, Array]] = None
        self._orphan_atoms = orphan_atoms if orphan_atoms else []
        self._orphan_coords = orphan_coords

    # ─────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def distances(self) -> Array:
        """(N,) array of bond lengths in Angstroms."""
        return self._distances

    @distances.setter
    def distances(self, value: Array) -> None:
        self._distances = value

    @property
    def angles(self) -> Array:
        """(N,) array of bond angles in radians."""
        return self._angles

    @angles.setter
    def angles(self, value: Array) -> None:
        self._angles = value

    @property
    def dihedrals(self) -> Array:
        """(N,) array of dihedral angles in radians."""
        return self._dihedrals

    @dihedrals.setter
    def dihedrals(self, value: Array) -> None:
        self._dihedrals = value

    @property
    def zmatrix(self) -> list["ZMatrixEntry"]:
        """Z-matrix defining coordinate references for each atom."""
        return self._zmatrix

    @property
    def source(self) -> "Polymer":
        """Reference to the source Polymer for metadata."""
        return self._source

    @property
    def size(self) -> int:
        """Number of atoms (including orphans)."""
        return len(self._zmatrix) + len(self._orphan_atoms)

    @property
    def backend(self) -> str:
        """Array backend ('numpy' or 'torch')."""
        return "torch" if is_torch(self._distances) else "numpy"

    # ─────────────────────────────────────────────────────────────────────
    # Conversion Methods
    # ─────────────────────────────────────────────────────────────────────

    def to_cartesian(self) -> "Polymer":
        """
        Convert internal coordinates to Cartesian coordinates.

        Uses the NERF (Natural Extension Reference Frame) algorithm for
        efficient and differentiable reconstruction.

        Returns:
            Polymer with Cartesian coordinates. Preserves all metadata
            from the source polymer (atoms, elements, sequence, etc.).
        """
        from .nerf import nerf_reconstruct

        # Total atoms includes orphans
        total_atoms = self._source.size()

        coords = nerf_reconstruct(
            self._distances,
            self._angles,
            self._dihedrals,
            self._zmatrix,
            n_atoms=total_atoms,
        )

        # Restore orphan atom coordinates (atoms with no bonds)
        if self._orphan_atoms and self._orphan_coords is not None:
            for i, atom_idx in enumerate(self._orphan_atoms):
                if is_torch(coords):
                    coords[atom_idx] = self._orphan_coords[i]
                else:
                    coords[atom_idx] = self._orphan_coords[i]

        result = copy(self._source)
        result.coordinates = coords
        return result

    def with_dihedrals(self, dihedrals: Array) -> "InternalPolymer":
        """
        Create copy with new dihedral angles.

        Args:
            dihedrals: New dihedral array (N,) in radians.

        Returns:
            New InternalPolymer with updated dihedrals.
        """
        return InternalPolymer(
            distances=self._distances,
            angles=self._angles,
            dihedrals=dihedrals,
            zmatrix=self._zmatrix,
            source_polymer=self._source,
            orphan_atoms=self._orphan_atoms,
            orphan_coords=self._orphan_coords,
        )

    def with_angles(self, angles: Array) -> "InternalPolymer":
        """
        Create copy with new bond angles.

        Args:
            angles: New angle array (N,) in radians.

        Returns:
            New InternalPolymer with updated angles.
        """
        return InternalPolymer(
            distances=self._distances,
            angles=angles,
            dihedrals=self._dihedrals,
            zmatrix=self._zmatrix,
            source_polymer=self._source,
            orphan_atoms=self._orphan_atoms,
            orphan_coords=self._orphan_coords,
        )

    def with_distances(self, distances: Array) -> "InternalPolymer":
        """
        Create copy with new bond lengths.

        Args:
            distances: New distance array (N,) in Angstroms.

        Returns:
            New InternalPolymer with updated distances.
        """
        return InternalPolymer(
            distances=distances,
            angles=self._angles,
            dihedrals=self._dihedrals,
            zmatrix=self._zmatrix,
            source_polymer=self._source,
            orphan_atoms=self._orphan_atoms,
            orphan_coords=self._orphan_coords,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Named Dihedral Accessors
    # ─────────────────────────────────────────────────────────────────────

    def backbone_dihedrals(self, name: str) -> Array:
        """
        Get backbone dihedral angles by name.

        Args:
            name: Dihedral name:
                - Proteins: 'phi', 'psi', 'omega'
                - Nucleic acids: 'alpha', 'beta', 'gamma', 'delta',
                                 'epsilon', 'zeta', 'chi'

        Returns:
            Array of dihedral values (one per residue where applicable).
            Missing values (e.g., phi for first residue) are not included.
        """
        if self._dihedral_indices is None:
            self._compute_dihedral_indices()

        indices = self._dihedral_indices.get(name)
        if indices is None:
            raise ValueError(f"Unknown dihedral name: {name}")

        return self._dihedrals[indices]

    def set_backbone_dihedrals(self, name: str, values: Array) -> "InternalPolymer":
        """
        Set backbone dihedral angles by name.

        Args:
            name: Dihedral name ('phi', 'psi', 'omega' for proteins, etc.)
            values: New values in radians.

        Returns:
            New InternalPolymer with updated dihedrals.
        """
        if self._dihedral_indices is None:
            self._compute_dihedral_indices()

        indices = self._dihedral_indices.get(name)
        if indices is None:
            raise ValueError(f"Unknown dihedral name: {name}")

        if is_torch(self._dihedrals):
            new_dihedrals = self._dihedrals.clone()
        else:
            new_dihedrals = self._dihedrals.copy()

        new_dihedrals[indices] = values

        return self.with_dihedrals(new_dihedrals)

    @property
    def phi(self) -> Array:
        """Protein phi angles (C-N-CA-C). Shape: (n_residues-1,)."""
        return self.backbone_dihedrals('phi')

    @property
    def psi(self) -> Array:
        """Protein psi angles (N-CA-C-N). Shape: (n_residues-1,)."""
        return self.backbone_dihedrals('psi')

    @property
    def omega(self) -> Array:
        """Protein omega angles (CA-C-N-CA). Shape: (n_residues-1,)."""
        return self.backbone_dihedrals('omega')

    def _compute_dihedral_indices(self) -> None:
        """Compute and cache indices for named dihedrals."""
        from .dihedrals import compute_dihedral_indices
        self._dihedral_indices = compute_dihedral_indices(
            self._source,
            self._zmatrix,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Backend Conversion
    # ─────────────────────────────────────────────────────────────────────

    def numpy(self) -> "InternalPolymer":
        """Convert all arrays to NumPy."""
        orphan_coords = to_numpy(self._orphan_coords) if self._orphan_coords is not None else None
        return InternalPolymer(
            distances=to_numpy(self._distances),
            angles=to_numpy(self._angles),
            dihedrals=to_numpy(self._dihedrals),
            zmatrix=self._zmatrix,
            source_polymer=self._source.numpy(),
            orphan_atoms=self._orphan_atoms,
            orphan_coords=orphan_coords,
        )

    def torch(self) -> "InternalPolymer":
        """Convert all arrays to PyTorch."""
        orphan_coords = to_torch(self._orphan_coords) if self._orphan_coords is not None else None
        return InternalPolymer(
            distances=to_torch(self._distances),
            angles=to_torch(self._angles),
            dihedrals=to_torch(self._dihedrals),
            zmatrix=self._zmatrix,
            source_polymer=self._source.torch(),
            orphan_atoms=self._orphan_atoms,
            orphan_coords=orphan_coords,
        )

    def to(self, device: str) -> "InternalPolymer":
        """
        Move PyTorch tensors to specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu', 'mps').

        Returns:
            New InternalPolymer with arrays on target device.

        Raises:
            RuntimeError: If backend is not PyTorch.
        """
        if not is_torch(self._distances):
            raise RuntimeError("to() requires PyTorch backend. Use .torch() first.")

        import torch

        orphan_coords = self._orphan_coords.to(device) if self._orphan_coords is not None else None
        return InternalPolymer(
            distances=self._distances.to(device),
            angles=self._angles.to(device),
            dihedrals=self._dihedrals.to(device),
            zmatrix=self._zmatrix,
            source_polymer=self._source.to(device),
            orphan_atoms=self._orphan_atoms,
            orphan_coords=orphan_coords,
        )

    def __repr__(self) -> str:
        return (
            f"InternalPolymer(size={self.size}, backend={self.backend})"
        )
