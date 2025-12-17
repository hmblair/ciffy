"""
Z-matrix representation and internal coordinate computation.

A Z-matrix represents molecular geometry using internal coordinates:
bond lengths, bond angles, and dihedral angles, relative to reference atoms.
This module provides:
- ZMatrix class: primary data structure for internal coordinate representation
- cartesian_to_internal: conversion from Cartesian to internal coordinates
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .topology import TopologyInfo

from ..backend import Array, is_torch, to_numpy, to_torch
from ..backend.dispatch import cartesian_to_internal


# =============================================================================
# ZMATRIX CLASS
# =============================================================================


class ZMatrix:
    """
    Z-matrix representation as (M, 4) array.

    Each row defines how an atom is placed relative to reference atoms:
    - Column 0: atom_idx - the atom being placed
    - Column 1: distance_ref - reference for bond length (-1 if none)
    - Column 2: angle_ref - reference for bond angle (-1 if none)
    - Column 3: dihedral_ref - reference for dihedral angle (-1 if none)

    Entries are in BFS order, so references always point to earlier atoms.

    Example:
        >>> zmatrix = ZMatrix.from_topology(topology)
        >>> print(len(zmatrix))  # Number of atoms in Z-matrix
        >>> print(zmatrix.atom_indices)  # Column 0
        >>> print(zmatrix[0])  # First row as array
    """

    __slots__ = ('_indices', '_dihedral_types', '_levels', '_level_offsets')

    def __init__(
        self,
        indices: Array,
        dihedral_types: Array | None = None,
        levels: Array | None = None,
    ) -> None:
        """
        Initialize Z-matrix from indices array.

        Args:
            indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 array mapping entry -> dihedral type (-1 if unnamed)
            levels: (M,) int32 array of BFS levels for parallel NERF reconstruction
        """
        self._indices = indices
        self._dihedral_types = dihedral_types
        self._levels = levels
        self._level_offsets = None  # Computed lazily

    @classmethod
    def from_topology(
        cls,
        topology: "TopologyInfo",
        csr_offsets: np.ndarray | None = None,
        csr_neighbors: np.ndarray | None = None,
    ) -> "ZMatrix":
        """
        Build Z-matrix from topology info using BFS traversal.

        Processes each chain independently with its own spanning tree.
        Returns entries in BFS order so references always point to
        earlier (already placed) atoms. The C extension performs
        dihedral-aware reference selection in a single pass.

        Args:
            topology: TopologyInfo containing structural metadata.
            csr_offsets: Optional pre-built CSR offsets array. If None, built from topology.
            csr_neighbors: Optional pre-built CSR neighbors array. If None, built from topology.

        Returns:
            ZMatrix with entries in placement order, dihedral type annotations, and BFS levels.
        """
        from .graph import _build_zmatrix_indices_from_topology

        # Build Z-matrix with dihedral-aware refs in single C pass
        indices, dihedral_types, levels = _build_zmatrix_indices_from_topology(
            topology, csr_offsets, csr_neighbors
        )

        if len(indices) == 0:
            return cls(indices, np.array([], dtype=np.int8), np.array([], dtype=np.int32))

        return cls(indices, dihedral_types, levels)

    @property
    def indices(self) -> Array:
        """Raw (M, 4) array."""
        return self._indices

    @property
    def atom_indices(self) -> Array:
        """Column 0: atom indices being placed."""
        return self._indices[:, 0]

    @property
    def distance_refs(self) -> Array:
        """Column 1: distance reference atoms (-1 for first atom)."""
        return self._indices[:, 1]

    @property
    def angle_refs(self) -> Array:
        """Column 2: angle reference atoms (-1 for first two atoms)."""
        return self._indices[:, 2]

    @property
    def dihedral_refs(self) -> Array:
        """Column 3: dihedral reference atoms (-1 for first three atoms)."""
        return self._indices[:, 3]

    @property
    def dihedral_types(self) -> Array | None:
        """(M,) int8 array mapping Z-matrix entry -> dihedral type (-1 if unnamed)."""
        return self._dihedral_types

    @property
    def levels(self) -> Array | None:
        """(M,) int32 BFS level per entry (0 for root atoms)."""
        return self._levels

    @property
    def level_offsets(self) -> Array:
        """
        (n_levels+1,) int32 cumulative count per level for parallel NERF.

        Level i's entries span indices level_offsets[i]:level_offsets[i+1].
        Computed lazily on first access.
        """
        if self._level_offsets is None and self._levels is not None:
            self._level_offsets = self._compute_level_offsets()
        return self._level_offsets

    def _compute_level_offsets(self) -> np.ndarray:
        """
        Convert per-entry levels to CSR-style offsets.

        Returns (n_levels+1,) int32 array where level i's entries span
        indices level_offsets[i]:level_offsets[i+1].

        Z-matrix entries are sorted by level at construction time, so
        bincount gives correct CSR offsets directly.
        """
        levels = to_numpy(self._levels)
        if len(levels) == 0:
            return np.zeros(1, dtype=np.int32)

        n_levels = int(levels.max()) + 1
        counts = np.bincount(levels, minlength=n_levels).astype(np.int32)
        offsets = np.zeros(n_levels + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:])
        return offsets

    def __len__(self) -> int:
        """Number of entries in Z-matrix."""
        return len(self._indices)

    def __getitem__(self, idx) -> Array:
        """Index into the Z-matrix array."""
        return self._indices[idx]

    def validate(self) -> None:
        """
        Validate Z-matrix structure using vectorized operations.

        Checks that:
        - All reference atoms are either -1 or point to earlier atoms
        - Reference progression is correct (dist before angle before dihedral)

        Raises:
            ValueError: If validation fails.
        """
        n_entries = len(self._indices)
        if n_entries == 0:
            return

        # Extract columns
        atom_indices = self._indices[:, 0].astype(np.int64)
        dist_refs = self._indices[:, 1].astype(np.int64)
        ang_refs = self._indices[:, 2].astype(np.int64)
        dih_refs = self._indices[:, 3].astype(np.int64)

        # Build entry_order: atom_idx -> entry position where it was placed
        # Atoms not in Z-matrix get n_entries (meaning "not placed")
        max_atom = int(atom_indices.max()) + 1
        entry_order = np.full(max_atom, n_entries, dtype=np.int64)
        entry_order[atom_indices] = np.arange(n_entries)

        # Entry positions for comparison
        entry_positions = np.arange(n_entries, dtype=np.int64)

        # Check distance references: ref must be placed before current entry
        valid_dist = dist_refs >= 0
        dist_ref_entries = np.where(
            valid_dist & (dist_refs < max_atom),
            entry_order[np.clip(dist_refs, 0, max_atom - 1)],
            -1  # Invalid refs default to -1 (always < entry position)
        )
        dist_violations = valid_dist & (dist_ref_entries >= entry_positions)
        if np.any(dist_violations):
            first = int(np.argmax(dist_violations))
            raise ValueError(f"Entry {first}: distance_ref {dist_refs[first]} not yet placed")

        # Check angle references
        valid_ang = ang_refs >= 0
        ang_ref_entries = np.where(
            valid_ang & (ang_refs < max_atom),
            entry_order[np.clip(ang_refs, 0, max_atom - 1)],
            -1
        )
        ang_violations = valid_ang & (ang_ref_entries >= entry_positions)
        if np.any(ang_violations):
            first = int(np.argmax(ang_violations))
            raise ValueError(f"Entry {first}: angle_ref {ang_refs[first]} not yet placed")

        # Check dihedral references
        valid_dih = dih_refs >= 0
        dih_ref_entries = np.where(
            valid_dih & (dih_refs < max_atom),
            entry_order[np.clip(dih_refs, 0, max_atom - 1)],
            -1
        )
        dih_violations = valid_dih & (dih_ref_entries >= entry_positions)
        if np.any(dih_violations):
            first = int(np.argmax(dih_violations))
            raise ValueError(f"Entry {first}: dihedral_ref {dih_refs[first]} not yet placed")

        # Check progression: can't have angle without distance
        invalid_progression_ang = valid_ang & (dist_refs < 0)
        if np.any(invalid_progression_ang):
            first = int(np.argmax(invalid_progression_ang))
            raise ValueError(f"Entry {first}: has angle_ref but no distance_ref")

        # Check progression: can't have dihedral without angle
        invalid_progression_dih = valid_dih & (ang_refs < 0)
        if np.any(invalid_progression_dih):
            first = int(np.argmax(invalid_progression_dih))
            raise ValueError(f"Entry {first}: has dihedral_ref but no angle_ref")

    def numpy(self) -> "ZMatrix":
        """Convert indices to NumPy array."""
        dihedral_types = to_numpy(self._dihedral_types) if self._dihedral_types is not None else None
        levels = to_numpy(self._levels) if self._levels is not None else None
        return ZMatrix(to_numpy(self._indices), dihedral_types, levels)

    def torch(self) -> "ZMatrix":
        """Convert indices to PyTorch tensor."""
        dihedral_types = to_torch(self._dihedral_types) if self._dihedral_types is not None else None
        levels = to_torch(self._levels) if self._levels is not None else None
        return ZMatrix(to_torch(self._indices), dihedral_types, levels)

    def to(self, device: str) -> "ZMatrix":
        """Move to specified device (PyTorch only)."""
        if not is_torch(self._indices):
            raise RuntimeError("to() requires PyTorch backend")
        dihedral_types = self._dihedral_types.to(device) if self._dihedral_types is not None else None
        levels = self._levels.to(device) if self._levels is not None else None
        return ZMatrix(self._indices.to(device), dihedral_types, levels)

    def __repr__(self) -> str:
        backend = "torch" if is_torch(self._indices) else "numpy"
        return f"ZMatrix({len(self)} entries, {backend})"
