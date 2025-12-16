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

# C extension (required)
from .._c import _cartesian_to_internal as _c_cartesian_to_internal


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

    __slots__ = ('_indices', '_dihedral_types')

    def __init__(self, indices: Array, dihedral_types: Array | None = None) -> None:
        """
        Initialize Z-matrix from indices array.

        Args:
            indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 array mapping entry -> dihedral type (-1 if unnamed)
        """
        self._indices = indices
        self._dihedral_types = dihedral_types

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
            ZMatrix with entries in placement order and dihedral type annotations.
        """
        from .graph import _build_zmatrix_indices_from_topology

        # Build Z-matrix with dihedral-aware refs in single C pass
        indices, dihedral_types = _build_zmatrix_indices_from_topology(
            topology, csr_offsets, csr_neighbors
        )

        if len(indices) == 0:
            return cls(indices, np.array([], dtype=np.int8))

        return cls(indices, dihedral_types)

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
        return ZMatrix(to_numpy(self._indices), dihedral_types)

    def torch(self) -> "ZMatrix":
        """Convert indices to PyTorch tensor."""
        dihedral_types = to_torch(self._dihedral_types) if self._dihedral_types is not None else None
        return ZMatrix(to_torch(self._indices), dihedral_types)

    def to(self, device: str) -> "ZMatrix":
        """Move to specified device (PyTorch only)."""
        if not is_torch(self._indices):
            raise RuntimeError("to() requires PyTorch backend")
        dihedral_types = self._dihedral_types.to(device) if self._dihedral_types is not None else None
        return ZMatrix(self._indices.to(device), dihedral_types)

    def __repr__(self) -> str:
        backend = "torch" if is_torch(self._indices) else "numpy"
        return f"ZMatrix({len(self)} entries, {backend})"


# =============================================================================
# CARTESIAN TO INTERNAL CONVERSION
# =============================================================================


def cartesian_to_internal(
    coords: Array,
    zmatrix_indices: Array,
) -> tuple[Array, Array, Array]:
    """
    Convert Cartesian coordinates to internal coordinates.

    Uses CUDA kernels when available for GPU tensors, otherwise falls back
    to CPU C extension. For PyTorch tensors that require gradients, uses
    autograd functions with backward passes.

    Args:
        coords: (N, 3) array of Cartesian coordinates.
        zmatrix_indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]

    Returns:
        Tuple of (distances, angles, dihedrals), each (M,) float32.
    """
    # For PyTorch tensors, check for CUDA or autograd path
    if is_torch(coords):
        import torch
        from ..backend.cuda_ops import is_cuda_available, cuda_cartesian_to_internal

        # Ensure indices are on same device as coords
        if not is_torch(zmatrix_indices):
            indices_tensor = torch.from_numpy(zmatrix_indices).to(coords.device)
        else:
            indices_tensor = zmatrix_indices.to(coords.device)

        # Use autograd path for tensors requiring gradients
        if coords.requires_grad:
            from ..backend.autograd import cartesian_to_internal as autograd_c2i
            return autograd_c2i(coords, indices_tensor)

        # Use CUDA kernels for GPU tensors (inference mode)
        if is_cuda_available(coords):
            return cuda_cartesian_to_internal(
                coords.to(torch.float32).contiguous(),
                indices_tensor.to(torch.int64).contiguous()
            )

        # CPU PyTorch tensor: use C extension
        device = coords.device
        dtype = coords.dtype
        coords_f32 = coords.detach().cpu().to(torch.float32).numpy()
        indices_np = indices_tensor.cpu().numpy().astype(np.int64)

        distances_np, angles_np, dihedrals_np = _c_cartesian_to_internal(
            coords_f32, indices_np
        )

        distances = torch.from_numpy(distances_np).to(device=device, dtype=dtype)
        angles = torch.from_numpy(angles_np).to(device=device, dtype=dtype)
        dihedrals = torch.from_numpy(dihedrals_np).to(device=device, dtype=dtype)
        return distances, angles, dihedrals

    # NumPy path
    if is_torch(zmatrix_indices):
        indices_np = zmatrix_indices.cpu().numpy()
    else:
        indices_np = np.asarray(zmatrix_indices)

    coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)
    return _c_cartesian_to_internal(coords_f32, indices_np)
