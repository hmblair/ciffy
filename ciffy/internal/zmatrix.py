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
    from ..polymer import Polymer
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
        >>> zmatrix = ZMatrix.from_polymer(polymer)
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
    def from_polymer(cls, polymer: "Polymer") -> "ZMatrix":
        """
        Build Z-matrix from polymer using canonical references.

        Uses biochemically-correct canonical references from codegen for
        all atoms that have them defined (backbone, base atoms). Falls back
        to BFS-based references for other atoms (sidechains).

        This ensures:
        1. Named dihedrals (PHI, PSI, ALPHA, CHI, etc.) are captured
        2. Ring dihedrals preserve ring geometry
        3. Deterministic, biochemically-meaningful Z-matrix

        Args:
            polymer: Polymer structure.

        Returns:
            ZMatrix with entries in placement order and dihedral type annotations.
        """
        from .graph import build_canonical_zmatrix
        indices, dihedral_types = build_canonical_zmatrix(polymer)
        return cls(indices, dihedral_types)

    @classmethod
    def from_topology(cls, topology: "TopologyInfo") -> "ZMatrix":
        """
        Build Z-matrix from topology info using BFS traversal.

        Processes each chain independently with its own spanning tree.
        Returns entries in BFS order so references always point to
        earlier (already placed) atoms. Post-processes to annotate
        named dihedral types and update references for dihedral owners.

        Args:
            topology: TopologyInfo containing structural metadata.

        Returns:
            ZMatrix with entries in placement order and dihedral type annotations.
        """
        from .graph import _build_zmatrix_indices_from_topology, annotate_dihedral_types

        # Build Z-matrix using BFS
        indices = _build_zmatrix_indices_from_topology(topology)

        if len(indices) == 0:
            return cls(indices, np.array([], dtype=np.int8))

        # Compute residue start offsets
        residue_starts = np.concatenate([[0], np.cumsum(topology.residue_sizes)])

        # Get chain boundaries (residue indices where new chains start)
        chain_boundaries = np.concatenate([[0], np.cumsum(topology.chain_lengths)[:-1]])

        # Post-process to annotate dihedral types and update references
        indices, dihedral_types = annotate_dihedral_types(
            indices,
            topology.atoms,
            topology.sequence,
            residue_starts,
            chain_boundaries,
        )

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
        Validate Z-matrix structure.

        Checks that:
        - All reference atoms are either -1 or point to earlier atoms
        - Reference progression is correct (dist before angle before dihedral)

        Raises:
            ValueError: If validation fails.
        """
        placed = set()
        for i in range(len(self._indices)):
            atom_idx = int(self._indices[i, 0])
            dist_ref = int(self._indices[i, 1])
            ang_ref = int(self._indices[i, 2])
            dih_ref = int(self._indices[i, 3])

            # Check distance reference
            if dist_ref >= 0 and dist_ref not in placed:
                raise ValueError(
                    f"Entry {i}: distance_ref {dist_ref} not yet placed"
                )

            # Check angle reference
            if ang_ref >= 0 and ang_ref not in placed:
                raise ValueError(
                    f"Entry {i}: angle_ref {ang_ref} not yet placed"
                )

            # Check dihedral reference
            if dih_ref >= 0 and dih_ref not in placed:
                raise ValueError(
                    f"Entry {i}: dihedral_ref {dih_ref} not yet placed"
                )

            # Check progression: can't have angle without distance, etc.
            if ang_ref >= 0 and dist_ref < 0:
                raise ValueError(
                    f"Entry {i}: has angle_ref but no distance_ref"
                )
            if dih_ref >= 0 and ang_ref < 0:
                raise ValueError(
                    f"Entry {i}: has dihedral_ref but no angle_ref"
                )

            placed.add(atom_idx)

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

    Uses C extension for optimal performance. For PyTorch tensors that
    require gradients, uses autograd functions with C backward passes.

    Args:
        coords: (N, 3) array of Cartesian coordinates.
        zmatrix_indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]

    Returns:
        Tuple of (distances, angles, dihedrals), each (M,) float32.
    """
    # Use autograd functions for PyTorch tensors that require gradients
    if is_torch(coords) and coords.requires_grad:
        from ..backend.autograd import cartesian_to_internal as autograd_c2i
        import torch
        indices_tensor = zmatrix_indices if is_torch(zmatrix_indices) else torch.from_numpy(zmatrix_indices).to(coords.device)
        return autograd_c2i(coords, indices_tensor)

    # Use C extension for all other cases
    # Convert indices to numpy if needed
    if is_torch(zmatrix_indices):
        indices_np = zmatrix_indices.cpu().numpy()
    else:
        indices_np = np.asarray(zmatrix_indices)

    if is_torch(coords):
        import torch
        device = coords.device
        dtype = coords.dtype
        coords_f32 = coords.detach().cpu().to(torch.float32).numpy()
    else:
        coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)

    # Call C extension
    distances_np, angles_np, dihedrals_np = _c_cartesian_to_internal(
        coords_f32, indices_np
    )

    if is_torch(coords):
        import torch
        distances = torch.from_numpy(distances_np).to(device=device, dtype=dtype)
        angles = torch.from_numpy(angles_np).to(device=device, dtype=dtype)
        dihedrals = torch.from_numpy(dihedrals_np).to(device=device, dtype=dtype)
        return distances, angles, dihedrals

    return distances_np, angles_np, dihedrals_np
