"""
Device-agnostic dispatch for internal coordinate operations.

This module is the **internal dispatch layer** that routes coordinate conversion
operations to the optimal backend implementation. For most use cases, prefer
using the higher-level public API via ``ciffy.internal`` or ``Polymer`` methods.

Implementation selection based on array type and device:

- NumPy arrays → C extension
- PyTorch CPU tensors → C extension (via numpy conversion)
- PyTorch CUDA tensors → CUDA kernels
- PyTorch tensors with requires_grad → autograd functions

Import Paths
------------
- **Public API**: ``ciffy.internal.nerf_reconstruct``, ``Polymer.coordinates``
- **Internal dispatch** (this module): ``ciffy.backend.dispatch``
- **Implementation details**: ``ciffy.backend.autograd`` (do not import directly)

Usage
-----
>>> from ciffy.backend.dispatch import cartesian_to_internal, nerf_reconstruct
>>>
>>> # Works with any array type on any device
>>> distances, angles, dihedrals = cartesian_to_internal(coords, indices)
>>> coords = nerf_reconstruct(indices, distances, angles, dihedrals, n_atoms)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import Array, is_torch
from .cuda_ops import HAS_CUDA_EXTENSION, is_cuda_available

if TYPE_CHECKING:
    import torch

# C extension imports (required)
from .._c import (
    _cartesian_to_internal as _c_cartesian_to_internal,
    _nerf_reconstruct as _c_nerf_reconstruct,
    _nerf_reconstruct_leveled as _c_nerf_reconstruct_leveled,
)

__all__ = [
    # Coordinate conversion
    "cartesian_to_internal",
    "nerf_reconstruct",
    # Graph building
    "build_bond_graph",
    "build_bond_graph_csr",
    "find_connected_components",
    # Data structures
    "ZMatrix",
    "ConnectedComponents",
    "TopologyInfo",
    # Alignment
    "kabsch_rotation",
]


# =============================================================================
# RE-EXPORTS FROM BACKEND MODULES
# =============================================================================

# Graph building and data structures (re-exported from backend.graph)
from .graph import (
    ZMatrix,
    ConnectedComponents,
    TopologyInfo,
    build_bond_graph,
    build_bond_graph_csr,
    find_connected_components,
)

# Kabsch rotation for coordinate alignment
from ..operations.alignment import kabsch_rotation


def cartesian_to_internal(
    coords: Array,
    indices: Array,
) -> tuple[Array, Array, Array]:
    """
    Convert Cartesian coordinates to internal coordinates.

    Automatically dispatches to the optimal implementation:
    - CUDA kernels for GPU tensors
    - C extension for CPU tensors and NumPy arrays
    - Autograd functions when gradients are required

    Args:
        coords: (N, 3) array of Cartesian coordinates.
        indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]

    Returns:
        Tuple of (distances, angles, dihedrals), each (M,) float32.
    """
    if is_torch(coords):
        return _torch_cartesian_to_internal(coords, indices)
    return _numpy_cartesian_to_internal(coords, indices)


def nerf_reconstruct(
    indices: Array,
    distances: Array,
    angles: Array,
    dihedrals: Array,
    n_atoms: int,
    level_offsets: Array | None = None,
    anchor_coords: Array | None = None,
    component_ids: Array | None = None,
) -> Array:
    """
    Reconstruct Cartesian coordinates using NERF algorithm.

    Automatically dispatches to the optimal implementation:
    - CUDA kernels for GPU tensors (uses level_offsets for parallelism)
    - C extension for CPU tensors and NumPy arrays
    - Autograd functions when gradients are required

    Args:
        indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
        distances: (M,) bond lengths in Angstroms.
        angles: (M,) bond angles in radians.
        dihedrals: (M,) dihedral angles in radians.
        n_atoms: Total number of atoms.
        level_offsets: (n_levels+1,) int32 CSR-style offsets for level-parallel CUDA.
            When provided, enables parallel NERF on CUDA by processing atoms
            at the same BFS level simultaneously.
        anchor_coords: (n_components, 3, 3) anchor positions for each component.
            When provided, atoms are placed directly in the reference frame
            defined by these anchors, eliminating need for Kabsch rotation.
        component_ids: (M,) int32 component index per Z-matrix entry.
            Required when anchor_coords is provided.

    Returns:
        (N, 3) array of Cartesian coordinates.
    """
    if is_torch(distances):
        return _torch_nerf_reconstruct(indices, distances, angles, dihedrals, n_atoms, level_offsets, anchor_coords, component_ids)
    return _numpy_nerf_reconstruct(indices, distances, angles, dihedrals, n_atoms, level_offsets, anchor_coords, component_ids)


# =============================================================================
# NUMPY DISPATCH
# =============================================================================


def _numpy_cartesian_to_internal(
    coords: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy path: use C extension directly."""
    coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)
    indices_i64 = np.ascontiguousarray(indices, dtype=np.int64)
    return _c_cartesian_to_internal(coords_f32, indices_i64)


def _numpy_nerf_reconstruct(
    indices: np.ndarray,
    distances: np.ndarray,
    angles: np.ndarray,
    dihedrals: np.ndarray,
    n_atoms: int,
    level_offsets: np.ndarray | None = None,
    anchor_coords: np.ndarray | None = None,
    component_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    NumPy path: use C extension directly.

    Note: For anchored NERF, the leveled C extension is used since anchored
    reconstruction requires the leveled approach. For non-anchored, sequential
    version is used to avoid OpenMP overhead.
    """
    from .._c import _nerf_reconstruct_leveled_anchored as _c_nerf_reconstruct_anchored

    indices_i64 = np.ascontiguousarray(indices, dtype=np.int64)
    dist_f32 = np.ascontiguousarray(distances, dtype=np.float32)
    ang_f32 = np.ascontiguousarray(angles, dtype=np.float32)
    dih_f32 = np.ascontiguousarray(dihedrals, dtype=np.float32)

    # Anchored path: use leveled anchored C extension
    if anchor_coords is not None and component_ids is not None and level_offsets is not None:
        anchor_f32 = np.ascontiguousarray(anchor_coords, dtype=np.float32)
        comp_ids_i32 = np.ascontiguousarray(component_ids, dtype=np.int32)
        level_off_i32 = np.ascontiguousarray(level_offsets, dtype=np.int32)
        return _c_nerf_reconstruct_anchored(
            indices_i64, dist_f32, ang_f32, dih_f32, n_atoms,
            level_off_i32, anchor_f32, comp_ids_i32
        )

    # Non-anchored: use sequential version (avoid OpenMP overhead)
    return _c_nerf_reconstruct(indices_i64, dist_f32, ang_f32, dih_f32, n_atoms)


# =============================================================================
# TORCH DISPATCH
# =============================================================================


def _torch_cartesian_to_internal(
    coords: "torch.Tensor",
    indices: Array,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    PyTorch dispatch for Cartesian to internal conversion.

    Routes to:
    - Autograd functions if requires_grad=True
    - CUDA kernels for CUDA tensors
    - C extension for CPU tensors
    """
    import torch
    from .cuda_ops import cuda_cartesian_to_internal

    device = coords.device
    dtype = coords.dtype

    # Ensure indices are tensor on same device (skip if already correct)
    if is_torch(indices) and indices.device == device:
        indices_tensor = indices
    elif is_torch(indices):
        indices_tensor = indices.to(device)
    else:
        indices_tensor = torch.from_numpy(np.asarray(indices)).to(device)

    # Autograd path for gradient computation
    if coords.requires_grad:
        from .autograd import cartesian_to_internal as autograd_c2i
        return autograd_c2i(coords, indices_tensor)

    # CUDA path for GPU tensors
    if is_cuda_available(coords):
        distances, angles, dihedrals = cuda_cartesian_to_internal(
            coords.to(torch.float32).contiguous(),
            indices_tensor.to(torch.int64).contiguous()
        )
        return distances.to(dtype), angles.to(dtype), dihedrals.to(dtype)

    # CPU path: convert to numpy for C extension
    coords_f32 = coords.detach().cpu().to(torch.float32).numpy()
    indices_np = indices_tensor.cpu().numpy().astype(np.int64)

    distances_np, angles_np, dihedrals_np = _c_cartesian_to_internal(
        coords_f32, indices_np
    )

    distances = torch.from_numpy(distances_np).to(device=device, dtype=dtype)
    angles = torch.from_numpy(angles_np).to(device=device, dtype=dtype)
    dihedrals = torch.from_numpy(dihedrals_np).to(device=device, dtype=dtype)
    return distances, angles, dihedrals


def _torch_nerf_reconstruct(
    indices: Array,
    distances: "torch.Tensor",
    angles: "torch.Tensor",
    dihedrals: "torch.Tensor",
    n_atoms: int,
    level_offsets: Array | None = None,
    anchor_coords: Array | None = None,
    component_ids: Array | None = None,
) -> "torch.Tensor":
    """
    PyTorch dispatch for NERF reconstruction.

    Routes to:
    - Autograd functions if any input requires_grad=True
    - CUDA level-parallel kernels if level_offsets provided and on CUDA
    - C extension for CPU tensors

    When level_offsets is provided, CUDA can process atoms at the same BFS
    level in parallel, reducing kernel launches from O(atoms) to O(levels).

    When anchor_coords and component_ids are provided, uses anchored NERF
    which places atoms directly in the reference frame defined by anchors.
    """
    import torch
    from .cuda_ops import cuda_nerf_reconstruct_leveled_anchored, HAS_ANCHORED_NERF

    device = distances.device
    dtype = distances.dtype

    # Ensure indices are tensor on same device (skip if already correct)
    if is_torch(indices) and indices.device == device:
        indices_tensor = indices
    elif is_torch(indices):
        indices_tensor = indices.to(device)
    else:
        indices_tensor = torch.from_numpy(np.asarray(indices)).to(device)

    # Convert anchor_coords and component_ids if provided
    anchor_tensor = None
    comp_ids_tensor = None
    if anchor_coords is not None and component_ids is not None:
        if is_torch(anchor_coords) and anchor_coords.device == device:
            anchor_tensor = anchor_coords
        elif is_torch(anchor_coords):
            anchor_tensor = anchor_coords.to(device)
        else:
            anchor_tensor = torch.from_numpy(np.asarray(anchor_coords)).to(device)

        if is_torch(component_ids) and component_ids.device == device:
            comp_ids_tensor = component_ids
        elif is_torch(component_ids):
            comp_ids_tensor = component_ids.to(device)
        else:
            comp_ids_tensor = torch.from_numpy(np.asarray(component_ids)).to(device)

    # Autograd path for gradient computation
    if distances.requires_grad or angles.requires_grad or dihedrals.requires_grad:
        from .autograd import nerf_reconstruct as autograd_nerf
        return autograd_nerf(indices_tensor, distances, angles, dihedrals, n_atoms, level_offsets, anchor_tensor, comp_ids_tensor)

    # CUDA path with anchored or level-parallel reconstruction
    if is_cuda_available(distances) and level_offsets is not None:
        # Convert level_offsets to tensor
        if is_torch(level_offsets) and level_offsets.device == device:
            level_offsets_tensor = level_offsets
        elif is_torch(level_offsets):
            level_offsets_tensor = level_offsets.to(device)
        else:
            level_offsets_tensor = torch.from_numpy(np.asarray(level_offsets)).to(device)

        # Anchored CUDA path
        if anchor_tensor is not None and comp_ids_tensor is not None and HAS_ANCHORED_NERF:
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=device)
            cuda_nerf_reconstruct_leveled_anchored(
                coords,
                indices_tensor.to(torch.int64).contiguous(),
                distances.to(torch.float32).contiguous(),
                angles.to(torch.float32).contiguous(),
                dihedrals.to(torch.float32).contiguous(),
                level_offsets_tensor.to(torch.int32).contiguous(),
                anchor_tensor.to(torch.float32).contiguous(),
                comp_ids_tensor.to(torch.int32).contiguous(),
            )
            return coords.to(dtype)

        # Non-anchored CUDA path
        from .cuda_ops import cuda_nerf_reconstruct_leveled, HAS_LEVELED_NERF
        if HAS_LEVELED_NERF:
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=device)
            cuda_nerf_reconstruct_leveled(
                coords,
                indices_tensor.to(torch.int64).contiguous(),
                distances.to(torch.float32).contiguous(),
                angles.to(torch.float32).contiguous(),
                dihedrals.to(torch.float32).contiguous(),
                level_offsets_tensor.to(torch.int32).contiguous(),
            )
            return coords.to(dtype)

    # CPU path: use C extension
    indices_np = indices_tensor.cpu().numpy().astype(np.int64)
    dist_f32 = distances.detach().cpu().to(torch.float32).numpy()
    ang_f32 = angles.detach().cpu().to(torch.float32).numpy()
    dih_f32 = dihedrals.detach().cpu().to(torch.float32).numpy()

    # Anchored CPU path
    if anchor_tensor is not None and comp_ids_tensor is not None and level_offsets is not None:
        from .._c import _nerf_reconstruct_leveled_anchored as _c_nerf_reconstruct_anchored
        anchor_np = anchor_tensor.detach().cpu().numpy().astype(np.float32)
        comp_ids_np = comp_ids_tensor.detach().cpu().numpy().astype(np.int32)
        if is_torch(level_offsets):
            level_off_np = level_offsets.detach().cpu().numpy().astype(np.int32)
        else:
            level_off_np = np.asarray(level_offsets, dtype=np.int32)
        coords_np = _c_nerf_reconstruct_anchored(
            indices_np, dist_f32, ang_f32, dih_f32, n_atoms,
            level_off_np, anchor_np, comp_ids_np
        )
        return torch.from_numpy(coords_np).to(device=device, dtype=dtype)

    # Non-anchored CPU path
    coords_np = _c_nerf_reconstruct(indices_np, dist_f32, ang_f32, dih_f32, n_atoms)
    return torch.from_numpy(coords_np).to(device=device, dtype=dtype)
