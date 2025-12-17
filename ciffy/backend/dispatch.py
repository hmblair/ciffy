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
>>> internal = cartesian_to_internal(coords, indices)  # (M, 3) [dist, ang, dih]
>>> coords = nerf_reconstruct(indices, internal, n_atoms)
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
    _nerf_reconstruct_leveled_anchored as _c_nerf_reconstruct_anchored,
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
    ) -> Array:
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
        internal: (N, 3) array of internal coordinates.
    """
    if is_torch(coords):
        return _torch_cartesian_to_internal(coords, indices)
    return _numpy_cartesian_to_internal(coords, indices)


def nerf_reconstruct(
    indices: Array,
    internal: Array,
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
        internal: (M, 3) array of internal coordinates.
            Each row: [distance, angle, dihedral].
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
    if is_torch(internal):
        return _torch_nerf_reconstruct(indices, internal, n_atoms, level_offsets, anchor_coords, component_ids)
    return _numpy_nerf_reconstruct(indices, internal, n_atoms, level_offsets, anchor_coords, component_ids)


# =============================================================================
# NUMPY DISPATCH
# =============================================================================


def _numpy_cartesian_to_internal(
    coords: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """NumPy path: use C extension directly. Returns (M, 3) internal array."""
    coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)
    indices_i64 = np.ascontiguousarray(indices, dtype=np.int64)
    return _c_cartesian_to_internal(coords_f32, indices_i64)


def _numpy_nerf_reconstruct(
    indices: np.ndarray,
    internal: np.ndarray,
    n_atoms: int,
    level_offsets: np.ndarray | None = None,
    anchor_coords: np.ndarray | None = None,
    component_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    NumPy path: use anchored C extension.

    Requires level_offsets, anchor_coords, and component_ids for anchored
    reconstruction which places atoms directly in the reference frame.
    """
    if level_offsets is None or anchor_coords is None or component_ids is None:
        raise ValueError(
            "nerf_reconstruct requires level_offsets, anchor_coords, and component_ids. "
            "Use CoordinateManager for automatic setup of these parameters."
        )

    indices_i64 = np.ascontiguousarray(indices, dtype=np.int64)
    internal_f32 = np.ascontiguousarray(internal, dtype=np.float32)
    anchor_f32 = np.ascontiguousarray(anchor_coords, dtype=np.float32)
    comp_ids_i32 = np.ascontiguousarray(component_ids, dtype=np.int32)
    level_off_i32 = np.ascontiguousarray(level_offsets, dtype=np.int32)

    return _c_nerf_reconstruct_anchored(
        indices_i64, internal_f32, n_atoms,
        level_off_i32, anchor_f32, comp_ids_i32
    )


# =============================================================================
# TORCH DISPATCH
# =============================================================================


def _torch_cartesian_to_internal(
    coords: "torch.Tensor",
    indices: Array,
) -> "torch.Tensor":
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
        internal = cuda_cartesian_to_internal(
            coords.to(torch.float32).contiguous(),
            indices_tensor.to(torch.int64).contiguous()
        )
        return internal.to(dtype)

    # CPU path: convert to numpy for C extension
    coords_f32 = coords.detach().cpu().to(torch.float32).numpy()
    indices_np = indices_tensor.cpu().numpy().astype(np.int64)

    internal_np = _c_cartesian_to_internal(
        coords_f32, indices_np
    )

    return torch.from_numpy(internal_np).to(device=device, dtype=dtype)


def _torch_nerf_reconstruct(
    indices: "torch.Tensor",
    internal: "torch.Tensor",
    n_atoms: int,
    level_offsets: Array | None = None,
    anchor_coords: Array | None = None,
    component_ids: Array | None = None,
) -> "torch.Tensor":
    """
    PyTorch dispatch for anchored NERF reconstruction.

    Routes to:
    - Autograd functions if any input requires_grad=True
    - CUDA level-parallel kernels for CUDA tensors
    - C extension for CPU tensors

    Requires level_offsets, anchor_coords, and component_ids for anchored
    reconstruction which places atoms directly in the reference frame.
    """
    import torch
    from .cuda_ops import cuda_nerf_reconstruct_leveled_anchored, HAS_ANCHORED_NERF

    if level_offsets is None or anchor_coords is None or component_ids is None:
        raise ValueError(
            "nerf_reconstruct requires level_offsets, anchor_coords, and component_ids. "
            "Use CoordinateManager for automatic setup of these parameters."
        )

    device = internal.device
    dtype = internal.dtype

    # Ensure indices are tensor on same device
    if is_torch(indices) and indices.device == device:
        indices_tensor = indices
    elif is_torch(indices):
        indices_tensor = indices.to(device)
    else:
        indices_tensor = torch.from_numpy(np.asarray(indices)).to(device)

    # Convert anchor_coords and component_ids to tensors on same device
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

    # Convert level_offsets to tensor
    if is_torch(level_offsets) and level_offsets.device == device:
        level_offsets_tensor = level_offsets
    elif is_torch(level_offsets):
        level_offsets_tensor = level_offsets.to(device)
    else:
        level_offsets_tensor = torch.from_numpy(np.asarray(level_offsets)).to(device)

    # Autograd path for gradient computation
    if internal.requires_grad:
        from .autograd import nerf_reconstruct as autograd_nerf
        return autograd_nerf(indices_tensor, internal, n_atoms, level_offsets_tensor, anchor_tensor, comp_ids_tensor)

    # CUDA path with anchored level-parallel reconstruction
    if is_cuda_available(internal) and HAS_ANCHORED_NERF:
        coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=device)
        cuda_nerf_reconstruct_leveled_anchored(
            coords,
            indices_tensor.to(torch.int64).contiguous(),
            internal.to(torch.float32).contiguous(),
            level_offsets_tensor.to(torch.int32).contiguous(),
            anchor_tensor.to(torch.float32).contiguous(),
            comp_ids_tensor.to(torch.int32).contiguous(),
        )
        return coords.to(dtype)

    # CPU path: use anchored C extension
    indices_np = indices_tensor.cpu().numpy().astype(np.int64)
    internal_f32 = internal.detach().cpu().to(torch.float32).numpy()
    anchor_np = anchor_tensor.detach().cpu().numpy().astype(np.float32)
    comp_ids_np = comp_ids_tensor.detach().cpu().numpy().astype(np.int32)
    level_off_np = level_offsets_tensor.detach().cpu().numpy().astype(np.int32)

    coords_np = _c_nerf_reconstruct_anchored(
        indices_np, internal_f32, n_atoms,
        level_off_np, anchor_np, comp_ids_np
    )
    return torch.from_numpy(coords_np).to(device=device, dtype=dtype)
