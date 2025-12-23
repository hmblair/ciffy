"""
Device-agnostic dispatch for internal coordinate operations.

This module is the **internal dispatch layer** that routes coordinate conversion
operations to the optimal backend implementation. For most use cases, prefer
using the higher-level public API via ``ciffy.internal`` or ``Polymer`` methods.

Implementation selection based on array type and device:

- NumPy arrays → C extension (parent-based)
- PyTorch CPU tensors → C extension (via numpy conversion)
- PyTorch CUDA tensors → CUDA kernels (uses bridge for zmatrix_indices)
- PyTorch tensors with requires_grad → autograd functions (uses bridge)

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
>>> internal = cartesian_to_internal(coords, parent)  # (N, 3) [dist, ang, dih]
>>> coords = nerf_reconstruct(parent, levels, internal, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import Array, is_torch
from .cuda_ops import CUDA_EXTENSION_AVAILABLE, is_cuda_available

if TYPE_CHECKING:
    import torch
    from ..internal.tree import ReconstructionData

# C extension imports (required) - parent-based functions
from .._c import (
    _cartesian_to_internal_parent as _c_cartesian_to_internal_parent,
    _nerf_reconstruct_parent as _c_nerf_reconstruct_parent,
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
    "TopologyInfo",
    # Alignment
    "kabsch_rotation",
]


# =============================================================================
# RE-EXPORTS FROM BACKEND MODULES
# =============================================================================

# Graph building and data structures (re-exported from backend.graph)
from .graph import (
    TopologyInfo,
    build_bond_graph,
    build_bond_graph_csr,
    find_connected_components,
)

# Kabsch rotation for coordinate alignment
from ..operations.alignment import kabsch_rotation


def cartesian_to_internal(
    coords: Array,
    parent: Array,
    ) -> Array:
    """
    Convert Cartesian coordinates to internal coordinates.

    Automatically dispatches to the optimal implementation:
    - CUDA kernels for GPU tensors (derives zmatrix from parent)
    - C extension for CPU tensors and NumPy arrays (native parent-based)
    - Autograd functions when gradients are required (derives zmatrix)

    Args:
        coords: (N, 3) array of Cartesian coordinates.
        parent: (N,) int64 array of parent indices. References are derived:
            dist_ref[k] = parent[k]
            ang_ref[k] = parent[parent[k]]
            dih_ref[k] = parent[parent[parent[k]]]

    Returns:
        internal: (N, 3) array of internal coordinates [distance, angle, dihedral].
    """
    if is_torch(coords):
        return _torch_cartesian_to_internal(coords, parent)
    return _numpy_cartesian_to_internal(coords, parent)


def nerf_reconstruct(
    parent: Array,
    levels: Array,
    internal: Array,
    level_offsets: Array,
    level_atoms: Array,
    fixed_coords: Array,
    anchor_coords: Array | None = None,
    component_ids: Array | None = None,
    center_offsets: Array | None = None,
) -> Array:
    """
    Reconstruct Cartesian coordinates using NERF algorithm.

    Automatically dispatches to the optimal implementation:
    - CUDA kernels for GPU tensors (derives zmatrix from parent)
    - C extension for CPU tensors and NumPy arrays (native parent-based)
    - Autograd functions when gradients are required (derives zmatrix)

    Args:
        parent: (N,) int64 parent indices for each atom (-1 for roots).
        levels: (N,) int32 tree depth for each atom.
        internal: (N, 3) array of internal coordinates [distance, angle, dihedral].
        level_offsets: (max_level+2,) int32 CSR-style offsets by level.
        level_atoms: (N,) int64 atom indices sorted by level.
        fixed_coords: (N, 3) float32 reference coordinates for atoms at levels 0-2.
        anchor_coords: (n_components, 3, 3) float32 anchor positions for CUDA path.
        component_ids: (N,) int32 component index per atom (for center offsets).
        center_offsets: (n_components, 3) float32 per-component centering offsets.

    Returns:
        (N, 3) array of Cartesian coordinates.
    """
    if is_torch(internal):
        return _torch_nerf_reconstruct(
            parent, levels, internal, level_offsets, level_atoms,
            fixed_coords, anchor_coords, component_ids, center_offsets
        )
    return _numpy_nerf_reconstruct(
        parent, levels, internal, level_offsets, level_atoms,
        fixed_coords, component_ids, center_offsets
    )


# =============================================================================
# NUMPY DISPATCH
# =============================================================================


def _numpy_cartesian_to_internal(
    coords: np.ndarray,
    parent: np.ndarray,
) -> np.ndarray:
    """NumPy path: use parent-based C extension directly. Returns (N, 3) internal array."""
    coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)
    parent_i64 = np.ascontiguousarray(parent, dtype=np.int64)
    return _c_cartesian_to_internal_parent(coords_f32, parent_i64)


def _numpy_nerf_reconstruct(
    parent: np.ndarray,
    levels: np.ndarray,
    internal: np.ndarray,
    level_offsets: np.ndarray,
    level_atoms: np.ndarray,
    fixed_coords: np.ndarray,
    component_ids: np.ndarray | None = None,
    center_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """
    NumPy path: use parent-based C extension.

    Uses level-parallel NERF reconstruction with parent array.
    """
    n_levels = int(levels.max()) + 1 if len(levels) > 0 else 0

    parent_i64 = np.ascontiguousarray(parent, dtype=np.int64)
    levels_i32 = np.ascontiguousarray(levels, dtype=np.int32)
    internal_f32 = np.ascontiguousarray(internal, dtype=np.float32)
    level_offsets_i32 = np.ascontiguousarray(level_offsets, dtype=np.int32)
    level_atoms_i64 = np.ascontiguousarray(level_atoms, dtype=np.int64)
    fixed_f32 = np.ascontiguousarray(fixed_coords, dtype=np.float32)

    coords = _c_nerf_reconstruct_parent(
        parent_i64, levels_i32, internal_f32,
        level_offsets_i32, level_atoms_i64, n_levels,
        fixed_f32
    )

    # Apply per-component center offsets if provided
    if center_offsets is not None and component_ids is not None:
        coords += center_offsets[component_ids]

    return coords


# =============================================================================
# TORCH DISPATCH
# =============================================================================


def _torch_cartesian_to_internal(
    coords: "torch.Tensor",
    parent: Array,
) -> "torch.Tensor":
    """
    PyTorch dispatch for Cartesian to internal conversion.

    Routes to:
    - Autograd functions if requires_grad=True (derives zmatrix from parent)
    - CUDA kernels for CUDA tensors (derives zmatrix from parent)
    - C extension for CPU tensors (native parent-based)
    """
    import torch
    from .cuda_ops import cuda_cartesian_to_internal

    device = coords.device
    dtype = coords.dtype

    # Ensure parent is tensor on same device
    if is_torch(parent) and parent.device == device:
        parent_tensor = parent
    elif is_torch(parent):
        parent_tensor = parent.to(device)
    else:
        parent_tensor = torch.from_numpy(np.asarray(parent)).to(device)

    # For CUDA and autograd paths, derive zmatrix_indices from parent
    # (bridge pattern for backward compatibility)
    needs_zmatrix = coords.requires_grad or is_cuda_available(coords)
    if needs_zmatrix:
        from ..internal.tree import derive_zmatrix_from_parent
        parent_np = parent_tensor.cpu().numpy() if is_torch(parent_tensor) else np.asarray(parent)
        zmatrix_np = derive_zmatrix_from_parent(parent_np)
        indices_tensor = torch.from_numpy(zmatrix_np).to(device)

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

    # CPU path: use parent-based C extension via buffer protocol
    import warnings

    if not coords.is_cpu:
        warnings.warn(
            f"Tensor on {device} falling back to CPU for C extension. "
            "Consider using CUDA tensors with the CUDA extension for best performance.",
            stacklevel=3
        )
        coords = coords.cpu()
        parent_tensor = parent_tensor.cpu()

    # Ensure contiguous layout for buffer protocol
    coords_f32 = coords.detach().to(torch.float32).contiguous()
    parent_i64 = parent_tensor.detach().to(torch.int64).contiguous()

    # Call parent-based C extension
    internal_np = _c_cartesian_to_internal_parent(coords_f32, parent_i64)

    return torch.from_numpy(internal_np).to(device=device, dtype=dtype)


def _torch_nerf_reconstruct(
    parent: "torch.Tensor",
    levels: "torch.Tensor",
    internal: "torch.Tensor",
    level_offsets: Array,
    level_atoms: Array,
    fixed_coords: Array,
    anchor_coords: Array | None = None,
    component_ids: Array | None = None,
    center_offsets: Array | None = None,
) -> "torch.Tensor":
    """
    PyTorch dispatch for parent-based NERF reconstruction.

    Routes to:
    - Autograd functions if internal requires_grad=True (derives zmatrix)
    - CUDA kernels for CUDA tensors (derives zmatrix)
    - C extension for CPU tensors (native parent-based)
    """
    import torch
    from .cuda_ops import cuda_nerf_reconstruct_leveled_anchored, ANCHORED_NERF_AVAILABLE

    n_atoms = len(parent)
    device = internal.device
    dtype = internal.dtype

    # Convert all arrays to tensors on same device
    def to_tensor(arr, target_dtype=None):
        if arr is None:
            return None
        if is_torch(arr) and arr.device == device:
            t = arr
        elif is_torch(arr):
            t = arr.to(device)
        else:
            t = torch.from_numpy(np.asarray(arr)).to(device)
        if target_dtype:
            t = t.to(target_dtype)
        return t

    parent_tensor = to_tensor(parent, torch.int64)
    levels_tensor = to_tensor(levels, torch.int32)
    level_offsets_tensor = to_tensor(level_offsets, torch.int32)
    level_atoms_tensor = to_tensor(level_atoms, torch.int64)
    fixed_tensor = to_tensor(fixed_coords, torch.float32)
    anchor_tensor = to_tensor(anchor_coords, torch.float32)
    comp_ids_tensor = to_tensor(component_ids, torch.int32)
    center_offsets_tensor = to_tensor(center_offsets, torch.float32)

    # For CUDA and autograd paths, derive zmatrix_indices from parent
    # (bridge pattern for backward compatibility with existing CUDA kernels)
    needs_zmatrix = internal.requires_grad or (is_cuda_available(internal) and ANCHORED_NERF_AVAILABLE)
    if needs_zmatrix:
        from ..internal.tree import derive_zmatrix_from_parent
        parent_np = parent_tensor.cpu().numpy()
        zmatrix_np = derive_zmatrix_from_parent(parent_np)
        indices_tensor = torch.from_numpy(zmatrix_np).to(device)

        # Use provided anchor_coords or derive from fixed_coords
        if anchor_tensor is None:
            if component_ids is not None:
                n_components = int(comp_ids_tensor.max().item()) + 1 if len(comp_ids_tensor) > 0 else 1
                anchor_coords_list = []
                for comp in range(n_components):
                    comp_mask = comp_ids_tensor == comp
                    level_mask = levels_tensor <= 2
                    combined_mask = comp_mask & level_mask
                    anchor_atoms = torch.where(combined_mask)[0][:3]
                    if len(anchor_atoms) < 3:
                        while len(anchor_atoms) < 3:
                            anchor_atoms = torch.cat([anchor_atoms, anchor_atoms[:1]])
                    anchor_coords_list.append(fixed_tensor[anchor_atoms])
                anchor_tensor = torch.stack(anchor_coords_list)
            else:
                anchor_tensor = fixed_tensor[:3].unsqueeze(0)

    # Autograd path for gradient computation
    if internal.requires_grad:
        from .autograd import nerf_reconstruct as autograd_nerf
        return autograd_nerf(indices_tensor, internal, level_offsets_tensor, anchor_tensor, comp_ids_tensor)

    # CUDA path with anchored component-parallel reconstruction
    if is_cuda_available(internal) and ANCHORED_NERF_AVAILABLE:
        coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=device)
        cuda_nerf_reconstruct_leveled_anchored(
            coords,
            indices_tensor.to(torch.int64).contiguous(),
            internal.to(torch.float32).contiguous(),
            level_offsets_tensor.to(torch.int32).contiguous(),
            anchor_tensor.to(torch.float32).contiguous(),
            comp_ids_tensor.to(torch.int32).contiguous(),
        )
        # Apply center offsets using advanced indexing
        if center_offsets_tensor is not None and comp_ids_tensor is not None:
            coords += center_offsets_tensor[comp_ids_tensor]
        return coords.to(dtype)

    # CPU path: use parent-based C extension via buffer protocol
    import warnings

    if not internal.is_cpu:
        warnings.warn(
            f"Tensor on {device} falling back to CPU for C extension. "
            "Consider using CUDA tensors with the CUDA extension for best performance.",
            stacklevel=3
        )
        parent_tensor = parent_tensor.cpu()
        levels_tensor = levels_tensor.cpu()
        internal = internal.cpu()
        level_offsets_tensor = level_offsets_tensor.cpu()
        level_atoms_tensor = level_atoms_tensor.cpu()
        fixed_tensor = fixed_tensor.cpu()
        if comp_ids_tensor is not None:
            comp_ids_tensor = comp_ids_tensor.cpu()
        if center_offsets_tensor is not None:
            center_offsets_tensor = center_offsets_tensor.cpu()

    # Compute n_levels
    n_levels = int(levels_tensor.max().item()) + 1 if len(levels_tensor) > 0 else 0

    # Ensure contiguous layout for buffer protocol
    parent_i64 = parent_tensor.detach().to(torch.int64).contiguous()
    levels_i32 = levels_tensor.detach().to(torch.int32).contiguous()
    internal_f32 = internal.detach().to(torch.float32).contiguous()
    level_offsets_i32 = level_offsets_tensor.detach().to(torch.int32).contiguous()
    level_atoms_i64 = level_atoms_tensor.detach().to(torch.int64).contiguous()
    fixed_f32 = fixed_tensor.detach().to(torch.float32).contiguous()

    # Call parent-based C extension
    coords_np = _c_nerf_reconstruct_parent(
        parent_i64, levels_i32, internal_f32,
        level_offsets_i32, level_atoms_i64, n_levels,
        fixed_f32
    )

    coords = torch.from_numpy(coords_np).to(device=device, dtype=dtype)

    # Apply center offsets using advanced indexing
    if center_offsets_tensor is not None and comp_ids_tensor is not None:
        # Convert offset array to tensor once, then index by component IDs
        offsets_on_device = center_offsets_tensor.to(device=device, dtype=dtype)
        comp_ids_on_device = comp_ids_tensor.to(device=device)
        coords += offsets_on_device[comp_ids_on_device]

    return coords
