"""
NERF (Natural Extension Reference Frame) algorithm for coordinate reconstruction.

Reconstructs Cartesian coordinates from internal coordinates (bond lengths,
bond angles, dihedral angles) in a differentiable manner suitable for
gradient-based optimization and machine learning.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch

# C extension (required)
from .._c import _nerf_reconstruct as _c_nerf_reconstruct


def nerf_reconstruct(
    zmatrix_indices: Array,
    distances: Array,
    angles: Array,
    dihedrals: Array,
    n_atoms: int | None = None,
) -> Array:
    """
    Reconstruct Cartesian coordinates using NERF algorithm.

    Uses CUDA kernels when available for GPU tensors, otherwise falls back
    to CPU C extension. For PyTorch tensors that require gradients, uses
    autograd functions with backward passes.

    The Natural Extension Reference Frame algorithm places each atom
    by constructing a local coordinate system from three previously
    placed atoms, then positioning the new atom using spherical-like
    coordinates (distance, angle, dihedral).

    Args:
        zmatrix_indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
        distances: (M,) bond lengths in Angstroms (in BFS order).
        angles: (M,) bond angles in radians (in BFS order).
        dihedrals: (M,) dihedral angles in radians (in BFS order).
        n_atoms: Total number of atoms (including orphans). If None,
            inferred from max Z-matrix index.

    Returns:
        (N, 3) array of Cartesian coordinates in original atom order.
    """
    n_entries = len(zmatrix_indices)

    # Find max atom index to allocate coords array
    if n_atoms is None:
        if n_entries > 0:
            max_idx = int(zmatrix_indices[:, 0].max())
            n_atoms = max_idx + 1
        else:
            n_atoms = 0

    # For PyTorch tensors, check for CUDA or autograd path
    if is_torch(distances):
        import torch
        from ..backend.cuda_ops import is_cuda_available, cuda_nerf_reconstruct

        device = distances.device
        dtype = distances.dtype

        # Ensure indices are on same device
        if not is_torch(zmatrix_indices):
            indices_tensor = torch.from_numpy(zmatrix_indices).to(device)
        else:
            indices_tensor = zmatrix_indices.to(device)

        # Use autograd path for tensors requiring gradients
        if distances.requires_grad or angles.requires_grad or dihedrals.requires_grad:
            from ..backend.autograd import nerf_reconstruct as autograd_nerf
            return autograd_nerf(indices_tensor, distances, angles, dihedrals, n_atoms)

        # Use CUDA kernels for GPU tensors (inference mode)
        if is_cuda_available(distances):
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=device)
            cuda_nerf_reconstruct(
                coords,
                indices_tensor.to(torch.int64).contiguous(),
                distances.to(torch.float32).contiguous(),
                angles.to(torch.float32).contiguous(),
                dihedrals.to(torch.float32).contiguous()
            )
            return coords.to(dtype=dtype)

        # CPU PyTorch tensor: use C extension
        indices_np = indices_tensor.cpu().numpy().astype(np.int64)
        dist_f32 = distances.detach().cpu().to(torch.float32).numpy()
        ang_f32 = angles.detach().cpu().to(torch.float32).numpy()
        dih_f32 = dihedrals.detach().cpu().to(torch.float32).numpy()

        coords_np = _c_nerf_reconstruct(indices_np, dist_f32, ang_f32, dih_f32, n_atoms)
        return torch.from_numpy(coords_np).to(device=device, dtype=dtype)

    # NumPy path
    if is_torch(zmatrix_indices):
        indices_np = zmatrix_indices.cpu().numpy()
    else:
        indices_np = np.asarray(zmatrix_indices)

    dist_f32 = np.ascontiguousarray(distances, dtype=np.float32)
    ang_f32 = np.ascontiguousarray(angles, dtype=np.float32)
    dih_f32 = np.ascontiguousarray(dihedrals, dtype=np.float32)

    return _c_nerf_reconstruct(indices_np, dist_f32, ang_f32, dih_f32, n_atoms)
