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

    Uses C extension for optimal performance. For PyTorch tensors that
    require gradients, uses the Python implementation which is fully
    differentiable.

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

    # Use autograd functions for PyTorch tensors that require gradients
    if is_torch(distances):
        if distances.requires_grad or angles.requires_grad or dihedrals.requires_grad:
            from ..backend.autograd import nerf_reconstruct as autograd_nerf
            import torch
            indices_tensor = zmatrix_indices if is_torch(zmatrix_indices) else torch.from_numpy(zmatrix_indices).to(distances.device)
            return autograd_nerf(indices_tensor, distances, angles, dihedrals, n_atoms)

    # Use C extension for all other cases
    # Ensure indices are numpy int64
    if is_torch(zmatrix_indices):
        indices_np = zmatrix_indices.cpu().numpy()
    else:
        indices_np = np.asarray(zmatrix_indices)

    if is_torch(distances):
        import torch
        device = distances.device
        dtype = distances.dtype
        dist_f32 = distances.detach().cpu().to(torch.float32).numpy()
        ang_f32 = angles.detach().cpu().to(torch.float32).numpy()
        dih_f32 = dihedrals.detach().cpu().to(torch.float32).numpy()
    else:
        dist_f32 = np.ascontiguousarray(distances, dtype=np.float32)
        ang_f32 = np.ascontiguousarray(angles, dtype=np.float32)
        dih_f32 = np.ascontiguousarray(dihedrals, dtype=np.float32)

    # Call C extension
    coords_np = _c_nerf_reconstruct(indices_np, dist_f32, ang_f32, dih_f32, n_atoms)

    if is_torch(distances):
        import torch
        return torch.from_numpy(coords_np).to(device=device, dtype=dtype)
    return coords_np
