"""
NERF (Natural Extension Reference Frame) algorithm for coordinate reconstruction.

Reconstructs Cartesian coordinates from internal coordinates (bond lengths,
bond angles, dihedral angles) in a differentiable manner suitable for
gradient-based optimization and machine learning.

This module re-exports the nerf_reconstruct function from the dispatch layer
for backwards compatibility. The actual implementation handles device dispatch
(NumPy, CPU PyTorch, CUDA) automatically.
"""

from __future__ import annotations

from ..backend import Array
from ..backend.dispatch import nerf_reconstruct as _nerf_reconstruct


def nerf_reconstruct(
    zmatrix_indices: Array,
    distances: Array,
    angles: Array,
    dihedrals: Array,
    n_atoms: int | None = None,
    level_offsets: Array | None = None,
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
        level_offsets: (n_levels+1,) int32 CSR-style offsets for level-parallel CUDA.
            When provided, enables parallel NERF on CUDA by processing atoms
            at the same BFS level simultaneously. Can be obtained from ZMatrix.level_offsets.

    Returns:
        (N, 3) array of Cartesian coordinates in original atom order.
    """
    # Infer n_atoms if not provided
    if n_atoms is None:
        n_entries = len(zmatrix_indices)
        if n_entries > 0:
            max_idx = int(zmatrix_indices[:, 0].max())
            n_atoms = max_idx + 1
        else:
            n_atoms = 0

    return _nerf_reconstruct(zmatrix_indices, distances, angles, dihedrals, n_atoms, level_offsets)
