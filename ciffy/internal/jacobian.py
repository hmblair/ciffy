"""
Jacobian computation and DOF discovery for constraint systems.

This module provides:
1. Analytical Jacobian computation using DFS timestamps for O(1) ancestry
2. DOF discovery via QR decomposition with pivoting

The key insight is that we can compute ∂position/∂torsion analytically:
when a torsion rotates, all downstream atoms rotate around the bond axis.
Ancestry is checked in O(1) using DFS timestamps (Euler tour).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from .constraints import ClosureConstraints, ConstraintSystem


def discover_dof(
    parent: np.ndarray,
    level: np.ndarray,
    dfs_enter: np.ndarray,
    dfs_exit: np.ndarray,
    closures: "ClosureConstraints",
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discover independent vs dependent torsions via Jacobian analysis.

    Uses DFS timestamps for O(1) ancestry queries and QR decomposition
    with column pivoting to identify linearly independent torsions.

    Args:
        parent: (N,) int64 parent array from spanning tree
        level: (N,) int32 level array
        dfs_enter: (N,) int32 DFS entry times
        dfs_exit: (N,) int32 DFS exit times
        closures: ClosureConstraints with closure bond info
        coords: (N, 3) reference Cartesian coordinates

    Returns:
        (independent_idx, dependent_idx) - arrays of atom indices
    """
    # All potential torsions (atoms at level >= 3)
    all_torsions = np.where(level >= 3)[0].astype(np.int64)

    if closures.n_closures == 0:
        # No closures - all torsions are independent
        return all_torsions, np.zeros(0, dtype=np.int64)

    # Find ring torsions using DFS timestamps (those that affect any closure)
    ring_torsions = _find_ring_torsions(
        all_torsions, closures.closure_bonds, dfs_enter, dfs_exit
    )

    if len(ring_torsions) == 0:
        return all_torsions, np.zeros(0, dtype=np.int64)

    ring_torsion_set = set(ring_torsions.tolist())

    # Non-ring torsions are always independent
    non_ring_torsions = np.array(
        [t for t in all_torsions if t not in ring_torsion_set],
        dtype=np.int64
    )

    # Build Jacobian for ring torsions only
    J = _compute_jacobian_vectorized(
        coords, parent, dfs_enter, dfs_exit, closures.closure_bonds, ring_torsions
    )

    # QR with column pivoting to find rank and select dependent torsions
    if J.shape[0] == 0 or J.shape[1] == 0:
        return all_torsions, np.zeros(0, dtype=np.int64)

    Q, R, perm = scipy.linalg.qr(J, pivoting=True)

    # Rank = number of non-zero diagonal elements in R
    tol = 1e-10 * max(J.shape) * np.abs(R).max() if R.size > 0 else 1e-10
    diag_R = np.diag(R) if min(R.shape) > 0 else np.array([])
    rank = np.sum(np.abs(diag_R) > tol)

    # First `rank` columns (after pivoting) are the dependent ones
    n_dependent = min(rank, len(ring_torsions))
    dependent_local = perm[:n_dependent]
    independent_local = perm[n_dependent:]

    # Map back to atom indices
    dependent_idx = ring_torsions[dependent_local]
    independent_ring = ring_torsions[independent_local]

    # Combine non-ring (always independent) with independent ring torsions
    independent_idx = np.concatenate([non_ring_torsions, independent_ring])
    independent_idx = np.sort(independent_idx)

    return independent_idx, dependent_idx


def _find_ring_torsions(
    all_torsions: np.ndarray,
    closure_bonds: np.ndarray,
    dfs_enter: np.ndarray,
    dfs_exit: np.ndarray,
) -> np.ndarray:
    """
    Find torsions that affect at least one closure using DFS timestamps.

    A torsion affects a closure if exactly one of the closure endpoints
    is a descendant of the torsion atom.
    """
    if len(all_torsions) == 0 or len(closure_bonds) == 0:
        return np.zeros(0, dtype=np.int64)

    closure_i = closure_bonds[:, 0]
    closure_j = closure_bonds[:, 1]

    # Vectorized descendant check: (T, C)
    enter_t = dfs_enter[all_torsions, None]  # (T, 1)
    exit_t = dfs_exit[all_torsions, None]    # (T, 1)
    enter_i = dfs_enter[closure_i]           # (C,)
    enter_j = dfs_enter[closure_j]           # (C,)

    i_is_desc = (enter_t <= enter_i) & (enter_i <= exit_t)  # (T, C)
    j_is_desc = (enter_t <= enter_j) & (enter_j <= exit_t)  # (T, C)

    # Torsion affects closure if exactly one endpoint moves
    affects_any = (i_is_desc != j_is_desc).any(axis=1)  # (T,)

    return all_torsions[affects_any]


def compute_jacobian_analytical(
    internal: np.ndarray,
    coords: np.ndarray,
    system: "ConstraintSystem",
    torsion_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute Jacobian analytically using DFS timestamps.

    For each closure constraint and each torsion, computes:
    ∂(constraint)/∂(torsion)

    Args:
        internal: (N, 3) internal coordinates (unused, kept for API compat)
        coords: (N, 3) current Cartesian coordinates
        system: ConstraintSystem with DFS timestamps
        torsion_indices: (D,) which torsions to compute Jacobian for

    Returns:
        (3C, D) Jacobian matrix where C = closures, D = torsions
    """
    return _compute_jacobian_vectorized(
        coords,
        system.parent,
        system.dfs_enter,
        system.dfs_exit,
        system.closures.closure_bonds,
        torsion_indices,
    )


def _compute_jacobian_vectorized(
    coords: np.ndarray,
    parent: np.ndarray,
    dfs_enter: np.ndarray,
    dfs_exit: np.ndarray,
    closure_bonds: np.ndarray,
    torsion_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute Jacobian using vectorized operations and DFS timestamps.

    Args:
        coords: (N, 3) Cartesian coordinates
        parent: (N,) parent array
        dfs_enter: (N,) DFS entry times
        dfs_exit: (N,) DFS exit times
        closure_bonds: (C, 2) closure atom pairs
        torsion_indices: (D,) torsion atom indices

    Returns:
        (3C, D) Jacobian matrix
    """
    D = len(torsion_indices)
    C = len(closure_bonds)

    if D == 0 or C == 0:
        return np.zeros((3 * C, D), dtype=np.float32)

    # Get rotation axes for all torsions: (D, 3) each
    parents = parent[torsion_indices]
    grandparents = parent[parents]

    # Filter out invalid torsions (missing grandparent)
    valid = (parents >= 0) & (grandparents >= 0)
    if not valid.all():
        # Handle edge case: some torsions don't have valid rotation axes
        # For now, just skip the invalid ones - they contribute 0 to Jacobian
        pass

    axis_origins = coords[parents]                          # (D, 3)
    axis_dirs = coords[grandparents] - axis_origins         # (D, 3)
    axis_norms = np.linalg.norm(axis_dirs, axis=1, keepdims=True)
    axis_norms = np.maximum(axis_norms, 1e-10)
    axis_dirs = axis_dirs / axis_norms                      # (D, 3)

    # Vectorized descendant check: (D, C)
    closure_i = closure_bonds[:, 0]
    closure_j = closure_bonds[:, 1]

    enter_t = dfs_enter[torsion_indices, None]  # (D, 1)
    exit_t = dfs_exit[torsion_indices, None]    # (D, 1)
    enter_i = dfs_enter[closure_i]              # (C,)
    enter_j = dfs_enter[closure_j]              # (C,)

    i_is_desc = (enter_t <= enter_i) & (enter_i <= exit_t)  # (D, C)
    j_is_desc = (enter_t <= enter_j) & (enter_j <= exit_t)  # (D, C)
    effective = (i_is_desc != j_is_desc).T                  # (C, D)
    i_moves = i_is_desc.T                                   # (C, D)

    # Position differences for closures
    pos_i = coords[closure_i]  # (C, 3)
    pos_j = coords[closure_j]  # (C, 3)
    diff = pos_i - pos_j       # (C, 3)
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    dist = np.maximum(dist, 1e-10)
    diff_norm = diff / dist    # (C, 3)

    # r vectors: position relative to axis origin
    # r_i[c, d] = pos_i[c] - axis_origins[d]
    r_i = pos_i[:, None, :] - axis_origins[None, :, :]  # (C, D, 3)
    r_j = pos_j[:, None, :] - axis_origins[None, :, :]  # (C, D, 3)

    # Select r based on which atom moves
    r = np.where(i_moves[:, :, None], r_i, r_j)  # (C, D, 3)

    # dpos = axis × r
    # axis_dirs is (D, 3), need to broadcast to (C, D, 3)
    dpos = np.cross(axis_dirs[None, :, :], r)  # (C, D, 3)

    # Project onto distance direction: sign matters
    sign = np.where(i_moves, 1.0, -1.0)  # (C, D)

    # J_dist[c, d] = sign * (diff_norm[c] · dpos[c, d])
    J_dist = sign * np.einsum('ci,cdi->cd', diff_norm, dpos)  # (C, D)
    J_dist = np.where(effective, J_dist, 0.0)

    # Assemble (3C, D) matrix - only distance constraints for now
    J = np.zeros((3 * C, D), dtype=np.float32)
    J[0::3, :] = J_dist

    return J


def _get_rotation_axis(
    torsion_atom: int,
    parent: np.ndarray,
    coords: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Get the rotation axis for a torsion.

    The torsion at atom k rotates around the bond from parent[k] toward
    the grandparent (parent[parent[k]]).

    Returns:
        (axis_origin, axis_direction) or (None, None) if invalid
    """
    p = int(parent[torsion_atom])
    if p < 0:
        return None, None

    pp = int(parent[p])
    if pp < 0:
        return None, None

    axis_origin = coords[p].copy()
    axis_dir = coords[pp] - coords[p]
    norm = np.linalg.norm(axis_dir)
    if norm < 1e-10:
        return None, None

    axis_dir = axis_dir / norm
    return axis_origin, axis_dir


def compute_jacobian_for_backward(
    internal: np.ndarray,
    coords: np.ndarray,
    system: "ConstraintSystem",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobians needed for backward pass (implicit differentiation).

    Returns:
        (J_dep, J_ind) - Jacobians w.r.t. dependent and independent torsions
    """
    J_dep = compute_jacobian_analytical(
        internal, coords, system, system.dependent_idx
    )
    J_ind = compute_jacobian_analytical(
        internal, coords, system, system.independent_idx
    )

    return J_dep, J_ind
