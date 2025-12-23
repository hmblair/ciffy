"""
Ring closure solver using Cyclic Coordinate Descent (CCD).

This module provides algorithms for solving ring closure constraints
in internal coordinate systems. When bonds and angles are fixed,
rings create dependencies between dihedral angles - some dihedrals
are "dependent" and must be computed to close the ring.

The CCD algorithm iteratively adjusts dependent dihedrals to minimize
the closure error (distance between where the closing atom is vs where
it should be).

Mathematical Background
-----------------------
For a ring with k atoms and fixed bonds/angles:
- k-3 dihedrals are independent (can be set freely)
- 3 dihedrals are dependent (determined by ring closure)

CCD works by optimizing one dihedral at a time:
1. For each dependent dihedral, the closure atom traces a circle
   when that dihedral is rotated
2. Find the rotation angle that brings the closure atom closest
   to its target position
3. This 1D optimization has an analytical solution

References
----------
- Canutescu & Dunbrack (2003): Cyclic coordinate descent
- Coutsias et al. (2004): Kinematic closure for loop modeling
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch
from .geometry import (
    norm,
    normalize,
    clone,
    to_scalar,
    optimal_rotation_to_target,
)
from .ring_analysis import RingAnalyzer

if TYPE_CHECKING:
    pass


# =============================================================================
# CCD Ring Closure Solver
# =============================================================================

class CCDSolver:
    """
    Cyclic Coordinate Descent solver for ring closure.

    Given a ring with fixed bonds and angles, and some independent dihedral
    values, computes the dependent dihedral values that close the ring.

    Attributes:
        max_iterations: Maximum CCD iterations per ring
        tolerance: Convergence tolerance for closure distance (Angstroms)
        verbose: Whether to print convergence information
        tree: SpanningTree for NERF reconstruction
        fixed_coords: Reference coordinates for atoms at levels 0-2
        offsets: Per-component centering offsets
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False,
        tree=None,
        fixed_coords=None,
        offsets=None,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.tree = tree
        self.fixed_coords = fixed_coords
        self.offsets = offsets

    def _full_reconstruct(self, internal: Array, zmatrix_indices: np.ndarray) -> Array:
        """Do full NERF reconstruction with current internal coordinates."""
        if self.tree is None:
            raise RuntimeError("SpanningTree not set - cannot reconstruct")

        # Use SpanningTree's unified reconstruction
        internal_np = internal.astype(np.float32) if not is_torch(internal) else internal.cpu().numpy().astype(np.float32)
        coords = self.tree.internal_to_cartesian(internal_np, self.fixed_coords, self.offsets)

        # Convert back to torch if needed
        if is_torch(internal):
            import torch
            coords = torch.from_numpy(coords).to(internal.device)

        return coords

    def solve_ring(
        self,
        internal: Array,
        zmatrix_indices: np.ndarray,
        coords: Array,
        ring_atoms: np.ndarray,
        dependent_dihedrals: np.ndarray,
        closure_bond: tuple[int, int],
        expected_closure_distance: float,
    ) -> Array:
        """
        Solve ring closure for a single ring.

        Args:
            internal: (N, 3) internal coordinates [distance, angle, dihedral]
            zmatrix_indices: (N, 4) Z-matrix indices
            coords: (N, 3) current Cartesian coordinates
            ring_atoms: (k,) atom indices forming the ring
            dependent_dihedrals: (3,) indices of dependent dihedral atoms
            closure_bond: (i, j) atom pair forming the closure bond
            expected_closure_distance: Expected length of closure bond

        Returns:
            (N, 3) updated internal coordinates with dependent dihedrals solved
        """
        internal = clone(internal)
        n_rows = len(internal)

        closure_atom_i, closure_atom_j = closure_bond

        # Build atom -> row mapping for Z-matrix lookup
        atom_to_row = {}
        for row in range(len(zmatrix_indices)):
            atom = int(zmatrix_indices[row, 0])
            atom_to_row[atom] = row

        n_atoms = len(atom_to_row)

        # Find effective dihedrals using ring_analysis (already pre-computed correctly)
        # These are dihedrals where rotating them moves EXACTLY ONE closure atom
        effective_dihedrals = RingAnalyzer.find_effective_dihedrals(
            zmatrix_indices, closure_bond, ring_atoms, n_atoms
        )

        if self.verbose and effective_dihedrals:
            print(f"  Effective dihedrals: {[(d, 'moves_i' if mi else 'moves_j') for d, mi, mj in effective_dihedrals]}")

        if not effective_dihedrals:
            if self.verbose:
                print(f"  No effective dihedrals for this ring - cannot close")
            return internal

        for iteration in range(self.max_iterations):
            # Do full reconstruction to get current coordinates
            coords = self._full_reconstruct(internal, zmatrix_indices)

            # Compute current closure error
            current_distance = norm(coords[closure_atom_j] - coords[closure_atom_i])
            closure_error = abs(to_scalar(current_distance) - expected_closure_distance)

            if self.verbose:
                print(f"  Iteration {iteration}: closure error = {closure_error:.6f} A")

            if closure_error < self.tolerance:
                if self.verbose:
                    print(f"  Converged in {iteration} iterations")
                break

            # Adjust each effective dependent dihedral
            for dep_atom, i_affected, j_affected in effective_dihedrals:
                # Get the Z-matrix row for this atom
                dep_row = atom_to_row.get(dep_atom, -1)
                if dep_row < 0:
                    continue

                # Get rotation axis from Z-matrix
                # The dihedral at this atom rotates about the bond from
                # ang_ref to dist_ref (the axis points from ang_ref toward dist_ref)
                dist_ref = int(zmatrix_indices[dep_row, 1])
                ang_ref = int(zmatrix_indices[dep_row, 2])

                if dist_ref < 0 or ang_ref < 0:
                    continue

                # Which closure atom moves when we rotate this dihedral?
                moving_atom = closure_atom_i if i_affected else closure_atom_j
                anchor_atom = closure_atom_j if i_affected else closure_atom_i

                # Rotation axis: from dist_ref toward ang_ref
                # In NERF, the z-axis is (ang_ref - dist_ref), so positive dihedral
                # rotation is counterclockwise around this axis (right-hand rule)
                axis_origin = coords[dist_ref]
                axis_direction = normalize(coords[ang_ref] - coords[dist_ref])

                # Target: we want moving_atom to be at distance expected_closure_distance
                # from anchor_atom. The target is on a sphere, so we aim for the
                # closest point on that sphere to current position.
                current_pos = coords[moving_atom]
                anchor_pos = coords[anchor_atom]

                # Direction from anchor to current moving atom
                direction = current_pos - anchor_pos
                dir_norm = norm(direction)
                if to_scalar(dir_norm) < 1e-10:
                    continue
                direction = direction / dir_norm

                # Target position: on the sphere at expected distance
                if is_torch(anchor_pos):
                    import torch
                    target = anchor_pos + direction * torch.tensor(
                        expected_closure_distance, dtype=anchor_pos.dtype, device=anchor_pos.device
                    )
                else:
                    target = anchor_pos + direction * expected_closure_distance

                # Find optimal rotation angle using geometry module
                optimal_angle = optimal_rotation_to_target(
                    current_pos, target, axis_origin, axis_direction
                )

                # Update the dihedral (use Z-matrix row index, not atom index)
                current_dihedral = to_scalar(internal[dep_row, 2])
                new_dihedral = current_dihedral + optimal_angle

                # Wrap to [-pi, pi]
                while new_dihedral > np.pi:
                    new_dihedral -= 2 * np.pi
                while new_dihedral < -np.pi:
                    new_dihedral += 2 * np.pi

                if is_torch(internal):
                    import torch
                    internal[dep_row, 2] = torch.tensor(
                        new_dihedral, dtype=internal.dtype, device=internal.device
                    )
                else:
                    internal[dep_row, 2] = new_dihedral

                # Do full reconstruction after updating this dihedral
                coords = self._full_reconstruct(internal, zmatrix_indices)

        return internal


class HybridRingSolver:
    """
    Hybrid ring closure solver: analytical for simple rings, optimization for complex.

    Uses analytical circle-sphere intersection for 5-6 member unfused rings,
    which is exact and fast. Falls back to multi-start optimization for fused
    rings or when analytical solution fails.

    The optimization fallback optimizes ALL ring dihedrals together (not just
    "dependent" ones) because ring dihedrals are coupled through the closure
    constraint - some combinations of "independent" values cannot be closed
    by adjusting only the dependent dihedrals.

    Attributes:
        analytical: AnalyticalRingSolver for simple rings
        polynomial: PolynomialRingSolver for optimization-based closure
        tree: SpanningTree for NERF reconstruction
        fixed_coords: Reference coordinates for atoms at levels 0-2
        offsets: Per-component centering offsets
    """

    def __init__(
        self,
        tree,
        fixed_coords: np.ndarray,
        offsets: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 0.01,
    ):
        """
        Initialize hybrid solver.

        Args:
            tree: SpanningTree for NERF reconstruction
            fixed_coords: Reference coordinates for atoms at levels 0-2
            offsets: Per-component centering offsets
            max_iterations: Maximum iterations (unused, for API compatibility)
            tolerance: Convergence tolerance in Angstroms
        """
        from .analytical_closure import AnalyticalRingSolver
        from .polynomial_closure import PolynomialRingSolver

        self.tree = tree
        self.fixed_coords = fixed_coords
        self.offsets = offsets
        self.tolerance = tolerance

        self.analytical = AnalyticalRingSolver(tree, fixed_coords, offsets)
        self.polynomial = PolynomialRingSolver(
            tree=tree,
            fixed_coords=fixed_coords,
            offsets=offsets,
            tolerance=tolerance,
        )

    def solve_ring(
        self,
        internal: Array,
        ring_constraint,
        coords: Array,
    ) -> tuple[Array, bool]:
        """
        Solve ring closure using best available method.

        Tries analytical first for simple rings, falls back to optimization.

        Args:
            internal: (N, 3) internal coordinates
            ring_constraint: RingConstraint to solve
            coords: (N, 3) original Cartesian coordinates

        Returns:
            (updated_internal, success) tuple
        """
        # Try analytical first for simple rings
        if self.analytical.can_solve_analytically(ring_constraint):
            result, success = self.analytical.solve_ring(
                internal, ring_constraint, coords
            )
            if success:
                return result, True

        # Fall back to multi-start optimization
        # This optimizes ALL ring dihedrals, not just dependent ones
        result, success = self.polynomial.solve_ring(
            internal, ring_constraint, coords
        )

        return result, success


def _group_fused_rings(ring_constraints: list) -> list[list]:
    """
    Group fused rings together for simultaneous solving.

    Fused rings share atoms, so modifying dihedrals in one ring affects
    the other. They must be solved together to ensure all closures are satisfied.

    Args:
        ring_constraints: List of RingConstraint objects

    Returns:
        List of groups, where each group is a list of ring indices that should
        be solved together. Unfused rings are in singleton groups.
    """
    n = len(ring_constraints)
    if n == 0:
        return []

    # Build adjacency for fused rings
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue

        # BFS to find all rings fused with ring i
        group = []
        queue = deque([i])
        visited[i] = True

        while queue:
            curr = queue.popleft()
            group.append(curr)

            # Add all rings fused with current ring
            if ring_constraints[curr].fused_with:
                for j in ring_constraints[curr].fused_with:
                    if not visited[j]:
                        visited[j] = True
                        queue.append(j)

        groups.append(group)

    return groups


def solve_ring_closure(
    internal: Array,
    parent: np.ndarray,
    coords: Array,
    ring_constraints: list,
    tree=None,
    fixed_coords: np.ndarray = None,
    offsets: np.ndarray = None,
    max_iterations: int = 100,
    tolerance: float = 0.01,
    use_hybrid: bool = True,
) -> Array:
    """
    Solve all ring closure constraints.

    Uses L-BFGS-B optimization to find dihedral values that close all rings.
    Handles fused rings by solving them together.

    Args:
        internal: (N, 3) internal coordinates [distance, angle, dihedral]
        parent: (N,) int64 parent array from spanning tree
        coords: (N, 3) original Cartesian coordinates (for closure distances)
        ring_constraints: List of RingConstraint objects
        tree: SpanningTree for NERF reconstruction
        fixed_coords: (N, 3) reference coordinates for atoms at levels 0-2
        offsets: (n_components, 3) per-component centering offsets
        max_iterations: Maximum iterations (unused, for API compatibility)
        tolerance: Convergence tolerance in Angstroms
        use_hybrid: Unused, kept for API compatibility

    Returns:
        (N, 3) updated internal coordinates with ring closures solved
    """
    if not ring_constraints:
        return internal

    if tree is None:
        raise ValueError("SpanningTree must be provided for ring closure solving")

    internal = clone(internal)

    # Group fused rings together
    groups = _group_fused_rings(ring_constraints)

    # Solve each group (single rings or fused ring sets)
    for group in groups:
        rings = [ring_constraints[i] for i in group]
        internal = _solve_rings_lbfgs(
            internal, rings, tree, fixed_coords, offsets, coords, tolerance
        )

    return internal


def _solve_rings_lbfgs(
    internal: Array,
    rings: list,
    tree,
    fixed_coords: np.ndarray,
    offsets: np.ndarray,
    original_coords: Array,
    tolerance: float,
) -> Array:
    """
    Solve ring closure using L-BFGS-B optimization.

    Optimizes ALL dihedrals in the given rings to satisfy closure constraints.
    Fast and robust for both single and fused rings.

    Args:
        internal: (N, 3) internal coordinates
        rings: List of RingConstraint objects to solve together
        tree: SpanningTree for NERF
        fixed_coords: Reference coordinates
        offsets: Per-component offsets
        original_coords: Original coordinates for closure distances
        tolerance: Convergence tolerance

    Returns:
        Updated internal coordinates
    """
    from scipy.optimize import minimize

    internal = clone(internal)
    internal_np = internal if isinstance(internal, np.ndarray) else internal.cpu().numpy()
    internal_np = internal_np.astype(np.float32)
    original_np = original_coords if isinstance(original_coords, np.ndarray) else original_coords.cpu().numpy()

    # Collect all ring atoms
    all_ring_atoms = set()
    for ring in rings:
        all_ring_atoms.update(ring.ring_atoms.tolist())

    # With parent-based storage, atom k's data is at row k (identity mapping)
    # Ring dihedral rows are atoms with level >= 3 (have valid dihedrals)
    levels = tree.level
    ring_rows = sorted([int(a) for a in all_ring_atoms if levels[int(a)] >= 3])

    if not ring_rows:
        return internal

    # Collect closure constraints
    constraints = []
    for ring in rings:
        closure_i, closure_j = ring.closure_bond
        expected_dist = float(np.linalg.norm(original_np[closure_j] - original_np[closure_i]))
        constraints.append((closure_i, closure_j, expected_dist))

    # Early exit if already closed
    current_coords = tree.internal_to_cartesian(internal_np, fixed_coords, offsets)
    max_error = max(
        abs(np.linalg.norm(current_coords[cj] - current_coords[ci]) - d)
        for ci, cj, d in constraints
    )
    if max_error < max(tolerance, 0.001):
        return internal

    # Objective: sum of squared closure errors
    def objective(dihedrals):
        internal_work = internal_np.copy()
        for i, row in enumerate(ring_rows):
            internal_work[row, 2] = dihedrals[i]
        coords = tree.internal_to_cartesian(internal_work, fixed_coords, offsets)
        return sum(
            (np.linalg.norm(coords[cj] - coords[ci]) - d) ** 2
            for ci, cj, d in constraints
        )

    # Optimize from current values
    initial = np.array([internal_np[r, 2] for r in ring_rows], dtype=np.float32)
    result = minimize(objective, initial, method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-9})

    # Update if successful
    if np.sqrt(result.fun / len(constraints)) < tolerance:
        for i, row in enumerate(ring_rows):
            internal_np[row, 2] = result.x[i]
            internal_np[row, 2] = np.mod(internal_np[row, 2] + np.pi, 2 * np.pi) - np.pi
        if is_torch(internal):
            import torch
            return torch.from_numpy(internal_np).to(internal.device)
        return internal_np

    # Multi-start fallback (only if first attempt failed)
    rng = np.random.default_rng(42)
    best_x, best_f = result.x, result.fun

    for _ in range(5):
        x0 = rng.uniform(-np.pi, np.pi, len(ring_rows)).astype(np.float32)
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50})
        if res.fun < best_f:
            best_x, best_f = res.x, res.fun
            if np.sqrt(best_f / len(constraints)) < tolerance:
                break

    for i, row in enumerate(ring_rows):
        internal_np[row, 2] = best_x[i]
        internal_np[row, 2] = np.mod(internal_np[row, 2] + np.pi, 2 * np.pi) - np.pi

    if is_torch(internal):
        import torch
        return torch.from_numpy(internal_np).to(internal.device)
    return internal_np
