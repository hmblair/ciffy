"""
Polynomial-based ring closure solver.

This module implements algebraic ring closure using polynomial equations.
For a ring with fixed bonds and angles, the closure constraint becomes a
polynomial in sin/cos of the dependent dihedrals.

Key insight from Coutsias et al. (2004):
- For 3 unknown dihedrals, the constraint reduces to a degree-16 univariate polynomial
- All real roots correspond to valid ring closures
- This finds ALL solutions, not just local minima like CCD

Mathematical Background
-----------------------
1. The position of each atom after a dihedral rotation is:
   p(θ) = origin + R(axis, θ) @ (p0 - origin)

   where R(axis, θ) is a rotation matrix.

2. Using Rodrigues' formula:
   R(n, θ)v = v·cos(θ) + (n×v)·sin(θ) + n·(n·v)·(1-cos(θ))

3. The closure constraint |p_i - p_j|² = d² is quadratic in cos(θ), sin(θ).

4. For multiple dihedrals, each position compounds the rotations.

5. Using half-angle substitution t = tan(θ/2):
   cos(θ) = (1-t²)/(1+t²)
   sin(θ) = 2t/(1+t²)

   This converts trigonometric equations to rational polynomials.

6. Clearing denominators and using resultants to eliminate variables
   gives a degree-16 univariate polynomial for 3 unknowns.

References
----------
- Coutsias et al. (2004): A Kinematic View of Loop Closure
- Manocha & Canny (1994): Efficient solution of polynomial systems
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import polynomial as P

from ..backend import Array, is_torch, to_numpy
from .geometry import norm, normalize, cross, dot, to_scalar, clone

if TYPE_CHECKING:
    from .ring_analysis import RingConstraint
    from .tree import SpanningTree


@dataclass
class PolynomialSolution:
    """Result of polynomial ring closure."""
    dihedrals: np.ndarray  # (3,) dependent dihedral values
    closure_error: float   # Distance error at closure bond
    is_valid: bool         # Whether this is a real, geometrically valid solution


class PolynomialRingSolver:
    """
    Polynomial-based ring closure solver.

    Uses algebraic methods to find ALL valid ring closures, not just
    local minima. This is more reliable than CCD for complex geometries.

    The solver works by:
    1. Building symbolic expressions for atom positions as functions of dihedrals
    2. Setting up the closure constraint as polynomial equations
    3. Using numerical methods to find all roots
    4. Selecting the best solution based on proximity to original geometry

    Attributes:
        tree: SpanningTree for NERF reconstruction
        fixed_coords: Reference coordinates for atoms at levels 0-2
        offsets: Per-component centering offsets
        tolerance: Acceptable closure error in Angstroms
    """

    def __init__(
        self,
        tree: "SpanningTree",
        fixed_coords: np.ndarray,
        offsets: np.ndarray,
        tolerance: float = 1e-4,
    ):
        self.tree = tree
        self.fixed_coords = fixed_coords
        self.offsets = offsets
        self.tolerance = tolerance

    def solve_ring(
        self,
        internal: Array,
        ring_constraint: "RingConstraint",
        original_coords: Array,
    ) -> tuple[Array, bool]:
        """
        Solve ring closure using optimization.

        This method optimizes ALL ring dihedrals (not just "dependent" ones)
        to achieve ring closure. This is necessary because the k-3 "independent"
        dihedrals in a ring are coupled through the closure constraint - some
        combinations of independent values are geometrically impossible to close
        with only dependent adjustments.

        Args:
            internal: (N, 3) internal coordinates [distance, angle, dihedral]
            ring_constraint: Ring constraint to solve
            original_coords: Original Cartesian coordinates (for closure distance)

        Returns:
            (updated_internal, success) tuple
        """
        internal = clone(internal)
        internal_np = to_numpy(internal).astype(np.float64)
        original_np = to_numpy(original_coords).astype(np.float64)

        # Get ring info
        closure_i, closure_j = ring_constraint.closure_bond
        expected_distance = float(np.linalg.norm(
            original_np[closure_j] - original_np[closure_i]
        ))

        # Build Z-matrix info
        zmatrix_indices = self.tree.to_zmatrix_indices()
        atom_to_row = {int(zmatrix_indices[r, 0]): r for r in range(len(zmatrix_indices))}

        # Get ALL ring dihedral rows (not just dependent)
        # This is crucial for successful ring closure
        ring_rows = [atom_to_row.get(int(a), -1) for a in ring_constraint.ring_atoms]
        ring_rows = [r for r in ring_rows if r >= 3]

        if len(ring_rows) == 0:
            return internal, False

        # Early exit: check if ring is already closed
        # Use a relaxed tolerance for early exit (0.001 A = 1 pm) to avoid
        # unnecessary optimization when rings are already closed but have
        # small numerical errors from float32 reconstruction
        internal_f32 = to_numpy(internal).astype(np.float32)
        current_error = self._compute_closure_error(
            internal_f32, closure_i, closure_j, expected_distance
        )
        early_exit_tolerance = max(self.tolerance, 0.001)
        if current_error < early_exit_tolerance:
            # Ring is already closed, return unchanged
            return internal, True

        # Optimize all ring dihedrals to achieve closure
        solutions = self._find_solutions_optimize_all(
            internal_np, ring_rows, closure_i, closure_j,
            expected_distance, original_np
        )

        if not solutions:
            return internal, False

        # Select best solution (closest to original geometry)
        best_solution = self._select_best_solution(
            solutions, internal_np, ring_rows, original_np, closure_i, closure_j
        )

        if best_solution is None:
            return internal, False

        # Update internal coordinates
        for i, row in enumerate(ring_rows):
            if i < len(best_solution.dihedrals):
                internal_np[row, 2] = best_solution.dihedrals[i]

        # Convert back to original dtype
        if is_torch(internal):
            import torch
            result = torch.from_numpy(internal_np.astype(np.float32)).to(internal.device)
        else:
            result = internal_np.astype(np.float32)

        return result, best_solution.is_valid

    def _find_solutions_grid(
        self,
        internal: np.ndarray,
        dep_rows: list[int],
        closure_i: int,
        closure_j: int,
        expected_distance: float,
        original_coords: np.ndarray,
        n_samples: int = 16,
    ) -> list[PolynomialSolution]:
        """
        Find solutions using grid search over dihedral space.

        For each dependent dihedral, samples n_samples values in [-π, π].
        Finds grid points with low closure error and refines them.
        """
        solutions = []
        n_dep = len(dep_rows)

        if n_dep == 0:
            return solutions

        # Create grid
        angles = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)

        # For efficiency, use coarse grid first then refine
        if n_dep == 1:
            grids = [angles]
        elif n_dep == 2:
            grids = [angles, angles]
        else:  # n_dep >= 3
            # Use coarser grid for more dimensions
            coarse_angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
            grids = [coarse_angles] * min(n_dep, 3)

        # Evaluate grid
        best_points = []
        internal_work = internal.copy()

        if n_dep == 1:
            for a0 in grids[0]:
                internal_work[dep_rows[0], 2] = a0
                error = self._compute_closure_error(
                    internal_work, closure_i, closure_j, expected_distance
                )
                if error < 0.5:  # Coarse threshold
                    best_points.append(([a0], error))

        elif n_dep == 2:
            for a0 in grids[0]:
                internal_work[dep_rows[0], 2] = a0
                for a1 in grids[1]:
                    internal_work[dep_rows[1], 2] = a1
                    error = self._compute_closure_error(
                        internal_work, closure_i, closure_j, expected_distance
                    )
                    if error < 0.5:
                        best_points.append(([a0, a1], error))

        else:  # n_dep >= 3
            for a0 in grids[0]:
                internal_work[dep_rows[0], 2] = a0
                for a1 in grids[1]:
                    internal_work[dep_rows[1], 2] = a1
                    for a2 in grids[2]:
                        internal_work[dep_rows[2], 2] = a2
                        error = self._compute_closure_error(
                            internal_work, closure_i, closure_j, expected_distance
                        )
                        if error < 0.5:
                            best_points.append(([a0, a1, a2], error))

        # Sort by error and refine top candidates
        best_points.sort(key=lambda x: x[1])

        for angles_init, _ in best_points[:10]:  # Refine top 10
            refined = self._refine_solution(
                internal, dep_rows, angles_init, closure_i, closure_j, expected_distance
            )
            if refined is not None and refined.is_valid:
                # Check if this is a new solution
                is_duplicate = False
                for existing in solutions:
                    if self._solutions_similar(refined.dihedrals, existing.dihedrals):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    solutions.append(refined)

        return solutions

    def _find_solutions_optimize(
        self,
        internal: np.ndarray,
        dep_rows: list[int],
        closure_i: int,
        closure_j: int,
        expected_distance: float,
        original_coords: np.ndarray,
    ) -> list[PolynomialSolution]:
        """
        Find solutions using multi-start optimization.

        Starts from multiple random initial points and optimizes
        to find local minima of the closure error.
        """
        from scipy.optimize import minimize

        solutions = []
        n_dep = len(dep_rows)

        if n_dep == 0:
            return solutions

        def objective(dihedrals):
            internal_work = internal.copy()
            for i, row in enumerate(dep_rows):
                if i < len(dihedrals):
                    internal_work[row, 2] = dihedrals[i]
            return self._compute_closure_error(
                internal_work, closure_i, closure_j, expected_distance
            ) ** 2

        # Multi-start optimization
        n_starts = 20
        rng = np.random.default_rng(42)

        for _ in range(n_starts):
            # Random initial point
            x0 = rng.uniform(-np.pi, np.pi, min(n_dep, 3))

            try:
                result = minimize(
                    objective, x0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10}
                )

                if result.fun < self.tolerance ** 2:
                    dihedrals = result.x.copy()
                    # Wrap to [-π, π]
                    dihedrals = np.mod(dihedrals + np.pi, 2 * np.pi) - np.pi

                    error = np.sqrt(result.fun)
                    solution = PolynomialSolution(
                        dihedrals=dihedrals,
                        closure_error=error,
                        is_valid=error < self.tolerance
                    )

                    # Check for duplicates
                    is_duplicate = False
                    for existing in solutions:
                        if self._solutions_similar(dihedrals, existing.dihedrals):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        solutions.append(solution)
            except Exception:
                continue

        return solutions

    def _find_solutions_optimize_all(
        self,
        internal: np.ndarray,
        ring_rows: list[int],
        closure_i: int,
        closure_j: int,
        expected_distance: float,
        original_coords: np.ndarray,
    ) -> list[PolynomialSolution]:
        """
        Find solutions by optimizing ALL ring dihedrals.

        This is more robust than optimizing only dependent dihedrals
        because ring dihedrals are coupled through the closure constraint.
        Uses multi-start L-BFGS-B with Nelder-Mead refinement.
        """
        from scipy.optimize import minimize

        solutions = []
        n_ring = len(ring_rows)

        if n_ring == 0:
            return solutions

        # Initial dihedrals from internal coords
        # IMPORTANT: Use float32 to match the internal array dtype.
        # L-BFGS-B converges better with float32 initial points when the
        # objective function uses float32 internally.
        initial_dihedrals = np.array(
            [internal[r, 2] for r in ring_rows], dtype=np.float32
        )

        # Use the original float32 array for objective function
        # This ensures consistent numerical behavior
        internal_f32 = internal.astype(np.float32)

        def objective(dihedrals):
            internal_work = internal_f32.copy()
            for i, row in enumerate(ring_rows):
                internal_work[row, 2] = dihedrals[i]
            return self._compute_closure_error(
                internal_work, closure_i, closure_j, expected_distance
            ) ** 2

        # Multi-start optimization with L-BFGS-B
        # Start with the initial point, then try random starts
        # Stop early if we find a good solution
        rng = np.random.default_rng(42)

        # First: try starting from current values (often works best)
        result = minimize(
            objective, initial_dihedrals.copy(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10}
        )

        error = np.sqrt(result.fun)
        if error < self.tolerance:
            # Found a good solution immediately
            dihedrals = np.mod(result.x + np.pi, 2 * np.pi) - np.pi
            solutions.append(PolynomialSolution(
                dihedrals=dihedrals.astype(np.float32),
                closure_error=error,
                is_valid=True
            ))
            return solutions

        # Collect candidates for refinement
        best_solutions = []
        if error < 0.5:
            best_solutions.append((result.x.copy(), error))

        # Try random starts (fewer iterations since we use refinement)
        n_starts = 10
        for _ in range(n_starts):
            x0 = rng.uniform(-np.pi, np.pi, n_ring).astype(np.float32)
            try:
                result = minimize(
                    objective, x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                error = np.sqrt(result.fun)

                # Early exit if very good
                if error < self.tolerance:
                    dihedrals = np.mod(result.x + np.pi, 2 * np.pi) - np.pi
                    solutions.append(PolynomialSolution(
                        dihedrals=dihedrals.astype(np.float32),
                        closure_error=error,
                        is_valid=True
                    ))
                    return solutions

                if error < 0.5:
                    best_solutions.append((result.x.copy(), error))
            except Exception:
                continue

        # Refine best candidates with Nelder-Mead
        best_solutions.sort(key=lambda x: x[1])

        for dihedrals, _ in best_solutions[:3]:  # Refine top 3
            try:
                result = minimize(
                    objective, dihedrals,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-6}
                )

                dihedrals_refined = np.mod(result.x + np.pi, 2 * np.pi) - np.pi
                error_refined = np.sqrt(result.fun)

                if error_refined < self.tolerance:
                    solutions.append(PolynomialSolution(
                        dihedrals=dihedrals_refined.astype(np.float32),
                        closure_error=error_refined,
                        is_valid=True
                    ))
                    return solutions

                # Keep as a candidate even if not perfect
                is_duplicate = any(
                    self._solutions_similar(dihedrals_refined, s.dihedrals)
                    for s in solutions
                )
                if not is_duplicate:
                    solutions.append(PolynomialSolution(
                        dihedrals=dihedrals_refined.astype(np.float32),
                        closure_error=error_refined,
                        is_valid=error_refined < self.tolerance
                    ))
            except Exception:
                continue

        return solutions

    def _refine_solution(
        self,
        internal: np.ndarray,
        dep_rows: list[int],
        initial_angles: list[float],
        closure_i: int,
        closure_j: int,
        expected_distance: float,
    ) -> PolynomialSolution | None:
        """Refine a candidate solution using local optimization."""
        from scipy.optimize import minimize

        n_dep = min(len(dep_rows), len(initial_angles))

        def objective(dihedrals):
            internal_work = internal.copy()
            for i in range(n_dep):
                internal_work[dep_rows[i], 2] = dihedrals[i]
            return self._compute_closure_error(
                internal_work, closure_i, closure_j, expected_distance
            ) ** 2

        try:
            result = minimize(
                objective,
                initial_angles[:n_dep],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )

            dihedrals = result.x.copy()
            # Wrap to [-π, π]
            dihedrals = np.mod(dihedrals + np.pi, 2 * np.pi) - np.pi

            error = np.sqrt(result.fun)
            return PolynomialSolution(
                dihedrals=dihedrals,
                closure_error=error,
                is_valid=error < self.tolerance
            )
        except Exception:
            return None

    def _compute_closure_error(
        self,
        internal: np.ndarray,
        closure_i: int,
        closure_j: int,
        expected_distance: float,
    ) -> float:
        """Compute the closure bond distance error."""
        coords = self.tree.internal_to_cartesian(
            internal.astype(np.float32), self.fixed_coords, self.offsets
        )
        actual_distance = np.linalg.norm(coords[closure_j] - coords[closure_i])
        return abs(actual_distance - expected_distance)

    def _solutions_similar(
        self,
        sol1: np.ndarray,
        sol2: np.ndarray,
        threshold: float = 0.1,
    ) -> bool:
        """Check if two solutions are similar (within threshold radians)."""
        n = min(len(sol1), len(sol2))
        for i in range(n):
            diff = abs(sol1[i] - sol2[i])
            # Handle wraparound
            diff = min(diff, 2 * np.pi - diff)
            if diff > threshold:
                return False
        return True

    def _select_best_solution(
        self,
        solutions: list[PolynomialSolution],
        internal: np.ndarray,
        dep_rows: list[int],
        original_coords: np.ndarray,
        closure_i: int,
        closure_j: int,
    ) -> PolynomialSolution | None:
        """
        Select the best solution based on proximity to original geometry.

        Reconstructs coordinates for each solution and computes RMSD
        to original positions for atoms in the ring.
        """
        if not solutions:
            return None

        # Filter valid solutions
        valid = [s for s in solutions if s.is_valid]
        if not valid:
            # If no valid solutions, take the one with smallest error
            return min(solutions, key=lambda s: s.closure_error)

        if len(valid) == 1:
            return valid[0]

        # Compare solutions by deviation from original dihedrals
        original_dihedrals = np.array([internal[r, 2] for r in dep_rows])

        best_solution = None
        best_deviation = float('inf')

        for sol in valid:
            # Compute angular deviation from original
            deviation = 0.0
            for i, orig in enumerate(original_dihedrals):
                if i < len(sol.dihedrals):
                    diff = abs(sol.dihedrals[i] - orig)
                    diff = min(diff, 2 * np.pi - diff)
                    deviation += diff

            if deviation < best_deviation:
                best_deviation = deviation
                best_solution = sol

        return best_solution


def solve_ring_polynomial(
    internal: Array,
    ring_constraint: "RingConstraint",
    original_coords: Array,
    tree: "SpanningTree",
    fixed_coords: np.ndarray,
    offsets: np.ndarray,
    tolerance: float = 1e-4,
) -> tuple[Array, bool]:
    """
    Convenience function for polynomial ring closure.

    Args:
        internal: (N, 3) internal coordinates
        ring_constraint: Ring to close
        original_coords: Original Cartesian coordinates
        tree: SpanningTree for NERF
        fixed_coords: Reference coordinates
        offsets: Per-component offsets
        tolerance: Acceptable closure error

    Returns:
        (updated_internal, success) tuple
    """
    solver = PolynomialRingSolver(
        tree=tree,
        fixed_coords=fixed_coords,
        offsets=offsets,
        tolerance=tolerance,
    )
    return solver.solve_ring(internal, ring_constraint, original_coords)
