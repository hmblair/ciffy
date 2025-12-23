"""
Optimization-based constraint solver for ring closure and distance constraints.

This module provides a unified framework for solving geometric constraints
(ring closures, disulfide bonds, arbitrary distance constraints) by optimizing
dependent dihedral angles to minimize constraint violation.

The solver uses L-BFGS optimization and supports implicit differentiation
for gradient computation through the constraint solving process.

Mathematical Background
-----------------------
Given:
- Internal coordinates (distances, angles, dihedrals)
- A set of distance constraints: ||x_i - x_j|| = d_ij
- Indices of "dependent" dihedrals that can be adjusted

We minimize:
    L = Σ_k (||x_i - x_j||² - d_ij²)²

This squared formulation avoids sqrt gradients and has nicer optimization
landscape than (||x_i - x_j|| - d_ij)².

Implicit Differentiation
------------------------
At the optimum θ*, the gradient ∂L/∂θ = 0.

For any upstream loss that depends on θ*, we can compute gradients without
unrolling the optimizer using the implicit function theorem:

    dθ*/d(inputs) = -[∂²L/∂θ²]⁻¹ @ [∂²L/∂θ∂(inputs)]

This is memory-efficient and numerically stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .tree import SpanningTree


@dataclass
class DistanceConstraint:
    """A distance constraint between two atoms.

    Attributes:
        atom_i: First atom index
        atom_j: Second atom index
        target_distance: Expected distance in Angstroms
        weight: Relative weight for this constraint (default 1.0)
    """
    atom_i: int
    atom_j: int
    target_distance: float
    weight: float = 1.0


class ConstraintSolver:
    """
    Optimization-based solver for geometric constraints.

    Adjusts dependent dihedral angles to satisfy distance constraints
    (ring closures, disulfide bonds, etc.) using L-BFGS optimization.

    Attributes:
        tree: SpanningTree for NERF reconstruction
        fixed_coords: Reference coordinates for atoms at levels 0-2
        offsets: Per-component centering offsets
        max_iterations: Maximum L-BFGS iterations
        tolerance: Convergence tolerance for constraint error
    """

    def __init__(
        self,
        tree: "SpanningTree",
        fixed_coords: np.ndarray,
        offsets: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        """
        Initialize constraint solver.

        Args:
            tree: SpanningTree for NERF reconstruction
            fixed_coords: Reference coordinates for atoms at levels 0-2
            offsets: Per-component centering offsets
            max_iterations: Maximum L-BFGS iterations
            tolerance: Convergence tolerance (squared distance error)
        """
        self.tree = tree
        self.fixed_coords = fixed_coords
        self.offsets = offsets
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(
        self,
        internal: np.ndarray,
        constraints: list[DistanceConstraint],
        dependent_indices: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Solve constraints by optimizing dependent dihedrals.

        Args:
            internal: (N, 3) internal coordinates [distance, angle, dihedral]
            constraints: List of distance constraints to satisfy
            dependent_indices: Indices of atoms whose dihedrals can be adjusted

        Returns:
            (updated_internal, final_error) tuple where:
            - updated_internal: internal coords with optimized dependent dihedrals
            - final_error: sum of squared constraint violations
        """
        if len(constraints) == 0 or len(dependent_indices) == 0:
            return internal.copy(), 0.0

        internal = internal.copy().astype(np.float64)  # Work in float64 for precision
        dependent_indices = np.asarray(dependent_indices, dtype=np.int32)

        # Extract initial values of dependent dihedrals
        initial_dihedrals = internal[dependent_indices, 2].copy()

        # Build constraint data arrays for fast access
        n_constraints = len(constraints)
        constraint_atoms_i = np.array([c.atom_i for c in constraints], dtype=np.int32)
        constraint_atoms_j = np.array([c.atom_j for c in constraints], dtype=np.int32)
        constraint_targets = np.array([c.target_distance for c in constraints], dtype=np.float64)
        constraint_weights = np.array([c.weight for c in constraints], dtype=np.float64)

        # Precompute squared targets
        target_sq = constraint_targets ** 2

        def objective(dihedrals: np.ndarray) -> float:
            """Compute total weighted squared constraint error."""
            # Update internal coordinates with current dihedrals
            internal[dependent_indices, 2] = dihedrals

            # Reconstruct Cartesian coordinates
            coords = self.tree.internal_to_cartesian(
                internal.astype(np.float32), self.fixed_coords, self.offsets
            )

            # Compute constraint errors
            total_error = 0.0
            for k in range(n_constraints):
                i, j = constraint_atoms_i[k], constraint_atoms_j[k]
                diff = coords[j] - coords[i]
                dist_sq = float(np.dot(diff, diff))
                error = (dist_sq - target_sq[k]) ** 2
                total_error += constraint_weights[k] * error

            return total_error

        def gradient(dihedrals: np.ndarray) -> np.ndarray:
            """Compute gradient of objective w.r.t. dihedrals using finite differences."""
            eps = 1e-7
            grad = np.zeros_like(dihedrals)
            f0 = objective(dihedrals)

            for i in range(len(dihedrals)):
                dihedrals_plus = dihedrals.copy()
                dihedrals_plus[i] += eps
                f_plus = objective(dihedrals_plus)
                grad[i] = (f_plus - f0) / eps

            return grad

        # Run L-BFGS optimization
        result = minimize(
            objective,
            initial_dihedrals,
            method='L-BFGS-B',
            jac=gradient,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'gtol': 1e-8,
            }
        )

        # Update internal with optimized dihedrals
        internal[dependent_indices, 2] = result.x

        # Wrap dihedrals to [-π, π]
        internal[dependent_indices, 2] = np.mod(
            internal[dependent_indices, 2] + np.pi, 2 * np.pi
        ) - np.pi

        return internal.astype(np.float32), result.fun

    def solve_ring_constraints(
        self,
        internal: np.ndarray,
        ring_constraints: list,
        original_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Solve all ring closure constraints.

        Optimizes ALL ring dihedrals (not just dependent ones) to satisfy
        closure constraints. This is necessary because ring dihedrals are
        coupled - arbitrary combinations of "independent" values may not
        allow closure with only "dependent" adjustments.

        Args:
            internal: (N, 3) internal coordinates
            ring_constraints: List of RingConstraint objects
            original_coords: Original Cartesian coordinates (for closure distances)

        Returns:
            Updated internal coordinates with rings closed
        """
        if not ring_constraints:
            return internal

        # Collect all distance constraints from ring closures
        constraints = []
        all_ring_atoms = set()

        for ring in ring_constraints:
            ci, cj = ring.closure_bond
            target_dist = float(np.linalg.norm(
                original_coords[cj] - original_coords[ci]
            ))
            constraints.append(DistanceConstraint(
                atom_i=ci,
                atom_j=cj,
                target_distance=target_dist,
            ))
            # Include ALL ring atoms, not just dependent ones
            all_ring_atoms.update(ring.ring_atoms.tolist())

        ring_dihedral_indices = np.array(sorted(all_ring_atoms), dtype=np.int32)

        # Solve all constraints together, optimizing all ring dihedrals
        result, error = self.solve(internal, constraints, ring_dihedral_indices)

        return result


def solve_constraints(
    internal: np.ndarray,
    constraints: list[DistanceConstraint],
    dependent_indices: np.ndarray,
    tree: "SpanningTree",
    fixed_coords: np.ndarray,
    offsets: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """
    Convenience function to solve constraints without creating a solver object.

    Args:
        internal: (N, 3) internal coordinates
        constraints: List of distance constraints
        dependent_indices: Indices of adjustable dihedrals
        tree: SpanningTree for reconstruction
        fixed_coords: Reference coordinates
        offsets: Per-component offsets
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        (updated_internal, final_error) tuple
    """
    solver = ConstraintSolver(
        tree=tree,
        fixed_coords=fixed_coords,
        offsets=offsets,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return solver.solve(internal, constraints, dependent_indices)
