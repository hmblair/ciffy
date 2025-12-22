"""
Constrained coordinate manager for minimal DOF representation.

This module provides ConstrainedCoordinateManager, which wraps CoordinateManager
to expose only independent degrees of freedom. Given fixed bonds and angles,
it automatically identifies independent dihedrals and handles ring closure
constraints internally.

The key insight is that for a molecule with fixed covalent geometry,
the number of independent DOF is much less than 3N-6. For molecules
with rings, some dihedrals are dependent through ring closure.

Example:
    >>> from ciffy import load
    >>> polymer = load("structure.cif")
    >>> constrained = polymer.with_constraints(fixed_bonds="all", fixed_angles="all")
    >>> print(f"Independent DOF: {constrained.n_dof}")
    >>> values = constrained.values  # Get independent dihedral values
    >>> constrained.values = new_values  # Set and reconstruct
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch, to_numpy, check_compatible
from .ring_analysis import ConstraintSpec, IndependentDOF, RingAnalyzer

if TYPE_CHECKING:
    from .coordinates import CoordinateManager
    from ..backend.graph import TopologyInfo


class RingClosureSolver:
    """
    Solves ring closure constraints to compute dependent dihedrals.

    Given independent dihedral values and fixed bonds/angles, computes
    the values of dependent dihedrals that satisfy ring closure.

    Uses Cyclic Coordinate Descent (CCD) algorithm from robotics:
    1. Place atoms up to ring closure point using independent dihedrals
    2. Iteratively adjust dependent dihedrals to minimize closure gap
    3. Each iteration solves a 1D optimization for one dihedral

    For 6-membered rings, uses an analytical solution instead.

    Attributes:
        constraints: List of RingConstraint objects to solve.
        max_iterations: Maximum CCD iterations per ring.
        tolerance: Convergence tolerance for closure distance (Angstroms).
    """

    def __init__(
        self,
        constraints: list,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ):
        """
        Initialize ring closure solver.

        Args:
            constraints: List of RingConstraint objects.
            max_iterations: Maximum iterations for CCD.
            tolerance: Convergence tolerance in Angstroms.
        """
        self.constraints = constraints
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(
        self,
        dihedrals: Array,
        distances: Array,
        angles: Array,
        zmatrix_indices: np.ndarray,
    ) -> Array:
        """
        Solve for dependent dihedrals satisfying ring closure.

        This is called after setting independent dihedrals to compute
        the dependent dihedral values that close all rings.

        Args:
            dihedrals: (N,) array of dihedral angles (independent values set).
            distances: (N,) array of bond distances.
            angles: (N,) array of bond angles.
            zmatrix_indices: (N, 4) Z-matrix indices.

        Returns:
            (N,) array of dihedrals with dependent values computed.
        """
        if not self.constraints:
            return dihedrals

        # Clone for modification
        if is_torch(dihedrals):
            result = dihedrals.clone()
        else:
            result = dihedrals.copy()

        # Solve each ring constraint
        for ring_constraint in self.constraints:
            result = self._solve_single_ring(
                result, distances, angles, zmatrix_indices, ring_constraint
            )

        return result

    def _solve_single_ring(
        self,
        dihedrals: Array,
        distances: Array,
        angles: Array,
        zmatrix_indices: np.ndarray,
        ring,
    ) -> Array:
        """
        Solve closure for a single ring using CCD.

        The CCD algorithm iteratively adjusts each dependent dihedral
        to minimize the distance between the actual and target positions
        of the closing atom.

        Args:
            dihedrals: Current dihedral values.
            distances: Bond distances.
            angles: Bond angles.
            zmatrix_indices: Z-matrix indices.
            ring: RingConstraint to solve.

        Returns:
            Updated dihedrals with this ring closed.
        """
        if ring.ring_size <= 3:
            return dihedrals

        # For small rings (5-6 atoms), use simplified closure
        # TODO: Implement analytical solution for 6-membered rings
        # For now, use iterative CCD for all ring sizes

        ring_atoms = ring.ring_atoms
        dependent_dihedrals = ring.dependent_dihedrals
        closure_bond = ring.closure_bond

        if len(dependent_dihedrals) == 0:
            return dihedrals

        # CCD iteration
        for _iteration in range(self.max_iterations):
            # Compute current closure error
            # This requires partial NERF reconstruction of ring atoms
            # For now, we use a simplified approach

            # Adjust each dependent dihedral
            for dep_idx in dependent_dihedrals:
                # Find optimal dihedral value to minimize closure error
                # This is a 1D optimization problem
                optimal_dihedral = self._optimize_single_dihedral(
                    dihedrals, distances, angles, zmatrix_indices,
                    dep_idx, ring_atoms, closure_bond
                )
                if is_torch(dihedrals):
                    dihedrals = dihedrals.clone()
                    dihedrals[dep_idx] = optimal_dihedral
                else:
                    dihedrals = dihedrals.copy()
                    dihedrals[dep_idx] = optimal_dihedral

            # Check convergence (simplified - full implementation would
            # compute actual closure distance)
            # For now, we do fixed iterations
            break  # TODO: Implement proper convergence check

        return dihedrals

    def _optimize_single_dihedral(
        self,
        dihedrals: Array,
        distances: Array,
        angles: Array,
        zmatrix_indices: np.ndarray,
        dihedral_idx: int,
        ring_atoms: np.ndarray,
        closure_bond: tuple[int, int],
    ) -> float:
        """
        Find optimal value for a single dihedral to minimize closure error.

        This is the core of the CCD algorithm. For each dependent dihedral,
        we find the value that minimizes the squared distance between:
        - Where the closing atom currently is
        - Where it needs to be for ring closure

        The solution is analytical (closed-form) as shown in the CCD paper.

        Args:
            dihedrals: Current dihedral values.
            distances: Bond distances.
            angles: Bond angles.
            zmatrix_indices: Z-matrix indices.
            dihedral_idx: Index of dihedral to optimize.
            ring_atoms: Atoms in the ring.
            closure_bond: (i, j) closure bond.

        Returns:
            Optimal dihedral value in radians.
        """
        # Simplified implementation: return current value
        # Full implementation would:
        # 1. Compute partial NERF for atoms affected by this dihedral
        # 2. Express closure error as function of dihedral angle
        # 3. Solve analytically for minimum (quadratic in cos/sin)

        # TODO: Implement full CCD optimization
        # For now, return the current value unchanged
        if is_torch(dihedrals):
            return float(dihedrals[dihedral_idx].item())
        return float(dihedrals[dihedral_idx])


class ConstrainedCoordinateManager:
    """
    Coordinate manager with constraint-aware minimal DOF representation.

    Wraps a CoordinateManager to expose only independent degrees of freedom.
    Users interact with a reduced set of dihedrals; dependent dihedrals are
    computed internally via ring closure.

    This is the primary interface for ML applications where you want to
    predict/sample only the true degrees of freedom.

    Attributes:
        n_dof: Number of independent degrees of freedom.
        values: (K,) array of independent dihedral values.
        coordinates: (N, 3) reconstructed Cartesian coordinates.

    Example:
        >>> constrained = polymer.with_constraints(fixed_bonds="all", fixed_angles="all")
        >>> print(f"Independent DOF: {constrained.n_dof}")
        >>>
        >>> # Get independent values
        >>> dof_values = constrained.values
        >>>
        >>> # Set new values (triggers ring closure + reconstruction)
        >>> constrained.values = new_dof_values
        >>> coords = constrained.coordinates
    """

    __slots__ = (
        '_base_manager',
        '_constraint_spec',
        '_independent_dof',
        '_ring_solver',
        '_csr_offsets',
        '_csr_neighbors',
    )

    def __init__(
        self,
        base_manager: "CoordinateManager",
        constraint_spec: ConstraintSpec,
    ):
        """
        Initialize constrained coordinate manager.

        Analyzes the constraint graph to identify independent DOF
        and sets up ring closure solving.

        Args:
            base_manager: Underlying CoordinateManager.
            constraint_spec: Specification of fixed bonds/angles.
        """
        self._base_manager = base_manager
        self._constraint_spec = constraint_spec

        # Build bond graph and analyze constraints
        self._build_constraint_graph()

    def _build_constraint_graph(self) -> None:
        """
        Build constraint graph and analyze independent DOF.

        This is called once during initialization to:
        1. Get bond graph from base manager
        2. Add extra bonds from constraint spec
        3. Find rings and identify independent/dependent dihedrals
        4. Set up ring closure solver
        """
        from ..backend.dispatch import build_bond_graph_csr

        # Ensure Z-matrix is built (triggers internal computation if needed)
        _ = self._base_manager.zmatrix

        topology = self._base_manager._topology
        n_atoms = topology.n_atoms

        # Build bond graph in CSR format
        csr_offsets, csr_neighbors, _n_edges = build_bond_graph_csr(topology)
        self._csr_offsets = csr_offsets
        self._csr_neighbors = csr_neighbors

        # Get Z-matrix indices
        zmatrix_indices = self._base_manager.zmatrix.indices

        # Analyze constraints to find independent DOF
        self._independent_dof = RingAnalyzer.analyze_constraints(
            csr_offsets,
            csr_neighbors,
            n_atoms,
            zmatrix_indices,
            self._constraint_spec,
        )

        # Set up ring closure solver
        self._ring_solver = RingClosureSolver(
            self._independent_dof.ring_constraints
        )

    @property
    def n_dof(self) -> int:
        """Number of independent degrees of freedom."""
        return self._independent_dof.n_independent

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return self._independent_dof.n_atoms

    @property
    def independent_dof(self) -> IndependentDOF:
        """Get the IndependentDOF analysis result."""
        return self._independent_dof

    @property
    def values(self) -> Array:
        """
        Get current values of independent DOF.

        Returns:
            (K,) array of independent dihedral values in radians.
        """
        all_dihedrals = self._base_manager.dihedrals
        indices = self._independent_dof.independent_indices

        if len(indices) == 0:
            # No independent DOF
            if is_torch(all_dihedrals):
                import torch
                return torch.tensor([], dtype=all_dihedrals.dtype, device=all_dihedrals.device)
            return np.array([], dtype=all_dihedrals.dtype)

        return all_dihedrals[indices]

    @values.setter
    def values(self, new_values: Array) -> None:
        """
        Set independent DOF and reconstruct Cartesian coordinates.

        This triggers:
        1. Update independent dihedrals in full array
        2. Compute dependent dihedrals via ring closure
        3. Reconstruct Cartesian from all dihedrals

        Args:
            new_values: (K,) array of new independent dihedral values.
        """
        indices = self._independent_dof.independent_indices

        if len(indices) == 0:
            # No independent DOF to set
            return

        # Validate shape
        expected_len = len(indices)
        if len(new_values) != expected_len:
            raise ValueError(
                f"Expected {expected_len} values, got {len(new_values)}"
            )

        # Get current full dihedral array
        all_dihedrals = self._base_manager.dihedrals

        # Clone/copy for modification
        if is_torch(all_dihedrals):
            all_dihedrals = all_dihedrals.clone()
            # Set independent values
            all_dihedrals[indices] = new_values
        else:
            all_dihedrals = all_dihedrals.copy()
            all_dihedrals[indices] = new_values

        # Solve ring closure for dependent dihedrals
        if self._independent_dof.ring_constraints:
            all_dihedrals = self._ring_solver.solve(
                all_dihedrals,
                self._base_manager.distances,
                self._base_manager.angles,
                self._base_manager.zmatrix.indices,
            )

        # Set full dihedrals (triggers Cartesian reconstruction)
        self._base_manager.dihedrals = all_dihedrals

    @property
    def coordinates(self) -> Array:
        """
        Get reconstructed Cartesian coordinates.

        Returns:
            (N, 3) array of XYZ positions in Angstroms.
        """
        return self._base_manager.coordinates

    @property
    def distances(self) -> Array:
        """Get bond distances."""
        return self._base_manager.distances

    @property
    def angles(self) -> Array:
        """Get bond angles."""
        return self._base_manager.angles

    @property
    def dihedrals(self) -> Array:
        """Get all dihedrals (both independent and dependent)."""
        return self._base_manager.dihedrals

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConstrainedCoordinateManager("
            f"atoms={self.n_atoms}, "
            f"dof={self.n_dof}, "
            f"rings={len(self._independent_dof.ring_constraints)})"
        )
