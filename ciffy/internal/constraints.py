"""
Unified constraint system for internal coordinates.

This module provides a general system for specifying arbitrary bond and angle
constraints, automatically discovering independent degrees of freedom via
Jacobian analysis, and solving for dependent coordinates using Newton-Raphson.

Key classes:
    ClosureConstraints: Minimal representation of non-tree edge constraints
    ConstraintSystem: Complete system for DOF <-> Cartesian mapping

The key insight is that the spanning tree handles most constraints implicitly.
Only non-tree edges (closure bonds) create actual constraints that need solving,
making the constraint Jacobian very small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..backend.dispatch import TopologyInfo


# =============================================================================
# Closure Constraints
# =============================================================================


@dataclass
class ClosureConstraints:
    """
    Minimal representation of ring closure constraints.

    Only non-tree edges create actual constraints. Each closure bond
    creates a distance constraint: ||x_i - x_j|| = d_ij

    Attributes:
        closure_bonds: (C, 2) int64 - atom pairs forming closure bonds
        closure_distances: (C,) float32 - expected distances
        independent_idx: (K,) int64 - independent torsion atoms (DOF)
        dependent_idx: (D,) int64 - dependent torsion atoms (solved)
    """

    closure_bonds: np.ndarray        # (C, 2) int64
    closure_distances: np.ndarray    # (C,) float32
    independent_idx: np.ndarray      # (K,) int64
    dependent_idx: np.ndarray        # (D,) int64

    @property
    def n_closures(self) -> int:
        """Number of closure constraints."""
        return len(self.closure_bonds)

    @property
    def n_constraints(self) -> int:
        """Total number of scalar constraints (3 per closure for distance)."""
        return 3 * self.n_closures

    @property
    def n_independent(self) -> int:
        """Number of independent torsions."""
        return len(self.independent_idx)

    @property
    def n_dependent(self) -> int:
        """Number of dependent torsions."""
        return len(self.dependent_idx)

    @classmethod
    def empty(cls) -> "ClosureConstraints":
        """Create empty constraints (for molecules with no rings)."""
        return cls(
            closure_bonds=np.zeros((0, 2), dtype=np.int64),
            closure_distances=np.zeros(0, dtype=np.float32),
            independent_idx=np.zeros(0, dtype=np.int64),
            dependent_idx=np.zeros(0, dtype=np.int64),
        )

    def __repr__(self) -> str:
        return (
            f"ClosureConstraints(closures={self.n_closures}, "
            f"independent={self.n_independent}, dependent={self.n_dependent})"
        )


# =============================================================================
# Constraint System
# =============================================================================


@dataclass
class ConstraintSystem:
    """
    Complete system for DOF <-> Cartesian coordinate mapping.

    This is the main interface for working with constrained internal coordinates.
    It supports arbitrary bond and angle constraints, automatically discovers
    independent DOF via Jacobian analysis, and provides forward/backward mapping.

    Attributes:
        n_atoms: Total number of atoms
        parent: (N,) int64 - spanning tree parent array
        level: (N,) int32 - tree depth for each atom
        dfs_enter: (N,) int32 - DFS entry time for O(1) ancestry queries
        dfs_exit: (N,) int32 - DFS exit time for O(1) ancestry queries
        closures: ClosureConstraints for non-tree edges
        base_internal: (N, 3) float32 - reference internal coordinates
        fixed_coords: (N, 3) float32 - reference Cartesian for NERF
        center_offsets: (n_components, 3) float32 or None - centering offsets

    Example:
        >>> # From arbitrary constraints
        >>> system = ConstraintSystem.from_constraints(
        ...     n_atoms=100,
        ...     coords=xyz,
        ...     bond_constraints=[(0, 1, 1.52), (5, 50, 2.9)],  # includes H-bond!
        ...     angle_constraints=[(0, 1, 2, 1.91)],
        ... )
        >>> print(system.n_dof)  # Discovered DOF count

        >>> # From molecular topology (convenience)
        >>> system = ConstraintSystem.from_topology(
        ...     topology=polymer.topology,
        ...     coords=polymer.coordinates,
        ... )
    """

    n_atoms: int
    parent: np.ndarray                    # (N,) int64
    level: np.ndarray                     # (N,) int32
    dfs_enter: np.ndarray                 # (N,) int32 - DFS entry time
    dfs_exit: np.ndarray                  # (N,) int32 - DFS exit time
    closures: ClosureConstraints
    base_internal: np.ndarray             # (N, 3) float32
    fixed_coords: np.ndarray              # (N, 3) float32
    center_offsets: np.ndarray | None     # (n_components, 3) float32

    @property
    def n_dof(self) -> int:
        """Number of independent degrees of freedom."""
        # All torsions at level >= 3 that are independent
        n_torsions = np.sum(self.level >= 3)
        return n_torsions - self.closures.n_dependent

    @property
    def independent_idx(self) -> np.ndarray:
        """Indices of atoms with independent torsions."""
        return self.closures.independent_idx

    @property
    def dependent_idx(self) -> np.ndarray:
        """Indices of atoms with dependent torsions."""
        return self.closures.dependent_idx

    def is_descendant(self, ancestor: int, node: int) -> bool:
        """Check if node is a descendant of ancestor using O(1) DFS timestamps."""
        return (self.dfs_enter[ancestor] <= self.dfs_enter[node] and
                self.dfs_enter[node] <= self.dfs_exit[ancestor])

    @classmethod
    def from_constraints(
        cls,
        n_atoms: int,
        coords: np.ndarray,
        bond_constraints: list[tuple[int, int, float]],
        angle_constraints: list[tuple[int, int, int, float]] | None = None,
    ) -> "ConstraintSystem":
        """
        Create system from arbitrary bond and angle constraints.

        Args:
            n_atoms: Number of atoms
            coords: (N, 3) reference Cartesian coordinates
            bond_constraints: [(i, j, distance), ...] - any atom pairs
            angle_constraints: [(i, j, k, angle), ...] - any atom triples

        Returns:
            ConstraintSystem with DOF automatically discovered
        """
        from .jacobian import discover_dof
        from .tree import SpanningTree

        coords = np.asarray(coords, dtype=np.float32)
        if angle_constraints is None:
            angle_constraints = []

        # Build bond graph from constraints
        bond_pairs = [(i, j) for i, j, d in bond_constraints]
        edges = np.array(bond_pairs, dtype=np.int64)

        # Build CSR representation
        csr_offsets, csr_neighbors = _edges_to_csr(edges, n_atoms)

        # Build spanning tree
        tree = SpanningTree.from_bond_graph(csr_offsets, csr_neighbors, n_atoms)

        # Build DFS timestamps early for O(1) ancestry queries
        dfs_enter, dfs_exit = _build_dfs_timestamps(tree.parent)

        # Find non-tree edges (closure bonds)
        tree_edges = set()
        for k in range(n_atoms):
            p = int(tree.parent[k])
            if p >= 0:
                tree_edges.add((min(k, p), max(k, p)))

        closure_bonds = []
        closure_distances = []
        for i, j, d in bond_constraints:
            edge = (min(i, j), max(i, j))
            if edge not in tree_edges:
                closure_bonds.append([i, j])
                closure_distances.append(d)

        if len(closure_bonds) == 0:
            # No closures - all torsions are independent
            all_torsions = np.where(tree.level >= 3)[0].astype(np.int64)
            closures = ClosureConstraints(
                closure_bonds=np.zeros((0, 2), dtype=np.int64),
                closure_distances=np.zeros(0, dtype=np.float32),
                independent_idx=all_torsions,
                dependent_idx=np.zeros(0, dtype=np.int64),
            )
        else:
            closure_bonds_arr = np.array(closure_bonds, dtype=np.int64)
            closure_distances_arr = np.array(closure_distances, dtype=np.float32)

            # Build preliminary closures (DOF will be filled in by discover_dof)
            closures = ClosureConstraints(
                closure_bonds=closure_bonds_arr,
                closure_distances=closure_distances_arr,
                independent_idx=np.zeros(0, dtype=np.int64),  # placeholder
                dependent_idx=np.zeros(0, dtype=np.int64),     # placeholder
            )

            # Discover DOF via Jacobian analysis (uses DFS timestamps)
            independent_idx, dependent_idx = discover_dof(
                tree.parent, tree.level, dfs_enter, dfs_exit, closures, coords
            )
            closures = ClosureConstraints(
                closure_bonds=closure_bonds_arr,
                closure_distances=closure_distances_arr,
                independent_idx=independent_idx,
                dependent_idx=dependent_idx,
            )

        # Compute internal coordinates
        internal, center_offsets = tree.cartesian_to_internal(coords, center=True)

        # Compute fixed coords (centered)
        if center_offsets is not None:
            fixed_coords = coords - center_offsets[tree.component_id]
        else:
            fixed_coords = coords.copy()

        return cls(
            n_atoms=n_atoms,
            parent=tree.parent,
            level=tree.level,
            dfs_enter=dfs_enter,
            dfs_exit=dfs_exit,
            closures=closures,
            base_internal=internal,
            fixed_coords=fixed_coords,
            center_offsets=center_offsets,
        )

    @classmethod
    def from_topology(
        cls,
        topology: "TopologyInfo",
        coords: np.ndarray,
        fix_covalent_bonds: bool = True,
        fix_covalent_angles: bool = True,
        extra_bonds: list[tuple[int, int, float]] | None = None,
        extra_angles: list[tuple[int, int, int, float]] | None = None,
    ) -> "ConstraintSystem":
        """
        Create system from molecular topology.

        Convenience method that extracts covalent bonds and angles from
        topology, then adds any extra constraints (H-bonds, disulfides, etc.)

        Args:
            topology: TopologyInfo with molecular structure
            coords: (N, 3) reference coordinates
            fix_covalent_bonds: Whether to constrain all covalent bonds
            fix_covalent_angles: Whether to constrain all covalent angles
            extra_bonds: Additional bond constraints
            extra_angles: Additional angle constraints

        Returns:
            ConstraintSystem with DOF discovered
        """
        from ..backend.graph import build_bond_graph_from_topology

        coords = np.asarray(coords, dtype=np.float32)

        # Get covalent bonds from topology
        edges, _ = build_bond_graph_from_topology(topology)

        bond_constraints = []
        if fix_covalent_bonds:
            for i, j in edges:
                if i < j:  # Avoid duplicates
                    dist = float(np.linalg.norm(coords[j] - coords[i]))
                    bond_constraints.append((int(i), int(j), dist))

        # Add extra bonds
        if extra_bonds:
            bond_constraints.extend(extra_bonds)

        # TODO: Extract covalent angles if fix_covalent_angles is True
        angle_constraints = []
        if extra_angles:
            angle_constraints.extend(extra_angles)

        return cls.from_constraints(
            n_atoms=topology.n_atoms,
            coords=coords,
            bond_constraints=bond_constraints,
            angle_constraints=angle_constraints,
        )

    def __repr__(self) -> str:
        return (
            f"ConstraintSystem(n_atoms={self.n_atoms}, n_dof={self.n_dof}, "
            f"closures={self.closures.n_closures})"
        )


# =============================================================================
# Newton-Raphson Ring Closure Solver
# =============================================================================


def solve_closure(
    internal: np.ndarray,
    system: ConstraintSystem,
    coords: np.ndarray | None = None,
    max_iter: int = 10,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Solve ring closure constraints via Newton-Raphson.

    Given internal coordinates with independent torsions set, solve for
    the dependent torsions that close all rings.

    Args:
        internal: (N, 3) internal coordinates [distance, angle, dihedral]
        system: ConstraintSystem with closure constraints
        coords: (N, 3) current Cartesian (for residual computation)
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance (Angstroms)

    Returns:
        (N, 3) updated internal coordinates with closures solved
    """
    from .jacobian import compute_jacobian_analytical

    if system.closures.n_closures == 0:
        return internal  # No closures to solve

    internal = internal.copy()
    dep_idx = system.closures.dependent_idx

    if len(dep_idx) == 0:
        return internal

    # Get current coordinates if not provided
    if coords is None:
        from .tree import SpanningTree
        tree = SpanningTree(
            parent=system.parent,
            level=system.level,
            component_id=np.zeros(system.n_atoms, dtype=np.int32),
            n_components=1,
        )
        coords = tree.internal_to_cartesian(
            internal, system.fixed_coords, system.center_offsets
        )

    for iteration in range(max_iter):
        # Compute current coordinates
        from .tree import SpanningTree
        tree = SpanningTree(
            parent=system.parent,
            level=system.level,
            component_id=np.zeros(system.n_atoms, dtype=np.int32),
            n_components=1,
        )
        current_coords = tree.internal_to_cartesian(
            internal, system.fixed_coords, system.center_offsets
        )

        # Compute residual (constraint violations)
        residual = _compute_residual(current_coords, system.closures)

        # Check convergence
        max_error = np.abs(residual).max()
        if max_error < tol:
            break

        # Compute Jacobian w.r.t. dependent torsions
        J = compute_jacobian_analytical(
            internal, current_coords, system, dep_idx
        )

        # Newton step: solve J @ delta = -residual
        delta, *_ = np.linalg.lstsq(J, -residual, rcond=1e-10)

        # Update dependent torsions
        internal[dep_idx, 2] += delta

        # Wrap to [-pi, pi]
        internal[dep_idx, 2] = np.mod(
            internal[dep_idx, 2] + np.pi, 2 * np.pi
        ) - np.pi

    return internal


def _compute_residual(
    coords: np.ndarray,
    closures: ClosureConstraints,
) -> np.ndarray:
    """
    Compute constraint residuals (violations).

    Returns:
        (3C,) residual vector with distance errors at indices 0, 3, 6, ...
    """
    n_closures = closures.n_closures
    if n_closures == 0:
        return np.zeros(0, dtype=np.float32)

    # Vectorized distance computation
    diff = coords[closures.closure_bonds[:, 1]] - coords[closures.closure_bonds[:, 0]]
    actual_dist = np.linalg.norm(diff, axis=1)

    residual = np.zeros(3 * n_closures, dtype=np.float32)
    residual[0::3] = actual_dist - closures.closure_distances
    return residual


# =============================================================================
# Helper Functions
# =============================================================================


def _edges_to_csr(
    edges: np.ndarray,
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert edge list to CSR format."""
    if len(edges) == 0:
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        neighbors = np.zeros(0, dtype=np.int64)
        return offsets, neighbors

    # Make symmetric
    edges_sym = np.vstack([edges, edges[:, ::-1]])

    # Sort by source
    order = np.lexsort((edges_sym[:, 1], edges_sym[:, 0]))
    edges_sorted = edges_sym[order]

    # Build CSR
    neighbors = edges_sorted[:, 1].astype(np.int64)
    sources = edges_sorted[:, 0]

    offsets = np.zeros(n_atoms + 1, dtype=np.int64)
    for src in sources:
        offsets[src + 1] += 1
    np.cumsum(offsets, out=offsets)

    return offsets, neighbors


def _build_dfs_timestamps(parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build DFS enter/exit timestamps for O(1) ancestry queries.

    Uses iterative DFS to avoid recursion limits on large molecules.
    After building, is_descendant(ancestor, node) can be checked in O(1) as:
        dfs_enter[ancestor] <= dfs_enter[node] <= dfs_exit[ancestor]

    Args:
        parent: (N,) int64 - parent array from spanning tree

    Returns:
        (dfs_enter, dfs_exit) - both (N,) int32 arrays
    """
    n = len(parent)
    dfs_enter = np.zeros(n, dtype=np.int32)
    dfs_exit = np.zeros(n, dtype=np.int32)

    if n == 0:
        return dfs_enter, dfs_exit

    # Build children array using np.bincount
    valid_parents = parent[parent >= 0]
    if len(valid_parents) > 0:
        children_count = np.bincount(valid_parents.astype(np.int64), minlength=n)
    else:
        children_count = np.zeros(n, dtype=np.int64)

    children_indptr = np.zeros(n + 1, dtype=np.int32)
    children_indptr[1:] = np.cumsum(children_count)
    children_indices = np.zeros(int(children_indptr[-1]), dtype=np.int64)

    # Fill children_indices
    insert_pos = children_indptr[:-1].copy()
    for child in range(n):
        p = int(parent[child])
        if p >= 0:
            children_indices[insert_pos[p]] = child
            insert_pos[p] += 1

    # Find root(s) - atoms with parent == -1
    roots = np.where(parent < 0)[0]
    if len(roots) == 0:
        return dfs_enter, dfs_exit

    # Iterative DFS from each root (handles disconnected components)
    time = 0
    for root in roots:
        stack = [(int(root), False)]

        while stack:
            node, exiting = stack.pop()
            if exiting:
                dfs_exit[node] = time
                time += 1
            else:
                dfs_enter[node] = time
                time += 1
                stack.append((node, True))
                # Add children in reverse order for correct DFS order
                start, end = children_indptr[node], children_indptr[node + 1]
                for i in range(end - 1, start - 1, -1):
                    stack.append((int(children_indices[i]), False))

    return dfs_enter, dfs_exit
