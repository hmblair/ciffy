"""
Ring detection and degrees of freedom analysis for constrained internal coordinates.

This module provides algorithms for:
- Finding fundamental cycles (rings) in molecular bond graphs
- Computing independent vs dependent degrees of freedom
- Analyzing ring closure constraints

The core insight is that for a molecule with N atoms, if we fix B bonds
and A angles, the remaining degrees of freedom are:

    DOF = 3N - 6 - B - A

For acyclic molecules (trees), this equals N-3 independent dihedrals.
For molecules with rings, some dihedrals become dependent through ring
closure constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class RingConstraint:
    """
    Ring closure constraint for a single ring.

    A k-membered ring with fixed bonds and angles has k-3 independent
    dihedrals. The remaining 3 are determined by ring closure (analogous
    to a 6-DOF robotic arm reaching a target position).

    Attributes:
        ring_atoms: (k,) array of atom indices forming the ring, in order.
        ring_size: Number of atoms in the ring (k).
        independent_dihedrals: (k-3,) indices of dihedrals that are independent.
        dependent_dihedrals: (3,) indices of dihedrals determined by ring closure.
        closure_bond: (i, j) atom pair forming the "closing" bond.
    """

    ring_atoms: np.ndarray
    ring_size: int
    independent_dihedrals: np.ndarray
    dependent_dihedrals: np.ndarray
    closure_bond: tuple[int, int]

    def __repr__(self) -> str:
        return (
            f"RingConstraint(size={self.ring_size}, "
            f"independent={len(self.independent_dihedrals)}, "
            f"dependent={len(self.dependent_dihedrals)})"
        )


@dataclass
class ConstraintSpec:
    """
    Specification of which internal coordinates are fixed.

    Allows users to specify constraints at different granularities:
    - "all": Fix all bonds/angles (default for rigid geometry)
    - "none": No fixed bonds/angles (full flexibility)
    - Boolean mask: Selective fixing of individual coordinates

    Attributes:
        fixed_bonds: Which bonds are fixed. "all", "none", or (B,) bool mask.
        fixed_angles: Which angles are fixed. "all", "none", or (A,) bool mask.
        extra_bonds: Additional bond constraints beyond topology.
            List of (atom_i, atom_j, distance) tuples.
        extra_angles: Additional angle constraints beyond topology.
            List of (atom_i, atom_j, atom_k, angle) tuples.

    Example:
        >>> # Fix all covalent geometry (most common case)
        >>> spec = ConstraintSpec(fixed_bonds="all", fixed_angles="all")
        >>>
        >>> # Add hydrogen bond constraints
        >>> spec = ConstraintSpec(
        ...     fixed_bonds="all",
        ...     fixed_angles="all",
        ...     extra_bonds=[(10, 50, 2.9)],  # H-bond distance
        ... )
    """

    fixed_bonds: Literal["all", "none"] | np.ndarray = "all"
    fixed_angles: Literal["all", "none"] | np.ndarray = "all"
    extra_bonds: list[tuple[int, int, float]] | None = None
    extra_angles: list[tuple[int, int, int, float]] | None = None

    def __post_init__(self):
        if self.extra_bonds is None:
            self.extra_bonds = []
        if self.extra_angles is None:
            self.extra_angles = []


@dataclass
class IndependentDOF:
    """
    Computed independent degrees of freedom.

    After analyzing the constraint graph, this structure contains
    which dihedrals are truly independent (can be set freely) vs
    dependent (computed from ring closure).

    Attributes:
        independent_mask: (N,) bool array. True = independent dihedral.
        independent_indices: (K,) indices of independent dihedrals.
        dependent_indices: (D,) indices of dependent dihedrals.
        ring_constraints: List of RingConstraint for each detected ring.
        n_atoms: Total number of atoms.
        n_independent: Number of independent DOF (K).
        n_dependent: Number of dependent DOF (D).
    """

    independent_mask: np.ndarray
    independent_indices: np.ndarray
    dependent_indices: np.ndarray
    ring_constraints: list[RingConstraint]
    n_atoms: int
    n_independent: int
    n_dependent: int

    def __repr__(self) -> str:
        return (
            f"IndependentDOF(atoms={self.n_atoms}, "
            f"independent={self.n_independent}, "
            f"dependent={self.n_dependent}, "
            f"rings={len(self.ring_constraints)})"
        )


class RingAnalyzer:
    """
    Analyzes molecular bond graphs for rings and computes DOF constraints.

    The analyzer uses graph algorithms to:
    1. Detect all fundamental cycles (rings) in the bond graph
    2. Identify which dihedrals are independent vs dependent
    3. Build RingConstraint objects for ring closure solving

    The algorithm is based on finding a spanning tree and identifying
    non-tree edges, each of which creates one fundamental cycle.
    """

    @staticmethod
    def find_fundamental_cycles(
        csr_offsets: np.ndarray,
        csr_neighbors: np.ndarray,
        n_atoms: int,
    ) -> list[np.ndarray]:
        """
        Find all fundamental cycles (rings) in the bond graph.

        Uses the spanning tree method:
        1. Build a spanning tree using BFS
        2. Each non-tree edge creates one fundamental cycle
        3. Trace the path in the tree to find cycle atoms

        Args:
            csr_offsets: (N+1,) CSR offsets for bond graph.
            csr_neighbors: (E,) CSR neighbor indices.
            n_atoms: Total number of atoms.

        Returns:
            List of (k,) arrays, each containing atom indices of a ring
            in traversal order.

        Example:
            >>> # For a 6-membered ring (benzene-like)
            >>> cycles = RingAnalyzer.find_fundamental_cycles(offsets, neighbors, 6)
            >>> len(cycles)
            1
            >>> len(cycles[0])
            6
        """
        if n_atoms == 0:
            return []

        # Build adjacency list for easier traversal
        adj = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            start = csr_offsets[i]
            end = csr_offsets[i + 1]
            for j in range(start, end):
                neighbor = csr_neighbors[j]
                adj[i].append(neighbor)

        # Find spanning tree using BFS and collect non-tree edges
        visited = np.zeros(n_atoms, dtype=bool)
        parent = np.full(n_atoms, -1, dtype=np.int64)
        depth = np.zeros(n_atoms, dtype=np.int64)
        non_tree_edges = []

        # Handle disconnected components
        for start in range(n_atoms):
            if visited[start]:
                continue

            # BFS from this start node
            queue = [start]
            visited[start] = True
            depth[start] = 0

            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        depth[neighbor] = depth[node] + 1
                        queue.append(neighbor)
                    elif parent[node] != neighbor:
                        # Non-tree edge found (not the edge we came from)
                        # Only add each edge once (when node < neighbor)
                        if node < neighbor:
                            non_tree_edges.append((node, neighbor))

        # For each non-tree edge, find the fundamental cycle
        cycles = []
        for u, v in non_tree_edges:
            cycle = RingAnalyzer._find_cycle_from_edge(u, v, parent, depth)
            if cycle is not None and len(cycle) >= 3:
                cycles.append(np.array(cycle, dtype=np.int64))

        return cycles

    @staticmethod
    def _find_cycle_from_edge(
        u: int,
        v: int,
        parent: np.ndarray,
        depth: np.ndarray,
    ) -> list[int] | None:
        """
        Find the cycle created by adding edge (u, v) to the spanning tree.

        The cycle is formed by the path from u to LCA(u, v) plus
        the path from v to LCA(u, v), where LCA is the lowest common ancestor.

        Args:
            u, v: Endpoints of the non-tree edge.
            parent: Parent array from BFS.
            depth: Depth array from BFS.

        Returns:
            List of atom indices in cycle order, or None if no valid cycle.
        """
        # Trace paths from u and v up to their lowest common ancestor
        path_u = []
        path_v = []

        # Bring u and v to the same depth
        u_curr, v_curr = u, v
        while depth[u_curr] > depth[v_curr]:
            path_u.append(u_curr)
            u_curr = parent[u_curr]
        while depth[v_curr] > depth[u_curr]:
            path_v.append(v_curr)
            v_curr = parent[v_curr]

        # Move up together until we find LCA
        while u_curr != v_curr:
            path_u.append(u_curr)
            path_v.append(v_curr)
            u_curr = parent[u_curr]
            v_curr = parent[v_curr]
            if u_curr == -1 or v_curr == -1:
                return None  # Shouldn't happen in connected component

        # u_curr == v_curr is the LCA
        path_u.append(u_curr)

        # Combine paths: u -> LCA -> v (reversed)
        cycle = path_u + path_v[::-1]

        return cycle

    @staticmethod
    def analyze_constraints(
        csr_offsets: np.ndarray,
        csr_neighbors: np.ndarray,
        n_atoms: int,
        zmatrix_indices: np.ndarray,
        constraint_spec: ConstraintSpec,
    ) -> IndependentDOF:
        """
        Analyze constraints and compute independent DOF.

        Given the bond graph, Z-matrix, and constraint specification,
        determines which dihedrals are independent vs dependent.

        For acyclic molecules: all N-3 dihedrals are independent.
        For molecules with rings: some dihedrals become dependent
        through ring closure constraints.

        Args:
            csr_offsets: (N+1,) CSR offsets for bond graph.
            csr_neighbors: (E,) CSR neighbor indices.
            n_atoms: Total number of atoms.
            zmatrix_indices: (N, 4) Z-matrix indices array.
            constraint_spec: User-specified constraints.

        Returns:
            IndependentDOF with analysis results.
        """
        if n_atoms <= 3:
            # First 3 atoms have no independent dihedrals
            return IndependentDOF(
                independent_mask=np.zeros(n_atoms, dtype=bool),
                independent_indices=np.array([], dtype=np.int64),
                dependent_indices=np.array([], dtype=np.int64),
                ring_constraints=[],
                n_atoms=n_atoms,
                n_independent=0,
                n_dependent=0,
            )

        # Find all fundamental cycles
        cycles = RingAnalyzer.find_fundamental_cycles(
            csr_offsets, csr_neighbors, n_atoms
        )

        # Add cycles from extra bonds (long-range constraints)
        if constraint_spec.extra_bonds:
            extra_cycles = RingAnalyzer._find_cycles_from_extra_bonds(
                csr_offsets, csr_neighbors, n_atoms, constraint_spec.extra_bonds
            )
            cycles.extend(extra_cycles)

        # Start with all dihedrals as independent (for atoms >= 3)
        # In Z-matrix, first 3 atoms don't have full dihedral definitions
        independent_mask = np.zeros(n_atoms, dtype=bool)
        independent_mask[3:] = True  # Atoms 3+ have dihedrals

        # Process each ring to mark dependent dihedrals
        ring_constraints = []
        for cycle in cycles:
            ring_constraint = RingAnalyzer._create_ring_constraint(
                cycle, zmatrix_indices, independent_mask
            )
            if ring_constraint is not None:
                ring_constraints.append(ring_constraint)
                # Mark dependent dihedrals
                for dep_idx in ring_constraint.dependent_dihedrals:
                    independent_mask[dep_idx] = False

        # Extract indices
        independent_indices = np.where(independent_mask)[0].astype(np.int64)
        dependent_indices = np.where(~independent_mask & (np.arange(n_atoms) >= 3))[0].astype(np.int64)

        return IndependentDOF(
            independent_mask=independent_mask,
            independent_indices=independent_indices,
            dependent_indices=dependent_indices,
            ring_constraints=ring_constraints,
            n_atoms=n_atoms,
            n_independent=len(independent_indices),
            n_dependent=len(dependent_indices),
        )

    @staticmethod
    def _find_cycles_from_extra_bonds(
        csr_offsets: np.ndarray,
        csr_neighbors: np.ndarray,
        n_atoms: int,
        extra_bonds: list[tuple[int, int, float]],
    ) -> list[np.ndarray]:
        """
        Find cycles created by adding extra bonds (e.g., H-bonds).

        Each extra bond creates a new cycle by connecting two atoms
        that are already connected through the covalent bond graph.

        Args:
            csr_offsets: (N+1,) CSR offsets for covalent bond graph.
            csr_neighbors: (E,) CSR neighbor indices.
            n_atoms: Total number of atoms.
            extra_bonds: List of (atom_i, atom_j, distance) extra constraints.

        Returns:
            List of cycle arrays for cycles created by extra bonds.
        """
        cycles = []

        for atom_i, atom_j, _distance in extra_bonds:
            # Find shortest path between atom_i and atom_j in the covalent graph
            path = RingAnalyzer._find_shortest_path(
                csr_offsets, csr_neighbors, n_atoms, atom_i, atom_j
            )
            if path is not None and len(path) >= 2:
                # The cycle is: path from i to j + edge back from j to i
                cycles.append(np.array(path, dtype=np.int64))

        return cycles

    @staticmethod
    def _find_shortest_path(
        csr_offsets: np.ndarray,
        csr_neighbors: np.ndarray,
        n_atoms: int,
        start: int,
        end: int,
    ) -> list[int] | None:
        """
        Find shortest path between two atoms using BFS.

        Args:
            csr_offsets: (N+1,) CSR offsets.
            csr_neighbors: (E,) CSR neighbors.
            n_atoms: Total atoms.
            start: Starting atom index.
            end: Target atom index.

        Returns:
            List of atom indices from start to end, or None if not connected.
        """
        if start == end:
            return [start]

        visited = np.zeros(n_atoms, dtype=bool)
        parent = np.full(n_atoms, -1, dtype=np.int64)
        queue = [start]
        visited[start] = True

        while queue:
            node = queue.pop(0)
            if node == end:
                # Reconstruct path
                path = []
                curr = end
                while curr != -1:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]

            # Visit neighbors
            for j in range(csr_offsets[node], csr_offsets[node + 1]):
                neighbor = csr_neighbors[j]
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    queue.append(neighbor)

        return None  # Not connected

    @staticmethod
    def _create_ring_constraint(
        cycle: np.ndarray,
        zmatrix_indices: np.ndarray,
        current_independent: np.ndarray,
    ) -> RingConstraint | None:
        """
        Create a RingConstraint from a detected cycle.

        For a k-membered ring with fixed bonds/angles:
        - k-3 dihedrals are independent
        - 3 dihedrals are dependent (determined by ring closure)

        We choose the 3 dependent dihedrals to be those closest to
        the "closing" bond, as this minimizes the closure computation.

        Args:
            cycle: (k,) array of atom indices in ring.
            zmatrix_indices: (N, 4) Z-matrix indices.
            current_independent: Current independent mask (may be updated).

        Returns:
            RingConstraint, or None if cycle is too small.
        """
        k = len(cycle)
        if k < 4:
            return None  # Need at least 4 atoms for a dihedral constraint

        # Find which atoms in the cycle have dihedrals in Z-matrix
        cycle_set = set(cycle)
        cycle_dihedrals = []
        for atom_idx in cycle:
            # Check if this atom has a valid dihedral in Z-matrix
            if atom_idx >= 3:  # First 3 atoms don't have dihedrals
                # Check if this dihedral is still marked as independent
                if current_independent[atom_idx]:
                    cycle_dihedrals.append(atom_idx)

        if len(cycle_dihedrals) < 3:
            # Not enough dihedrals to constrain
            return None

        # Choose last 3 dihedrals as dependent (arbitrary but consistent choice)
        # In practice, these should be near the "closing" point of the ring
        dependent = np.array(cycle_dihedrals[-3:], dtype=np.int64)
        independent = np.array(cycle_dihedrals[:-3], dtype=np.int64)

        # The closure bond is between first and last atom in cycle
        closure_bond = (int(cycle[0]), int(cycle[-1]))

        return RingConstraint(
            ring_atoms=cycle,
            ring_size=k,
            independent_dihedrals=independent,
            dependent_dihedrals=dependent,
            closure_bond=closure_bond,
        )
