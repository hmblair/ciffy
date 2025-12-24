"""
Ring detection and degrees of freedom analysis for constrained internal coordinates.

This module provides algorithms for:
- Finding fundamental cycles (rings) in molecular bond graphs
- Classifying rings by chemistry (ribose, purine, proline, etc.)
- Computing independent vs dependent degrees of freedom
- Analyzing ring closure constraints

The core insight is that for a molecule with N atoms, if we fix B bonds
and A angles, the remaining degrees of freedom are:

    DOF = 3N - 6 - B - A

For acyclic molecules (trees), this equals N-3 independent dihedrals.
For molecules with rings, some dihedrals become dependent through ring
closure constraints.

Ring classification determines flexibility:
- Flexible rings (ribose, proline): use Cremer-Pople puckering coordinates
- Rigid rings (purine, pyrimidine, aromatics): planar, 0 internal DOF
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .tree import SpanningTree


# =============================================================================
# Ring Classification
# =============================================================================


class RingType(Enum):
    """
    Classification of ring flexibility by chemistry.

    Used to determine how to represent ring degrees of freedom:
    - FLEXIBLE_5: 5-membered flexible ring (ribose, proline) → Cremer-Pople (q₂, φ₂)
    - FLEXIBLE_6: 6-membered flexible ring (cyclohexane-like) → Cremer-Pople (Q, θ, φ)
    - RIGID_PLANAR: Aromatic or base rings (purine, pyrimidine) → 0 internal DOF
    - UNKNOWN: Fallback, treated as rigid
    """
    FLEXIBLE_5 = auto()
    FLEXIBLE_6 = auto()
    RIGID_PLANAR = auto()
    UNKNOWN = auto()


@dataclass
class ClassifiedRing:
    """
    A ring with its chemistry-based type classification.

    This dataclass captures the CHEMICAL properties of a ring - whether it's
    aromatic, what elements it contains, and whether it can pucker. This is
    distinct from RingConstraint which captures the DOF GEOMETRY.

    Design Note - Separation from RingConstraint:
        ClassifiedRing and RingConstraint serve different purposes:
        - ClassifiedRing: Chemistry (aromatic? which elements? can it pucker?)
        - RingConstraint: Geometry (which dihedrals close the ring?)

        A flexible ClassifiedRing still has a RingConstraint - the ring can
        pucker, but its dihedrals are still coupled through closure. These
        are intentionally separate dataclasses to avoid conflating chemistry
        with coordinate constraints.

    Attributes:
        atoms: Atom indices in the ring, in cyclic order.
        ring_type: Chemistry-based classification (flexible vs rigid).
        n_dof: Number of puckering DOF (0 for rigid, 2 for 5-ring, 3 for 6-ring flexible).
        element_pattern: Element composition for debugging.
    """
    atoms: np.ndarray
    ring_type: RingType
    n_dof: int
    element_pattern: str = ""

    def __repr__(self) -> str:
        return (
            f"ClassifiedRing(size={len(self.atoms)}, type={self.ring_type.name}, "
            f"dof={self.n_dof}, pattern='{self.element_pattern}')"
        )


def classify_ring(
    ring_atoms: np.ndarray,
    atom_elements: list[str],
    is_aromatic: np.ndarray | None = None,
) -> ClassifiedRing:
    """
    Classify a ring by its chemistry.

    Uses element composition and aromaticity to determine if a ring is:
    - Flexible (ribose, proline): has puckering DOF
    - Rigid (purine, pyrimidine, aromatics): planar, no internal DOF

    Args:
        ring_atoms: Atom indices in the ring (in order).
        atom_elements: Element symbols for ALL atoms in the molecule.
        is_aromatic: Optional (N,) bool array of aromaticity flags.

    Returns:
        ClassifiedRing with type and DOF count.

    Classification rules:
    1. 5-ring with 1 O + 4 C → ribose/deoxyribose → FLEXIBLE_5 (2 DOF)
    2. 5-ring with 1 N + 4 C → proline → FLEXIBLE_5 (2 DOF)
    3. Any ring marked aromatic → RIGID_PLANAR (0 DOF)
    4. 5 or 6-ring with ≥2 N → purine/pyrimidine base → RIGID_PLANAR (0 DOF)
    5. Default → RIGID_PLANAR (0 DOF)
    """
    k = len(ring_atoms)
    elements = [atom_elements[int(i)] for i in ring_atoms]
    element_counts = Counter(elements)
    element_pattern = "".join(sorted(elements))

    # Check for aromatic rings first (always rigid)
    if is_aromatic is not None:
        if all(is_aromatic[int(i)] for i in ring_atoms):
            return ClassifiedRing(
                atoms=ring_atoms,
                ring_type=RingType.RIGID_PLANAR,
                n_dof=0,
                element_pattern=element_pattern,
            )

    # Check for ribose/deoxyribose: 5-ring with 1 O and 4 C (sugar ring)
    if k == 5:
        n_oxygen = element_counts.get('O', 0)
        n_carbon = element_counts.get('C', 0)
        if n_oxygen == 1 and n_carbon == 4:
            return ClassifiedRing(
                atoms=ring_atoms,
                ring_type=RingType.FLEXIBLE_5,
                n_dof=2,
                element_pattern=element_pattern,
            )

    # Check for proline: 5-ring with 1 N and 4 C (pyrrolidine ring)
    if k == 5:
        n_nitrogen = element_counts.get('N', 0)
        n_carbon = element_counts.get('C', 0)
        if n_nitrogen == 1 and n_carbon == 4:
            return ClassifiedRing(
                atoms=ring_atoms,
                ring_type=RingType.FLEXIBLE_5,
                n_dof=2,
                element_pattern=element_pattern,
            )

    # Check for purine/pyrimidine bases (N-rich heterocycles)
    if k == 5 or k == 6:
        n_nitrogen = element_counts.get('N', 0)
        if n_nitrogen >= 2:
            # Likely a nucleobase ring (imidazole, pyrimidine)
            return ClassifiedRing(
                atoms=ring_atoms,
                ring_type=RingType.RIGID_PLANAR,
                n_dof=0,
                element_pattern=element_pattern,
            )

    # Check for 6-membered all-carbon ring (could be cyclohexane-like)
    if k == 6:
        n_carbon = element_counts.get('C', 0)
        if n_carbon == 6:
            # All-carbon 6-ring: could be aromatic (benzene) or flexible (cyclohexane)
            # Without aromaticity info, default to rigid (conservative)
            # In proteins, Phe/Tyr/Trp rings are aromatic, so this is usually correct
            return ClassifiedRing(
                atoms=ring_atoms,
                ring_type=RingType.RIGID_PLANAR,
                n_dof=0,
                element_pattern=element_pattern,
            )

    # Default: treat as rigid (conservative)
    return ClassifiedRing(
        atoms=ring_atoms,
        ring_type=RingType.RIGID_PLANAR,
        n_dof=0,
        element_pattern=element_pattern,
    )


def classify_rings(
    cycles: list[np.ndarray],
    atom_elements: list[str],
    is_aromatic: np.ndarray | None = None,
) -> tuple[list[ClassifiedRing], list[ClassifiedRing]]:
    """
    Classify all rings and separate into flexible vs rigid.

    Args:
        cycles: List of ring atom arrays from find_fundamental_cycles.
        atom_elements: Element symbols for all atoms.
        is_aromatic: Optional aromaticity flags.

    Returns:
        (flexible_rings, rigid_rings): Two lists of ClassifiedRing.
    """
    flexible = []
    rigid = []

    for cycle in cycles:
        classified = classify_ring(cycle, atom_elements, is_aromatic)
        if classified.ring_type in (RingType.FLEXIBLE_5, RingType.FLEXIBLE_6):
            flexible.append(classified)
        else:
            rigid.append(classified)

    return flexible, rigid


@dataclass
class RingConstraint:
    """
    Ring closure constraint for a single ring.

    A k-membered ring with fixed bonds and angles has k-3 independent
    dihedrals. The remaining 3 are determined by ring closure (analogous
    to a 6-DOF robotic arm reaching a target position).

    Design Note - Separation from ClassifiedRing:
        RingConstraint captures DOF GEOMETRY while ClassifiedRing captures
        CHEMISTRY. Both are needed:
        - RingConstraint: Used by ring_closure.py to solve dependent dihedrals
        - ClassifiedRing: Used for puckering analysis and flexible ring detection

        The same physical ring may appear in both: as a RingConstraint for DOF
        reduction and as a ClassifiedRing for chemistry classification.

    Attributes:
        ring_atoms: (k,) array of atom indices forming the ring, in order.
        ring_size: Number of atoms in the ring (k).
        independent_dihedrals: (k-3,) indices of dihedrals that are independent.
        dependent_dihedrals: (3,) indices of dihedrals determined by ring closure.
        closure_bond: (i, j) atom pair forming the "closing" bond.
        is_fused: True if this ring shares atoms with another ring.
        fused_with: List of indices of rings this one is fused with.
    """

    ring_atoms: np.ndarray
    ring_size: int
    independent_dihedrals: np.ndarray
    dependent_dihedrals: np.ndarray
    closure_bond: tuple[int, int]
    is_fused: bool = False
    fused_with: list[int] | None = None

    def __post_init__(self):
        if self.fused_with is None:
            self.fused_with = []

    def __repr__(self) -> str:
        fused_str = ", fused" if self.is_fused else ""
        return (
            f"RingConstraint(size={self.ring_size}, "
            f"independent={len(self.independent_dihedrals)}, "
            f"dependent={len(self.dependent_dihedrals)}{fused_str})"
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


@dataclass
class UnifiedDOF:
    """
    Unified DOF representation - all degrees of freedom as generalized torsions.

    This provides a single flat array of DOF where all values are angles (radians).
    Ring closure is handled analytically: setting independent dihedrals automatically
    computes dependent dihedrals to close rings.

    Key insight: Puckering and ring dihedrals are mathematically equivalent -
    different parametrizations of the same geometric DOF. By using analytical
    ring closure, ring geometry (including puckering) emerges naturally.

    Attributes:
        n_atoms: Total number of atoms.
        n_dof: Number of independent torsion DOF.
        dof_to_atom: (K,) array mapping DOF index -> atom index.
        atom_to_dof: (N,) array mapping atom index -> DOF index (-1 if dependent).
        independent_mask: (N,) bool array. True = independent dihedral.
        ring_constraints: List of RingConstraint for ring closure.

    Example:
        >>> unified = UnifiedDOF.from_independent_dof(independent_dof)
        >>> dof_values = unified.get_dof(internal)  # (K,) angles
        >>> internal = unified.set_dof(internal, new_values, tree, fixed_coords, offsets, original_coords)
    """

    n_atoms: int
    n_dof: int
    dof_to_atom: np.ndarray  # (K,) int64 - which atom each DOF controls
    atom_to_dof: np.ndarray  # (N,) int64 - DOF index for each atom (-1 if dependent)
    independent_mask: np.ndarray  # (N,) bool - True if atom has independent dihedral
    ring_constraints: list[RingConstraint]

    @classmethod
    def from_independent_dof(cls, independent_dof: IndependentDOF) -> "UnifiedDOF":
        """
        Create UnifiedDOF from IndependentDOF.

        Args:
            independent_dof: The analyzed independent DOF structure.

        Returns:
            UnifiedDOF with mappings for DOF access.
        """
        n_atoms = independent_dof.n_atoms
        n_dof = independent_dof.n_independent

        # DOF to atom mapping: sorted independent indices
        dof_to_atom = independent_dof.independent_indices.copy()

        # Atom to DOF mapping: -1 for dependent, DOF index for independent
        atom_to_dof = np.full(n_atoms, -1, dtype=np.int64)
        for dof_idx, atom_idx in enumerate(dof_to_atom):
            atom_to_dof[atom_idx] = dof_idx

        return cls(
            n_atoms=n_atoms,
            n_dof=n_dof,
            dof_to_atom=dof_to_atom,
            atom_to_dof=atom_to_dof,
            independent_mask=independent_dof.independent_mask.copy(),
            ring_constraints=independent_dof.ring_constraints,
        )

    def get_dof(self, internal: np.ndarray) -> np.ndarray:
        """
        Extract DOF values from internal coordinates.

        Simply reads the dihedral (column 2) at independent atom positions.

        Args:
            internal: (N, 3) internal coordinates [distance, angle, dihedral].

        Returns:
            (K,) array of DOF values (angles in radians).
        """
        if self.n_dof == 0:
            return np.array([], dtype=internal.dtype)
        return internal[self.dof_to_atom, 2].copy()

    def set_dof(
        self,
        internal: np.ndarray,
        values: np.ndarray,
        tree: "SpanningTree",
        fixed_coords: np.ndarray,
        offsets: np.ndarray | None,
        original_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Set DOF values and solve ring closure.

        .. deprecated::
            Use the new ConstraintSystem from ciffy.internal.constraints instead.
            This method is kept for backwards compatibility but redirects to the
            new Newton-Raphson solver.

        Args:
            internal: (N, 3) internal coordinates [distance, angle, dihedral].
            values: (K,) new DOF values (angles in radians).
            tree: SpanningTree for NERF reconstruction.
            fixed_coords: Reference coordinates for atoms at levels 0-2.
            offsets: Per-component centering offsets, or None.
            original_coords: Original Cartesian coordinates for solution selection.

        Returns:
            (N, 3) updated internal coordinates with rings closed.
        """
        internal = internal.copy()

        # 1. Set independent dihedrals
        if self.n_dof > 0 and len(values) > 0:
            internal[self.dof_to_atom, 2] = values

        # 2. For ring closure, users should use the new ConstraintSystem
        # This legacy method just sets independent DOF without solving closures
        # Ring closure is handled by MolecularGeometry via the new system

        return internal

    def __repr__(self) -> str:
        return (
            f"UnifiedDOF(atoms={self.n_atoms}, "
            f"dof={self.n_dof}, "
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

    Key insight for ring closure:
    - The Z-matrix represents atoms as a tree (each atom has ONE parent via dist_ref)
    - In a ring, one bond is NOT represented in this tree - this is the "closure bond"
    - We identify this bond and select dependent dihedrals that can close it
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
            queue = deque([start])
            visited[start] = True
            depth[start] = 0

            while queue:
                node = queue.popleft()
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
        parent: np.ndarray,
        constraint_spec: ConstraintSpec,
    ) -> IndependentDOF:
        """
        Analyze constraints and compute independent DOF.

        Given the bond graph, parent array, and constraint specification,
        determines which dihedrals are independent vs dependent.

        For acyclic molecules: all N-3 dihedrals are independent.
        For molecules with rings: some dihedrals become dependent
        through ring closure constraints.

        Args:
            csr_offsets: (N+1,) CSR offsets for bond graph.
            csr_neighbors: (E,) CSR neighbor indices.
            n_atoms: Total number of atoms.
            parent: (N,) int64 parent array from spanning tree.
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

        # Compute tree depth (level) for each atom to determine valid dihedrals
        # Atoms need level >= 3 to have valid dihedral (parent, grandparent, great-grandparent)
        level = RingAnalyzer._compute_levels(parent, n_atoms)

        # Start with dihedrals as independent for atoms at level >= 3
        independent_mask = level >= 3

        # Process each ring to mark dependent dihedrals
        ring_constraints = []
        for cycle in cycles:
            ring_constraint = RingAnalyzer._create_ring_constraint(
                cycle, parent, level, independent_mask
            )
            if ring_constraint is not None:
                ring_constraints.append(ring_constraint)
                # Mark dependent dihedrals
                for dep_idx in ring_constraint.dependent_dihedrals:
                    independent_mask[dep_idx] = False

        # Detect fused rings (rings sharing atoms)
        ring_constraints = RingAnalyzer.detect_fused_rings(ring_constraints)

        # Extract indices - atoms at level >= 3 and are independent
        independent_indices = np.where(independent_mask)[0].astype(np.int64)
        # Dependent = atoms at level >= 3 that are NOT independent
        has_dihedral = level >= 3
        dependent_indices = np.where(~independent_mask & has_dihedral)[0].astype(np.int64)

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
    def _compute_levels(parent: np.ndarray, n_atoms: int) -> np.ndarray:
        """
        Compute tree depth for each atom from parent array.

        Uses O(n) BFS from roots instead of O(n × depth) parent traversal.
        """
        level = np.full(n_atoms, -1, dtype=np.int32)

        # Build children list from parent array (O(n))
        children: list[list[int]] = [[] for _ in range(n_atoms)]
        roots = []
        for k in range(n_atoms):
            p = int(parent[k])
            if p >= 0:
                children[p].append(k)
            else:
                roots.append(k)
                level[k] = 0

        # BFS from roots (O(n) total)
        queue = deque(roots)
        while queue:
            node = queue.popleft()
            child_level = level[node] + 1
            for child in children[node]:
                level[child] = child_level
                queue.append(child)

        return level

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
        queue = deque([start])
        visited[start] = True

        while queue:
            node = queue.popleft()
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
    def identify_closure_bond(
        cycle: np.ndarray,
        parent: np.ndarray,
    ) -> tuple[int, int]:
        """
        Find which bond in the cycle is NOT represented in the spanning tree.

        The parent array forms a tree: each atom has exactly one parent.
        In a ring, one bond must be "missing" from this tree - this is the
        closure bond that we need to satisfy through CCD.

        Args:
            cycle: (k,) atom indices forming the ring in order.
            parent: (N,) int64 parent array.

        Returns:
            (atom_i, atom_j) - the closure bond endpoints (sorted).

        Example:
            For cycle [4, 5, 6, 7, 8] where parent has:
            - parent[5] = 4 (bond 4-5)
            - parent[6] = 5 (bond 5-6)
            - parent[7] = 6 (bond 6-7)
            - parent[8] = 7 (bond 7-8)
            The closure bond is (4, 8) since there's no parent edge 8→4 or 4→8.
        """
        cycle_set = set(int(a) for a in cycle)
        k = len(cycle)

        # Build set of tree bonds within the cycle
        tree_bonds = set()
        for atom in cycle:
            atom = int(atom)
            p = int(parent[atom])
            if p >= 0 and p in cycle_set:
                bond = tuple(sorted([atom, p]))
                tree_bonds.add(bond)

        # Check each adjacent pair in the cycle to find the missing bond
        for i in range(k):
            atom_a = int(cycle[i])
            atom_b = int(cycle[(i + 1) % k])  # Next atom (wraps around)
            bond = tuple(sorted([atom_a, atom_b]))

            if bond not in tree_bonds:
                return bond

        # Fallback: if all bonds are in tree (shouldn't happen for rings),
        # use the bond between first and last atoms
        return (int(cycle[0]), int(cycle[-1]))

    @staticmethod
    def find_affected_atoms(
        parent: np.ndarray,
        dihedral_atom: int,
        n_atoms: int,
    ) -> set[int]:
        """
        Find all atoms whose position is affected by rotating a dihedral.

        When we rotate the dihedral at atom `dihedral_atom`, all atoms that
        are descendants of this atom in the parent tree will move.

        Args:
            parent: (N,) int64 parent array.
            dihedral_atom: Atom index whose dihedral is being rotated.
            n_atoms: Total number of atoms.

        Returns:
            Set of atom indices affected by rotating this dihedral.
        """
        affected = set()
        affected.add(dihedral_atom)

        # Find all descendants of dihedral_atom in the tree
        # An atom is a descendant if its parent chain includes dihedral_atom
        for k in range(n_atoms):
            if k == dihedral_atom:
                continue
            # Trace parent chain to see if it includes dihedral_atom
            curr = k
            while curr >= 0:
                if curr == dihedral_atom:
                    affected.add(k)
                    break
                curr = int(parent[curr])

        return affected

    @staticmethod
    def find_effective_dihedrals(
        parent: np.ndarray,
        level: np.ndarray,
        closure_bond: tuple[int, int],
        cycle: np.ndarray,
        n_atoms: int,
    ) -> list[tuple[int, bool, bool]]:
        """
        Find dihedrals that can effectively close the ring.

        A dihedral is "effective" for ring closure if rotating it moves
        EXACTLY ONE of the closure bond atoms. If both atoms move together
        (or neither moves), the distance between them doesn't change.

        Args:
            parent: (N,) int64 parent array.
            level: (N,) int32 level array.
            closure_bond: (atom_i, atom_j) - the closure bond endpoints.
            cycle: (k,) atom indices forming the ring.
            n_atoms: Total number of atoms.

        Returns:
            List of (dihedral_atom, moves_i, moves_j) tuples where:
            - dihedral_atom: Atom index of an effective dihedral
            - moves_i: Whether rotating this dihedral moves closure_bond[0]
            - moves_j: Whether rotating this dihedral moves closure_bond[1]
            Only returns dihedrals where exactly one of (moves_i, moves_j) is True.
        """
        closure_i, closure_j = closure_bond
        effective = []

        # Check each atom in the cycle
        for atom_idx in cycle:
            atom_idx = int(atom_idx)
            if level[atom_idx] < 3:  # Need level >= 3 for valid dihedral
                continue

            affected = RingAnalyzer.find_affected_atoms(parent, atom_idx, n_atoms)
            i_affected = closure_i in affected
            j_affected = closure_j in affected

            # Effective if exactly one closure atom moves
            if i_affected != j_affected:
                effective.append((atom_idx, i_affected, j_affected))

        return effective

    @staticmethod
    def _create_ring_constraint(
        cycle: np.ndarray,
        parent: np.ndarray,
        level: np.ndarray,
        current_independent: np.ndarray,
    ) -> RingConstraint | None:
        """
        Create a RingConstraint from a detected cycle.

        For a k-membered ring with fixed bonds/angles:
        - k-3 dihedrals are independent
        - 3 dihedrals are dependent (determined by ring closure)

        The dependent dihedrals are chosen from those that can EFFECTIVELY
        close the ring - i.e., dihedrals where rotating them moves exactly
        one of the closure bond atoms.

        Args:
            cycle: (k,) array of atom indices in ring.
            parent: (N,) int64 parent array.
            level: (N,) int32 level array.
            current_independent: Current independent mask (may be updated).

        Returns:
            RingConstraint, or None if cycle is too small.
        """
        k = len(cycle)
        if k < 4:
            return None  # Need at least 4 atoms for a dihedral constraint

        n_atoms = len(parent)

        # Find the closure bond (the one NOT in tree)
        closure_bond = RingAnalyzer.identify_closure_bond(cycle, parent)

        # Find effective dihedrals (those that move exactly one closure atom)
        effective = RingAnalyzer.find_effective_dihedrals(
            parent, level, closure_bond, cycle, n_atoms
        )

        # Filter to only currently independent dihedrals
        effective_independent = [
            (d, mi, mj) for d, mi, mj in effective
            if current_independent[d]
        ]

        if len(effective_independent) < 1:
            # No effective dihedrals available
            return None

        # Find all dihedrals in the cycle that are currently independent
        cycle_dihedrals = []
        for atom_idx in cycle:
            atom_idx = int(atom_idx)
            # Check level (not row) for dihedral validity
            if level[atom_idx] >= 3 and current_independent[atom_idx]:
                cycle_dihedrals.append(atom_idx)

        if len(cycle_dihedrals) < 3:
            return None

        # Select dependent dihedrals: prefer effective ones
        # We need up to 3 dependent dihedrals for ring closure
        effective_indices = [d for d, _, _ in effective_independent]

        # Take effective dihedrals as dependent (up to 3)
        dependent_list = effective_indices[:3]

        # If we don't have 3 effective dihedrals, that's okay - CCD can still work
        # with fewer (especially for small rings)
        if not dependent_list:
            # Fallback: use the last dihedrals in the cycle
            dependent_list = cycle_dihedrals[-3:] if len(cycle_dihedrals) >= 3 else cycle_dihedrals

        # Independent dihedrals: all cycle dihedrals except the dependent ones
        dependent_set = set(dependent_list)
        independent_list = [d for d in cycle_dihedrals if d not in dependent_set]

        dependent = np.array(dependent_list, dtype=np.int64)
        independent = np.array(independent_list, dtype=np.int64)

        return RingConstraint(
            ring_atoms=cycle,
            ring_size=k,
            independent_dihedrals=independent,
            dependent_dihedrals=dependent,
            closure_bond=closure_bond,
        )

    @staticmethod
    def detect_fused_rings(
        ring_constraints: list[RingConstraint],
    ) -> list[RingConstraint]:
        """
        Detect and mark fused rings (rings that share atoms).

        Two rings are fused if they share at least 2 atoms (i.e., share an edge).
        This is common in purines (adenine, guanine) which have a 5-member
        imidazole ring fused with a 6-member pyrimidine ring.

        Fused rings require special handling during ring closure because
        closing one ring may affect the geometry of the fused ring.

        Args:
            ring_constraints: List of RingConstraint objects

        Returns:
            Same list with is_fused and fused_with fields updated
        """
        n_rings = len(ring_constraints)

        if n_rings < 2:
            return ring_constraints

        # Build sets of ring atoms for fast intersection
        ring_atom_sets = [
            set(int(a) for a in rc.ring_atoms)
            for rc in ring_constraints
        ]

        # Check each pair of rings for fusion
        for i in range(n_rings):
            for j in range(i + 1, n_rings):
                shared = ring_atom_sets[i] & ring_atom_sets[j]
                if len(shared) >= 2:
                    # Rings share an edge - they are fused
                    ring_constraints[i].is_fused = True
                    ring_constraints[j].is_fused = True
                    ring_constraints[i].fused_with.append(j)
                    ring_constraints[j].fused_with.append(i)

        return ring_constraints


def find_cycles(edges: np.ndarray | list, n_atoms: int) -> list[np.ndarray]:
    """
    Find all fundamental cycles (rings) in a bond graph.

    Convenience wrapper around RingAnalyzer.find_fundamental_cycles that
    accepts edge list format instead of CSR.

    Args:
        edges: (E, 2) array or list of (i, j) tuples representing bonds.
        n_atoms: Total number of atoms.

    Returns:
        List of (k,) arrays, each containing atom indices of a ring.
    """
    if n_atoms == 0:
        return []

    # Convert edges to list of tuples if numpy array
    if isinstance(edges, np.ndarray):
        edge_list = [(int(e[0]), int(e[1])) for e in edges]
    else:
        edge_list = list(edges)

    if len(edge_list) == 0:
        return []

    # Build CSR format
    counts = np.zeros(n_atoms, dtype=np.int64)
    for i, j in edge_list:
        counts[i] += 1
        counts[j] += 1

    offsets = np.zeros(n_atoms + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    n_edges = len(edge_list) * 2
    neighbors = np.zeros(n_edges, dtype=np.int64)
    current = np.zeros(n_atoms, dtype=np.int64)

    for i, j in edge_list:
        idx = int(offsets[i] + current[i])
        neighbors[idx] = j
        current[i] += 1

        idx = int(offsets[j] + current[j])
        neighbors[idx] = i
        current[j] += 1

    return RingAnalyzer.find_fundamental_cycles(offsets, neighbors, n_atoms)
