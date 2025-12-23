"""
Spanning tree representation for internal coordinates.

This module provides SpanningTree, the core data structure for the internal
coordinate system. It stores parent relationships from DFS traversal and
derives Z-matrix references on-the-fly.

The key insight is that for NERF reconstruction, we only need:
- parent[k]: the parent of atom k in the spanning tree
- level[k]: the depth of atom k (distance from root)

References are derived as:
    dist_ref[k] = parent[k]
    ang_ref[k] = parent[parent[k]]
    dih_ref[k] = parent[parent[parent[k]]]

DFS (not BFS) is used to build the tree because it creates chain-like paths:
    0 → 1 → 2 → 3 → ...
rather than star structures:
    0 → {1, 2, 3, ...}

Chain structures ensure that most atoms have valid parent chains for all
three references (dist, ang, dih).

This eliminates the need for multiple Z-matrix modes and ensures
atom k's data is always at index k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..backend import Array


@dataclass(frozen=True)
class SpanningTree:
    """
    Spanning tree of molecular bond graph.

    Stores parent relationships from DFS traversal. References for internal
    coordinates (dist_ref, ang_ref, dih_ref) are derived on-the-fly from
    the parent chain.

    Invariants:
        - len(parent) == n_atoms
        - parent[k] == -1 for root atoms (one per connected component)
        - level[k] = depth from root (root has level 0)
        - For non-root atoms: level[parent[k]] == level[k] - 1 (parent is exactly one level up)
        - component_id[k] identifies which connected component atom k belongs to

    Attributes:
        parent: (N,) int64 array where parent[k] is the parent of atom k, -1 for roots.
        level: (N,) int32 array of depth levels (root = 0, children = 1, etc).
        component_id: (N,) int32 array mapping each atom to its connected component.
        n_components: Number of connected components.

    Example:
        >>> tree = SpanningTree.from_bond_graph(csr_offsets, csr_neighbors, n_atoms)
        >>> dist_ref, ang_ref, dih_ref = tree.get_references(atom_idx)
        >>> indices = tree.to_zmatrix_indices()  # (N, 4) for C backend
    """

    parent: np.ndarray        # (N,) int64
    level: np.ndarray         # (N,) int32
    component_id: np.ndarray  # (N,) int32
    n_components: int

    @classmethod
    def from_bond_graph(
        cls,
        csr_offsets: np.ndarray,
        csr_neighbors: np.ndarray,
        n_atoms: int,
    ) -> "SpanningTree":
        """
        Build spanning tree via DFS on each connected component.

        Uses DFS (depth-first search) rather than BFS because it creates
        chain-like paths where most atoms have valid parent chains:
            0 → 1 → 2 → 3 → ...

        This is essential for NERF reconstruction, where we need:
            dist_ref = parent[k]
            ang_ref = parent[parent[k]]
            dih_ref = parent[parent[parent[k]]]

        With DFS, atoms at depth >= 3 have all valid references from the
        parent chain. Only the first few atoms per component need fallback.

        Args:
            csr_offsets: (N+1,) int64 CSR offsets array.
            csr_neighbors: (E,) int64 CSR neighbor indices.
            n_atoms: Total number of atoms.

        Returns:
            SpanningTree with parent and level arrays.
        """
        if n_atoms == 0:
            return cls(
                parent=np.array([], dtype=np.int64),
                level=np.array([], dtype=np.int32),
                component_id=np.array([], dtype=np.int32),
                n_components=0,
            )

        # Initialize arrays
        parent = np.full(n_atoms, -1, dtype=np.int64)
        level = np.full(n_atoms, -1, dtype=np.int32)
        component_id = np.zeros(n_atoms, dtype=np.int32)

        # DFS to discover components and build tree
        comp_idx = 0
        for start in range(n_atoms):
            if level[start] >= 0:  # Already visited
                continue

            # New component - DFS from start
            level[start] = 0
            component_id[start] = comp_idx
            stack = [start]

            while stack:
                current = stack.pop()
                current_level = level[current]

                # Visit neighbors
                neighbor_start = csr_offsets[current]
                neighbor_end = csr_offsets[current + 1]

                for idx in range(neighbor_start, neighbor_end):
                    neighbor = csr_neighbors[idx]
                    if level[neighbor] == -1:  # Not visited
                        parent[neighbor] = current
                        level[neighbor] = current_level + 1
                        component_id[neighbor] = comp_idx
                        stack.append(neighbor)

            comp_idx += 1

        return cls(
            parent=parent,
            level=level,
            component_id=component_id,
            n_components=comp_idx,
        )

    @property
    def n_atoms(self) -> int:
        """Total number of atoms."""
        return len(self.parent)

    def get_references(self, atom: int) -> tuple[int, int, int]:
        """
        Derive distance, angle, and dihedral references for an atom.

        Basic strategy is to follow the parent chain:
            dist_ref = parent[atom]
            ang_ref = parent[parent[atom]]
            dih_ref = parent[parent[parent[atom]]]

        However, when parent chain is too short (e.g., near root), we search
        for valid references among atoms at lower levels in the same component.

        Args:
            atom: Atom index.

        Returns:
            Tuple of (dist_ref, ang_ref, dih_ref). Each is -1 if not available.
        """
        dist_ref = int(self.parent[atom])
        if dist_ref < 0:
            return -1, -1, -1

        my_level = self.level[atom]
        my_comp = self.component_id[atom]

        # Try parent chain first
        ang_ref = int(self.parent[dist_ref])

        if ang_ref < 0:
            # Parent chain too short - search for valid ang_ref
            # Need an atom at level < my_level, in same component, not dist_ref
            ang_ref = self._find_reference(atom, dist_ref, -1, my_level, my_comp)

        if ang_ref < 0:
            return dist_ref, -1, -1

        # Try parent chain for dih_ref
        dih_ref = int(self.parent[ang_ref])

        if dih_ref < 0:
            # Search for valid dih_ref
            # Need an atom at level < my_level, in same component, not dist_ref or ang_ref
            dih_ref = self._find_reference(atom, dist_ref, ang_ref, my_level, my_comp)

        return dist_ref, ang_ref, dih_ref

    def _find_reference(
        self,
        atom: int,
        exclude1: int,
        exclude2: int,
        max_level: int,
        component: int,
    ) -> int:
        """
        Find a valid reference atom for angle or dihedral.

        Searches for an atom that:
        - Has level < max_level (already placed when we place `atom`)
        - Is in the same component
        - Is not in the exclude list

        Args:
            atom: The atom we're finding references for.
            exclude1: First atom to exclude (dist_ref).
            exclude2: Second atom to exclude (ang_ref), or -1 if not applicable.
            max_level: Maximum level (exclusive) to search in.
            component: Component ID to restrict search to.

        Returns:
            Valid reference atom index, or -1 if none found.
        """
        # Search atoms in order of decreasing level (prefer closer atoms)
        # This is a simple linear search; could be optimized with level indexing
        best_ref = -1
        best_level = -1

        for k in range(self.n_atoms):
            if k == atom or k == exclude1 or k == exclude2:
                continue
            if self.component_id[k] != component:
                continue
            if self.level[k] >= max_level:
                continue

            # Prefer higher level (closer in tree to atom)
            if self.level[k] > best_level:
                best_level = self.level[k]
                best_ref = k

        return best_ref

    def to_zmatrix_indices(self) -> np.ndarray:
        """
        Convert to (N, 4) indices array for C backend compatibility.

        Each row is [atom_idx, dist_ref, ang_ref, dih_ref].
        This is compatible with the existing NERF C functions.

        Returns:
            (N, 4) int64 array of Z-matrix indices.
        """
        n = self.n_atoms
        indices = np.zeros((n, 4), dtype=np.int64)
        indices[:, 0] = np.arange(n)  # atom_idx

        for k in range(n):
            dist_ref, ang_ref, dih_ref = self.get_references(k)
            indices[k, 1] = dist_ref
            indices[k, 2] = ang_ref
            indices[k, 3] = dih_ref

        return indices

    def get_descendants(self, atom: int) -> np.ndarray:
        """
        Get all descendants of an atom in the spanning tree.

        This is used for partial reconstruction: when a dihedral at `atom`
        is rotated, only its descendants need to be reconstructed.

        Args:
            atom: Root atom to find descendants of.

        Returns:
            (K,) int64 array of descendant atom indices (includes `atom` itself),
            sorted by level for correct reconstruction order.
        """
        n = self.n_atoms
        if n == 0:
            return np.array([], dtype=np.int64)

        # Build children list (inverse of parent)
        # children[k] = list of atoms whose parent is k
        children = [[] for _ in range(n)]
        for k in range(n):
            p = self.parent[k]
            if p >= 0:
                children[p].append(k)

        # BFS from atom to find all descendants
        descendants = [atom]
        head = 0
        while head < len(descendants):
            current = descendants[head]
            head += 1
            descendants.extend(children[current])

        # Sort by level for correct reconstruction order
        desc_array = np.array(descendants, dtype=np.int64)
        desc_levels = self.level[desc_array]
        sort_order = np.argsort(desc_levels)

        return desc_array[sort_order]

    def get_component_offsets(self) -> np.ndarray:
        """
        Get CSR-style offsets for atoms grouped by component.

        Atoms are grouped by component_id, with offsets[i] being the
        start index for component i's atoms when sorted by component.

        Returns:
            (n_components+1,) int32 array of offsets.
        """
        if self.n_components == 0:
            return np.zeros(1, dtype=np.int32)

        counts = np.bincount(self.component_id, minlength=self.n_components)
        offsets = np.zeros(self.n_components + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:])
        return offsets

    def get_level_offsets(self) -> np.ndarray:
        """
        Get CSR-style offsets for atoms grouped by level.

        This is used for level-parallel NERF reconstruction.
        Atoms at level L can be processed after all atoms at levels < L.

        Returns:
            (max_level+2,) int32 array of offsets.
        """
        if self.n_atoms == 0:
            return np.zeros(1, dtype=np.int32)

        max_level = int(self.level.max())
        counts = np.bincount(self.level, minlength=max_level + 1)
        offsets = np.zeros(max_level + 2, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:])
        return offsets

    def cartesian_to_internal(
        self,
        coords: np.ndarray,
        center: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Convert Cartesian coordinates to internal coordinates.

        Optionally centers each connected component before conversion to reduce
        float32 precision loss. Per-component centering is more effective than
        global centering for multi-chain structures where chains are far apart.

        Args:
            coords: (N, 3) float32 array of Cartesian coordinates.
            center: If True (default), center each connected component at origin
                before conversion. This reduces accumulated NERF error for large
                structures by preserving float32 precision.

        Returns:
            Tuple of:
                internal: (N, 3) float32 array [distance, angle, dihedral].
                offsets: (n_components, 3) float32 per-component center offsets,
                    or None if center=False. Used to restore original frame.
        """
        from .._c import _cartesian_to_internal_parent

        coords = np.asarray(coords, dtype=np.float32)

        if center and self.n_components > 0:
            # Compute per-component centers
            offsets = np.zeros((self.n_components, 3), dtype=np.float32)
            coords_centered = coords.copy()

            for comp_idx in range(self.n_components):
                # Find atoms in this component
                mask = self.component_id == comp_idx
                comp_coords = coords[mask]

                # Compute component center and store offset
                center_offset = comp_coords.mean(axis=0)
                offsets[comp_idx] = center_offset

                # Center this component's atoms
                coords_centered[mask] = comp_coords - center_offset

            internal = _cartesian_to_internal_parent(coords_centered, self.parent)
            return internal, offsets
        else:
            internal = _cartesian_to_internal_parent(coords, self.parent)
            return internal, None

    def internal_to_cartesian(
        self,
        internal: np.ndarray,
        fixed_coords: np.ndarray,
        offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct Cartesian coordinates from internal coordinates.

        Uses NERF (Natural Extension Reference Frame) algorithm to place
        atoms level-by-level. Atoms at levels 0-2 are copied from fixed_coords
        since they lack sufficient parent chain for full NERF placement.

        Args:
            internal: (N, 3) float32 array [distance, angle, dihedral].
            fixed_coords: (N, 3) float32 array of reference coordinates.
                Atoms at levels 0-2 are copied from here. Should be in the
                same frame as the internal coordinates (centered if centering
                was used in cartesian_to_internal).
            offsets: (n_components, 3) float32 per-component center offsets
                from cartesian_to_internal. If provided, each component's
                atoms are shifted by its offset to restore the original frame.

        Returns:
            (N, 3) float32 array of reconstructed Cartesian coordinates.
        """
        from .._c import _nerf_reconstruct_parent

        internal = np.asarray(internal, dtype=np.float32)
        fixed_coords = np.asarray(fixed_coords, dtype=np.float32)

        # Prepare level data for NERF
        level_offsets = self.get_level_offsets()
        level_atoms = np.argsort(self.level).astype(np.int64)
        n_levels = int(self.level.max()) + 1 if self.n_atoms > 0 else 0

        # Reconstruct
        coords = _nerf_reconstruct_parent(
            self.parent, self.level, internal,
            level_offsets, level_atoms, n_levels,
            fixed_coords,
        )

        # Add per-component offsets if provided
        if offsets is not None:
            for comp_idx in range(self.n_components):
                mask = self.component_id == comp_idx
                coords[mask] += offsets[comp_idx]

        return coords

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpanningTree(atoms={self.n_atoms}, "
            f"components={self.n_components}, "
            f"max_level={int(self.level.max()) if self.n_atoms > 0 else 0})"
        )
