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

Classes:
    SpanningTree: Core tree structure with parent/level/component data
    ReconstructionData: Frozen bundle of all data needed for NERF reconstruction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..backend import Array


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def derive_zmatrix_from_parent(parent: np.ndarray) -> np.ndarray:
    """
    Derive (N, 4) zmatrix indices from (N,) parent array.

    This is used as a bridge for autograd backward compatibility.
    References are derived as:
        dist_ref[k] = parent[k]
        ang_ref[k] = parent[parent[k]]
        dih_ref[k] = parent[parent[parent[k]]]

    Args:
        parent: (N,) int64 array where parent[k] is parent of atom k, -1 for roots

    Returns:
        (N, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
    """
    n = len(parent)
    indices = np.zeros((n, 4), dtype=np.int64)
    indices[:, 0] = np.arange(n)  # atom_idx
    indices[:, 1] = parent  # dist_ref

    for k in range(n):
        p = int(parent[k])
        if p >= 0:
            pp = int(parent[p])
            indices[k, 2] = pp  # ang_ref
            if pp >= 0:
                indices[k, 3] = int(parent[pp])  # dih_ref
            else:
                indices[k, 3] = -1
        else:
            indices[k, 2] = -1
            indices[k, 3] = -1

    return indices


# =============================================================================
# RECONSTRUCTION DATA BUNDLE
# =============================================================================


@dataclass
class ReconstructionData:
    """
    Frozen bundle of all data needed for NERF reconstruction.

    This is the single source of truth for autograd functions and dispatch.
    Generated from SpanningTree.get_reconstruction_data().

    All arrays are NumPy. Use to_torch() to convert for GPU operations.

    Attributes:
        parent: (N,) int64 parent of each atom (-1 for roots)
        component_ids: (N,) int32 component ID for each atom
        component_offsets: (n_components+1,) int32 CSR offsets
        anchor_coords: (n_components, 3, 3) float32 anchor positions
        levels: (N,) int32 tree depth for each atom
        level_offsets: (max_level+2,) int32 CSR offsets by level
        level_atoms: (N,) int64 atom indices sorted by level
        fixed_coords: (N, 3) float32 reference coordinates for NERF
        center_offsets: (n_components, 3) float32 or None - per-component centering
    """

    # Tree structure (replaces zmatrix_indices)
    parent: np.ndarray  # (N,) int64 - parent[k] is parent of atom k

    # Component information
    component_ids: np.ndarray  # (N,) int32
    component_offsets: np.ndarray  # (n_components+1,) int32
    anchor_coords: np.ndarray  # (n_components, 3, 3) float32

    # Level information for parallel NERF
    levels: np.ndarray  # (N,) int32
    level_offsets: np.ndarray  # (max_level+2,) int32
    level_atoms: np.ndarray  # (N,) int64 - atoms sorted by level

    # Reference frame data
    fixed_coords: np.ndarray  # (N, 3) float32
    center_offsets: np.ndarray | None = None  # (n_components, 3) float32

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return len(self.parent)

    @property
    def n_components(self) -> int:
        """Number of connected components."""
        return len(self.component_offsets) - 1

    @property
    def n_levels(self) -> int:
        """Number of tree levels (max level + 1)."""
        return int(self.levels.max()) + 1 if self.n_atoms > 0 else 0

    def to_torch(self, device: str | None = None):
        """
        Convert to PyTorch tensors.

        Args:
            device: Target device ('cpu', 'cuda', 'mps'). None for default.

        Returns:
            Dict of tensors with same keys as dataclass attributes.
        """
        import torch

        def convert(arr, dtype=None):
            if arr is None:
                return None
            t = torch.from_numpy(arr)
            if dtype:
                t = t.to(dtype)
            if device:
                t = t.to(device)
            return t

        return {
            "parent": convert(self.parent, torch.int64),
            "component_ids": convert(self.component_ids, torch.int32),
            "component_offsets": convert(self.component_offsets, torch.int32),
            "anchor_coords": convert(self.anchor_coords, torch.float32),
            "levels": convert(self.levels, torch.int32),
            "level_offsets": convert(self.level_offsets, torch.int32),
            "level_atoms": convert(self.level_atoms, torch.int64),
            "fixed_coords": convert(self.fixed_coords, torch.float32),
            "center_offsets": convert(self.center_offsets, torch.float32),
        }

    def __repr__(self) -> str:
        return (
            f"ReconstructionData(n_atoms={self.n_atoms}, "
            f"n_components={self.n_components})"
        )


# =============================================================================
# SPANNING TREE
# =============================================================================


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
        >>> parent = tree.parent  # (N,) parent indices for C backend
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

    def get_anchor_atom_indices(self) -> np.ndarray:
        """
        Get indices of anchor atoms (first 3 atoms per component).

        Anchor atoms are the first atoms discovered by DFS in each component:
        - anchor0: root atom (level 0)
        - anchor1: first child of root (level 1)
        - anchor2: first grandchild (level 2)

        Returns:
            (n_components, 3) int64 array of anchor atom indices.
            -1 indicates padding for components with fewer than 3 atoms.
        """
        if self.n_components == 0:
            return np.zeros((0, 3), dtype=np.int64)

        anchor_indices = np.full((self.n_components, 3), -1, dtype=np.int64)

        for comp_idx in range(self.n_components):
            # Find atoms in this component
            comp_mask = self.component_id == comp_idx

            # Find root (level 0) - should be exactly one per component
            root_mask = comp_mask & (self.level == 0)
            roots = np.where(root_mask)[0]
            if len(roots) == 0:
                continue
            root = roots[0]
            anchor_indices[comp_idx, 0] = root

            # Find first child of root (level 1, parent = root)
            child_mask = comp_mask & (self.parent == root)
            children = np.where(child_mask)[0]
            if len(children) == 0:
                continue
            child = children[0]
            anchor_indices[comp_idx, 1] = child

            # Find first grandchild (level 2, parent = child)
            grandchild_mask = comp_mask & (self.parent == child)
            grandchildren = np.where(grandchild_mask)[0]
            if len(grandchildren) == 0:
                continue
            anchor_indices[comp_idx, 2] = grandchildren[0]

        return anchor_indices

    def get_anchor_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Extract anchor coordinates from Cartesian coordinates.

        Args:
            coords: (N, 3) float32 array of Cartesian coordinates.

        Returns:
            (n_components, 3, 3) float32 array of anchor positions.
            Zero-padded for components with fewer than 3 atoms.
        """
        coords = np.asarray(coords, dtype=np.float32)
        anchor_indices = self.get_anchor_atom_indices()

        if self.n_components == 0:
            return np.zeros((0, 3, 3), dtype=np.float32)

        anchor_coords = np.zeros((self.n_components, 3, 3), dtype=np.float32)

        # Vectorized gather with masking for invalid indices
        valid_mask = anchor_indices >= 0
        safe_indices = np.where(valid_mask, anchor_indices, 0)

        # Gather coordinates
        anchor_coords = coords[safe_indices]  # (n_components, 3, 3)

        # Zero out invalid entries
        anchor_coords = anchor_coords * valid_mask[:, :, np.newaxis]

        return anchor_coords

    def get_reconstruction_data(
        self,
        coords: np.ndarray,
        center_offsets: np.ndarray | None = None,
    ) -> ReconstructionData:
        """
        Create a ReconstructionData bundle for NERF reconstruction.

        This bundles all data needed for coordinate reconstruction into
        a single immutable object that can be passed to dispatch/autograd.

        Args:
            coords: (N, 3) float32 reference coordinates. These are the
                coordinates used for atoms at levels 0-2 (anchor atoms).
            center_offsets: (n_components, 3) float32 per-component centering
                offsets, or None. If provided, these are applied after
                reconstruction to restore the original frame.

        Returns:
            ReconstructionData bundle with all reconstruction parameters.
        """
        coords = np.asarray(coords, dtype=np.float32)

        return ReconstructionData(
            parent=self.parent.copy(),
            component_ids=self.component_id.copy(),
            component_offsets=self.get_component_offsets(),
            anchor_coords=self.get_anchor_coords(coords),
            levels=self.level.copy(),
            level_offsets=self.get_level_offsets(),
            level_atoms=np.argsort(self.level).astype(np.int64),
            fixed_coords=coords.copy(),
            center_offsets=center_offsets.copy() if center_offsets is not None else None,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpanningTree(atoms={self.n_atoms}, "
            f"components={self.n_components}, "
            f"max_level={int(self.level.max()) if self.n_atoms > 0 else 0})"
        )
