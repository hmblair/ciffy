"""
Bond graph construction, Z-matrix representation, and connected components.

This module provides the core data structures and algorithms for internal
coordinate representation:

- ZMatrix: Z-matrix representation as (M, 4) array
- ConnectedComponents: Connected component storage in CSR format
- Bond graph construction from topology
- NERF reconstruction wrapper

.. note::
    This is an **internal backend module**. For coordinate operations, use
    the higher-level ``ciffy.internal.CoordinateManager`` or ``Polymer`` APIs.
    The ``backend.dispatch`` module provides the coordinate conversion functions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import Array, is_torch, to_numpy, to_torch

if TYPE_CHECKING:
    from .._c import (
        _build_bond_graph,
        _edges_to_csr,
        _build_zmatrix_parallel,
        _find_connected_components,
    )

# C extension imports (required)
from .._c import _build_bond_graph as _build_bond_graph_c
from .._c import _edges_to_csr as _edges_to_csr_c
from .._c import _build_zmatrix_parallel as _build_zmatrix_parallel_c
from .._c import _find_connected_components as _find_connected_components_c

__all__ = [
    # Data structures
    "ZMatrix",
    "ConnectedComponents",
    "TopologyInfo",
    # Bond graph functions
    "build_bond_graph",
    "build_bond_graph_from_topology",
    "build_bond_graph_csr",
    "edges_to_csr",
    "find_connected_components",
    "build_zmatrix_from_components",
    # NERF reconstruction wrapper
    "nerf_reconstruct",
]


# =============================================================================
# TOPOLOGY INFO (moved here to avoid circular imports)
# =============================================================================


@dataclass(frozen=True)
class TopologyInfo:
    """
    Immutable topology information for coordinate operations.

    Captures all structural information needed for Z-matrix building and
    coordinate reconstruction without requiring a Polymer reference.

    Attributes:
        atoms: (N,) int32 array of atom type indices.
        sequence: (R,) int32 array of residue type indices.
        residue_sizes: (R,) int32 array of atom counts per residue.
        chain_lengths: (C,) int32 array of residue counts per chain.
        chain_atom_offsets: (C+1,) int64 array of cumulative atom counts per chain.
        chain_residue_offsets: (C+1,) int64 array of cumulative residue counts per chain.
        n_atoms: Total number of atoms.
        n_residues: Total number of residues.
        n_chains: Total number of chains.

    Example:
        >>> topology = TopologyInfo.from_polymer(polymer)
        >>> zmatrix = ZMatrix.from_topology(topology)
    """

    atoms: np.ndarray
    sequence: np.ndarray
    residue_sizes: np.ndarray
    chain_lengths: np.ndarray
    chain_atom_offsets: np.ndarray
    chain_residue_offsets: np.ndarray
    n_atoms: int
    n_residues: int
    n_chains: int

    @classmethod
    def from_polymer(cls, polymer) -> "TopologyInfo":
        """
        Create TopologyInfo from a Polymer instance.

        Args:
            polymer: Polymer structure to extract topology from.

        Returns:
            TopologyInfo with all structural information.
        """
        from ..types import Scale

        # Convert to numpy for storage (topology is always CPU)
        atoms = to_numpy(polymer.atoms).astype(np.int32)
        sequence = to_numpy(polymer.sequence).astype(np.int32)
        residue_sizes = to_numpy(polymer.sizes(Scale.RESIDUE)).astype(np.int32)
        chain_lengths = to_numpy(polymer.lengths).astype(np.int32)

        n_atoms = len(atoms)
        n_residues = len(sequence)
        n_chains = len(chain_lengths)

        # Compute cumulative offsets
        chain_residue_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        chain_residue_offsets[1:] = np.cumsum(chain_lengths)

        chain_atom_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        res_offset = 0
        for chain_idx in range(n_chains):
            chain_len = int(chain_lengths[chain_idx])
            chain_atom_count = int(residue_sizes[res_offset:res_offset + chain_len].sum())
            chain_atom_offsets[chain_idx + 1] = chain_atom_offsets[chain_idx] + chain_atom_count
            res_offset += chain_len

        return cls(
            atoms=atoms,
            sequence=sequence,
            residue_sizes=residue_sizes,
            chain_lengths=chain_lengths,
            chain_atom_offsets=chain_atom_offsets,
            chain_residue_offsets=chain_residue_offsets,
            n_atoms=n_atoms,
            n_residues=n_residues,
            n_chains=n_chains,
        )

    def get_chain_atom_range(self, chain_idx: int) -> tuple[int, int]:
        """Get atom index range for a chain."""
        return int(self.chain_atom_offsets[chain_idx]), int(self.chain_atom_offsets[chain_idx + 1])

    def get_chain_residue_range(self, chain_idx: int) -> tuple[int, int]:
        """Get residue index range for a chain."""
        return int(self.chain_residue_offsets[chain_idx]), int(self.chain_residue_offsets[chain_idx + 1])

    def get_residue_atom_range(self, residue_idx: int) -> tuple[int, int]:
        """Get atom index range for a residue."""
        residue_atom_offsets = np.zeros(self.n_residues + 1, dtype=np.int64)
        residue_atom_offsets[1:] = np.cumsum(self.residue_sizes)
        return int(residue_atom_offsets[residue_idx]), int(residue_atom_offsets[residue_idx + 1])

    def slice_atoms(self, mask: np.ndarray, new_residue_sizes: np.ndarray, new_chain_lengths: np.ndarray) -> "TopologyInfo":
        """Create sliced TopologyInfo for a subset of atoms."""
        mask_np = to_numpy(mask)
        new_atoms = self.atoms[mask_np].astype(np.int32)

        residue_atom_offsets = np.zeros(self.n_residues + 1, dtype=np.int64)
        residue_atom_offsets[1:] = np.cumsum(self.residue_sizes)

        new_sequence_list = []
        for res_idx in range(self.n_residues):
            start = int(residue_atom_offsets[res_idx])
            end = int(residue_atom_offsets[res_idx + 1])
            if mask_np[start:end].any():
                new_sequence_list.append(self.sequence[res_idx])

        new_sequence = np.array(new_sequence_list, dtype=np.int32) if new_sequence_list else np.array([], dtype=np.int32)
        new_residue_sizes = to_numpy(new_residue_sizes).astype(np.int32)
        new_chain_lengths = to_numpy(new_chain_lengths).astype(np.int32)

        n_atoms = len(new_atoms)
        n_residues = len(new_sequence)
        n_chains = len(new_chain_lengths)

        chain_residue_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        chain_residue_offsets[1:] = np.cumsum(new_chain_lengths)

        chain_atom_offsets = np.zeros(n_chains + 1, dtype=np.int64)
        res_offset = 0
        for chain_idx in range(n_chains):
            chain_len = int(new_chain_lengths[chain_idx])
            if chain_len > 0:
                chain_atom_count = int(new_residue_sizes[res_offset:res_offset + chain_len].sum())
            else:
                chain_atom_count = 0
            chain_atom_offsets[chain_idx + 1] = chain_atom_offsets[chain_idx] + chain_atom_count
            res_offset += chain_len

        return TopologyInfo(
            atoms=new_atoms,
            sequence=new_sequence,
            residue_sizes=new_residue_sizes,
            chain_lengths=new_chain_lengths,
            chain_atom_offsets=chain_atom_offsets,
            chain_residue_offsets=chain_residue_offsets,
            n_atoms=n_atoms,
            n_residues=n_residues,
            n_chains=n_chains,
        )


# =============================================================================
# CONNECTED COMPONENTS
# =============================================================================


@dataclass
class ConnectedComponents:
    """
    Connected component storage.

    Stores anchor atom indices for efficient coordinate lookup during NERF
    reconstruction. The anchor_atom_indices allow vectorized extraction of
    anchor coordinates without Python loops.

    Attributes:
        offsets: (C+1,) int64 CSR offsets array for component boundaries.
        anchor_atom_indices: (C, 3) int64 array of atom indices for first 3 atoms
            per component. For components with <3 atoms, remaining indices are -1.
        anchor_coords: (C, 3, 3) array of anchor positions for each component.
            anchor_coords[c] contains [anchor0, anchor1, anchor2] for component c.
            For components with <3 atoms, remaining anchors are zero-padded.
            These are the initial anchor positions computed at construction time.
        contiguous: List of bool indicating if component atoms are contiguous.

    Example:
        >>> components = ConnectedComponents.from_bond_graph(csr_offsets, csr_neighbors, coords, n_atoms)
        >>> # Get current anchor coords efficiently
        >>> anchor_coords = components.get_anchor_coords(current_coordinates)
    """

    offsets: np.ndarray
    anchor_atom_indices: np.ndarray  # (n_components, 3) int64, -1 for padding
    anchor_coords: Array  # (n_components, 3, 3) anchor positions (initial)
    contiguous: list[bool]

    @classmethod
    def from_bond_graph(
        cls,
        csr_offsets: np.ndarray,
        csr_neighbors: np.ndarray,
        coordinates: Array,
        n_atoms: int,
    ) -> "ConnectedComponents":
        """
        Build connected components from bond graph in CSR format.

        Finds all connected components including isolated atoms (no bonds).
        Stores anchor coordinates (first 3 atoms' positions) for each component,
        which are used to place atoms directly in the correct frame during NERF.

        Args:
            csr_offsets: (N+1,) CSR offsets array for bond graph.
            csr_neighbors: (E,) CSR neighbor indices.
            coordinates: (N, 3) array of Cartesian coordinates.
            n_atoms: Total number of atoms.

        Returns:
            ConnectedComponents with all components (bonded and isolated).
        """
        if n_atoms == 0:
            if is_torch(coordinates):
                import torch
                anchor_coords = torch.zeros(0, 3, 3, dtype=coordinates.dtype, device=coordinates.device)
            else:
                anchor_coords = np.zeros((0, 3, 3), dtype=coordinates.dtype)
            return cls(
                offsets=np.array([0], dtype=np.int64),
                anchor_atom_indices=np.zeros((0, 3), dtype=np.int64),
                anchor_coords=anchor_coords,
                contiguous=[],
            )

        # Find all connected components (includes isolated atoms as single-atom components)
        comp_atoms, comp_offsets, n_components = find_connected_components(
            csr_offsets, csr_neighbors, n_atoms
        )

        if n_components == 0:
            if is_torch(coordinates):
                import torch
                anchor_coords = torch.zeros(0, 3, 3, dtype=coordinates.dtype, device=coordinates.device)
            else:
                anchor_coords = np.zeros((0, 3, 3), dtype=coordinates.dtype)
            return cls(
                offsets=np.array([0], dtype=np.int64),
                anchor_atom_indices=np.zeros((0, 3), dtype=np.int64),
                anchor_coords=anchor_coords,
                contiguous=[],
            )

        # Build anchor_atom_indices array (always numpy, since it's just indices)
        # -1 indicates padding for components with fewer than 3 atoms
        anchor_atom_indices = np.full((n_components, 3), -1, dtype=np.int64)

        # Build anchor_coords on the same device as coordinates
        if is_torch(coordinates):
            import torch
            anchor_coords = torch.zeros(
                n_components, 3, 3, dtype=coordinates.dtype, device=coordinates.device
            )
        else:
            anchor_coords = np.zeros((n_components, 3, 3), dtype=coordinates.dtype)

        contiguous_list = []

        for i in range(n_components):
            start = comp_offsets[i]
            end = comp_offsets[i + 1]
            component_atoms = comp_atoms[start:end]

            # Check if atoms are contiguous in memory
            is_contiguous = (
                len(component_atoms) > 0 and
                (len(component_atoms) == 1 or np.all(np.diff(component_atoms) == 1))
            )
            contiguous_list.append(is_contiguous)

            # Store anchor atom indices (first 3 atoms)
            n_anchor = min(3, len(component_atoms))
            if n_anchor > 0:
                anchor_atom_indices[i, :n_anchor] = component_atoms[:n_anchor]
                # Also store initial anchor coords
                component_coords = coordinates[component_atoms[:n_anchor]]
                anchor_coords[i, :n_anchor] = component_coords

        # Detach tensors to avoid keeping grad history - these are frozen reference values
        if is_torch(anchor_coords):
            anchor_coords = anchor_coords.detach()

        return cls(
            offsets=comp_offsets,
            anchor_atom_indices=anchor_atom_indices,
            anchor_coords=anchor_coords,
            contiguous=contiguous_list,
        )

    @property
    def n_components(self) -> int:
        """Number of connected components."""
        return len(self.offsets) - 1

    def get_anchor_coords(self, coordinates: Array) -> Array:
        """
        Get current anchor coordinates from Cartesian coordinates.

        Uses vectorized gather operation for efficiency. This is O(n_components)
        GPU memory operations instead of O(n_components) Python loop iterations.

        Args:
            coordinates: (N, 3) array of current Cartesian coordinates.

        Returns:
            (n_components, 3, 3) array of anchor positions for each component.
            For components with <3 atoms, extra positions are zero-padded.
        """
        n_components = self.n_components
        if n_components == 0:
            if is_torch(coordinates):
                import torch
                return torch.zeros(0, 3, 3, dtype=coordinates.dtype, device=coordinates.device)
            else:
                return np.zeros((0, 3, 3), dtype=coordinates.dtype)

        # Vectorized gather: replace -1 indices with 0 for valid indexing
        # We'll mask out the invalid entries afterward
        valid_mask = self.anchor_atom_indices >= 0  # (n_components, 3)
        safe_indices = np.where(valid_mask, self.anchor_atom_indices, 0)

        if is_torch(coordinates):
            import torch
            # Convert indices to torch tensor on same device
            safe_indices_t = torch.from_numpy(safe_indices).to(coordinates.device)
            valid_mask_t = torch.from_numpy(valid_mask).to(coordinates.device)

            # Gather: coords[safe_indices] -> (n_components, 3, 3)
            anchor_coords = coordinates[safe_indices_t]  # (n_components, 3, 3)

            # Zero out invalid entries
            anchor_coords = anchor_coords * valid_mask_t.unsqueeze(-1).float()

            # Detach to avoid keeping grad history
            anchor_coords = anchor_coords.detach()
        else:
            # NumPy version
            anchor_coords = coordinates[safe_indices]  # (n_components, 3, 3)
            # Zero out invalid entries
            anchor_coords = anchor_coords * valid_mask[:, :, np.newaxis]

        return anchor_coords


# =============================================================================
# ZMATRIX CLASS
# =============================================================================


class ZMatrix:
    """
    Z-matrix representation as (M, 4) array.

    Each row defines how an atom is placed relative to reference atoms:
    - Column 0: atom_idx - the atom being placed
    - Column 1: distance_ref - reference for bond length (-1 if none)
    - Column 2: angle_ref - reference for bond angle (-1 if none)
    - Column 3: dihedral_ref - reference for dihedral angle (-1 if none)

    Entries are in BFS order, so references always point to earlier atoms.

    Example:
        >>> zmatrix = ZMatrix.from_topology(topology)
        >>> print(len(zmatrix))  # Number of atoms in Z-matrix
        >>> print(zmatrix.atom_indices)  # Column 0
    """

    __slots__ = ('_indices', '_dihedral_types', '_levels', '_component_offsets', '_component_ids')

    def __init__(
        self,
        indices: Array,
        dihedral_types: Array | None = None,
        levels: Array | None = None,
        component_ids: Array | None = None,
    ) -> None:
        """
        Initialize Z-matrix from indices array.

        Args:
            indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 array mapping entry -> dihedral type (-1 if unnamed)
            levels: (M,) int32 array of BFS levels for parallel NERF reconstruction
            component_ids: (M,) int32 array mapping entry -> component index
        """
        self._indices = indices
        self._dihedral_types = dihedral_types
        self._levels = levels
        self._component_offsets = None  # Computed lazily
        self._component_ids = component_ids

    @classmethod
    def from_topology(
        cls,
        topology: TopologyInfo,
        csr_offsets: np.ndarray | None = None,
        csr_neighbors: np.ndarray | None = None,
    ) -> "ZMatrix":
        """
        Build Z-matrix from topology info using BFS traversal.

        Processes each chain independently with its own spanning tree.
        Returns entries in BFS order so references always point to
        earlier (already placed) atoms. The C extension performs
        dihedral-aware reference selection in a single pass.

        Args:
            topology: TopologyInfo containing structural metadata.
            csr_offsets: Optional pre-built CSR offsets array. If None, built from topology.
            csr_neighbors: Optional pre-built CSR neighbors array. If None, built from topology.

        Returns:
            ZMatrix with entries in placement order, dihedral type annotations, BFS levels, and component IDs.
        """
        # Build Z-matrix with dihedral-aware refs in single C pass
        indices, dihedral_types, levels, component_ids = _build_zmatrix_indices_from_topology(
            topology, csr_offsets, csr_neighbors
        )

        if len(indices) == 0:
            return cls(indices, np.array([], dtype=np.int8), np.array([], dtype=np.int32), np.array([], dtype=np.int32))

        return cls(indices, dihedral_types, levels, component_ids)

    @property
    def indices(self) -> Array:
        """Raw (M, 4) array."""
        return self._indices

    @property
    def atom_indices(self) -> Array:
        """Column 0: atom indices being placed."""
        return self._indices[:, 0]

    @property
    def distance_refs(self) -> Array:
        """Column 1: distance reference atoms (-1 for first atom)."""
        return self._indices[:, 1]

    @property
    def angle_refs(self) -> Array:
        """Column 2: angle reference atoms (-1 for first two atoms)."""
        return self._indices[:, 2]

    @property
    def dihedral_refs(self) -> Array:
        """Column 3: dihedral reference atoms (-1 for first three atoms)."""
        return self._indices[:, 3]

    @property
    def dihedral_types(self) -> Array | None:
        """(M,) int8 array mapping Z-matrix entry -> dihedral type (-1 if unnamed)."""
        return self._dihedral_types

    @property
    def levels(self) -> Array | None:
        """(M,) int32 BFS level per entry (0 for root atoms)."""
        return self._levels

    @property
    def component_ids(self) -> Array | None:
        """(M,) int32 component index per entry."""
        return self._component_ids

    @property
    def component_offsets(self) -> Array:
        """
        (n_components+1,) int32 cumulative count per component for parallel NERF.

        Component i's entries span indices component_offsets[i]:component_offsets[i+1].
        Computed lazily on first access.
        """
        if self._component_offsets is None and self._component_ids is not None:
            self._component_offsets = self._compute_component_offsets()
        return self._component_offsets

    def _compute_component_offsets(self) -> np.ndarray:
        """Convert per-entry component_ids to CSR-style offsets."""
        comp_ids = to_numpy(self._component_ids)
        if len(comp_ids) == 0:
            return np.zeros(1, dtype=np.int32)

        n_components = int(comp_ids.max()) + 1
        counts = np.bincount(comp_ids, minlength=n_components).astype(np.int32)
        offsets = np.zeros(n_components + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:])
        return offsets

    def __len__(self) -> int:
        """Number of entries in Z-matrix."""
        return len(self._indices)

    def __getitem__(self, idx) -> Array:
        """Index into the Z-matrix array."""
        return self._indices[idx]

    def validate(self) -> None:
        """
        Validate Z-matrix structure using vectorized operations.

        Checks that all reference atoms are either -1 or point to earlier atoms.

        Raises:
            ValueError: If validation fails.
        """
        n_entries = len(self._indices)
        if n_entries == 0:
            return

        atom_indices = self._indices[:, 0].astype(np.int64)
        dist_refs = self._indices[:, 1].astype(np.int64)
        ang_refs = self._indices[:, 2].astype(np.int64)
        dih_refs = self._indices[:, 3].astype(np.int64)

        max_atom = int(atom_indices.max()) + 1
        entry_order = np.full(max_atom, n_entries, dtype=np.int64)
        entry_order[atom_indices] = np.arange(n_entries)
        entry_positions = np.arange(n_entries, dtype=np.int64)

        # Check distance references
        valid_dist = dist_refs >= 0
        dist_ref_entries = np.where(
            valid_dist & (dist_refs < max_atom),
            entry_order[np.clip(dist_refs, 0, max_atom - 1)],
            -1
        )
        dist_violations = valid_dist & (dist_ref_entries >= entry_positions)
        if np.any(dist_violations):
            first = int(np.argmax(dist_violations))
            raise ValueError(f"Entry {first}: distance_ref {dist_refs[first]} not yet placed")

        # Check angle references
        valid_ang = ang_refs >= 0
        ang_ref_entries = np.where(
            valid_ang & (ang_refs < max_atom),
            entry_order[np.clip(ang_refs, 0, max_atom - 1)],
            -1
        )
        ang_violations = valid_ang & (ang_ref_entries >= entry_positions)
        if np.any(ang_violations):
            first = int(np.argmax(ang_violations))
            raise ValueError(f"Entry {first}: angle_ref {ang_refs[first]} not yet placed")

        # Check dihedral references
        valid_dih = dih_refs >= 0
        dih_ref_entries = np.where(
            valid_dih & (dih_refs < max_atom),
            entry_order[np.clip(dih_refs, 0, max_atom - 1)],
            -1
        )
        dih_violations = valid_dih & (dih_ref_entries >= entry_positions)
        if np.any(dih_violations):
            first = int(np.argmax(dih_violations))
            raise ValueError(f"Entry {first}: dihedral_ref {dih_refs[first]} not yet placed")

        # Check progression
        invalid_progression_ang = valid_ang & (dist_refs < 0)
        if np.any(invalid_progression_ang):
            first = int(np.argmax(invalid_progression_ang))
            raise ValueError(f"Entry {first}: has angle_ref but no distance_ref")

        invalid_progression_dih = valid_dih & (ang_refs < 0)
        if np.any(invalid_progression_dih):
            first = int(np.argmax(invalid_progression_dih))
            raise ValueError(f"Entry {first}: has dihedral_ref but no angle_ref")

    def numpy(self) -> "ZMatrix":
        """Convert indices to NumPy array."""
        dihedral_types = to_numpy(self._dihedral_types) if self._dihedral_types is not None else None
        levels = to_numpy(self._levels) if self._levels is not None else None
        component_ids = to_numpy(self._component_ids) if self._component_ids is not None else None
        return ZMatrix(to_numpy(self._indices), dihedral_types, levels, component_ids)

    def torch(self) -> "ZMatrix":
        """Convert indices to PyTorch tensor."""
        dihedral_types = to_torch(self._dihedral_types) if self._dihedral_types is not None else None
        levels = to_torch(self._levels) if self._levels is not None else None
        component_ids = to_torch(self._component_ids) if self._component_ids is not None else None
        return ZMatrix(to_torch(self._indices), dihedral_types, levels, component_ids)

    def to(self, device: str) -> "ZMatrix":
        """
        Move to specified device.

        Converts numpy arrays to PyTorch tensors if needed, then moves to device.
        This enables efficient CUDA operations by caching indices on GPU.
        """
        import torch

        # Convert indices to torch if needed, then move to device
        if is_torch(self._indices):
            indices = self._indices.to(device)
        else:
            indices = torch.from_numpy(self._indices).to(device)

        # Convert dihedral_types
        if self._dihedral_types is not None:
            if is_torch(self._dihedral_types):
                dihedral_types = self._dihedral_types.to(device)
            else:
                dihedral_types = torch.from_numpy(self._dihedral_types).to(device)
        else:
            dihedral_types = None

        # Convert levels
        if self._levels is not None:
            if is_torch(self._levels):
                levels = self._levels.to(device)
            else:
                levels = torch.from_numpy(self._levels).to(device)
        else:
            levels = None

        # Convert component_ids
        if self._component_ids is not None:
            if is_torch(self._component_ids):
                component_ids = self._component_ids.to(device)
            else:
                component_ids = torch.from_numpy(self._component_ids).to(device)
        else:
            component_ids = None

        return ZMatrix(indices, dihedral_types, levels, component_ids)

    def __repr__(self) -> str:
        backend = "torch" if is_torch(self._indices) else "numpy"
        return f"ZMatrix({len(self)} entries, {backend})"


# =============================================================================
# BOND GRAPH CONSTRUCTION
# =============================================================================


def build_bond_graph(polymer) -> tuple[np.ndarray, int]:
    """
    Build edge list representation of molecular bonds.

    Constructs bonds as an (E, 2) array for array-based processing.
    Combines intra-residue bonds from Residue.bond_indices and inter-residue
    bonds from LINKING_BY_TYPE.

    Args:
        polymer: Polymer structure with sequence and atoms.

    Returns:
        Tuple of:
            edges: (E, 2) int64 array of [atom_i, atom_j] pairs (symmetric)
            n_atoms: Total number of atoms
    """
    from ..types import Scale

    n_atoms = polymer.size()
    res_sizes = polymer.sizes(Scale.RESIDUE)
    edges = _build_bond_graph_c(
        np.ascontiguousarray(to_numpy(polymer.atoms), dtype=np.int32),
        np.ascontiguousarray(to_numpy(polymer.sequence), dtype=np.int32),
        np.ascontiguousarray(to_numpy(res_sizes), dtype=np.int32),
        np.ascontiguousarray(to_numpy(polymer.lengths), dtype=np.int32),
    )
    return edges, n_atoms


def build_bond_graph_from_topology(topology: TopologyInfo) -> tuple[np.ndarray, int]:
    """
    Build edge list representation of molecular bonds from topology info.

    Args:
        topology: TopologyInfo containing atoms, sequence, residue_sizes, chain_lengths.

    Returns:
        Tuple of:
            edges: (E, 2) int64 array of [atom_i, atom_j] pairs (symmetric)
            n_atoms: Total number of atoms
    """
    n_atoms = topology.n_atoms
    edges = _build_bond_graph_c(
        np.ascontiguousarray(topology.atoms, dtype=np.int32),
        np.ascontiguousarray(topology.sequence, dtype=np.int32),
        np.ascontiguousarray(topology.residue_sizes, dtype=np.int32),
        np.ascontiguousarray(topology.chain_lengths, dtype=np.int32),
    )
    return edges, n_atoms


def edges_to_csr(edges: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert edge list to CSR-style neighbor lists.

    Args:
        edges: (E, 2) int64 array of directed edges
        n_atoms: Total number of atoms

    Returns:
        Tuple of:
            offsets: (n_atoms+1,) int64 cumulative neighbor counts
            neighbors: (E,) int64 flattened neighbor indices, grouped by source
    """
    return _edges_to_csr_c(
        np.ascontiguousarray(edges, dtype=np.int64),
        n_atoms
    )


def build_bond_graph_csr(topology: TopologyInfo) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build CSR bond graph from topology info.

    Convenience function combining build_bond_graph_from_topology and edges_to_csr.

    Args:
        topology: TopologyInfo containing structural metadata.

    Returns:
        Tuple of:
            offsets: (N+1,) int64 CSR offsets array
            neighbors: (E,) int64 CSR neighbor indices
            n_atoms: Total number of atoms
    """
    edges, n_atoms = build_bond_graph_from_topology(topology)
    if len(edges) == 0:
        return np.zeros(n_atoms + 1, dtype=np.int64), np.array([], dtype=np.int64), n_atoms
    offsets, neighbors = edges_to_csr(edges, n_atoms)
    return offsets, neighbors, n_atoms


def find_connected_components(
    offsets: np.ndarray, neighbors: np.ndarray, n_atoms: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Find all connected components in a CSR-format graph.

    Args:
        offsets: (N+1,) CSR offsets array
        neighbors: (E,) CSR neighbor indices
        n_atoms: Total number of atoms

    Returns:
        Tuple of:
            atoms: (N,) int64 atom indices grouped by component
            component_offsets: (n_components+1,) int64 offsets into atoms array
            n_components: Number of components found
    """
    atoms, component_offsets, n_components = _find_connected_components_c(
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(neighbors, dtype=np.int64),
        n_atoms
    )
    return atoms, component_offsets, int(n_components)


# =============================================================================
# Z-MATRIX CONSTRUCTION HELPERS
# =============================================================================


def _build_zmatrix_indices_from_topology(
    topology: TopologyInfo,
    csr_offsets: np.ndarray | None = None,
    csr_neighbors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Z-matrix as (M, 4) int64 array from topology info.

    Internal function used by ZMatrix.from_topology().

    Returns:
        Tuple of (indices, dihedral_types, levels, component_ids).
    """
    n_atoms = topology.n_atoms

    # Build CSR if not provided
    if csr_offsets is None or csr_neighbors is None:
        edges, n_atoms = build_bond_graph_from_topology(topology)
        if len(edges) == 0:
            return (np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8),
                    np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        offsets, neighbors = edges_to_csr(edges, n_atoms)
    else:
        offsets, neighbors = csr_offsets, csr_neighbors

    # Find all connected components in the bond graph
    comp_atoms, comp_offsets, n_components = find_connected_components(offsets, neighbors, n_atoms)

    if n_components == 0:
        return (np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8),
                np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    # Extract component info for Z-matrix construction
    component_sizes = np.diff(comp_offsets).astype(np.int64)
    component_starts = comp_atoms[comp_offsets[:-1]].astype(np.int64)
    roots = component_starts.copy()

    # Build Z-matrix with dihedral-aware reference selection
    return build_zmatrix_from_components(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(neighbors, dtype=np.int64),
        n_atoms,
        component_starts,
        component_sizes,
        roots,
        atoms=np.ascontiguousarray(topology.atoms, dtype=np.int32),
        sequence=np.ascontiguousarray(topology.sequence, dtype=np.int32),
        res_sizes=np.ascontiguousarray(topology.residue_sizes, dtype=np.int32),
    )


def _compute_nerf_levels(indices: np.ndarray) -> np.ndarray:
    """
    Compute correct NERF dependency levels from Z-matrix indices.

    For parallel NERF reconstruction, an atom's level must be strictly
    greater than all its reference atoms' levels.
    """
    n_entries = len(indices)
    if n_entries == 0:
        return np.array([], dtype=np.int32)

    # Build atom_idx -> entry_idx mapping
    max_atom = int(indices[:, 0].max()) + 1
    atom_to_entry = np.full(max_atom, -1, dtype=np.int32)
    for i in range(n_entries):
        atom_to_entry[indices[i, 0]] = i

    # Compute levels iteratively
    levels = np.zeros(n_entries, dtype=np.int32)
    for i in range(n_entries):
        dist_ref = indices[i, 1]
        ang_ref = indices[i, 2]
        dih_ref = indices[i, 3]

        max_ref_level = -1

        if dist_ref >= 0 and dist_ref < max_atom:
            ref_entry = atom_to_entry[dist_ref]
            if ref_entry >= 0:
                max_ref_level = max(max_ref_level, levels[ref_entry])

        if ang_ref >= 0 and ang_ref < max_atom:
            ref_entry = atom_to_entry[ang_ref]
            if ref_entry >= 0:
                max_ref_level = max(max_ref_level, levels[ref_entry])

        if dih_ref >= 0 and dih_ref < max_atom:
            ref_entry = atom_to_entry[dih_ref]
            if ref_entry >= 0:
                max_ref_level = max(max_ref_level, levels[ref_entry])

        levels[i] = max_ref_level + 1

    return levels




def build_zmatrix_from_components(
    offsets: np.ndarray,
    neighbors: np.ndarray,
    n_atoms: int,
    component_starts: np.ndarray,
    component_sizes: np.ndarray,
    roots: np.ndarray,
    atoms: np.ndarray | None = None,
    sequence: np.ndarray | None = None,
    res_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Z-matrix from CSR graph for multiple connected components.

    When atoms, sequence, and res_sizes are provided, uses dihedral-aware
    reference selection to ensure named dihedrals (PHI, PSI, etc.) use
    the correct reference atoms.

    Args:
        offsets: (N+1,) CSR offsets array
        neighbors: (E,) CSR neighbor indices
        n_atoms: Total number of atoms
        component_starts: Start indices for each component
        component_sizes: Number of atoms in each component
        roots: Root atom for each component
        atoms: (N,) int32 atom types (optional, for dihedral-aware mode)
        sequence: (R,) int32 residue types (optional)
        res_sizes: (R,) int32 atoms per residue (optional)

    Returns:
        Tuple of:
            zmatrix: (M, 4) Z-matrix array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 dihedral type per entry (-1 if not named)
            levels: (M,) int32 BFS level per entry
            component_ids: (M,) int32 component index per entry
    """
    # Use parallel C implementation with optional dihedral-aware mode
    zmatrix, dihedral_types, levels, counts = _build_zmatrix_parallel_c(
        offsets, neighbors, n_atoms,
        component_starts, component_sizes, roots,
        atoms, sequence, res_sizes
    )
    # Trim to actual entries and build component_ids
    total_entries = int(counts.sum())
    if total_entries < len(zmatrix):
        result = np.zeros((total_entries, 4), dtype=np.int64)
        result_dtypes = np.full(total_entries, -1, dtype=np.int8)
        result_levels = np.zeros(total_entries, dtype=np.int32)
        result_comp_ids = np.zeros(total_entries, dtype=np.int32)
        src_offset = 0
        dst_offset = 0
        for comp_idx, (size, count) in enumerate(zip(component_sizes, counts)):
            count = int(count)
            result[dst_offset:dst_offset + count] = zmatrix[src_offset:src_offset + count]
            result_dtypes[dst_offset:dst_offset + count] = dihedral_types[src_offset:src_offset + count]
            result_levels[dst_offset:dst_offset + count] = levels[src_offset:src_offset + count]
            result_comp_ids[dst_offset:dst_offset + count] = comp_idx
            src_offset += size
            dst_offset += count
    else:
        result = zmatrix[:total_entries]
        result_dtypes = dihedral_types[:total_entries]
        result_levels = levels[:total_entries]
        # Build component_ids from counts
        result_comp_ids = np.zeros(total_entries, dtype=np.int32)
        dst_offset = 0
        for comp_idx, count in enumerate(counts):
            count = int(count)
            result_comp_ids[dst_offset:dst_offset + count] = comp_idx
            dst_offset += count

    # Return with entries sorted by component (from construction order)
    # This enables component-level parallelism in NERF reconstruction
    return result, result_dtypes, result_levels, result_comp_ids


# =============================================================================
# NERF RECONSTRUCTION WRAPPER
# =============================================================================


def nerf_reconstruct(
    zmatrix_indices: Array,
    internal: Array,
    component_offsets: Array | None = None,
    anchor_coords: Array | None = None,
    component_ids: Array | None = None,
) -> Array:
    """
    Reconstruct Cartesian coordinates using NERF algorithm.

    Uses CUDA kernels when available for GPU tensors, otherwise falls back
    to CPU C extension. For PyTorch tensors that require gradients, uses
    autograd functions with backward passes.

    The Natural Extension Reference Frame algorithm places each atom
    by constructing a local coordinate system from three previously
    placed atoms, then positioning the new atom using spherical-like
    coordinates (distance, angle, dihedral).

    Args:
        zmatrix_indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref].
            The number of atoms is inferred from the first dimension.
        internal: (M, 3) array of internal coordinates.
            Each row: [distance, angle, dihedral].
        component_offsets: (n_components+1,) int32 CSR-style offsets for component-parallel NERF.
            When provided, enables parallel NERF by processing each connected component
            independently. Can be obtained from ZMatrix.component_offsets.
        anchor_coords: (n_components, 3, 3) float32 anchor positions for each component.
            When provided with component_ids, atoms are placed directly in the
            reference frame defined by these anchors, eliminating Kabsch rotation.
        component_ids: (M,) int32 component index per Z-matrix entry.
            Required when anchor_coords is provided.

    Returns:
        (N, 3) array of Cartesian coordinates in original atom order.
    """
    # Late import to avoid circular dependency (dispatch imports graph)
    from .dispatch import nerf_reconstruct as _dispatch_nerf_reconstruct

    return _dispatch_nerf_reconstruct(
        zmatrix_indices, internal,
        component_offsets, anchor_coords, component_ids
    )
