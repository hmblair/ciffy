"""
Bond graph construction and topology utilities.

This module provides data structures and algorithms for molecular topology:

- TopologyInfo: Immutable container for polymer structure metadata
- Bond graph construction from topology (edge list and CSR formats)
- Connected component finding

All functions are backend-agnostic: inputs can be NumPy arrays or PyTorch
tensors, and outputs will match the input backend.

.. note::
    This is an **internal backend module**. For coordinate operations, use
    the higher-level ``ciffy.internal.MolecularGeometry`` or ``Polymer`` APIs.
    The ``backend.dispatch`` module provides the coordinate conversion functions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import Array, is_torch, to_numpy
from .ops import to_backend

if TYPE_CHECKING:
    from .._c import (
        _build_bond_graph,
        _edges_to_csr,
        _find_connected_components,
    )

# C extension imports (required)
from .._c import _build_bond_graph as _build_bond_graph_c
from .._c import _edges_to_csr as _edges_to_csr_c
from .._c import _find_connected_components as _find_connected_components_c

__all__ = [
    # Data structures
    "TopologyInfo",
    # Bond graph functions
    "build_bond_graph",
    "build_bond_graph_from_topology",
    "build_bond_graph_csr",
    "edges_to_csr",
    "find_connected_components",
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
        elements: (N,) int32 array of element indices (atomic numbers).
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
    elements: np.ndarray
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
        from ..biochemistry import Scale

        # Convert to numpy for storage (topology is always CPU)
        atoms = to_numpy(polymer.atoms).astype(np.int32)
        elements = to_numpy(polymer.elements).astype(np.int32)
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
            elements=elements,
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
        new_elements = self.elements[mask_np].astype(np.int32)

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
            elements=new_elements,
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
    from ..biochemistry import Scale

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


def edges_to_csr(edges: Array, n_atoms: int) -> tuple[Array, Array]:
    """
    Convert edge list to CSR-style neighbor lists.

    Args:
        edges: (E, 2) int64 array of directed edges (numpy or torch)
        n_atoms: Total number of atoms

    Returns:
        Tuple of:
            offsets: (n_atoms+1,) int64 cumulative neighbor counts
            neighbors: (E,) int64 flattened neighbor indices, grouped by source

        Output backend matches input backend.
    """
    edges_np = np.ascontiguousarray(to_numpy(edges), dtype=np.int64)
    offsets, neighbors = _edges_to_csr_c(edges_np, n_atoms)

    # Convert back to input backend
    if is_torch(edges):
        return to_backend(offsets, like=edges), to_backend(neighbors, like=edges)
    return offsets, neighbors


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
    offsets: Array, neighbors: Array, n_atoms: int
) -> tuple[Array, Array, int]:
    """
    Find all connected components in a CSR-format graph.

    Args:
        offsets: (N+1,) CSR offsets array (numpy or torch)
        neighbors: (E,) CSR neighbor indices (numpy or torch)
        n_atoms: Total number of atoms

    Returns:
        Tuple of:
            atoms: (N,) int64 atom indices grouped by component
            component_offsets: (n_components+1,) int64 offsets into atoms array
            n_components: Number of components found

        Output backend matches input backend.
    """
    offsets_np = np.ascontiguousarray(to_numpy(offsets), dtype=np.int64)
    neighbors_np = np.ascontiguousarray(to_numpy(neighbors), dtype=np.int64)

    atoms, component_offsets, n_components = _find_connected_components_c(
        offsets_np, neighbors_np, n_atoms
    )

    # Convert back to input backend
    if is_torch(offsets):
        return to_backend(atoms, like=offsets), to_backend(component_offsets, like=offsets), int(n_components)
    return atoms, component_offsets, int(n_components)
