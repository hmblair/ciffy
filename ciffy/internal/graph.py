"""
Bond graph construction and spanning tree traversal for Z-matrix generation.

Builds a complete bond graph by combining intra-residue bonds from
Residue.X.bonds with inter-residue linking from LinkingDefinition.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer
    from .topology import TopologyInfo

from ..backend import to_numpy

# C extension imports (required)
from .._c import _build_bond_graph as _build_bond_graph_c
from .._c import _edges_to_csr as _edges_to_csr_c
from .._c import _build_zmatrix_parallel as _build_zmatrix_parallel_c
from .._c import _find_connected_components as _find_connected_components_c


# ZMatrix class is now in zmatrix.py - re-export for backwards compatibility
from .zmatrix import ZMatrix


# =============================================================================
# BOND GRAPH CONSTRUCTION
# =============================================================================


def build_bond_graph(polymer: "Polymer") -> tuple[np.ndarray, int]:
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


def build_bond_graph_from_topology(topology: "TopologyInfo") -> tuple[np.ndarray, int]:
    """
    Build edge list representation of molecular bonds from topology info.

    Constructs bonds as an (E, 2) array for array-based processing.
    Combines intra-residue bonds from Residue.bond_indices and inter-residue
    bonds from LINKING_BY_TYPE.

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


def build_bond_graph_csr(topology: "TopologyInfo") -> tuple[np.ndarray, np.ndarray, int]:
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
# Z-MATRIX INDICES CONSTRUCTION
# =============================================================================


def _build_zmatrix_indices_from_topology(
    topology: "TopologyInfo",
    csr_offsets: np.ndarray | None = None,
    csr_neighbors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Z-matrix as (M, 4) int64 array from topology info.

    Internal function used by ZMatrix.from_topology().
    Processes all connected components in the bond graph, not just one per chain.
    Uses C extension with dihedral-aware reference selection for ~10-20x speedup.

    Args:
        topology: TopologyInfo containing structural metadata.
        csr_offsets: Optional pre-built CSR offsets array. If None, built from topology.
        csr_neighbors: Optional pre-built CSR neighbors array. If None, built from topology.

    Returns:
        Tuple of:
            indices: (M, 4) array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 array of dihedral types (-1 if not named)
            levels: (M,) int32 array of BFS levels
    """
    n_atoms = topology.n_atoms

    # Build CSR if not provided
    if csr_offsets is None or csr_neighbors is None:
        edges, n_atoms = build_bond_graph_from_topology(topology)
        if len(edges) == 0:
            return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8), np.array([], dtype=np.int32)
        offsets, neighbors = edges_to_csr(edges, n_atoms)
    else:
        offsets, neighbors = csr_offsets, csr_neighbors

    # Find all connected components in the bond graph
    comp_atoms, comp_offsets, n_components = find_connected_components(offsets, neighbors, n_atoms)

    if n_components == 0:
        return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8), np.array([], dtype=np.int32)

    # Extract component info for Z-matrix construction
    # For each component: first atom is root, size is offset diff
    component_sizes = np.diff(comp_offsets).astype(np.int64)
    component_starts = comp_atoms[comp_offsets[:-1]].astype(np.int64)  # First atom of each component
    roots = component_starts.copy()  # Use first atom as root

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


# =============================================================================
# BFS Z-MATRIX CONSTRUCTION
# =============================================================================


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    """
    # Use parallel C implementation with optional dihedral-aware mode
    zmatrix, dihedral_types, levels, counts = _build_zmatrix_parallel_c(
        offsets, neighbors, n_atoms,
        component_starts, component_sizes, roots,
        atoms, sequence, res_sizes
    )
    # Trim to actual entries
    total_entries = int(counts.sum())
    if total_entries < len(zmatrix):
        result = np.zeros((total_entries, 4), dtype=np.int64)
        result_dtypes = np.full(total_entries, -1, dtype=np.int8)
        result_levels = np.zeros(total_entries, dtype=np.int32)
        src_offset = 0
        dst_offset = 0
        for size, count in zip(component_sizes, counts):
            count = int(count)
            result[dst_offset:dst_offset + count] = zmatrix[src_offset:src_offset + count]
            result_dtypes[dst_offset:dst_offset + count] = dihedral_types[src_offset:src_offset + count]
            result_levels[dst_offset:dst_offset + count] = levels[src_offset:src_offset + count]
            src_offset += size
            dst_offset += count
        return result, result_dtypes, result_levels
    return zmatrix[:total_entries], dihedral_types[:total_entries], levels[:total_entries]




