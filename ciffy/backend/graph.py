"""
Bond graph construction utilities.

This module provides algorithms for molecular bond graph construction:

- Bond graph construction from polymer (edge list and CSR formats)
- Connected component finding

All functions are backend-agnostic: inputs can be NumPy arrays or PyTorch
tensors, and outputs will match the input backend.
"""

from __future__ import annotations

import numpy as np

from . import Array, is_torch, to_numpy
from .ops import to_backend

# C extension imports (required)
from .._c import _build_bond_graph as _build_bond_graph_c
from .._c import _edges_to_csr as _edges_to_csr_c
from .._c import _find_connected_components as _find_connected_components_c

__all__ = [
    "build_bond_graph",
    "edges_to_csr",
    "find_connected_components",
]


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
    res_sizes = polymer.counts(Scale.RESIDUE)
    edges = _build_bond_graph_c(
        np.ascontiguousarray(to_numpy(polymer.atoms), dtype=np.int32),
        np.ascontiguousarray(to_numpy(polymer.sequence), dtype=np.int32),
        np.ascontiguousarray(to_numpy(res_sizes), dtype=np.int32),
        np.ascontiguousarray(to_numpy(polymer.lengths), dtype=np.int32),
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
