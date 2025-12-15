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
from .._c import _build_canonical_zmatrix as _build_canonical_zmatrix_c


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


def find_connected_components(
    offsets: np.ndarray, neighbors: np.ndarray, n_atoms: int
) -> list[tuple[int, int]]:
    """
    Find all connected components in a CSR-format graph.

    Args:
        offsets: (N+1,) CSR offsets array
        neighbors: (E,) CSR neighbor indices
        n_atoms: Total number of atoms

    Returns:
        List of (root, size) tuples for each component.
    """
    roots, sizes, n_components = _find_connected_components_c(
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(neighbors, dtype=np.int64),
        n_atoms
    )
    return [(int(roots[i]), int(sizes[i])) for i in range(n_components)]


# =============================================================================
# SPANNING TREE CONSTRUCTION
# =============================================================================


def select_root_atom(
    polymer: "Polymer",
    chain_start_atom: int,
    chain_atom_count: int,
    chain_start_res: int,
) -> int:
    """
    Select appropriate root atom for a chain.

    Uses backbone atoms as roots:
    - Protein: N of first residue
    - Nucleic acid: P of first residue (or O5' if no P)

    Args:
        polymer: Polymer structure.
        chain_start_atom: Global atom index where this chain starts.
        chain_atom_count: Number of atoms in this chain.
        chain_start_res: Residue index where this chain starts.

    Returns:
        Global atom index for the root.
    """
    from ..biochemistry import Residue, ATOM_NAMES
    from ..types import Molecule

    # Get residue type to determine preferred root atom
    res_type_idx = int(polymer.sequence[chain_start_res])
    try:
        residue = Residue(res_type_idx)
        mol_type = residue.molecule_type
    except ValueError:
        mol_type = None

    # Determine preferred root atom names (Python convention: ' -> p)
    if mol_type in (Molecule.PROTEIN, Molecule.PROTEIN_D, Molecule.CYCLIC_PEPTIDE):
        preferred = ['N', 'CA', 'C']
    else:
        # Nucleic acids and others
        preferred = ['P', 'O5p', 'C5p']

    # Find preferred atom in first residue
    for local_idx in range(min(chain_atom_count, 20)):  # Check first 20 atoms
        atom_value = int(polymer.atoms[chain_start_atom + local_idx])
        atom_name = ATOM_NAMES.get(atom_value, "").replace("'", "p").replace('"', "pp")
        if atom_name in preferred:
            return chain_start_atom + local_idx

    # Fallback to first atom
    return chain_start_atom


# =============================================================================
# Z-MATRIX INDICES CONSTRUCTION
# =============================================================================


def _build_zmatrix_indices_from_topology(
    topology: "TopologyInfo",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Z-matrix as (M, 4) int64 array from topology info.

    Internal function used by ZMatrix.from_topology().
    Processes all connected components in the bond graph, not just one per chain.
    Uses C extension with dihedral-aware reference selection for ~10-20x speedup.

    Args:
        topology: TopologyInfo containing structural metadata.

    Returns:
        Tuple of:
            indices: (M, 4) array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 array of dihedral types (-1 if not named)
    """
    # Build array-based graph
    edges, n_atoms = build_bond_graph_from_topology(topology)

    if len(edges) == 0:
        return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8)

    # Convert to CSR format
    offsets, neighbors = edges_to_csr(edges, n_atoms)

    # Find all connected components in the bond graph
    components = find_connected_components(offsets, neighbors, n_atoms)

    if len(components) == 0:
        return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8)

    # Prepare component info for Z-matrix construction
    # Components can be either:
    # - (root, size) tuples from C extension
    # - list of atom indices from Python fallback
    component_starts = []
    component_sizes = []
    roots = []

    for component in components:
        if isinstance(component, tuple):
            # C extension format: (root, size)
            root, size = component
            component_starts.append(root)
            component_sizes.append(size)
            roots.append(root)
        else:
            # Python format: list of atom indices
            root = min(component)
            component_starts.append(root)
            component_sizes.append(len(component))
            roots.append(root)

    # Build Z-matrix with dihedral-aware reference selection
    return build_zmatrix_from_components(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(neighbors, dtype=np.int64),
        n_atoms,
        np.array(component_starts, dtype=np.int64),
        np.array(component_sizes, dtype=np.int64),
        np.array(roots, dtype=np.int64),
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
) -> tuple[np.ndarray, np.ndarray]:
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
    """
    # Use parallel C implementation with optional dihedral-aware mode
    zmatrix, dihedral_types, counts = _build_zmatrix_parallel_c(
        offsets, neighbors, n_atoms,
        component_starts, component_sizes, roots,
        atoms, sequence, res_sizes
    )
    # Trim to actual entries
    total_entries = int(counts.sum())
    if total_entries < len(zmatrix):
        result = np.zeros((total_entries, 4), dtype=np.int64)
        result_dtypes = np.full(total_entries, -1, dtype=np.int8)
        src_offset = 0
        dst_offset = 0
        for size, count in zip(component_sizes, counts):
            count = int(count)
            result[dst_offset:dst_offset + count] = zmatrix[src_offset:src_offset + count]
            result_dtypes[dst_offset:dst_offset + count] = dihedral_types[src_offset:src_offset + count]
            src_offset += size
            dst_offset += count
        return result, result_dtypes
    return zmatrix[:total_entries], dihedral_types[:total_entries]




