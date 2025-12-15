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
# DIHEDRAL TYPE ANNOTATION
# =============================================================================


def annotate_dihedral_types(
    zmatrix_indices: np.ndarray,
    atoms: np.ndarray,
    sequence: np.ndarray,
    residue_starts: np.ndarray,
    chain_boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Post-process Z-matrix to annotate named dihedral types.

    For atoms that "own" a named dihedral (PHI, PSI, ALPHA, etc.), this function:
    1. Updates the Z-matrix entry to use the correct reference atoms
    2. Records the dihedral type in a parallel array

    Args:
        zmatrix_indices: (M, 4) int64 - existing Z-matrix [atom, dist, ang, dih]
        atoms: (N,) int32 - atom enum values for all atoms
        sequence: (R,) int32 - residue enum values
        residue_starts: (R+1,) int64 - cumulative atom offsets per residue
        chain_boundaries: (C,) int64 - residue indices where chains start

    Returns:
        updated_zmatrix: (M, 4) int64 - Z-matrix with updated references
        dihedral_types: (M,) int8 - dihedral type for each entry (-1 if unnamed)
    """
    from ..biochemistry import ATOM_DIHEDRAL_TYPE, ATOM_DIHEDRAL_REFS, Residue

    n_entries = len(zmatrix_indices)
    n_residues = len(sequence)
    n_atoms = len(atoms)

    # Handle empty arrays
    if len(ATOM_DIHEDRAL_TYPE) == 0:
        return zmatrix_indices.copy(), np.full(n_entries, -1, dtype=np.int8)

    # Build atom_idx -> zmatrix_position mapping
    atom_to_z = {int(zmatrix_indices[i, 0]): i for i in range(n_entries)}

    # Build atom_idx -> residue_idx mapping
    atom_to_res = np.zeros(n_atoms, dtype=np.int64)
    for res_idx in range(n_residues):
        start = int(residue_starts[res_idx])
        end = int(residue_starts[res_idx + 1]) if res_idx + 1 < len(residue_starts) else n_atoms
        atom_to_res[start:end] = res_idx

    # Build chain boundary set for detecting chain breaks
    chain_start_residues = set(int(b) for b in chain_boundaries)

    # Cache: residue_type -> list of canonical atom types in order
    canonical_atoms_cache: dict[int, list[int]] = {}

    def get_canonical_atom_types(res_type: int) -> list[int]:
        """Get canonical atom types for a residue type."""
        if res_type not in canonical_atoms_cache:
            try:
                res = Residue(res_type)
                canonical_atoms_cache[res_type] = [atom.value for atom in res.atoms]
            except ValueError:
                canonical_atoms_cache[res_type] = []
        return canonical_atoms_cache[res_type]

    def find_atom_by_type(target_res: int, expected_type: int) -> int:
        """Find global atom index by type within a residue. Returns -1 if not found."""
        start = int(residue_starts[target_res])
        end = int(residue_starts[target_res + 1]) if target_res + 1 < len(residue_starts) else n_atoms
        for i in range(start, end):
            if int(atoms[i]) == expected_type:
                return i
        return -1

    # Initialize outputs
    updated_zmatrix = zmatrix_indices.copy()
    dihedral_types = np.full(n_entries, -1, dtype=np.int8)

    # Process each Z-matrix entry
    for z_idx in range(n_entries):
        atom_idx = int(zmatrix_indices[z_idx, 0])
        atom_type = int(atoms[atom_idx])

        # Check bounds for atom_type lookup
        if atom_type < 0 or atom_type >= len(ATOM_DIHEDRAL_TYPE):
            continue

        # Check if this atom owns a named dihedral
        dtype_idx = int(ATOM_DIHEDRAL_TYPE[atom_type])
        if dtype_idx < 0:
            continue  # Not a dihedral owner

        # Get residue index for this atom
        res_idx = int(atom_to_res[atom_idx])
        res_type = int(sequence[res_idx])

        # Get canonical atom types for the owner's residue type
        canonical_atoms = get_canonical_atom_types(res_type)
        if not canonical_atoms:
            continue

        # Get reference pattern: [dih_ref, ang_ref, dist_ref] as (offset, canonical_local_idx)
        refs = ATOM_DIHEDRAL_REFS[atom_type]  # (3, 2) array

        # Resolve references to global atom indices
        valid = True
        resolved_refs = []

        for ref_idx in range(3):
            offset = int(refs[ref_idx, 0])
            canonical_local_idx = int(refs[ref_idx, 1])

            target_res = res_idx + offset

            # Check residue bounds
            if target_res < 0 or target_res >= n_residues:
                valid = False
                break

            # Check chain boundary (don't span chains)
            if offset != 0:
                # Check if we cross a chain boundary
                min_res = min(res_idx, target_res)
                max_res = max(res_idx, target_res)
                for boundary_res in chain_start_residues:
                    if min_res < boundary_res <= max_res:
                        valid = False
                        break
                if not valid:
                    break

            # Get the expected atom TYPE from canonical ordering
            # For offset != 0, we need the canonical atoms of the TARGET residue
            if offset != 0:
                target_res_type = int(sequence[target_res])
                target_canonical = get_canonical_atom_types(target_res_type)
            else:
                target_canonical = canonical_atoms

            if canonical_local_idx >= len(target_canonical):
                valid = False
                break

            expected_atom_type = target_canonical[canonical_local_idx]

            # Find atom of this type in the target residue
            global_idx = find_atom_by_type(target_res, expected_atom_type)

            if global_idx < 0:
                valid = False
                break

            # Verify atom exists in Z-matrix (was reachable from bonds)
            if global_idx not in atom_to_z:
                valid = False
                break

            # Verify referenced atom appears BEFORE current atom in Z-matrix
            # (required for Z-matrix validity - references must point to earlier entries)
            ref_z_idx = atom_to_z[global_idx]
            if ref_z_idx >= z_idx:
                valid = False
                break

            resolved_refs.append(global_idx)

        if valid and len(resolved_refs) == 3:
            # Update Z-matrix entry with correct references
            # refs[0] = dih_ref, refs[1] = ang_ref, refs[2] = dist_ref
            updated_zmatrix[z_idx, 1] = resolved_refs[2]  # dist_ref
            updated_zmatrix[z_idx, 2] = resolved_refs[1]  # ang_ref
            updated_zmatrix[z_idx, 3] = resolved_refs[0]  # dih_ref
            dihedral_types[z_idx] = dtype_idx

    return updated_zmatrix, dihedral_types


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
# CANONICAL Z-MATRIX CONSTRUCTION
# =============================================================================


def build_canonical_zmatrix(
    polymer: "Polymer",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Z-matrix using canonical references from codegen.

    Two-phase approach:
    1. Build BFS Z-matrix for correct placement order (refs point to earlier atoms)
    2. Overlay canonical refs where possible for named dihedral capture

    Args:
        polymer: Polymer structure.

    Returns:
        Tuple of:
            zmatrix: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 array mapping entry -> dihedral type (-1 if unnamed)
    """
    from ..biochemistry import (
        ATOM_CANONICAL_REFS,
        ATOM_HAS_CANONICAL_REFS,
        ATOM_DIHEDRAL_TYPE,
    )
    from ..types import Scale

    n_atoms = polymer.size()
    if n_atoms == 0:
        return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8)

    # Phase 1: Build BFS Z-matrix (handles ordering correctly)
    # This reuses existing optimized C code path
    bfs_indices = _build_zmatrix_indices(polymer)

    if len(bfs_indices) == 0:
        return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8)

    # Check if we have canonical refs available
    if len(ATOM_HAS_CANONICAL_REFS) == 0:
        return bfs_indices, np.full(len(bfs_indices), -1, dtype=np.int8)

    # Phase 2: Overlay canonical refs for named dihedral capture
    # Pre-compute all lookups for efficiency
    atoms_np = to_numpy(polymer.atoms)
    sequence_np = to_numpy(polymer.sequence)
    n_entries = len(bfs_indices)
    n_residues = len(sequence_np)

    # Compute residue boundaries once
    res_sizes = to_numpy(polymer.sizes(Scale.RESIDUE))
    residue_starts = np.zeros(n_residues + 1, dtype=np.int64)
    residue_starts[1:] = np.cumsum(res_sizes)

    # Build chain boundary set
    chain_boundaries = np.zeros(len(polymer.lengths) + 1, dtype=np.int64)
    chain_boundaries[1:] = np.cumsum(to_numpy(polymer.lengths))
    chain_start_set = set(chain_boundaries[:-1].tolist())

    # Build atom_idx -> Z-matrix position mapping (for validation)
    atom_to_zidx = np.full(n_atoms, -1, dtype=np.int64)
    for z_idx in range(n_entries):
        atom_to_zidx[int(bfs_indices[z_idx, 0])] = z_idx

    # Build atom_idx -> residue_idx mapping
    atom_to_res = np.zeros(n_atoms, dtype=np.int64)
    for res_idx in range(n_residues):
        start = residue_starts[res_idx]
        end = residue_starts[res_idx + 1]
        atom_to_res[start:end] = res_idx

    # Initialize outputs
    canonical_indices = bfs_indices.copy()
    dihedral_types = np.full(n_entries, -1, dtype=np.int8)

    # Process each Z-matrix entry
    for z_idx in range(n_entries):
        atom_idx = int(bfs_indices[z_idx, 0])
        atom_type = int(atoms_np[atom_idx])

        # Check if we have canonical refs for this atom type
        if atom_type < 0 or atom_type >= len(ATOM_HAS_CANONICAL_REFS):
            continue
        if not ATOM_HAS_CANONICAL_REFS[atom_type]:
            continue

        # Get canonical refs
        refs = ATOM_CANONICAL_REFS[atom_type]
        res_idx = int(atom_to_res[atom_idx])

        # Resolve all three refs
        resolved = []
        valid = True

        for i in range(3):  # dist, ang, dih
            ref_type = int(refs[i])
            ref_off = int(refs[i + 3])

            if ref_type < 0:
                resolved.append(-1)
                continue

            target_res = res_idx + ref_off

            # Check residue bounds
            if target_res < 0 or target_res >= n_residues:
                valid = False
                break

            # Check chain boundary for inter-residue refs
            if ref_off != 0:
                min_res, max_res = min(res_idx, target_res), max(res_idx, target_res)
                if any(min_res < b <= max_res for b in chain_start_set):
                    valid = False
                    break

            # Find atom by type in target residue
            start = int(residue_starts[target_res])
            end = int(residue_starts[target_res + 1])
            global_idx = -1
            for j in range(start, end):
                if int(atoms_np[j]) == ref_type:
                    global_idx = j
                    break

            if global_idx < 0:
                valid = False
                break

            # Verify atom is placed before current atom in Z-matrix
            ref_zidx = atom_to_zidx[global_idx]
            if ref_zidx < 0 or ref_zidx >= z_idx:
                valid = False
                break

            resolved.append(global_idx)

        if not valid or len(resolved) != 3:
            continue

        # Update Z-matrix entry with canonical refs
        canonical_indices[z_idx, 1] = resolved[0]  # dist_ref
        canonical_indices[z_idx, 2] = resolved[1]  # ang_ref
        canonical_indices[z_idx, 3] = resolved[2]  # dih_ref

        # Set dihedral type if this atom owns a named dihedral
        if atom_type < len(ATOM_DIHEDRAL_TYPE):
            dtype_idx = int(ATOM_DIHEDRAL_TYPE[atom_type])
            if dtype_idx >= 0 and all(r >= 0 for r in resolved):
                dihedral_types[z_idx] = dtype_idx

    return canonical_indices, dihedral_types


def build_canonical_zmatrix_from_topology(
    topology: "TopologyInfo",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Z-matrix using canonical references from topology info.

    Two-phase approach:
    1. Build BFS Z-matrix for correct placement order (refs point to earlier atoms)
    2. Overlay canonical refs where possible for named dihedral capture

    Args:
        topology: TopologyInfo containing structural metadata.

    Returns:
        Tuple of:
            zmatrix: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
            dihedral_types: (M,) int8 array mapping entry -> dihedral type (-1 if unnamed)
    """
    from ..biochemistry import (
        ATOM_CANONICAL_REFS,
        ATOM_HAS_CANONICAL_REFS,
        ATOM_DIHEDRAL_TYPE,
    )

    n_atoms = topology.n_atoms
    if n_atoms == 0:
        return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8)

    # Phase 1: Build BFS Z-matrix (handles ordering correctly)
    # This reuses existing optimized C code path
    bfs_indices = _build_zmatrix_indices_from_topology(topology)

    if len(bfs_indices) == 0:
        return np.zeros((0, 4), dtype=np.int64), np.array([], dtype=np.int8)

    # Check if we have canonical refs available
    if len(ATOM_HAS_CANONICAL_REFS) == 0:
        return bfs_indices, np.full(len(bfs_indices), -1, dtype=np.int8)

    # Phase 2: Overlay canonical refs for named dihedral capture
    # Pre-compute all lookups for efficiency
    atoms_np = topology.atoms
    sequence_np = topology.sequence
    n_entries = len(bfs_indices)
    n_residues = len(sequence_np)

    # Compute residue boundaries once
    res_sizes = topology.residue_sizes
    residue_starts = np.zeros(n_residues + 1, dtype=np.int64)
    residue_starts[1:] = np.cumsum(res_sizes)

    # Build chain boundary set
    chain_boundaries = np.zeros(len(topology.chain_lengths) + 1, dtype=np.int64)
    chain_boundaries[1:] = np.cumsum(topology.chain_lengths)
    chain_start_set = set(chain_boundaries[:-1].tolist())

    # Build atom_idx -> Z-matrix position mapping (for validation)
    atom_to_zidx = np.full(n_atoms, -1, dtype=np.int64)
    for z_idx in range(n_entries):
        atom_to_zidx[int(bfs_indices[z_idx, 0])] = z_idx

    # Build atom_idx -> residue_idx mapping
    atom_to_res = np.zeros(n_atoms, dtype=np.int64)
    for res_idx in range(n_residues):
        start = residue_starts[res_idx]
        end = residue_starts[res_idx + 1]
        atom_to_res[start:end] = res_idx

    # Initialize outputs
    canonical_indices = bfs_indices.copy()
    dihedral_types = np.full(n_entries, -1, dtype=np.int8)

    # Process each Z-matrix entry
    for z_idx in range(n_entries):
        atom_idx = int(bfs_indices[z_idx, 0])
        atom_type = int(atoms_np[atom_idx])

        # Check if we have canonical refs for this atom type
        if atom_type < 0 or atom_type >= len(ATOM_HAS_CANONICAL_REFS):
            continue
        if not ATOM_HAS_CANONICAL_REFS[atom_type]:
            continue

        # Get canonical refs
        refs = ATOM_CANONICAL_REFS[atom_type]
        res_idx = int(atom_to_res[atom_idx])

        # Resolve all three refs
        resolved = []
        valid = True

        for i in range(3):  # dist, ang, dih
            ref_type = int(refs[i])
            ref_off = int(refs[i + 3])

            if ref_type < 0:
                resolved.append(-1)
                continue

            target_res = res_idx + ref_off

            # Check residue bounds
            if target_res < 0 or target_res >= n_residues:
                valid = False
                break

            # Check chain boundary for inter-residue refs
            if ref_off != 0:
                min_res, max_res = min(res_idx, target_res), max(res_idx, target_res)
                if any(min_res < b <= max_res for b in chain_start_set):
                    valid = False
                    break

            # Find atom by type in target residue
            start = int(residue_starts[target_res])
            end = int(residue_starts[target_res + 1])
            global_idx = -1
            for j in range(start, end):
                if int(atoms_np[j]) == ref_type:
                    global_idx = j
                    break

            if global_idx < 0:
                valid = False
                break

            # Verify atom is placed before current atom in Z-matrix
            ref_zidx = atom_to_zidx[global_idx]
            if ref_zidx < 0 or ref_zidx >= z_idx:
                valid = False
                break

            resolved.append(global_idx)

        if not valid or len(resolved) != 3:
            continue

        # Update Z-matrix entry with canonical refs
        canonical_indices[z_idx, 1] = resolved[0]  # dist_ref
        canonical_indices[z_idx, 2] = resolved[1]  # ang_ref
        canonical_indices[z_idx, 3] = resolved[2]  # dih_ref

        # Set dihedral type if this atom owns a named dihedral
        if atom_type < len(ATOM_DIHEDRAL_TYPE):
            dtype_idx = int(ATOM_DIHEDRAL_TYPE[atom_type])
            if dtype_idx >= 0 and all(r >= 0 for r in resolved):
                dihedral_types[z_idx] = dtype_idx

    return canonical_indices, dihedral_types


def _build_zmatrix_indices_from_topology(topology: "TopologyInfo") -> np.ndarray:
    """
    Build Z-matrix as (M, 4) int64 array from topology info.

    Internal function used by ZMatrix.from_topology().
    Processes all connected components in the bond graph, not just one per chain.
    Uses C extension when available for ~10-20x speedup.

    Args:
        topology: TopologyInfo containing structural metadata.

    Returns:
        (M, 4) array [atom_idx, dist_ref, ang_ref, dih_ref]
    """
    # Build array-based graph
    edges, n_atoms = build_bond_graph_from_topology(topology)

    if len(edges) == 0:
        return np.zeros((0, 4), dtype=np.int64)

    # Convert to CSR format
    offsets, neighbors = edges_to_csr(edges, n_atoms)

    # Find all connected components in the bond graph
    components = find_connected_components(offsets, neighbors, n_atoms)

    if len(components) == 0:
        return np.zeros((0, 4), dtype=np.int64)

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

    # Build Z-matrix using wrapper
    return build_zmatrix_from_components(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(neighbors, dtype=np.int64),
        n_atoms,
        np.array(component_starts, dtype=np.int64),
        np.array(component_sizes, dtype=np.int64),
        np.array(roots, dtype=np.int64),
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
) -> np.ndarray:
    """
    Build Z-matrix from CSR graph for multiple connected components.

    Args:
        offsets: (N+1,) CSR offsets array
        neighbors: (E,) CSR neighbor indices
        n_atoms: Total number of atoms
        component_starts: Start indices for each component
        component_sizes: Number of atoms in each component
        roots: Root atom for each component

    Returns:
        (M, 4) Z-matrix array [atom_idx, dist_ref, ang_ref, dih_ref]
    """
    # Use parallel C implementation
    zmatrix, counts = _build_zmatrix_parallel_c(
        offsets, neighbors, n_atoms,
        component_starts, component_sizes, roots
    )
    # Trim to actual entries
    total_entries = int(counts.sum())
    if total_entries < len(zmatrix):
        result = np.zeros((total_entries, 4), dtype=np.int64)
        src_offset = 0
        dst_offset = 0
        for size, count in zip(component_sizes, counts):
            count = int(count)
            result[dst_offset:dst_offset + count] = zmatrix[src_offset:src_offset + count]
            src_offset += size
            dst_offset += count
        return result
    return zmatrix[:total_entries]


def _build_zmatrix_indices(polymer: "Polymer") -> np.ndarray:
    """
    Build Z-matrix as (M, 4) int64 array.

    Internal function used by ZMatrix.from_polymer().
    Processes all connected components in the bond graph, not just one per chain.

    Args:
        polymer: Polymer structure.

    Returns:
        (M, 4) array [atom_idx, dist_ref, ang_ref, dih_ref]
    """
    # Build array-based graph
    edges, n_atoms = build_bond_graph(polymer)

    if len(edges) == 0:
        return np.zeros((0, 4), dtype=np.int64)

    # Convert to CSR format
    offsets, neighbors = edges_to_csr(edges, n_atoms)

    # Find all connected components in the bond graph
    # Returns list of (root, size) tuples
    components = find_connected_components(offsets, neighbors, n_atoms)

    if len(components) == 0:
        return np.zeros((0, 4), dtype=np.int64)

    # Prepare component info for Z-matrix construction
    component_starts = np.array([root for root, size in components], dtype=np.int64)
    component_sizes = np.array([size for root, size in components], dtype=np.int64)
    roots = component_starts.copy()

    # Build Z-matrix using C extension
    return build_zmatrix_from_components(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(neighbors, dtype=np.int64),
        n_atoms,
        component_starts,
        component_sizes,
        roots,
    )


