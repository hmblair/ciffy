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

from ..backend import Array, is_torch, to_numpy, to_torch

# Try to import C extension
try:
    from .._c import _build_bond_graph as _build_bond_graph_c
    from .._c import _edges_to_csr as _edges_to_csr_c
    from .._c import _build_zmatrix_from_csr as _build_zmatrix_from_csr_c
    from .._c import _build_zmatrix_parallel as _build_zmatrix_parallel_c
    from .._c import _find_connected_components as _find_connected_components_c
    _HAS_C_EXTENSION = True
except ImportError:
    _HAS_C_EXTENSION = False
    _edges_to_csr_c = None
    _build_zmatrix_from_csr_c = None
    _build_zmatrix_parallel_c = None
    _find_connected_components_c = None


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
        >>> zmatrix = ZMatrix.from_polymer(polymer)
        >>> print(len(zmatrix))  # Number of atoms in Z-matrix
        >>> print(zmatrix.atom_indices)  # Column 0
        >>> print(zmatrix[0])  # First row as array
    """

    __slots__ = ('_indices',)

    def __init__(self, indices: Array) -> None:
        """
        Initialize Z-matrix from indices array.

        Args:
            indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
        """
        self._indices = indices

    @classmethod
    def from_polymer(cls, polymer: "Polymer") -> "ZMatrix":
        """
        Build Z-matrix from polymer using BFS traversal.

        Processes each chain independently with its own spanning tree.
        Returns entries in BFS order so references always point to
        earlier (already placed) atoms.

        Args:
            polymer: Polymer structure.

        Returns:
            ZMatrix with entries in placement order.
        """
        indices = _build_zmatrix_indices(polymer)
        return cls(indices)

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

    def __len__(self) -> int:
        """Number of entries in Z-matrix."""
        return len(self._indices)

    def __getitem__(self, idx) -> Array:
        """Index into the Z-matrix array."""
        return self._indices[idx]

    def validate(self) -> None:
        """
        Validate Z-matrix structure.

        Checks that:
        - All reference atoms are either -1 or point to earlier atoms
        - Reference progression is correct (dist before angle before dihedral)

        Raises:
            ValueError: If validation fails.
        """
        placed = set()
        for i in range(len(self._indices)):
            atom_idx = int(self._indices[i, 0])
            dist_ref = int(self._indices[i, 1])
            ang_ref = int(self._indices[i, 2])
            dih_ref = int(self._indices[i, 3])

            # Check distance reference
            if dist_ref >= 0 and dist_ref not in placed:
                raise ValueError(
                    f"Entry {i}: distance_ref {dist_ref} not yet placed"
                )

            # Check angle reference
            if ang_ref >= 0 and ang_ref not in placed:
                raise ValueError(
                    f"Entry {i}: angle_ref {ang_ref} not yet placed"
                )

            # Check dihedral reference
            if dih_ref >= 0 and dih_ref not in placed:
                raise ValueError(
                    f"Entry {i}: dihedral_ref {dih_ref} not yet placed"
                )

            # Check progression: can't have angle without distance, etc.
            if ang_ref >= 0 and dist_ref < 0:
                raise ValueError(
                    f"Entry {i}: has angle_ref but no distance_ref"
                )
            if dih_ref >= 0 and ang_ref < 0:
                raise ValueError(
                    f"Entry {i}: has dihedral_ref but no angle_ref"
                )

            placed.add(atom_idx)

    def numpy(self) -> "ZMatrix":
        """Convert indices to NumPy array."""
        return ZMatrix(to_numpy(self._indices))

    def torch(self) -> "ZMatrix":
        """Convert indices to PyTorch tensor."""
        return ZMatrix(to_torch(self._indices))

    def to(self, device: str) -> "ZMatrix":
        """Move to specified device (PyTorch only)."""
        if not is_torch(self._indices):
            raise RuntimeError("to() requires PyTorch backend")
        return ZMatrix(self._indices.to(device))

    def __repr__(self) -> str:
        backend = "torch" if is_torch(self._indices) else "numpy"
        return f"ZMatrix({len(self)} entries, {backend})"


# =============================================================================
# BOND GRAPH CONSTRUCTION
# =============================================================================


def build_bond_graph(polymer: "Polymer") -> tuple[np.ndarray, int]:
    """
    Build edge list representation of molecular bonds.

    Constructs bonds as an (E, 2) array for array-based processing.
    Combines intra-residue bonds from Residue.bond_indices and inter-residue
    bonds from LINKING_BY_TYPE.

    Uses C extension when available for ~10-20x speedup on large structures.

    Args:
        polymer: Polymer structure with sequence and atoms.

    Returns:
        Tuple of:
            edges: (E, 2) int64 array of [atom_i, atom_j] pairs (symmetric)
            n_atoms: Total number of atoms
    """
    from ..types import Scale

    n_atoms = polymer.size()

    # Try C extension first
    if _HAS_C_EXTENSION:
        try:
            res_sizes = polymer.sizes(Scale.RESIDUE)
            edges = _build_bond_graph_c(
                np.ascontiguousarray(to_numpy(polymer.atoms), dtype=np.int32),
                np.ascontiguousarray(to_numpy(polymer.sequence), dtype=np.int32),
                np.ascontiguousarray(to_numpy(res_sizes), dtype=np.int32),
                np.ascontiguousarray(to_numpy(polymer.lengths), dtype=np.int32),
            )
            return edges, n_atoms
        except Exception:
            # Fall through to Python implementation
            pass

    # Python fallback
    return _build_bond_graph_python(polymer)


def _build_bond_graph_python(polymer: "Polymer") -> tuple[np.ndarray, int]:
    """
    Python implementation of bond graph building.

    Fallback when C extension is not available.
    """
    from ..biochemistry import Residue, LINKING_BY_TYPE, ATOM_NAMES
    from ..types import Scale

    intra_bonds = []
    atom_offset = 0
    res_sizes = polymer.sizes(Scale.RESIDUE)
    n_residues = len(res_sizes)

    # Build per-residue atom value mappings
    res_atom_info = []  # (offset, value_to_local, name_to_local)

    for res_idx in range(n_residues):
        res_size = int(res_sizes[res_idx])

        # Map: atom_value -> local_index and atom_name -> local_index
        value_to_local: dict[int, int] = {}
        name_to_local: dict[str, int] = {}

        for local_idx in range(res_size):
            atom_value = int(polymer.atoms[atom_offset + local_idx])
            value_to_local[atom_value] = local_idx

            # Get atom name from ATOM_NAMES
            atom_name = ATOM_NAMES.get(atom_value, "")
            py_name = atom_name.replace("'", "p").replace('"', "pp")
            name_to_local[py_name] = local_idx

        res_atom_info.append((atom_offset, value_to_local, name_to_local))
        atom_offset += res_size

    # Step 1: Collect intra-residue bonds
    for res_idx in range(n_residues):
        res_type_idx = int(polymer.sequence[res_idx])
        res_offset, value_to_local, _ = res_atom_info[res_idx]

        try:
            residue = Residue(res_type_idx)
            bond_idx = residue.bond_indices  # (M, 2) array from Residue

            if bond_idx is not None:
                # Translate from global atom values to global indices
                global_bonds = []
                for bond in bond_idx:
                    local1 = value_to_local.get(int(bond[0]))
                    local2 = value_to_local.get(int(bond[1]))
                    if local1 is not None and local2 is not None:
                        global_bonds.append([
                            res_offset + local1,
                            res_offset + local2
                        ])

                if global_bonds:
                    intra_bonds.append(np.array(global_bonds, dtype=np.int64))

        except ValueError:
            pass

    # Step 2: Build inter-residue bonds
    inter_bonds_list = []

    chain_start_res = 0
    for chain_len in polymer.lengths:
        chain_len_val = int(chain_len)
        chain_end_res = chain_start_res + chain_len_val

        if chain_len_val == 0:
            continue

        # Determine linking type from first residue
        first_res_type = int(polymer.sequence[chain_start_res])
        try:
            mol_type = Residue(first_res_type).molecule_type
            link_def = LINKING_BY_TYPE.get(mol_type)
        except ValueError:
            link_def = None

        if link_def is not None:
            # Add inter-residue bonds within this chain
            for res_idx in range(chain_start_res, chain_end_res - 1):
                curr_offset, _, curr_names = res_atom_info[res_idx]
                next_offset, _, next_names = res_atom_info[res_idx + 1]

                prev_local = curr_names.get(link_def.prev_atom)
                next_local = next_names.get(link_def.next_atom)

                if prev_local is not None and next_local is not None:
                    global_prev = curr_offset + prev_local
                    global_next = next_offset + next_local
                    inter_bonds_list.append([global_prev, global_next])

        chain_start_res = chain_end_res

    # Step 3: Concatenate all bonds
    all_bonds_list = intra_bonds
    if inter_bonds_list:
        all_bonds_list.append(np.array(inter_bonds_list, dtype=np.int64))

    if not all_bonds_list:
        return np.zeros((0, 2), dtype=np.int64), polymer.size()

    all_bonds = np.vstack(all_bonds_list)

    # Step 4: Symmetrize (add reverse edges for undirected graph)
    edges = np.vstack([all_bonds, all_bonds[:, [1, 0]]])

    return edges, polymer.size()


def edges_to_csr_neighbors(edges: Array, n_atoms: int) -> tuple[Array, Array]:
    """
    Convert edge list to CSR-style neighbor lists using counting sort.

    Uses O(n + n_atoms) counting sort instead of O(n log n) comparison sort
    for better performance on large graphs.

    Args:
        edges: (E, 2) array of directed edges
        n_atoms: Total number of atoms

    Returns:
        Tuple of:
            offsets: (n_atoms+1,) cumulative neighbor counts
            neighbors: (E,) flattened neighbor indices, grouped by source
    """
    n_edges = len(edges)

    if is_torch(edges):
        import torch
        # Count edges per source
        sources = edges[:, 0]
        counts = torch.bincount(sources, minlength=n_atoms)
        offsets = torch.cat([torch.tensor([0], device=edges.device, dtype=torch.int64),
                            counts.cumsum(0)])

        # Scatter destinations to output positions (counting sort)
        neighbors = torch.zeros(n_edges, dtype=edges.dtype, device=edges.device)
        write_pos = offsets[:-1].clone()
        for i in range(n_edges):
            src = int(edges[i, 0])
            neighbors[write_pos[src]] = edges[i, 1]
            write_pos[src] += 1
    else:
        # Count edges per source
        sources = edges[:, 0]
        counts = np.bincount(sources, minlength=n_atoms)
        offsets = np.concatenate([[0], counts.cumsum()]).astype(np.int64)

        # Scatter destinations to output positions (counting sort)
        neighbors = np.zeros(n_edges, dtype=edges.dtype)
        write_pos = offsets[:-1].copy()
        for i in range(n_edges):
            src = edges[i, 0]
            neighbors[write_pos[src]] = edges[i, 1]
            write_pos[src] += 1

    return offsets, neighbors


def find_connected_components(offsets: np.ndarray, neighbors: np.ndarray, n_atoms: int) -> list[list[int]]:
    """
    Find all connected components in a CSR-format graph.

    Args:
        offsets: (N+1,) CSR offsets array
        neighbors: (E,) CSR neighbor indices
        n_atoms: Total number of atoms

    Returns:
        List of components, where each component is a list of atom indices.
    """
    # Try C extension first
    if _HAS_C_EXTENSION and _find_connected_components_c is not None:
        try:
            roots, sizes, n_components = _find_connected_components_c(
                np.ascontiguousarray(offsets, dtype=np.int64),
                np.ascontiguousarray(neighbors, dtype=np.int64),
                n_atoms
            )
            # Convert to list of component atom lists
            # C returns (roots, sizes), we need to rebuild full component lists
            # But for Z-matrix building, we only need roots and sizes
            # Return a simplified format that _build_zmatrix_indices can use
            return [(int(roots[i]), int(sizes[i])) for i in range(n_components)]
        except Exception:
            pass  # Fall through to Python

    # Python fallback
    visited = np.zeros(n_atoms, dtype=bool)
    components = []

    for start in range(n_atoms):
        if visited[start]:
            continue

        # Check if this atom has any neighbors
        n_neighbors = offsets[start + 1] - offsets[start]
        if n_neighbors == 0:
            # Isolated atom - skip (will be handled as orphan)
            visited[start] = True
            continue

        # BFS to find component
        component = []
        queue = [start]
        visited[start] = True

        while queue:
            node = queue.pop(0)
            component.append(node)

            # Get neighbors from CSR
            for j in range(offsets[node], offsets[node + 1]):
                neighbor = neighbors[j]
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(component)

    return components


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


def build_spanning_tree(
    offsets: Array,
    neighbors: Array,
    root: int,
    atom_mask: Array | None = None
) -> tuple[Array, Array]:
    """
    BFS spanning tree using CSR-format graph.

    Args:
        offsets: (N+1,) CSR offsets
        neighbors: (E,) CSR neighbor indices
        root: Root atom index
        atom_mask: Optional (N,) boolean mask of valid atoms

    Returns:
        Tuple of:
            order: (M,) array of atom indices in BFS order
            parent: (N,) array mapping atom -> parent (-1 for root/unvisited)
    """
    n_atoms = len(offsets) - 1

    # Convert to numpy for CPU-based BFS
    if is_torch(offsets):
        offsets_np = offsets.cpu().numpy()
        neighbors_np = neighbors.cpu().numpy()
        if atom_mask is not None:
            atom_mask_np = atom_mask.cpu().numpy()
        else:
            atom_mask_np = None
        original_device = offsets.device
    else:
        offsets_np = offsets
        neighbors_np = neighbors
        atom_mask_np = atom_mask
        original_device = None

    # Initialize
    parent = np.full(n_atoms, -1, dtype=np.int64)
    visited = np.zeros(n_atoms, dtype=bool)
    order = []

    # BFS queue
    queue = [root]
    visited[root] = True

    while queue:
        current = queue.pop(0)
        order.append(current)

        # Get neighbors
        start = offsets_np[current]
        end = offsets_np[current + 1]
        current_neighbors = neighbors_np[start:end]

        # Sort for determinism
        current_neighbors = np.sort(current_neighbors)

        for neighbor in current_neighbors:
            if atom_mask_np is not None and not atom_mask_np[neighbor]:
                continue

            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = current
                queue.append(neighbor)

    # Convert back to original backend
    if original_device is not None:
        import torch
        return (
            torch.tensor(order, dtype=torch.int64, device=original_device),
            torch.from_numpy(parent).to(original_device)
        )
    else:
        return np.array(order, dtype=np.int64), parent


# =============================================================================
# Z-MATRIX CONSTRUCTION
# =============================================================================


def _build_zmatrix_indices(polymer: "Polymer") -> np.ndarray:
    """
    Build Z-matrix as (M, 4) int64 array.

    Internal function used by ZMatrix.from_polymer().
    Processes all connected components in the bond graph, not just one per chain.
    Uses C extension when available for ~10-20x speedup.

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
    if _HAS_C_EXTENSION and _edges_to_csr_c is not None:
        try:
            offsets, neighbors = _edges_to_csr_c(
                np.ascontiguousarray(edges, dtype=np.int64),
                n_atoms
            )
        except Exception:
            offsets, neighbors = edges_to_csr_neighbors(edges, n_atoms)
    else:
        offsets, neighbors = edges_to_csr_neighbors(edges, n_atoms)

    # Ensure numpy arrays for component finding
    if is_torch(offsets):
        offsets_np = offsets.cpu().numpy()
        neighbors_np = neighbors.cpu().numpy()
    else:
        offsets_np = np.asarray(offsets)
        neighbors_np = np.asarray(neighbors)

    # Find all connected components in the bond graph
    components = find_connected_components(offsets_np, neighbors_np, n_atoms)

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

    # Try C fast-path
    if _HAS_C_EXTENSION and _build_zmatrix_parallel_c is not None:
        try:
            # Use parallel C function
            zmatrix, counts = _build_zmatrix_parallel_c(
                offsets_np,
                neighbors_np,
                n_atoms,
                np.array(component_starts, dtype=np.int64),
                np.array(component_sizes, dtype=np.int64),
                np.array(roots, dtype=np.int64)
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
        except Exception:
            pass

    # Try sequential C fallback
    if _HAS_C_EXTENSION and _build_zmatrix_from_csr_c is not None:
        try:
            all_entries = []
            for comp_start, comp_size, root in zip(component_starts, component_sizes, roots):
                comp_zmatrix = _build_zmatrix_from_csr_c(
                    offsets_np, neighbors_np, n_atoms,
                    comp_start, comp_size, root
                )
                all_entries.append(comp_zmatrix)
            return np.vstack(all_entries)
        except Exception:
            pass

    # Python fallback - process each connected component
    # Use already-computed roots and sizes (works for both C and Python formats)
    all_entries = []

    for root, comp_size in zip(roots, component_sizes):
        # BFS from root (no atom mask needed - component is already connected)
        order, parent = build_spanning_tree(offsets_np, neighbors_np, root, atom_mask=None)

        # Take only the first comp_size entries (the component we care about)
        order_list = order.tolist()[:comp_size]

        parent_dict = {i: int(parent[i]) for i in range(len(parent)) if parent[i] >= 0}
        parent_dict[root] = -1

        # Build Z-matrix entries
        component_entries = _build_chain_zmatrix(order_list, parent_dict)
        all_entries.extend(component_entries)

    # Convert to (M, 4) array
    if len(all_entries) == 0:
        return np.zeros((0, 4), dtype=np.int64)

    return np.array(all_entries, dtype=np.int64)


def _build_chain_zmatrix(
    order: list[int],
    parent: dict[int, int],
) -> list[list[int]]:
    """
    Build Z-matrix entries for atoms in a chain.

    Returns entries in BFS order so references point to earlier atoms.

    Args:
        order: Atom indices in BFS traversal order.
        parent: Mapping from atom -> parent in spanning tree.

    Returns:
        List of [atom_idx, dist_ref, ang_ref, dih_ref] entries.
    """
    result: list[list[int]] = []
    grandparent: dict[int, int] = {}

    for i, atom in enumerate(order):
        p = parent[atom]

        if i == 0:
            # First atom: no references
            result.append([atom, -1, -1, -1])

        elif i == 1:
            # Second atom: distance to parent only
            result.append([atom, p, -1, -1])

        elif i == 2:
            # Third atom: distance and angle
            gp = parent[p]
            if gp == -1:
                # Parent was root, find another child of p
                gp = _find_child_of(order[:i], parent, p, exclude=atom)
            grandparent[atom] = gp
            result.append([atom, p, gp, -1])

        else:
            # Full Z-matrix entry
            gp = parent[p]
            if gp == -1:
                gp = _find_child_of(order[:i], parent, p, exclude=atom)

            # Find great-grandparent for dihedral
            ggp = grandparent.get(p)
            if ggp is None or ggp in (atom, p, gp, -1):
                ggp = parent.get(gp, -1)
            if ggp in (atom, p, gp, -1):
                ggp = _find_placed_neighbor(order[:i], parent, gp, exclude={atom, p, gp})

            grandparent[atom] = gp
            result.append([atom, p, gp, ggp])

    return result


def _find_child_of(
    placed: list[int],
    parent: dict[int, int],
    target: int,
    exclude: int | set[int] = -1,
) -> int:
    """
    Find a placed atom that is a child of target (has target as parent).
    """
    if isinstance(exclude, int):
        exclude = {exclude} if exclude >= 0 else set()

    for atom in placed:
        if atom in exclude or atom == target:
            continue
        if parent.get(atom) == target:
            return atom

    # Fallback: any placed atom not in exclude
    for atom in reversed(placed):
        if atom not in exclude and atom != target:
            return atom

    return -1


def _find_placed_neighbor(
    placed: list[int],
    parent: dict[int, int],
    target: int,
    exclude: int | set[int] = -1,
) -> int:
    """
    Find a placed atom that shares the same parent (sibling) or any placed atom.
    """
    if isinstance(exclude, int):
        exclude = {exclude} if exclude >= 0 else set()

    target_parent = parent.get(target, -1)

    # First, try to find a sibling
    for atom in placed:
        if atom in exclude or atom == target:
            continue
        if parent.get(atom) == target_parent:
            return atom

    # Fallback: any placed atom not in exclude
    for atom in reversed(placed):
        if atom not in exclude and atom != target:
            return atom

    return -1
