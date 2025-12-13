"""
Bond graph construction and spanning tree traversal for Z-matrix generation.

Builds a complete bond graph by combining intra-residue bonds from
Residue.X.bonds with inter-residue linking from LinkingDefinition.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections import deque
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer


@dataclass
class ZMatrixEntry:
    """
    Single entry in a Z-matrix.

    Each atom (except the first three) is defined by:
    - distance to a reference atom (bond length)
    - angle with two reference atoms (bond angle)
    - dihedral with three reference atoms (torsion angle)

    Attributes:
        atom_idx: Global atom index in the molecule.
        distance_ref: Atom index for bond length reference (-1 for first atom).
        angle_ref: Atom index for bond angle reference (-1 for first two atoms).
        dihedral_ref: Atom index for dihedral reference (-1 for first three atoms).
    """
    atom_idx: int
    distance_ref: int
    angle_ref: int
    dihedral_ref: int


def build_bond_graph(polymer: "Polymer") -> dict[int, set[int]]:
    """
    Build adjacency list representation of molecular bonds.

    Combines intra-residue bonds from Residue.X.bonds with inter-residue
    bonds from LinkingDefinition.

    Args:
        polymer: Polymer structure with sequence and atoms.

    Returns:
        Dict mapping atom index -> set of bonded atom indices.
    """
    from ..biochemistry import Residue, LINKING_BY_TYPE, ATOM_NAMES
    from ..types import Scale

    n_atoms = polymer.size()
    graph: dict[int, set[int]] = {i: set() for i in range(n_atoms)}

    # Get residue sizes
    res_sizes = polymer.sizes(Scale.RESIDUE)
    n_residues = len(res_sizes)

    # Build atom value -> local index mapping per residue
    atom_offset = 0
    res_atom_info: list[tuple[int, dict[int, int], dict[str, int]]] = []

    for res_idx in range(n_residues):
        res_size = int(res_sizes[res_idx])

        # Map: atom_value -> local_index and atom_name -> local_index
        value_to_local: dict[int, int] = {}
        name_to_local: dict[str, int] = {}

        for local_idx in range(res_size):
            atom_value = int(polymer.atoms[atom_offset + local_idx])
            value_to_local[atom_value] = local_idx

            # Get atom name and convert to Python naming (O3' -> O3p)
            atom_name = ATOM_NAMES.get(atom_value, "")
            py_name = atom_name.replace("'", "p").replace('"', "pp")
            name_to_local[py_name] = local_idx

        res_atom_info.append((atom_offset, value_to_local, name_to_local))
        atom_offset += res_size

    # Add intra-residue bonds
    atom_offset = 0
    for res_idx in range(n_residues):
        res_size = int(res_sizes[res_idx])
        res_type_idx = int(polymer.sequence[res_idx])

        try:
            residue = Residue(res_type_idx)
        except ValueError:
            atom_offset += res_size
            continue

        if not hasattr(residue, 'bonds') or residue.bonds is None:
            atom_offset += res_size
            continue

        # Get mapping for this residue
        _, value_to_local, _ = res_atom_info[res_idx]

        # Add bonds
        bond_indices = residue.bonds.indices()
        for bond in bond_indices:
            val1, val2 = int(bond[0]), int(bond[1])
            local1 = value_to_local.get(val1)
            local2 = value_to_local.get(val2)

            if local1 is not None and local2 is not None:
                global1 = atom_offset + local1
                global2 = atom_offset + local2
                graph[global1].add(global2)
                graph[global2].add(global1)

        atom_offset += res_size

    # Add inter-residue bonds
    _add_inter_residue_bonds(polymer, graph, res_atom_info)

    return graph


def _add_inter_residue_bonds(
    polymer: "Polymer",
    graph: dict[int, set[int]],
    res_atom_info: list[tuple[int, dict[int, int], dict[str, int]]],
) -> None:
    """Add bonds between consecutive residues based on molecule type."""
    from ..biochemistry import Residue, LINKING_BY_TYPE
    from ..types import Scale

    n_residues = len(res_atom_info)

    # Process each chain
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
                    graph[global_prev].add(global_next)
                    graph[global_next].add(global_prev)

        chain_start_res = chain_end_res


def build_spanning_tree(
    graph: dict[int, set[int]],
    root: int,
    atom_set: set[int] | None = None,
) -> tuple[list[int], dict[int, int]]:
    """
    Build spanning tree from bond graph using BFS.

    Args:
        graph: Bond adjacency list.
        root: Starting atom index.
        atom_set: Optional set of atoms to include (for chain subgraphs).

    Returns:
        Tuple of:
        - Ordered list of atom indices (BFS order)
        - Parent dict mapping atom -> parent atom in tree (-1 for root)
    """
    visited = {root}
    parent = {root: -1}
    order = [root]
    queue = deque([root])

    while queue:
        current = queue.popleft()
        neighbors = graph.get(current, set())

        # Sort for deterministic ordering
        for neighbor in sorted(neighbors):
            if neighbor in visited:
                continue
            if atom_set is not None and neighbor not in atom_set:
                continue

            visited.add(neighbor)
            parent[neighbor] = current
            order.append(neighbor)
            queue.append(neighbor)

    return order, parent


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


def build_zmatrix(polymer: "Polymer") -> list[ZMatrixEntry]:
    """
    Build complete Z-matrix for a polymer.

    Processes each chain independently with its own spanning tree.
    Returns entries in BFS (placement) order, so NERF can iterate
    through them sequentially and all reference atoms will have been
    placed before they are needed.

    The Z-matrix defines how each atom is positioned relative to
    previously placed atoms using internal coordinates:
    - First atom: placed at origin (no references)
    - Second atom: defined by distance to first (one reference)
    - Third atom: defined by distance and angle (two references)
    - All others: defined by distance, angle, and dihedral (three references)

    Args:
        polymer: Polymer structure.

    Returns:
        List of ZMatrixEntry in BFS order (placement order).
        Each entry's atom_idx gives the original atom index.
    """
    from ..types import Scale

    # Build full bond graph
    graph = build_bond_graph(polymer)

    # Result list - will be in BFS order
    result: list[ZMatrixEntry] = []

    # Get residue sizes
    res_sizes = polymer.sizes(Scale.RESIDUE)

    # Process each chain
    res_offset = 0
    atom_offset = 0

    for chain_idx, chain_len in enumerate(polymer.lengths):
        chain_len_val = int(chain_len)

        if chain_len_val == 0:
            continue

        # Calculate atom count for this chain
        chain_atom_count = sum(
            int(res_sizes[res_offset + i])
            for i in range(chain_len_val)
        )

        if chain_atom_count == 0:
            res_offset += chain_len_val
            continue

        # Get atom range for this chain
        chain_atoms = set(range(atom_offset, atom_offset + chain_atom_count))

        # Build spanning tree for this chain
        root = select_root_atom(polymer, atom_offset, chain_atom_count, res_offset)
        order, parent = build_spanning_tree(graph, root, chain_atoms)

        # Build Z-matrix entries for this chain (in BFS order)
        chain_entries = _build_chain_zmatrix_ordered(order, parent)
        result.extend(chain_entries)

        res_offset += chain_len_val
        atom_offset += chain_atom_count

    return result


def _build_chain_zmatrix_ordered(
    order: list[int],
    parent: dict[int, int],
) -> list[ZMatrixEntry]:
    """
    Build Z-matrix entries for atoms in a chain, in BFS order.

    Returns entries in the order atoms should be placed, so each
    entry's references point to previously placed atoms.
    """
    result: list[ZMatrixEntry] = []

    # Track grandparent for angle references
    grandparent: dict[int, int] = {}

    for i, atom in enumerate(order):
        p = parent[atom]

        if i == 0:
            # First atom: no references
            result.append(ZMatrixEntry(atom, -1, -1, -1))

        elif i == 1:
            # Second atom: distance to parent only
            result.append(ZMatrixEntry(atom, p, -1, -1))

        elif i == 2:
            # Third atom: distance and angle
            gp = parent[p]
            if gp == -1:
                # Parent was root, find another child of p (sibling of current atom)
                gp = _find_child_of(order[:i], parent, p, exclude=atom)
            grandparent[atom] = gp
            result.append(ZMatrixEntry(atom, p, gp, -1))

        else:
            # Full Z-matrix entry
            gp = parent[p]
            if gp == -1:
                # Parent was root, find another child of p
                gp = _find_child_of(order[:i], parent, p, exclude=atom)

            # Find great-grandparent for dihedral
            # Must be distinct from atom, p, and gp
            ggp = grandparent.get(p)
            if ggp is None or ggp in (atom, p, gp, -1):
                ggp = parent.get(gp, -1)
            if ggp in (atom, p, gp, -1):
                # Find any placed atom distinct from atom, p, gp
                ggp = _find_placed_neighbor(order[:i], parent, gp, exclude={atom, p, gp})

            grandparent[atom] = gp
            result.append(ZMatrixEntry(atom, p, gp, ggp))

    return result


def _build_chain_zmatrix(
    order: list[int],
    parent: dict[int, int],
    zmatrix: list[ZMatrixEntry | None],
) -> None:
    """Build Z-matrix entries for atoms in a chain (legacy, stores at atom index)."""
    # Track grandparent for angle references
    grandparent: dict[int, int] = {}

    for i, atom in enumerate(order):
        p = parent[atom]

        if i == 0:
            # First atom: no references
            zmatrix[atom] = ZMatrixEntry(atom, -1, -1, -1)

        elif i == 1:
            # Second atom: distance to parent only
            zmatrix[atom] = ZMatrixEntry(atom, p, -1, -1)

        elif i == 2:
            # Third atom: distance and angle
            gp = parent[p]
            if gp == -1:
                # Parent was root, use sibling if available
                gp = _find_placed_neighbor(order[:i], parent, p, exclude=atom)
            grandparent[atom] = gp
            zmatrix[atom] = ZMatrixEntry(atom, p, gp, -1)

        else:
            # Full Z-matrix entry
            gp = parent[p]
            if gp == -1:
                gp = _find_placed_neighbor(order[:i], parent, p, exclude=atom)

            # Find great-grandparent for dihedral
            ggp = grandparent.get(p)
            if ggp is None:
                ggp = parent.get(gp, -1)
            if ggp == -1:
                ggp = _find_placed_neighbor(order[:i], parent, gp, exclude={atom, p})

            grandparent[atom] = gp
            zmatrix[atom] = ZMatrixEntry(atom, p, gp, ggp)


def _find_child_of(
    placed: list[int],
    parent: dict[int, int],
    target: int,
    exclude: int | set[int] = -1,
) -> int:
    """
    Find a placed atom that is a child of target (has target as parent).

    Used to find angle_ref when grandparent is the root - we need another
    atom bonded to the distance_ref.
    """
    if isinstance(exclude, int):
        exclude = {exclude} if exclude >= 0 else set()

    # Find atoms whose parent is target (children of target)
    for atom in placed:
        if atom in exclude:
            continue
        if atom == target:
            continue
        if parent.get(atom) == target:
            return atom

    # Fallback: any placed atom not in exclude (and not target itself)
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
    """Find a placed atom that shares the same parent (sibling) or any placed atom."""
    if isinstance(exclude, int):
        exclude = {exclude} if exclude >= 0 else set()

    target_parent = parent.get(target, -1)

    # First, try to find a sibling (atom with same parent)
    for atom in placed:
        if atom in exclude:
            continue
        if atom == target:
            continue
        if parent.get(atom) == target_parent:
            return atom

    # Fallback: any placed atom not in exclude (and not target)
    for atom in reversed(placed):
        if atom not in exclude and atom != target:
            return atom

    return -1


def zmatrix_to_indices(zmatrix: list[ZMatrixEntry]) -> np.ndarray:
    """
    Build (M, 4) int64 indices array from Z-matrix entries.

    Converts a list of ZMatrixEntry objects to a contiguous NumPy array
    suitable for passing to C extensions.

    Each row contains: [atom_idx, distance_ref, angle_ref, dihedral_ref].
    Missing references are represented as -1.

    Args:
        zmatrix: List of Z-matrix entries in placement order.

    Returns:
        (M, 4) int64 NumPy array where M is len(zmatrix).
    """
    n_entries = len(zmatrix)
    indices = np.zeros((n_entries, 4), dtype=np.int64)
    for i, entry in enumerate(zmatrix):
        indices[i, 0] = entry.atom_idx
        indices[i, 1] = entry.distance_ref
        indices[i, 2] = entry.angle_ref
        indices[i, 3] = entry.dihedral_ref
    return indices
