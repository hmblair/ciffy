"""Type stubs for ciffy C extension module."""

from typing import Any
from numpy import ndarray

# I/O functions
def _load(
    path: str,
    load_descriptions: bool = False,
    metadata_only: bool = False,
    molecule_types: list[int] | None = None,
    chains: list[str] | None = None,
) -> dict[str, Any]: ...

def _save(
    path: str,
    coordinates: ndarray,
    atoms: ndarray,
    elements: ndarray,
    residues: ndarray,
    atoms_per_res: ndarray,
    chain_names: list[str],
    strand_names: list[str],
    res_per_chain: ndarray,
    pdb_id: str,
    molecule_types: ndarray,
    polymer_count: int,
    bfactors: ndarray | None = None,
) -> None: ...

# Bond graph functions
def _build_bond_graph(
    atoms: ndarray,
    sequence: ndarray,
    res_sizes: ndarray,
    lengths: ndarray,
) -> ndarray: ...

def _edges_to_csr(
    edges: ndarray,
    n_atoms: int,
) -> tuple[ndarray, ndarray]: ...

def _find_connected_components(
    offsets: ndarray,
    neighbors: ndarray,
    n_atoms: int,
) -> tuple[ndarray, ndarray, int]: ...

# Internal coordinate functions
def _cartesian_to_internal(
    coordinates: ndarray,
    parents: ndarray,
    grandparents: ndarray,
    great_grandparents: ndarray,
) -> tuple[ndarray, ndarray, ndarray]: ...

def _cartesian_to_internal_backward(
    grad_distances: ndarray,
    grad_angles: ndarray,
    grad_dihedrals: ndarray,
    coordinates: ndarray,
    parents: ndarray,
    grandparents: ndarray,
    great_grandparents: ndarray,
) -> ndarray: ...

def _cartesian_to_internal_parent(
    coordinates: ndarray,
    parents: ndarray,
) -> tuple[ndarray, ndarray, ndarray]: ...

# NeRF reconstruction functions
def _nerf_place_atom(
    parent_coords: ndarray,
    grandparent_coords: ndarray,
    great_grandparent_coords: ndarray,
    distance: float,
    angle: float,
    dihedral: float,
) -> ndarray: ...

def _nerf_reconstruct_leveled_anchored(
    distances: ndarray,
    angles: ndarray,
    dihedrals: ndarray,
    parents: ndarray,
    grandparents: ndarray,
    great_grandparents: ndarray,
    levels: ndarray,
    anchors: ndarray,
) -> ndarray: ...

def _nerf_reconstruct_backward_leveled_anchored(
    grad_output: ndarray,
    coordinates: ndarray,
    distances: ndarray,
    angles: ndarray,
    dihedrals: ndarray,
    parents: ndarray,
    grandparents: ndarray,
    great_grandparents: ndarray,
    levels: ndarray,
    anchors: ndarray,
) -> tuple[ndarray, ndarray, ndarray]: ...

def _nerf_reconstruct_parent(
    distances: ndarray,
    angles: ndarray,
    dihedrals: ndarray,
    parents: ndarray,
) -> ndarray: ...

# Profiling (optional, only present when built with CIFFY_PROFILE=1)
def _get_profile() -> dict[str, float]: ...
