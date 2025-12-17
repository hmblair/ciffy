"""
Bond graph construction and Z-matrix generation.

.. note::
    This module re-exports functions from ``ciffy.backend.graph`` for backwards
    compatibility. New code should import directly from ``ciffy.backend.dispatch``.
"""

from __future__ import annotations

# Re-export from backend for backwards compatibility
from ..backend.graph import (
    ZMatrix,
    build_bond_graph,
    build_bond_graph_from_topology,
    build_bond_graph_csr,
    edges_to_csr,
    find_connected_components,
    build_zmatrix_from_components,
)

__all__ = [
    "ZMatrix",
    "build_bond_graph",
    "build_bond_graph_from_topology",
    "build_bond_graph_csr",
    "edges_to_csr",
    "find_connected_components",
    "build_zmatrix_from_components",
]
