"""
Re-exports for backwards compatibility.

This module re-exports graph building and alignment functions from their
respective modules.
"""

from __future__ import annotations

__all__ = [
    # Graph building
    "build_bond_graph",
    "build_bond_graph_csr",
    "build_bond_graph_from_topology",
    "find_connected_components",
    # Data structures
    "TopologyInfo",
    # Alignment
    "kabsch_rotation",
]

# Graph building and data structures (from backend.graph)
from .graph import (
    TopologyInfo,
    build_bond_graph,
    build_bond_graph_csr,
    build_bond_graph_from_topology,
    find_connected_components,
)

# Kabsch rotation for coordinate alignment
from ..operations.alignment import kabsch_rotation
