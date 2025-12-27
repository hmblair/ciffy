"""
Re-exports for backwards compatibility.

This module re-exports graph building and alignment functions from their
respective modules.
"""

from __future__ import annotations

__all__ = [
    # Graph building
    "build_bond_graph",
    "edges_to_csr",
    "find_connected_components",
    # Alignment
    "kabsch_rotation",
]

# Graph building (from backend.graph)
from .graph import (
    build_bond_graph,
    edges_to_csr,
    find_connected_components,
)

# Kabsch rotation for coordinate alignment
from ..operations.alignment import kabsch_rotation
