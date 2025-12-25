"""
Re-exports for backwards compatibility.

DEPRECATED: The internal coordinate system has been removed.
Use ciffy.nn.flow.PolymerFlowModel for generative modeling.

This module now only re-exports graph building and alignment functions.
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


def cartesian_to_internal(*args, **kwargs):
    """DEPRECATED: Internal coordinate system removed."""
    raise NotImplementedError(
        "cartesian_to_internal is deprecated. "
        "Internal coordinate system has been removed. "
        "Use ciffy.nn.flow.PolymerFlowModel for generative modeling."
    )


def nerf_reconstruct(*args, **kwargs):
    """DEPRECATED: Internal coordinate system removed."""
    raise NotImplementedError(
        "nerf_reconstruct is deprecated. "
        "Internal coordinate system has been removed. "
        "Use ciffy.nn.flow.PolymerFlowModel for generative modeling."
    )
