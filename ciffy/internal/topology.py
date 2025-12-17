"""
Topology information for coordinate operations.

.. note::
    This module re-exports ``TopologyInfo`` and ``ConnectedComponents`` from
    ``ciffy.backend.graph`` for backwards compatibility. New code should import
    directly from ``ciffy.backend.dispatch``.
"""

from __future__ import annotations

# Re-export from backend for backwards compatibility
from ..backend.graph import TopologyInfo, ConnectedComponents

__all__ = ["TopologyInfo", "ConnectedComponents"]
