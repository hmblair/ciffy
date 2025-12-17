"""
Z-matrix representation and internal coordinate computation.

.. note::
    This module re-exports ``ZMatrix`` and ``cartesian_to_internal`` from
    ``ciffy.backend`` for backwards compatibility. New code should import
    directly from ``ciffy.backend.dispatch``.
"""

from __future__ import annotations

# Re-export from backend for backwards compatibility
from ..backend.graph import ZMatrix
from ..backend.dispatch import cartesian_to_internal

__all__ = ["ZMatrix", "cartesian_to_internal"]
