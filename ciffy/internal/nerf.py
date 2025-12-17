"""
NERF (Natural Extension Reference Frame) algorithm for coordinate reconstruction.

.. note::
    This module re-exports ``nerf_reconstruct`` from ``ciffy.backend.graph``
    for backwards compatibility. New code should import directly from
    ``ciffy.backend.dispatch`` (without n_atoms inference) or
    ``ciffy.backend.graph`` (with n_atoms inference).
"""

from __future__ import annotations

# Re-export from backend for backwards compatibility
from ..backend.graph import nerf_reconstruct

__all__ = ["nerf_reconstruct"]
