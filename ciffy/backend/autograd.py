"""
DEPRECATED: Internal coordinate autograd functions removed.

The internal coordinate system has been deprecated in favor of
ciffy.nn.flow.PolymerFlowModel for generative modeling.
"""

from __future__ import annotations

__all__ = [
    "cartesian_to_internal",
    "nerf_reconstruct",
]


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
