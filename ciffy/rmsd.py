"""
Backward-compatible re-exports from ciffy.operations.alignment.

New code should import from ciffy.operations.alignment directly.
"""

from .operations.alignment import (
    coordinate_covariance as _coordinate_covariance,
    kabsch_distance as _kabsch_distance,
)

__all__ = [
    "_coordinate_covariance",
    "_kabsch_distance",
]
