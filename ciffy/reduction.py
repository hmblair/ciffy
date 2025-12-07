"""
Backward-compatible re-exports from ciffy.operations.reduction.

New code should import from ciffy.operations.reduction directly.
"""

from .operations.reduction import (
    Reduction,
    REDUCTIONS,
    ReductionResult as _Reduction,
    scatter_collate as t_scatter_collate,
)

__all__ = [
    "Reduction",
    "REDUCTIONS",
    "_Reduction",
    "t_scatter_collate",
]
