"""
Operations on polymer structures.

Pure functions for geometry, selection, reduction, and alignment operations.
"""

from .reduction import Reduction, REDUCTIONS
from .alignment import (
    kabsch_rotation,
    kabsch_align,
    align,
    intersect,
)
from .extract import extract
from .gnm import graph_laplacian, gnm_correlations, gnm_variances
from .metrics import tm_score, lddt, rmsd, coordinate_covariance

# Legacy alias
kabsch_distance = rmsd

__all__ = [
    "Reduction",
    "REDUCTIONS",
    "coordinate_covariance",
    "kabsch_distance",
    "kabsch_rotation",
    "kabsch_align",
    "align",
    "intersect",
    "extract",
    # GNM utilities
    "graph_laplacian",
    "gnm_correlations",
    "gnm_variances",
    # Structure comparison metrics
    "tm_score",
    "lddt",
    "rmsd",
]
