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
from .gnm import GNM
from .metrics import tm_score, lddt, rmsd, coordinate_covariance
from .cluster import cluster, cluster_representatives, ClusterResult

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
    "GNM",
    # Structure comparison metrics
    "tm_score",
    "lddt",
    "rmsd",
    # Structural clustering
    "cluster",
    "cluster_representatives",
    "ClusterResult",
]
