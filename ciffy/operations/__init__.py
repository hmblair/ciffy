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
from .gnm import GNM, contact_map, inverse_square_map
from .metrics import tm_score, lddt, rmsd, coordinate_covariance, rg, clashes, sasa
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
    "contact_map",
    "inverse_square_map",
    # Structure comparison metrics
    "tm_score",
    "lddt",
    "rmsd",
    "rg",
    "clashes",
    # Structural clustering
    "cluster",
    "cluster_representatives",
    "ClusterResult",
]
