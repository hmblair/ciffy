"""
Operations on polymer structures.

Pure functions for geometry, alignment, analysis, and frame operations.
"""

from .reduction import Reduction, REDUCTIONS
from .alignment import (
    kabsch_rotation,
    kabsch_align,
    align,
    intersect,
)
from .gnm import GNM, contact_map, inverse_square_map
from .metrics import tm_score, lddt, rmsd, coordinate_covariance, rg, clashes, sasa
from .cluster import cluster, cluster_representatives, ClusterResult
from .packing import pack, unpack
from .chain import join

# Geometry operations (moved from Polymer)
from .geometry import (
    pairwise_distances,
    knn,
    adjacency,
    bonded_distances,
    moment,
    pca,
)

# Frame operations
from .frames import (
    Transforms,
    decompose,
    compose,
    gather,
    sort_atoms,
)

__all__ = [
    # Reduction
    "Reduction",
    "REDUCTIONS",
    # Alignment (Kabsch)
    "coordinate_covariance",
    "kabsch_rotation",
    "kabsch_align",
    "align",
    "intersect",
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
    "sasa",
    # Structural clustering
    "cluster",
    "cluster_representatives",
    "ClusterResult",
    # Packing
    "pack",
    "unpack",
    # Chain operations
    "join",
    # Geometry operations
    "pairwise_distances",
    "knn",
    "adjacency",
    "bonded_distances",
    "moment",
    "pca",
    # Frame operations
    "Transforms",
    "decompose",
    "compose",
    "gather",
    "sort_atoms",
]
