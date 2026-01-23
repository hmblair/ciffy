"""Tests for sequence-based clustering."""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.utils import get_test_cif, DATA_DIR


# =============================================================================
# Check MMseqs2 availability
# =============================================================================

def _mmseqs_available() -> bool:
    """Check if mmseqs is available."""
    candidates = [
        shutil.which("mmseqs"),
        "/Users/hmblair/mambaforge/envs/ciffy/bin/mmseqs",
        "/Users/hmblair/mambaforge/bin/mmseqs",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return True
    return False


MMSEQS_AVAILABLE = _mmseqs_available()

requires_mmseqs = pytest.mark.skipif(
    not MMSEQS_AVAILABLE,
    reason="MMseqs2 not installed (mamba install -c bioconda mmseqs2)"
)


# =============================================================================
# Test ClusterResult
# =============================================================================

class TestClusterResult:
    """Tests for ClusterResult data class."""

    def test_cluster_result_properties(self):
        """Test basic ClusterResult properties."""
        from ciffy.operations.cluster import ClusterResult

        paths = [Path(f"test_{i}.cif") for i in range(5)]
        labels = np.array([0, 0, 1, 1, 2])
        reps = [paths[0], paths[2], paths[4]]

        result = ClusterResult(
            paths=paths,
            labels=labels,
            representatives=reps,
            threshold=0.5,
        )

        assert result.n_structures == 5
        assert result.n_clusters == 3
        assert result.threshold == 0.5

    def test_cluster_result_get_cluster(self):
        """Test getting members of a cluster."""
        from ciffy.operations.cluster import ClusterResult

        paths = [Path(f"test_{i}.cif") for i in range(5)]
        labels = np.array([0, 0, 1, 1, 2])
        reps = [paths[0], paths[2], paths[4]]

        result = ClusterResult(paths=paths, labels=labels, representatives=reps)

        cluster_0 = result.get_cluster(0)
        assert len(cluster_0) == 2
        assert paths[0] in cluster_0
        assert paths[1] in cluster_0

        cluster_2 = result.get_cluster(2)
        assert len(cluster_2) == 1
        assert paths[4] in cluster_2

    def test_cluster_result_sizes(self):
        """Test cluster_sizes method."""
        from ciffy.operations.cluster import ClusterResult

        paths = [Path(f"test_{i}.cif") for i in range(5)]
        labels = np.array([0, 0, 1, 1, 2])
        reps = [paths[0], paths[2], paths[4]]

        result = ClusterResult(paths=paths, labels=labels, representatives=reps)

        sizes = result.cluster_sizes()
        assert sizes[0] == 2
        assert sizes[1] == 2
        assert sizes[2] == 1

    def test_cluster_result_summary(self):
        """Test summary string."""
        from ciffy.operations.cluster import ClusterResult

        paths = [Path(f"test_{i}.cif") for i in range(5)]
        labels = np.array([0, 0, 1, 1, 2])
        reps = [paths[0], paths[2], paths[4]]

        result = ClusterResult(paths=paths, labels=labels, representatives=reps)

        summary = result.summary()
        assert "structures=5" in summary
        assert "clusters=3" in summary


# =============================================================================
# Test Clustering Function
# =============================================================================

@requires_mmseqs
class TestClusterFunction:
    """Tests for cluster() function."""

    def test_cluster_single_structure(self):
        """Clustering single structure returns single cluster."""
        from ciffy.operations.cluster import cluster

        paths = [Path(get_test_cif("3SKW"))]
        result = cluster(paths, threshold=0.5)

        assert result.n_structures == 1
        assert result.n_clusters == 1
        assert result.representatives[0] == paths[0]

    def test_cluster_identical_structures(self):
        """Clustering same structure twice groups them together."""
        from ciffy.operations.cluster import cluster

        # Use same structure - should cluster together
        path = Path(get_test_cif("3SKW"))
        paths = [path, path]

        result = cluster(paths, threshold=0.5)

        # Same file should be deduplicated to one
        assert result.n_structures == 1

    def test_cluster_different_structures(self):
        """Clustering different structures may separate them."""
        from ciffy.operations.cluster import cluster

        paths = [
            Path(get_test_cif("3SKW")),
            Path(get_test_cif("9GCM")),
        ]

        result = cluster(paths, threshold=0.9)

        # At high threshold, different structures should be separate
        assert result.n_structures == 2
        # Number of clusters depends on actual sequences
        assert result.n_clusters >= 1

    def test_cluster_threshold_effect(self):
        """Lower threshold produces fewer clusters."""
        from ciffy.operations.cluster import cluster

        paths = [
            Path(get_test_cif("3SKW")),
            Path(get_test_cif("9GCM")),
        ]

        result_high = cluster(paths, threshold=0.9)
        result_low = cluster(paths, threshold=0.1)

        # Lower threshold = more permissive = fewer or equal clusters
        assert result_low.n_clusters <= result_high.n_clusters

    def test_cluster_returns_valid_representatives(self):
        """Representatives are valid paths from input."""
        from ciffy.operations.cluster import cluster

        paths = [
            Path(get_test_cif("3SKW")),
            Path(get_test_cif("9GCM")),
        ]

        result = cluster(paths, threshold=0.5)

        for rep in result.representatives:
            assert rep in result.paths
            assert rep.exists()

    def test_cluster_labels_valid(self):
        """Labels are valid indices into representatives."""
        from ciffy.operations.cluster import cluster

        paths = [
            Path(get_test_cif("3SKW")),
            Path(get_test_cif("9GCM")),
        ]

        result = cluster(paths, threshold=0.5)

        assert len(result.labels) == result.n_structures
        assert result.labels.min() >= 0
        assert result.labels.max() < result.n_clusters

    def test_cluster_empty_raises(self):
        """Clustering empty list raises ValueError."""
        from ciffy.operations.cluster import cluster

        with pytest.raises(ValueError, match="No valid"):
            cluster([], threshold=0.5)

    def test_cluster_missing_files_warns(self):
        """Clustering with missing files warns and continues."""
        from ciffy.operations.cluster import cluster

        paths = [
            Path(get_test_cif("3SKW")),
            Path("/nonexistent/file.cif"),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cluster(paths, threshold=0.5)

            # Should warn about missing file
            assert len(w) == 1
            assert "missing" in str(w[0].message).lower()

        # Should still cluster the valid file
        assert result.n_structures == 1


@requires_mmseqs
class TestClusterRepresentatives:
    """Tests for cluster_representatives() convenience function."""

    def test_returns_list_of_paths(self):
        """Returns list of Path objects."""
        from ciffy.operations.cluster import cluster_representatives

        paths = [Path(get_test_cif("3SKW"))]
        reps = cluster_representatives(paths, threshold=0.5)

        assert isinstance(reps, list)
        assert all(isinstance(p, Path) for p in reps)

    def test_representatives_subset(self):
        """Representatives are subset of input paths."""
        from ciffy.operations.cluster import cluster_representatives

        paths = [
            Path(get_test_cif("3SKW")),
            Path(get_test_cif("9GCM")),
        ]

        reps = cluster_representatives(paths, threshold=0.5)

        assert len(reps) <= len(paths)
        # All reps should be from input (after resolving to same paths)
        path_set = set(p.resolve() for p in paths)
        for rep in reps:
            assert rep.resolve() in path_set or rep in paths


# =============================================================================
# Test Edge Cases
# =============================================================================

@requires_mmseqs
class TestClusterEdgeCases:
    """Edge case tests for clustering."""

    def test_all_structures_cluster_together(self):
        """When all structures cluster together at low threshold."""
        from ciffy.operations.cluster import cluster

        # Same structure = definitely clusters together
        path = Path(get_test_cif("3SKW"))

        result = cluster([path], threshold=0.1)

        assert result.n_clusters == 1

    def test_very_high_threshold(self):
        """Very high threshold may put each structure in own cluster."""
        from ciffy.operations.cluster import cluster

        paths = [
            Path(get_test_cif("3SKW")),
            Path(get_test_cif("9GCM")),
        ]

        result = cluster(paths, threshold=0.99)

        # At 99% identity, different structures should be separate
        # (unless they happen to be nearly identical)
        assert result.n_clusters >= 1

    def test_coverage_parameter(self):
        """Coverage parameter affects clustering."""
        from ciffy.operations.cluster import cluster

        paths = [Path(get_test_cif("3SKW"))]

        # Should work with different coverage values
        result_low = cluster(paths, threshold=0.5, coverage=0.5)
        result_high = cluster(paths, threshold=0.5, coverage=0.9)

        # Both should succeed
        assert result_low.n_structures == 1
        assert result_high.n_structures == 1

    def test_threads_parameter(self):
        """Threads parameter doesn't affect results."""
        from ciffy.operations.cluster import cluster

        paths = [
            Path(get_test_cif("3SKW")),
            Path(get_test_cif("9GCM")),
        ]

        result_1 = cluster(paths, threshold=0.5, threads=1)
        result_4 = cluster(paths, threshold=0.5, threads=4)

        # Same results regardless of thread count
        assert result_1.n_clusters == result_4.n_clusters


# =============================================================================
# Test Imports
# =============================================================================

class TestImports:
    """Test that clustering is properly exported."""

    def test_import_from_operations(self):
        """Can import from ciffy.operations."""
        from ciffy.operations import cluster, cluster_representatives, ClusterResult

        assert callable(cluster)
        assert callable(cluster_representatives)
        assert ClusterResult is not None
