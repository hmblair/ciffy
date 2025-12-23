"""Tests for GNM (Gaussian Network Model) utilities.

Tests cover:
- graph_laplacian: Graph Laplacian computation
- gnm_correlations: GNM correlations
- gnm_variances: GNM position variances
"""
from __future__ import annotations

import pytest
import torch

from ciffy.operations.gnm import graph_laplacian, gnm_correlations, gnm_variances


# ============================================================================
# GRAPH LAPLACIAN TESTS
# ============================================================================


class TestGraphLaplacian:
    """Tests for graph_laplacian function."""

    def test_basic_example(self):
        """Test Laplacian computation for simple triangle graph."""
        adj = torch.tensor([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.]
        ])

        L = graph_laplacian(adj)

        expected = torch.tensor([
            [2., -1., -1.],
            [-1., 2., -1.],
            [-1., -1., 2.]
        ])
        assert torch.allclose(L, expected)

    def test_row_sum_zero(self):
        """Test that each row of Laplacian sums to zero."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)
        row_sums = L.sum(dim=1)

        assert torch.allclose(row_sums, torch.zeros(10), atol=1e-5)

    def test_symmetric(self):
        """Test that Laplacian is symmetric for symmetric adjacency."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)

        assert torch.allclose(L, L.T)

    def test_positive_semidefinite(self):
        """Test that Laplacian is positive semi-definite."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)
        eigenvalues = torch.linalg.eigvalsh(L)

        assert (eigenvalues >= -1e-5).all()  # Relaxed tolerance for numerical precision

    def test_smallest_eigenvalue_zero(self):
        """Test that smallest eigenvalue is zero for connected graph."""
        adj = torch.ones(5, 5) - torch.eye(5)

        L = graph_laplacian(adj)
        eigenvalues = torch.linalg.eigvalsh(L)

        assert torch.isclose(eigenvalues[0], torch.tensor(0.), atol=1e-6)

    def test_diagonal_is_degree(self):
        """Test that diagonal elements are node degrees."""
        adj = torch.tensor([
            [0., 1., 1., 0.],
            [1., 0., 1., 1.],
            [1., 1., 0., 0.],
            [0., 1., 0., 0.]
        ])

        L = graph_laplacian(adj)
        degrees = adj.sum(dim=1)

        assert torch.allclose(L.diagonal(), degrees)

    def test_off_diagonal_negated(self):
        """Test that off-diagonal elements are negated adjacency."""
        adj = torch.rand(5, 5)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)

        mask = ~torch.eye(5, dtype=torch.bool)
        assert torch.allclose(L[mask], -adj[mask])

    def test_empty_graph(self):
        """Test Laplacian of graph with no edges."""
        adj = torch.zeros(5, 5)

        L = graph_laplacian(adj)

        assert torch.allclose(L, torch.zeros(5, 5))

    def test_single_node(self):
        """Test Laplacian of single node graph."""
        adj = torch.tensor([[0.]])

        L = graph_laplacian(adj)

        assert torch.allclose(L, torch.tensor([[0.]]))


# ============================================================================
# GNM CORRELATIONS TESTS
# ============================================================================


class TestGNMCorrelations:
    """Tests for gnm_correlations function."""

    def test_output_shape(self):
        """Test output shape matches input."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        assert corr.shape == (10, 10)

    def test_symmetric(self):
        """Test correlations are symmetric."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        assert torch.allclose(corr, corr.T, atol=1e-4)

    def test_no_nan(self):
        """Test output contains no NaN."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        assert not torch.isnan(corr).any()

    def test_pseudoinverse_property(self):
        """Test that corr is pseudoinverse of Laplacian."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)
        corr = gnm_correlations(adj)

        result = L @ corr @ L
        assert torch.allclose(result, L, atol=1e-3)

    def test_complete_graph(self):
        """Test correlations for complete graph."""
        n = 5
        adj = torch.ones(n, n) - torch.eye(n)

        corr = gnm_correlations(adj)

        assert corr.shape == (n, n)
        assert not torch.isnan(corr).any()

    def test_positive_diagonal(self):
        """Test diagonal elements (variances) are positive."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        assert (corr.diagonal() >= -1e-5).all()


# ============================================================================
# GNM VARIANCES TESTS
# ============================================================================


class TestGNMVariances:
    """Tests for gnm_variances function."""

    def test_output_shape(self):
        """Test output is vector of correct size."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)

        assert var.shape == (10,)

    def test_matches_diagonal_of_correlations(self):
        """Test variances equal diagonal of correlation matrix."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)
        corr = gnm_correlations(adj)

        assert torch.allclose(var, corr.diagonal())

    def test_no_nan(self):
        """Test output contains no NaN."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)

        assert not torch.isnan(var).any()

    def test_positive(self):
        """Test variances are non-negative."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)

        assert (var >= -1e-5).all()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestGNMIntegration:
    """Integration tests for GNM functions."""

    def test_laplacian_to_correlations_pipeline(self):
        """Test full pipeline from adjacency to correlations."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)
        assert L.shape == (10, 10)

        corr = gnm_correlations(adj)
        assert corr.shape == (10, 10)

        var = gnm_variances(adj)
        assert var.shape == (10,)

        assert torch.allclose(var, corr.diagonal())
