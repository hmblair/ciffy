"""Tests for GNM (Gaussian Network Model) utilities.

Tests cover:
- graph_laplacian: Graph Laplacian computation
- gnm_correlations: GNM correlations
- gnm_variances: GNM position variances

All tests are parametrized to run on both numpy and torch backends.
"""
from __future__ import annotations

import pytest
import numpy as np

from ciffy.operations.gnm import graph_laplacian, gnm_correlations, gnm_variances


def make_symmetric_adj(n: int, backend: str, seed: int = 42):
    """Create a random symmetric adjacency matrix with zero diagonal."""
    np.random.seed(seed)
    adj = np.random.rand(n, n).astype(np.float32)
    adj = adj + adj.T
    np.fill_diagonal(adj, 0)

    if backend == "torch":
        import torch
        return torch.from_numpy(adj)
    return adj


def make_array(data, backend: str):
    """Create array from list/nested list for given backend."""
    arr = np.array(data, dtype=np.float32)
    if backend == "torch":
        import torch
        return torch.from_numpy(arr)
    return arr


def allclose(a, b, atol=1e-5):
    """Backend-agnostic allclose check."""
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    return np.allclose(a_np, b_np, atol=atol)


def get_diagonal(arr):
    """Backend-agnostic diagonal extraction."""
    if hasattr(arr, 'diagonal'):
        return arr.diagonal()
    return np.diag(arr)


def isnan_any(arr):
    """Backend-agnostic NaN check."""
    return np.isnan(np.asarray(arr)).any()


# ============================================================================
# GRAPH LAPLACIAN TESTS
# ============================================================================


class TestGraphLaplacian:
    """Tests for graph_laplacian function."""

    def test_basic_example(self, backend):
        """Test Laplacian computation for simple triangle graph."""
        adj = make_array([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.]
        ], backend)

        L = graph_laplacian(adj)

        expected = make_array([
            [2., -1., -1.],
            [-1., 2., -1.],
            [-1., -1., 2.]
        ], backend)
        assert allclose(L, expected)

    def test_row_sum_zero(self, backend):
        """Test that each row of Laplacian sums to zero."""
        adj = make_symmetric_adj(10, backend)

        L = graph_laplacian(adj)
        L_np = np.asarray(L)
        row_sums = L_np.sum(axis=1)

        assert allclose(row_sums, np.zeros(10))

    def test_symmetric(self, backend):
        """Test that Laplacian is symmetric for symmetric adjacency."""
        adj = make_symmetric_adj(10, backend)

        L = graph_laplacian(adj)
        L_np = np.asarray(L)

        assert allclose(L_np, L_np.T)

    def test_positive_semidefinite(self, backend):
        """Test that Laplacian is positive semi-definite."""
        adj = make_symmetric_adj(10, backend)

        L = graph_laplacian(adj)
        L_np = np.asarray(L)
        eigenvalues = np.linalg.eigvalsh(L_np)

        # Relaxed tolerance for numerical precision
        assert (eigenvalues >= -1e-5).all()

    def test_smallest_eigenvalue_zero(self, backend):
        """Test that smallest eigenvalue is zero for connected graph."""
        # Complete graph (fully connected)
        adj = make_array(np.ones((5, 5)) - np.eye(5), backend)

        L = graph_laplacian(adj)
        L_np = np.asarray(L)
        eigenvalues = np.linalg.eigvalsh(L_np)

        assert abs(eigenvalues[0]) < 1e-6

    def test_diagonal_is_degree(self, backend):
        """Test that diagonal elements are node degrees."""
        adj = make_array([
            [0., 1., 1., 0.],
            [1., 0., 1., 1.],
            [1., 1., 0., 0.],
            [0., 1., 0., 0.]
        ], backend)

        L = graph_laplacian(adj)
        L_np = np.asarray(L)
        adj_np = np.asarray(adj)
        degrees = adj_np.sum(axis=1)

        assert allclose(np.diag(L_np), degrees)

    def test_off_diagonal_negated(self, backend):
        """Test that off-diagonal elements are negated adjacency."""
        adj = make_symmetric_adj(5, backend)

        L = graph_laplacian(adj)
        L_np = np.asarray(L)
        adj_np = np.asarray(adj)

        mask = ~np.eye(5, dtype=bool)
        assert allclose(L_np[mask], -adj_np[mask])

    def test_empty_graph(self, backend):
        """Test Laplacian of graph with no edges."""
        adj = make_array(np.zeros((5, 5)), backend)

        L = graph_laplacian(adj)

        assert allclose(L, np.zeros((5, 5)))

    def test_single_node(self, backend):
        """Test Laplacian of single node graph."""
        adj = make_array([[0.]], backend)

        L = graph_laplacian(adj)

        assert allclose(L, [[0.]])


# ============================================================================
# GNM CORRELATIONS TESTS
# ============================================================================


class TestGNMCorrelations:
    """Tests for gnm_correlations function."""

    def test_output_shape(self, backend):
        """Test output shape matches input."""
        adj = make_symmetric_adj(10, backend)

        corr = gnm_correlations(adj)

        assert corr.shape == (10, 10)

    def test_symmetric(self, backend):
        """Test correlations are symmetric."""
        adj = make_symmetric_adj(10, backend)

        corr = gnm_correlations(adj)
        corr_np = np.asarray(corr)

        assert allclose(corr_np, corr_np.T, atol=1e-4)

    def test_no_nan(self, backend):
        """Test output contains no NaN."""
        adj = make_symmetric_adj(10, backend)

        corr = gnm_correlations(adj)

        assert not isnan_any(corr)

    def test_pseudoinverse_property(self, backend):
        """Test that corr is pseudoinverse of Laplacian."""
        adj = make_symmetric_adj(10, backend)

        L = graph_laplacian(adj)
        corr = gnm_correlations(adj)

        L_np = np.asarray(L)
        corr_np = np.asarray(corr)

        result = L_np @ corr_np @ L_np
        assert allclose(result, L_np, atol=1e-3)

    def test_complete_graph(self, backend):
        """Test correlations for complete graph."""
        n = 5
        adj = make_array(np.ones((n, n)) - np.eye(n), backend)

        corr = gnm_correlations(adj)

        assert corr.shape == (n, n)
        assert not isnan_any(corr)

    def test_positive_diagonal(self, backend):
        """Test diagonal elements (variances) are positive."""
        adj = make_symmetric_adj(10, backend)

        corr = gnm_correlations(adj)
        corr_np = np.asarray(corr)

        assert (np.diag(corr_np) >= -1e-5).all()


# ============================================================================
# GNM VARIANCES TESTS
# ============================================================================


class TestGNMVariances:
    """Tests for gnm_variances function."""

    def test_output_shape(self, backend):
        """Test output is vector of correct size."""
        adj = make_symmetric_adj(10, backend)

        var = gnm_variances(adj)

        assert var.shape == (10,)

    def test_matches_diagonal_of_correlations(self, backend):
        """Test variances equal diagonal of correlation matrix."""
        adj = make_symmetric_adj(10, backend)

        var = gnm_variances(adj)
        corr = gnm_correlations(adj)

        var_np = np.asarray(var)
        corr_np = np.asarray(corr)

        assert allclose(var_np, np.diag(corr_np))

    def test_no_nan(self, backend):
        """Test output contains no NaN."""
        adj = make_symmetric_adj(10, backend)

        var = gnm_variances(adj)

        assert not isnan_any(var)

    def test_positive(self, backend):
        """Test variances are non-negative."""
        adj = make_symmetric_adj(10, backend)

        var = gnm_variances(adj)
        var_np = np.asarray(var)

        assert (var_np >= -1e-5).all()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestGNMIntegration:
    """Integration tests for GNM functions."""

    def test_laplacian_to_correlations_pipeline(self, backend):
        """Test full pipeline from adjacency to correlations."""
        adj = make_symmetric_adj(10, backend)

        L = graph_laplacian(adj)
        assert L.shape == (10, 10)

        corr = gnm_correlations(adj)
        assert corr.shape == (10, 10)

        var = gnm_variances(adj)
        assert var.shape == (10,)

        var_np = np.asarray(var)
        corr_np = np.asarray(corr)
        assert allclose(var_np, np.diag(corr_np))

    def test_backend_consistency(self):
        """Test numpy and torch backends produce same results."""
        adj_np = make_symmetric_adj(10, "numpy")
        adj_torch = make_symmetric_adj(10, "torch")

        # Graph Laplacian
        L_np = graph_laplacian(adj_np)
        L_torch = graph_laplacian(adj_torch)
        assert allclose(L_np, L_torch)

        # GNM correlations
        corr_np = gnm_correlations(adj_np)
        corr_torch = gnm_correlations(adj_torch)
        assert allclose(corr_np, corr_torch, atol=1e-4)

        # GNM variances
        var_np = gnm_variances(adj_np)
        var_torch = gnm_variances(adj_torch)
        assert allclose(var_np, var_torch, atol=1e-4)
