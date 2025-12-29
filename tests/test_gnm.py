"""Tests for GNM (Gaussian Network Model) utilities.

Tests cover:
- contact_map: Build adjacency matrix from Polymer
- graph_laplacian: Graph Laplacian computation
- gnm_correlations: GNM correlations
- gnm_variances: GNM position variances
- GNM class: Cached computation wrapper

All tests are parametrized to run on both numpy and torch backends.
"""
from __future__ import annotations

import pytest
import numpy as np

import ciffy
from ciffy import Scale
# Import internal functions for testing (not publicly exported)
from ciffy.operations.gnm import graph_laplacian, gnm_correlations, gnm_variances
# Import the public API
from ciffy.operations import GNM, contact_map


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


# ============================================================================
# GNM CLASS TESTS
# ============================================================================


class TestGNMClass:
    """Tests for the GNM class."""

    def test_initialization(self, backend):
        """Test GNM class initialization."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)

        assert gnm.adj is adj
        assert gnm.laplacian.shape == (10, 10)
        # Verify Laplacian is computed correctly
        expected_L = graph_laplacian(adj)
        assert allclose(gnm.laplacian, expected_L)

    def test_correlations_property(self, backend):
        """Test correlations property matches function output."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        expected = gnm_correlations(adj)

        assert allclose(gnm.correlations, expected, atol=1e-4)

    def test_variances_property(self, backend):
        """Test variances property matches function output."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        expected = gnm_variances(adj)

        assert allclose(gnm.variances, expected, atol=1e-4)

    def test_cross_correlations_shape(self, backend):
        """Test cross_correlations has correct shape."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        cross_corr = gnm.cross_correlations

        assert cross_corr.shape == (10, 10)

    def test_cross_correlations_diagonal_ones(self, backend):
        """Test cross_correlations diagonal is all ones."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        cross_corr = gnm.cross_correlations
        cross_corr_np = np.asarray(cross_corr)

        assert allclose(np.diag(cross_corr_np), np.ones(10))

    def test_cross_correlations_range(self, backend):
        """Test cross_correlations values are in [-1, 1]."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        cross_corr = gnm.cross_correlations
        cross_corr_np = np.asarray(cross_corr)

        assert (cross_corr_np >= -1 - 1e-5).all()
        assert (cross_corr_np <= 1 + 1e-5).all()

    def test_cross_correlations_symmetric(self, backend):
        """Test cross_correlations is symmetric."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        cross_corr = gnm.cross_correlations
        cross_corr_np = np.asarray(cross_corr)

        assert allclose(cross_corr_np, cross_corr_np.T, atol=1e-5)

    def test_eigenvalues_property(self, backend):
        """Test eigenvalues property."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        eigenvalues = gnm.eigenvalues
        eigenvalues_np = np.asarray(eigenvalues)

        assert eigenvalues.shape == (10,)
        # Eigenvalues should be non-negative for positive semidefinite matrix
        assert (eigenvalues_np >= -1e-5).all()
        # First eigenvalue should be close to zero (connected graph)
        assert abs(eigenvalues_np[0]) < 1e-5

    def test_modes_all(self, backend):
        """Test modes() returns all non-trivial modes when k=None."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        eigenvalues, modes = gnm.modes()

        # Should skip first (zero) mode, return 9 modes
        assert eigenvalues.shape == (9,)
        assert modes.shape == (10, 9)

    def test_modes_k(self, backend):
        """Test modes() returns k slowest modes."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        eigenvalues, modes = gnm.modes(k=3)

        assert eigenvalues.shape == (3,)
        assert modes.shape == (10, 3)

    def test_modes_eigenvalues_ascending(self, backend):
        """Test modes() returns eigenvalues in ascending order."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        eigenvalues, _ = gnm.modes(k=5)
        eigenvalues_np = np.asarray(eigenvalues)

        # Should be sorted ascending
        assert (np.diff(eigenvalues_np) >= -1e-5).all()

    def test_modes_orthogonal(self, backend):
        """Test that eigenvectors (modes) are orthogonal."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)
        _, modes = gnm.modes()
        modes_np = np.asarray(modes)

        # Modes should be orthonormal
        orthogonality = modes_np.T @ modes_np
        assert allclose(orthogonality, np.eye(9), atol=1e-4)

    def test_caching_pinv(self, backend):
        """Test that pseudo-inverse is cached (computed once)."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)

        # Access correlations to trigger computation
        _ = gnm.correlations
        first_pinv = gnm._pinv

        # Access variances (should use cached pinv)
        _ = gnm.variances
        second_pinv = gnm._pinv

        # Should be the same object (not recomputed)
        assert first_pinv is second_pinv

    def test_caching_eigen(self, backend):
        """Test that eigendecomposition is cached (computed once)."""
        adj = make_symmetric_adj(10, backend)

        gnm = GNM(adj)

        # Access eigenvalues to trigger computation
        _ = gnm.eigenvalues
        first_eigenvalues = gnm._eigenvalues

        # Access modes (should use cached eigendecomposition)
        _ = gnm.modes()
        second_eigenvalues = gnm._eigenvalues

        # Should be the same object (not recomputed)
        assert first_eigenvalues is second_eigenvalues


class TestGNMClassIntegration:
    """Integration tests for GNM class."""

    def test_class_vs_functions(self, backend):
        """Test that GNM class produces same results as standalone functions."""
        adj = make_symmetric_adj(15, backend)

        gnm = GNM(adj)

        # Correlations
        assert allclose(gnm.correlations, gnm_correlations(adj), atol=1e-4)

        # Variances
        assert allclose(gnm.variances, gnm_variances(adj), atol=1e-4)

        # Laplacian
        assert allclose(gnm.laplacian, graph_laplacian(adj))

    def test_backend_consistency_class(self):
        """Test GNM class produces same results for numpy and torch."""
        adj_np = make_symmetric_adj(10, "numpy")
        adj_torch = make_symmetric_adj(10, "torch")

        gnm_np = GNM(adj_np)
        gnm_torch = GNM(adj_torch)

        # Correlations
        assert allclose(gnm_np.correlations, gnm_torch.correlations, atol=1e-4)

        # Variances
        assert allclose(gnm_np.variances, gnm_torch.variances, atol=1e-4)

        # Cross-correlations
        assert allclose(gnm_np.cross_correlations, gnm_torch.cross_correlations, atol=1e-4)

        # Eigenvalues
        assert allclose(gnm_np.eigenvalues, gnm_torch.eigenvalues, atol=1e-4)

        # Modes
        eig_np, modes_np = gnm_np.modes(k=5)
        eig_torch, modes_torch = gnm_torch.modes(k=5)
        assert allclose(eig_np, eig_torch, atol=1e-4)
        # Note: eigenvectors may differ by sign, so compare absolute values
        assert allclose(np.abs(modes_np), np.abs(np.asarray(modes_torch)), atol=1e-4)

    def test_complete_graph(self, backend):
        """Test GNM class with complete graph."""
        n = 5
        adj = make_array(np.ones((n, n)) - np.eye(n), backend)

        gnm = GNM(adj)

        assert gnm.correlations.shape == (n, n)
        assert gnm.variances.shape == (n,)
        assert gnm.cross_correlations.shape == (n, n)
        eigenvalues, modes = gnm.modes()
        assert eigenvalues.shape == (n - 1,)
        assert modes.shape == (n, n - 1)


# ============================================================================
# CONTACT MAP TESTS
# ============================================================================


class TestContactMap:
    """Tests for contact_map function."""

    @pytest.fixture
    def polymer(self, backend):
        """Load a test polymer with the specified backend."""
        return ciffy.load("tests/data/9MDS.cif", backend=backend).poly().chain(0)

    def test_output_shape_residue(self, backend, polymer):
        """Test contact map shape at residue scale."""
        adj = contact_map(polymer, cutoff=7.0, scale=Scale.RESIDUE)

        n_residues = polymer.size(Scale.RESIDUE)
        assert adj.shape == (n_residues, n_residues)

    def test_output_shape_atom(self, backend, polymer):
        """Test contact map shape at atom scale."""
        adj = contact_map(polymer, cutoff=4.0, scale=Scale.ATOM)

        n_atoms = polymer.size(Scale.ATOM)
        assert adj.shape == (n_atoms, n_atoms)

    def test_binary_values(self, backend, polymer):
        """Test contact map contains only 0 and 1."""
        adj = contact_map(polymer, cutoff=7.0)
        adj_np = np.asarray(adj)

        unique_values = np.unique(adj_np)
        assert len(unique_values) <= 2
        assert all(v in [0.0, 1.0] for v in unique_values)

    def test_zero_diagonal(self, backend, polymer):
        """Test contact map has zero diagonal (no self-contacts)."""
        adj = contact_map(polymer, cutoff=7.0)
        adj_np = np.asarray(adj)

        assert allclose(np.diag(adj_np), np.zeros(adj_np.shape[0]))

    def test_symmetric(self, backend, polymer):
        """Test contact map is symmetric."""
        adj = contact_map(polymer, cutoff=7.0)
        adj_np = np.asarray(adj)

        assert allclose(adj_np, adj_np.T)

    def test_larger_cutoff_more_contacts(self, backend, polymer):
        """Test that larger cutoff produces more contacts."""
        adj_small = contact_map(polymer, cutoff=5.0)
        adj_large = contact_map(polymer, cutoff=10.0)

        n_contacts_small = np.asarray(adj_small).sum()
        n_contacts_large = np.asarray(adj_large).sum()

        assert n_contacts_large >= n_contacts_small

    def test_very_small_cutoff_few_contacts(self, backend, polymer):
        """Test that very small cutoff produces few/no contacts."""
        adj = contact_map(polymer, cutoff=0.1)
        adj_np = np.asarray(adj)

        # With 0.1A cutoff, should have very few contacts
        assert adj_np.sum() < adj_np.size * 0.01  # Less than 1% contacts

    def test_very_large_cutoff_many_contacts(self, backend, polymer):
        """Test that very large cutoff produces many contacts."""
        adj = contact_map(polymer, cutoff=1000.0)
        adj_np = np.asarray(adj)

        n = adj_np.shape[0]
        # With huge cutoff, all pairs should be in contact (except diagonal)
        expected_contacts = n * (n - 1)  # All pairs
        assert adj_np.sum() == expected_contacts

    def test_backend_consistency(self):
        """Test numpy and torch backends produce similar contact counts.

        Note: Due to floating-point differences in distance calculations,
        contacts at exactly the cutoff boundary may differ. We check that
        the total number of contacts is very close.
        """
        polymer_np = ciffy.load("tests/data/9MDS.cif", backend="numpy").poly().chain(0)
        polymer_torch = ciffy.load("tests/data/9MDS.cif", backend="torch").poly().chain(0)

        adj_np = contact_map(polymer_np, cutoff=7.0)
        adj_torch = contact_map(polymer_torch, cutoff=7.0)

        # Contact counts should be very close (allowing for edge cases at cutoff)
        n_contacts_np = np.asarray(adj_np).sum()
        n_contacts_torch = np.asarray(adj_torch).sum()

        # Allow up to 1% difference in contact count
        assert abs(n_contacts_np - n_contacts_torch) / max(n_contacts_np, 1) < 0.01

    def test_works_with_gnm(self, backend, polymer):
        """Test contact map integrates with GNM class."""
        adj = contact_map(polymer, cutoff=7.0)

        gnm = GNM(adj)

        # Should produce valid GNM outputs
        assert gnm.variances.shape == (polymer.size(Scale.RESIDUE),)
        assert not isnan_any(gnm.variances)
        assert (np.asarray(gnm.variances) >= -1e-5).all()

    def test_default_scale_is_residue(self, backend, polymer):
        """Test default scale is RESIDUE."""
        adj_default = contact_map(polymer, cutoff=7.0)
        adj_residue = contact_map(polymer, cutoff=7.0, scale=Scale.RESIDUE)

        assert allclose(adj_default, adj_residue)

    def test_default_cutoff_is_7(self, backend, polymer):
        """Test default cutoff is 7.0 Angstroms."""
        adj_default = contact_map(polymer)
        adj_7 = contact_map(polymer, cutoff=7.0)

        assert allclose(adj_default, adj_7)
