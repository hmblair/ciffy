"""Tests for backbone dihedral sampling from Ramachandran distributions."""

import numpy as np
import pytest

import ciffy
from ciffy import DihedralType, Scale
from ciffy.sampling import randomize_backbone, sample_protein_dihedrals
from ciffy.utils.gmm import GaussianMixtureModel


class TestGaussianMixtureModel:
    """Tests for the GMM utility class."""

    def test_create_gmm(self):
        """Test basic GMM creation."""
        gmm = GaussianMixtureModel(
            means=np.array([[-1.0, -0.7], [-2.1, 2.3]]),
            covariances=np.array([[[0.1, 0], [0, 0.1]], [[0.2, 0], [0, 0.2]]]),
            weights=np.array([0.6, 0.4]),
        )
        assert gmm.n_components == 2
        assert gmm.n_features == 2

    def test_gmm_weights_must_sum_to_one(self):
        """Test that GMM validates weights sum to 1."""
        with pytest.raises(ValueError, match="weights must sum to 1"):
            GaussianMixtureModel(
                means=np.array([[0, 0], [1, 1]]),
                covariances=np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
                weights=np.array([0.3, 0.3]),  # Sum to 0.6, not 1
            )

    def test_gmm_sample_shape(self):
        """Test GMM sampling returns correct shape."""
        gmm = GaussianMixtureModel(
            means=np.array([[-1.0, -0.7], [-2.1, 2.3]]),
            covariances=np.array([[[0.1, 0], [0, 0.1]], [[0.2, 0], [0, 0.2]]]),
            weights=np.array([0.6, 0.4]),
        )
        samples = gmm.sample(100)
        assert samples.shape == (100, 2)

    def test_gmm_sample_reproducibility(self):
        """Test GMM sampling is reproducible with seed."""
        gmm = GaussianMixtureModel(
            means=np.array([[-1.0, -0.7]]),
            covariances=np.array([[[0.1, 0], [0, 0.1]]]),
            weights=np.array([1.0]),
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        samples1 = gmm.sample(10, rng=rng1)
        samples2 = gmm.sample(10, rng=rng2)

        np.testing.assert_array_equal(samples1, samples2)

    def test_gmm_fit(self):
        """Test GMM fitting with synthetic data."""
        # Generate data from two well-separated clusters
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=[-2, -2], scale=0.3, size=(100, 2))
        cluster2 = rng.normal(loc=[2, 2], scale=0.3, size=(100, 2))
        data = np.vstack([cluster1, cluster2])

        gmm = GaussianMixtureModel.fit(data, n_components=2, rng=rng)

        assert gmm.n_components == 2
        assert gmm.n_features == 2
        assert np.isclose(gmm.weights.sum(), 1.0)

        # Means should be near the true cluster centers
        sorted_means = gmm.means[np.argsort(gmm.means[:, 0])]
        assert np.abs(sorted_means[0, 0] - (-2)) < 0.5
        assert np.abs(sorted_means[1, 0] - 2) < 0.5

    def test_gmm_save_load(self, tmp_path):
        """Test GMM save and load round-trip."""
        original = GaussianMixtureModel(
            means=np.array([[-1.0, -0.7], [-2.1, 2.3]]),
            covariances=np.array([[[0.1, 0.01], [0.01, 0.1]], [[0.2, 0], [0, 0.2]]]),
            weights=np.array([0.6, 0.4]),
        )

        path = tmp_path / "test_gmm.npz"
        original.save(path)
        loaded = GaussianMixtureModel.load(path)

        np.testing.assert_array_almost_equal(original.means, loaded.means)
        np.testing.assert_array_almost_equal(original.covariances, loaded.covariances)
        np.testing.assert_array_almost_equal(original.weights, loaded.weights)


class TestSampleProteinDihedrals:
    """Tests for sampling protein dihedrals."""

    def test_sample_dihedrals_shape(self):
        """Test that sampled dihedrals have correct shape."""
        phi, psi, omega = sample_protein_dihedrals(10)

        assert phi.shape == (10,)
        assert psi.shape == (10,)
        assert omega.shape == (10,)

    def test_sample_dihedrals_nan_positions(self):
        """Test that NaN values are at correct positions."""
        phi, psi, omega = sample_protein_dihedrals(5)

        # First residue has no phi (no preceding residue)
        assert np.isnan(phi[0])
        # Last residue has no psi (no following residue)
        assert np.isnan(psi[-1])
        # Last residue has no omega
        assert np.isnan(omega[-1])

        # Other positions should not be NaN
        assert not np.isnan(phi[1:]).any()
        assert not np.isnan(psi[:-1]).any()
        assert not np.isnan(omega[:-1]).any()

    def test_sample_dihedrals_range(self):
        """Test that sampled angles are centered in reasonable range."""
        phi, psi, omega = sample_protein_dihedrals(100)

        # Extract valid angles
        valid_phi = phi[~np.isnan(phi)]
        valid_psi = psi[~np.isnan(psi)]
        valid_omega = omega[~np.isnan(omega)]

        # Mean values should be roughly in typical Ramachandran regions
        # (exact values depend on GMM fit, but should be finite)
        assert np.isfinite(valid_phi.mean())
        assert np.isfinite(valid_psi.mean())

        # Omega should be near pi (trans configuration)
        assert np.abs(valid_omega.mean() - np.pi) < 0.2

    def test_sample_dihedrals_reproducibility(self):
        """Test reproducibility with seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        phi1, psi1, omega1 = sample_protein_dihedrals(10, rng=rng1)
        phi2, psi2, omega2 = sample_protein_dihedrals(10, rng=rng2)

        np.testing.assert_array_equal(phi1, phi2)
        np.testing.assert_array_equal(psi1, psi2)
        np.testing.assert_array_equal(omega1, omega2)


class TestRandomizeBackbone:
    """Tests for the randomize_backbone function."""

    def test_randomize_backbone_changes_coordinates(self):
        """Test that randomize_backbone modifies coordinates."""
        polymer = ciffy.from_sequence("MGKLF")
        original_coords = polymer.coordinates.copy()

        polymer = randomize_backbone(polymer, seed=42)
        new_coords = polymer.coordinates

        # Coordinates should be different
        assert not np.allclose(original_coords, new_coords)

    def test_randomize_backbone_preserves_size(self):
        """Test that randomize_backbone preserves polymer size."""
        polymer = ciffy.from_sequence("MGKLF")
        original_size = polymer.size()
        original_n_res = polymer.size(Scale.RESIDUE)

        polymer = randomize_backbone(polymer, seed=42)

        assert polymer.size() == original_size
        assert polymer.size(Scale.RESIDUE) == original_n_res

    def test_randomize_backbone_reproducibility(self):
        """Test that randomize_backbone is reproducible with seed."""
        polymer1 = ciffy.from_sequence("MGKLF")
        polymer2 = ciffy.from_sequence("MGKLF")

        polymer1 = randomize_backbone(polymer1, seed=42)
        polymer2 = randomize_backbone(polymer2, seed=42)

        np.testing.assert_array_almost_equal(
            polymer1.coordinates, polymer2.coordinates
        )

    def test_randomize_backbone_different_seeds(self):
        """Test that different seeds produce different results."""
        polymer1 = ciffy.from_sequence("MGKLF")
        polymer2 = ciffy.from_sequence("MGKLF")

        polymer1 = randomize_backbone(polymer1, seed=42)
        polymer2 = randomize_backbone(polymer2, seed=123)

        assert not np.allclose(polymer1.coordinates, polymer2.coordinates)


class TestFromSequenceWithSampling:
    """Tests for from_sequence with sample_dihedrals parameter."""

    def test_from_sequence_sample_dihedrals(self):
        """Test from_sequence with sample_dihedrals=True."""
        # Without sampling
        polymer_ideal = ciffy.from_sequence("MGKLF")

        # With sampling
        polymer_sampled = ciffy.from_sequence("MGKLF", sample_dihedrals=True, seed=42)

        # Same structure, different coordinates
        assert polymer_ideal.size() == polymer_sampled.size()
        assert polymer_ideal.size(Scale.RESIDUE) == polymer_sampled.size(Scale.RESIDUE)
        assert not np.allclose(polymer_ideal.coordinates, polymer_sampled.coordinates)

    def test_from_sequence_sample_dihedrals_reproducibility(self):
        """Test that from_sequence with sampling is reproducible."""
        polymer1 = ciffy.from_sequence("MGKLF", sample_dihedrals=True, seed=42)
        polymer2 = ciffy.from_sequence("MGKLF", sample_dihedrals=True, seed=42)

        np.testing.assert_array_almost_equal(
            polymer1.coordinates, polymer2.coordinates
        )

    def test_from_sequence_default_no_sampling(self):
        """Test that from_sequence defaults to no sampling."""
        polymer1 = ciffy.from_sequence("MGKLF")
        polymer2 = ciffy.from_sequence("MGKLF")

        # Without sampling, coordinates should be identical (ideal CCD)
        np.testing.assert_array_almost_equal(
            polymer1.coordinates, polymer2.coordinates
        )
