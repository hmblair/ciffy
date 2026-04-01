"""Tests for structure comparison metrics."""

import pytest
import numpy as np

import ciffy
from ciffy import Scale, Molecule, tm_score, lddt, rmsd

from tests.utils import get_test_cif, random_coordinates, get_like, requires_torch
from tests.testing import get_tolerances
from ciffy.backend import to_backend


# =============================================================================
# Test TM-score
# =============================================================================

class TestTMScore:
    """Tests for tm_score function."""

    def test_tm_score_self(self, backend):
        """TM-score of structure with itself should be 1.0."""
        # 3SKW has non-standard residues (CCC, GTP) - should work with extended groups
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        score = tm_score(p, p)

        tol = get_tolerances()
        assert abs(score - 1.0) < tol.score_self

    def test_tm_score_range(self, backend):
        """TM-score should be between 0 and 1."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        score = tm_score(p, p)

        assert 0.0 <= score <= 1.0

    def test_tm_score_size_mismatch(self):
        """TM-score should raise error for mismatched sizes."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend="numpy")
        p2 = ciffy.load(get_test_cif("9GCM"), backend="numpy")

        # Different residue counts will cause size mismatch
        with pytest.raises(ValueError):
            tm_score(p1, p2)


# =============================================================================
# Test lDDT
# =============================================================================

class TestLDDT:
    """Tests for lddt function."""

    def test_lddt_self(self, backend):
        """lDDT of structure with itself should be 1.0."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self

    def test_lddt_range(self, backend):
        """lDDT should be between 0 and 1."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        assert 0.0 <= global_score <= 1.0

    def test_lddt_per_residue_shape(self, backend):
        """lDDT should return per-residue scores with correct shape."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        expected_shape = (p.size(Scale.RESIDUE),)
        assert per_res.shape == expected_shape

    def test_lddt_per_residue_self(self, backend):
        """Per-residue lDDT with itself should be mostly 1.0."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p)

        if backend == "torch":
            per_res = per_res.numpy()

        # Most per-residue scores should be 1.0
        # (some terminal/isolated residues may have 0.0 due to no neighbors within cutoff)
        assert np.mean(per_res == 1.0) > 0.9, "Most residues should have lDDT=1.0"

    def test_lddt_custom_thresholds(self):
        """Test lDDT with custom thresholds."""
        p = ciffy.load(get_test_cif("3SKW"), backend="numpy")

        # Custom thresholds
        global_score, _ = lddt(p, p, thresholds=(0.5, 1.0))
        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self

    def test_lddt_custom_cutoff(self):
        """Test lDDT with custom cutoff."""
        p = ciffy.load(get_test_cif("3SKW"), backend="numpy")

        # Very small cutoff should still work
        global_score, _ = lddt(p, p, cutoff=5.0)
        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self

    def test_lddt_size_mismatch(self):
        """lDDT should raise error for mismatched sizes."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend="numpy")
        p2 = ciffy.load(get_test_cif("9GCM"), backend="numpy")

        with pytest.raises(ValueError, match="sizes must match"):
            lddt(p1, p2)


# =============================================================================
# Edge Case Tests for Metrics
# =============================================================================

class TestTMScoreEdgeCases:
    """Edge case tests for tm_score."""

    def test_tm_score_very_small_structure(self, backend):
        """TM-score handles very small structures (d_0 edge case)."""
        # Create 5-residue structure
        p = ciffy.template("acgua", backend=backend)

        # Attach random non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend)

        # Templates don't have molecule_types, so specify explicitly
        score = tm_score(p, p, molecule_type=Molecule.RNA)

        assert 0.0 <= score <= 1.0
        # Self-comparison should be ~1.0
        assert score > 0.99

    def test_tm_score_single_residue(self, backend):
        """TM-score handles single-residue structure (may return NaN for L<5)."""
        p = ciffy.template("a", backend=backend)

        # Attach non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend)

        # Templates don't have molecule_types, so specify explicitly
        score = tm_score(p, p, molecule_type=Molecule.RNA)

        # TM-score should always return a valid float in [0, 1]
        assert 0.0 <= score <= 1.0

    def test_tm_score_two_residues(self, backend):
        """TM-score handles two-residue structure."""
        p = ciffy.template("ac", backend=backend)

        # Attach non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend)

        # Templates don't have molecule_types, so specify explicitly
        score = tm_score(p, p, molecule_type=Molecule.RNA)

        # TM-score should always return a valid float in [0, 1]
        assert 0.0 <= score <= 1.0

    def test_tm_score_works_for_templates(self, backend):
        """TM-score works for template-generated polymers (which have molecule_types)."""
        p = ciffy.template("acgu", backend=backend)
        p.coordinates = random_coordinates(p.size(), backend)

        # Templates have molecule_types, so this should work
        score = tm_score(p, p)
        assert 0.0 <= score <= 1.0

    def test_tm_score_nonstandard_residues(self, backend):
        """TM-score works with non-standard residues (modified nucleotides)."""
        # 3SKW includes non-standard residues (CCC, GTP) - tests extended atom groups
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        score = tm_score(p, p)

        assert 0.0 <= score <= 1.0
        assert score > 0.99


class TestLDDTEdgeCases:
    """Edge case tests for lddt."""

    def test_lddt_very_small_cutoff(self, backend):
        """lDDT with very small cutoff (few/no pairs)."""
        p = ciffy.template("acgu", backend=backend)

        # Place atoms very far apart
        n = p.size()
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, 0] = np.arange(n) * 100  # 100 angstroms apart
        p.coordinates = to_backend(coords, like=get_like(backend))

        # Very small cutoff = no pairs
        global_score, per_res = lddt(p, p, cutoff=1.0)

        # With no valid pairs, lDDT should return 0.0
        assert global_score == 0.0

    def test_lddt_single_threshold(self, backend):
        """lDDT with single threshold."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p, thresholds=(1.0,))

        assert 0.0 <= global_score <= 1.0

    def test_lddt_many_thresholds(self, backend):
        """lDDT with many thresholds."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        thresholds = tuple(i * 0.1 for i in range(1, 21))  # 0.1 to 2.0
        global_score, per_res = lddt(p, p, thresholds=thresholds)

        assert 0.0 <= global_score <= 1.0

    def test_lddt_single_residue(self, backend):
        """lDDT on single-residue structure."""
        p = ciffy.template("a", backend=backend)

        # Attach non-zero coordinates
        p.coordinates = random_coordinates(p.size(), backend, scale=1.0)

        global_score, per_res = lddt(p, p)

        # Single residue may have undefined lDDT (no pairs to compare)
        assert per_res.shape == (1,)

    def test_lddt_two_residues(self, backend):
        """lDDT on two-residue structure."""
        p = ciffy.template("ac", backend=backend)

        # Attach coordinates close together
        p.coordinates = random_coordinates(p.size(), backend, scale=1.0)

        global_score, per_res = lddt(p, p)

        assert 0.0 <= global_score <= 1.0
        assert per_res.shape == (2,)

    def test_lddt_large_cutoff(self, backend):
        """lDDT with very large cutoff (all pairs included)."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        global_score, per_res = lddt(p, p, cutoff=1000.0)

        # Self comparison should be 1.0
        tol = get_tolerances()
        assert abs(global_score - 1.0) < tol.score_self


# =============================================================================
# Test RMSD Function
# =============================================================================

class TestRmsdFunction:
    """Tests for the rmsd() function."""

    def test_rmsd_self_is_zero(self, backend):
        """RMSD of structure with itself should be 0."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        rmsd_val = rmsd(p, p)

        tol = get_tolerances()
        rmsd_np = np.asarray(rmsd_val)
        assert np.all(rmsd_np < tol.allclose_atol)

    def test_rmsd_symmetric(self, backend):
        """RMSD(a, b) == RMSD(b, a)."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        # Create a perturbed copy
        p2 = p1.copy(coordinates=p1.coordinates + 0.1)

        rmsd_ab = rmsd(p1, p2)
        rmsd_ba = rmsd(p2, p1)

        rmsd_ab_np = np.asarray(rmsd_ab)
        rmsd_ba_np = np.asarray(rmsd_ba)

        tol = get_tolerances()
        assert np.allclose(rmsd_ab_np, rmsd_ba_np, atol=tol.allclose_atol)

    def test_rmsd_at_molecule_scale(self, backend):
        """RMSD at molecule scale returns single value."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        rmsd_val = rmsd(p, p, scale=Scale.MOLECULE)

        # Should be shape (1,) for single molecule
        rmsd_np = np.asarray(rmsd_val)
        assert rmsd_np.shape == (1,)

    def test_rmsd_at_chain_scale(self, backend):
        """RMSD at chain scale returns one value per chain."""
        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        n_chains = p.size(Scale.CHAIN)
        rmsd_val = rmsd(p, p, scale=Scale.CHAIN)

        rmsd_np = np.asarray(rmsd_val)
        assert rmsd_np.shape == (n_chains,)

    def test_rmsd_at_residue_scale(self, backend):
        """RMSD at residue scale returns one value per residue."""
        # Use poly() to exclude hetero atoms (water, ions) that don't have residues
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        n_res = p.size(Scale.RESIDUE)
        rmsd_val = rmsd(p, p, scale=Scale.RESIDUE)

        rmsd_np = np.asarray(rmsd_val)
        assert rmsd_np.shape == (n_res,)

    def test_rmsd_nonnegative(self, backend):
        """RMSD values are always non-negative."""
        # Use poly() to exclude hetero atoms
        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)

        np.random.seed(42)
        noise = np.random.randn(p1.size(), 3).astype(np.float32) * 0.5
        noise = to_backend(noise, like=p1.coordinates)
        p2 = p1.copy(coordinates=p1.coordinates + noise)

        rmsd_val = rmsd(p1, p2)

        rmsd_np = np.asarray(rmsd_val)
        assert np.all(rmsd_np >= 0)

    def test_rmsd_size_mismatch_raises(self):
        """RMSD raises ValueError for mismatched sizes."""
        p1 = ciffy.template("acgu", backend="numpy")
        p2 = ciffy.template("acguacgu", backend="numpy")
        # Set coordinates (template doesn't populate them)
        p1.coordinates = random_coordinates(p1.size(), "numpy")
        p2.coordinates = random_coordinates(p2.size(), "numpy")

        # Just check that it raises ValueError (message may vary)
        with pytest.raises(ValueError):
            rmsd(p1, p2)

    def test_rmsd_array_input(self, backend):
        """RMSD works with raw coordinate arrays."""
        np.random.seed(42)
        coords1 = np.random.randn(20, 3).astype(np.float32)
        coords2 = coords1 + 0.1

        ref = get_like(backend)
        coords1 = to_backend(coords1, like=ref)
        coords2 = to_backend(coords2, like=ref)

        rmsd_val = rmsd(coords1, coords2)

        # Should be a scalar or 0-d array
        if hasattr(rmsd_val, 'item'):
            rmsd_scalar = rmsd_val.item()
        else:
            rmsd_scalar = float(rmsd_val)

        assert rmsd_scalar >= 0
        assert rmsd_scalar < 1.0  # Should be small for small perturbation


# =============================================================================
# Test Intersect
# =============================================================================

class TestIntersect:
    """Tests for ciffy.intersect function."""

    def test_intersect_identical(self, backend):
        """Intersect of identical polymers returns same atoms."""
        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        a, b = ciffy.intersect(p, p)

        assert len(a) == len(p)
        assert len(b) == len(p)

    def test_intersect_missing_atoms(self, backend):
        """Intersect handles missing atoms correctly."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        # Remove first 10 atoms from p2
        p2 = p1[10:]

        a, b = ciffy.intersect(p1, p2)

        # Both should have same size (the intersection)
        assert len(a) == len(b)
        # Should be smaller than p2 (which was already smaller)
        assert len(a) <= len(p2)

    def test_intersect_atoms_match(self, backend):
        """Intersected polymers have matching atom types."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        p2 = p1[10:]

        a, b = ciffy.intersect(p1, p2)

        # Atom types should be identical
        a_atoms = np.asarray(a.atoms)
        b_atoms = np.asarray(b.atoms)
        assert np.array_equal(a_atoms, b_atoms)

    def test_intersect_enables_rmsd(self, backend):
        """Intersect enables RMSD between polymers with different atoms."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend=backend)
        p2 = p1[10:].copy()

        # Add some noise to p2 coordinates
        np.random.seed(42)
        noise = np.random.randn(len(p2), 3).astype(np.float32) * 0.5
        noise = to_backend(noise, like=p1.coordinates)
        p2.coordinates = p2.coordinates + noise

        # Direct RMSD would fail (different sizes)
        # But intersect + RMSD should work
        a, b = ciffy.intersect(p1, p2)
        rmsd_val = rmsd(a, b)

        rmsd_np = np.asarray(rmsd_val)
        assert np.all(rmsd_np >= 0)

    def test_intersect_residue_mismatch_raises(self):
        """Intersect raises ValueError for different residue counts."""
        p1 = ciffy.load(get_test_cif("3SKW"))
        p2 = ciffy.load(get_test_cif("9GCM"))

        with pytest.raises(ValueError, match="residue count"):
            ciffy.intersect(p1, p2)

    @requires_torch
    def test_intersect_backend_mismatch_raises(self):
        """Intersect raises TypeError for mixed backends."""
        p1 = ciffy.load(get_test_cif("3SKW"), backend="numpy")
        p2 = ciffy.load(get_test_cif("3SKW"), backend="torch")

        with pytest.raises(TypeError, match="Backend mismatch"):
            ciffy.intersect(p1, p2)


# =============================================================================
# RMSD invariance tests
# =============================================================================


class TestRMSDInvariance:
    """Test RMSD invariance properties (rotation, translation, reflection).

    Uses parametrized any_cif fixture to run on all test PDBs.
    """

    def test_rotation_zero_rmsd(self, any_cif):
        """Rotating a polymer should give zero RMSD after alignment."""
        import copy

        polymer = ciffy.load(any_cif)

        theta = np.pi / 2
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        rotated = copy.deepcopy(polymer)
        rotated.coordinates = polymer.coordinates @ rotation.T

        dist = rmsd(polymer, rotated, Scale.MOLECULE)
        assert np.allclose(dist, 0, atol=1e-3)

    def test_translation_zero_rmsd(self, any_cif):
        """Translating a polymer should give zero RMSD after alignment."""
        import copy

        polymer = ciffy.load(any_cif)

        translated = copy.deepcopy(polymer)
        translated.coordinates = polymer.coordinates + np.array([100.0, -50.0, 25.0])

        dist = rmsd(polymer, translated, Scale.MOLECULE)
        assert np.allclose(dist, 0, atol=1e-3)

    def test_flip_nonzero_rmsd(self, any_cif):
        """Flipping/reflecting coordinates should give nonzero RMSD."""
        import copy

        polymer = ciffy.load(any_cif)

        flipped = copy.deepcopy(polymer)
        flipped.coordinates = polymer.coordinates * np.array([1, 1, -1])

        dist = rmsd(polymer, flipped, Scale.MOLECULE)
        assert dist > 0.1

    def test_numpy_backend(self, any_cif):
        """Test RMSD with NumPy backend explicitly."""
        polymer = ciffy.load(any_cif, backend="numpy")
        assert polymer.backend == "numpy"

        dist = rmsd(polymer, polymer)
        n_atoms = polymer.coordinates.shape[0]
        tolerance = max(1e-6, (n_atoms ** 0.5) * 1e-7 * 100)
        assert np.allclose(dist, 0, atol=tolerance)

    def test_default_scale(self, any_cif):
        """Test that rmsd defaults to MOLECULE scale."""
        polymer = ciffy.load(any_cif)

        dist_default = rmsd(polymer, polymer)
        dist_explicit = rmsd(polymer, polymer, Scale.MOLECULE)

        assert np.allclose(dist_default, dist_explicit)

    def test_identical_structures(self, any_cif):
        """Test RMSD of identical structures is zero."""
        polymer = ciffy.load(any_cif)
        dist = rmsd(polymer, polymer)

        n_atoms = polymer.coordinates.shape[0]
        tolerance = max(1e-6, (n_atoms ** 0.5) * 1e-7 * 100)
        assert np.allclose(dist, 0, atol=tolerance)

    def test_single_chain(self, any_cif):
        """Test RMSD works on single-chain polymers."""
        import copy

        polymer = ciffy.load(any_cif)
        chain = polymer.chain(0)

        perturbed = copy.deepcopy(chain)
        perturbed.coordinates = chain.coordinates + np.random.randn(*chain.coordinates.shape) * 0.1

        dist = rmsd(chain, perturbed)
        assert dist.shape == (1,)
        assert dist[0] > 0

    def test_chain_scale(self, any_cif):
        """Test RMSD at CHAIN scale."""
        import copy

        polymer = ciffy.load(any_cif)

        perturbed = copy.deepcopy(polymer)
        coords = perturbed.coordinates.copy()
        n_first_chain = polymer.counts(Scale.CHAIN)[0]
        coords[:n_first_chain] += np.random.randn(n_first_chain, 3) * 1.0
        perturbed.coordinates = coords

        dist = rmsd(polymer, perturbed, Scale.CHAIN)
        assert dist.shape[0] == polymer.size(Scale.CHAIN)
        assert dist[0] > dist[1:].mean()


# =============================================================================
# RMSD edge cases with small atom counts
# =============================================================================


class TestRMSDEdgeCases:
    """Edge case tests for rmsd with small atom counts.

    The Kabsch algorithm uses SVD on a 3x3 covariance matrix. With fewer than
    3 atoms, the covariance matrix is rank-deficient (degenerate), which can
    cause numerical instability.
    """

    @staticmethod
    def _create_small_polymer(n_atoms: int, backend: str = "numpy", seed: int = 42):
        """Create a minimal polymer with n_atoms for testing."""
        from ciffy import Polymer
        from ciffy.polymer.hierarchy import _Hierarchy
        from ciffy.polymer import Field

        rng = np.random.RandomState(seed)
        coords = rng.randn(n_atoms, 3).astype(np.float32) * 10.0

        atoms = np.zeros(n_atoms, dtype=np.int64)
        elements = np.ones(n_atoms, dtype=np.int64) * 8
        sequence = np.zeros(1, dtype=np.int64)

        sizes = {
            Scale.RESIDUE: np.array([n_atoms], dtype=np.int64),
            Scale.CHAIN: np.array([n_atoms], dtype=np.int64),
            Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
        }
        lengths = np.array([1], dtype=np.int64)

        hierarchy = _Hierarchy.from_sizes_and_lengths(
            sizes=sizes,
            lengths=lengths,
            ref=coords,
        )

        polymer = Polymer(
            hierarchy,
            coordinates=Field(coords, Scale.ATOM),
            atoms=Field(atoms, Scale.ATOM),
            elements=Field(elements, Scale.ATOM),
            sequence=Field(sequence, Scale.RESIDUE),
            pdb_id="test",
            names=["A"],
        )

        if backend == "torch":
            return polymer.torch()
        return polymer

    def test_single_atom_rmsd(self, backend):
        """rmsd with single atom should return exactly 0."""
        p = self._create_small_polymer(1, backend)

        dist = rmsd(p, p)

        assert dist.shape == (1,)
        if backend == "torch":
            assert dist.item() == 0.0
        else:
            assert dist[0] == 0.0

    def test_two_atom_rmsd(self, backend):
        """rmsd with two atoms should return ~0 for self-comparison."""
        p = self._create_small_polymer(2, backend)
        dist = rmsd(p, p)

        assert dist.shape == (1,)
        if backend == "torch":
            assert dist.item() < 0.02
        else:
            assert dist[0] < 1e-5

    def test_three_atom_rmsd(self, backend):
        """rmsd with three atoms should return ~0 for self-comparison."""
        p = self._create_small_polymer(3, backend)
        dist = rmsd(p, p)

        assert dist.shape == (1,)
        if backend == "torch":
            assert dist.item() < 0.02
        else:
            assert dist[0] < 1e-5
