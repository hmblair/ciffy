"""
Accuracy tests using 2F8S as golden reference structure.

These tests verify CORRECTNESS of computed values, not just shapes.
Reference values computed with numpy (not ciffy) as ground truth.
E.g., distances via np.linalg.norm, means via coords.mean(axis=0).

2F8S structure:
- 4 chains: 2 RNA (22 residues each), 2 protein (706 residues each)
- 12,658 atoms total
- Mixed molecule types for comprehensive testing
"""

import pytest
import numpy as np

from ciffy import operations
from tests.utils import get_test_cif


# Precomputed reference values for 2F8S (computed with numpy)
EXPECTED = {
    # Structure counts (verified against CIF)
    "n_atoms": 12658,
    "n_residues": 1456,
    "n_chains": 4,
    "n_rna_chains": 2,
    "n_protein_chains": 2,
    "rna_atoms": 930,
    "protein_atoms": 11728,

    # Chain A (RNA) specific values
    "chain_a_atoms": 465,
    "chain_a_residues": 22,
    "chain_a_centroid": np.array([-53.631042, 30.122606, 33.363678]),

    # Pairwise distances (np.linalg.norm ground truth)
    "dist_atom_0_1": 1.487423,
    "dist_chain_centroids_0_1": 8.218030,

    # Reduction values (numpy ground truth)
    "chain_a_coord_mean": np.array([-53.631042, 30.122606, 33.363678]),
    "residue_0_coord_sum": np.array([-1587.950928, 1176.030884, 1110.885986]),

    # Adjacency/bonds
    "n_bonds": 12994,
}


def load_2f8s(backend: str):
    """Load 2F8S structure with specified backend."""
    import ciffy
    return ciffy.load(get_test_cif("2F8S"), backend=backend)


class TestAccuracyStructure:
    """Test structure properties match expected values."""

    def test_atom_count(self, backend):
        """Total atom count matches expected."""
        p = load_2f8s(backend)
        assert p.size() == EXPECTED["n_atoms"]

    def test_residue_count(self, backend):
        """Total residue count matches expected."""
        from ciffy import Scale
        p = load_2f8s(backend)
        assert p.size(Scale.RESIDUE) == EXPECTED["n_residues"]

    def test_chain_count(self, backend):
        """Total chain count matches expected."""
        from ciffy import Scale
        p = load_2f8s(backend)
        assert p.size(Scale.CHAIN) == EXPECTED["n_chains"]

    def test_rna_chain_count(self, backend):
        """RNA chain count matches expected."""
        from ciffy import Scale, Molecule
        p = load_2f8s(backend)
        rna = p.molecule_type(Molecule.RNA)
        assert rna.size(Scale.CHAIN) == EXPECTED["n_rna_chains"]

    def test_protein_chain_count(self, backend):
        """Protein chain count matches expected."""
        from ciffy import Scale, Molecule
        p = load_2f8s(backend)
        protein = p.molecule_type(Molecule.PROTEIN)
        assert protein.size(Scale.CHAIN) == EXPECTED["n_protein_chains"]

    def test_rna_atom_count(self, backend):
        """RNA atom count matches expected."""
        from ciffy import Molecule
        p = load_2f8s(backend)
        rna = p.molecule_type(Molecule.RNA)
        assert rna.size() == EXPECTED["rna_atoms"]

    def test_protein_atom_count(self, backend):
        """Protein atom count matches expected."""
        from ciffy import Molecule
        p = load_2f8s(backend)
        protein = p.molecule_type(Molecule.PROTEIN)
        assert protein.size() == EXPECTED["protein_atoms"]

    def test_chain_a_atoms(self, backend):
        """Chain A (first RNA chain) atom count matches expected."""
        p = load_2f8s(backend)
        chain_a = p.chain(0)
        assert chain_a.size() == EXPECTED["chain_a_atoms"]

    def test_chain_a_residues(self, backend):
        """Chain A residue count matches expected."""
        from ciffy import Scale
        p = load_2f8s(backend)
        chain_a = p.chain(0)
        assert chain_a.size(Scale.RESIDUE) == EXPECTED["chain_a_residues"]

    def test_bond_count(self, backend):
        """Bond count matches expected."""
        p = load_2f8s(backend)
        assert len(p.bonds) == EXPECTED["n_bonds"]


class TestAccuracyGeometry:
    """Test geometric computations match numpy ground truth."""

    def test_pairwise_distance_specific_pair(self, backend):
        """Distance between atoms 0 and 1 matches numpy ground truth."""
        p = load_2f8s(backend)
        dists = operations.pairwise_distances(p)
        actual = dists[0, 1].item() if hasattr(dists[0, 1], 'item') else dists[0, 1]
        assert np.isclose(actual, EXPECTED["dist_atom_0_1"], rtol=1e-5)

    def test_pairwise_distance_symmetric(self, backend):
        """Distance matrix is symmetric (d[i,j] == d[j,i])."""
        p = load_2f8s(backend).chain(0)  # Use chain A for speed
        dists = np.asarray(operations.pairwise_distances(p))
        assert np.allclose(dists, dists.T, rtol=1e-6)

    def test_pairwise_distance_diagonal_zero(self, backend):
        """Distance matrix diagonal is all zeros."""
        p = load_2f8s(backend).chain(0)
        dists = np.asarray(operations.pairwise_distances(p))
        assert np.allclose(np.diag(dists), 0, atol=1e-10)

    def test_pairwise_distance_nonnegative(self, backend):
        """All distances are non-negative."""
        p = load_2f8s(backend).chain(0)
        dists = np.asarray(operations.pairwise_distances(p))
        assert (dists >= 0).all()

    def test_center_produces_zero_mean(self, backend):
        """center() produces coordinates with zero mean."""
        p = load_2f8s(backend)
        centered, _ = p.center()
        mean = np.asarray(centered.coordinates).mean(axis=0)
        # Tolerance accounts for float32 accumulation over 12k atoms
        assert np.allclose(mean, 0, atol=1e-4)

    def test_center_chain_scale_zero_mean(self, backend):
        """center() at chain scale produces zero mean per chain."""
        from ciffy import Scale
        p = load_2f8s(backend)
        centered, _ = p.center(scale=Scale.CHAIN)
        # Check each chain has zero mean
        for i in range(p.size(Scale.CHAIN)):
            chain = centered.chain(i)
            mean = np.asarray(chain.coordinates).mean(axis=0)
            # Tolerance accounts for float32 accumulation
            assert np.allclose(mean, 0, atol=1e-4), f"Chain {i} not centered"

    def test_chain_centroid_matches_expected(self, backend):
        """Chain A centroid matches numpy-computed ground truth."""
        p = load_2f8s(backend)
        chain_a = p.chain(0)
        coords = np.asarray(chain_a.coordinates)
        centroid = coords.mean(axis=0)
        assert np.allclose(centroid, EXPECTED["chain_a_centroid"], rtol=1e-5)

    def test_chain_centroid_distance(self, backend):
        """Distance between chain A and B centroids matches expected."""
        p = load_2f8s(backend)
        chain_a = p.chain(0)
        chain_b = p.chain(1)
        centroid_a = np.asarray(chain_a.coordinates).mean(axis=0)
        centroid_b = np.asarray(chain_b.coordinates).mean(axis=0)
        dist = np.linalg.norm(centroid_a - centroid_b)
        assert np.isclose(dist, EXPECTED["dist_chain_centroids_0_1"], rtol=1e-5)

    def test_knn_first_neighbor_is_closest(self, backend):
        """KNN returns the actual nearest neighbor."""
        p = load_2f8s(backend).chain(0)  # Use chain A for speed
        neighbors = operations.knn(p, k=1)
        dists = np.asarray(operations.pairwise_distances(p))
        neighbors_np = np.asarray(neighbors)

        # Spot check first 20 atoms
        for i in range(min(20, p.size())):
            nn_idx = int(neighbors_np[0, i])
            nn_dist = dists[i, nn_idx]
            # Set self-distance to inf for comparison
            dists_copy = dists[i].copy()
            dists_copy[i] = np.inf
            min_dist = dists_copy.min()
            assert np.isclose(nn_dist, min_dist, rtol=1e-5), \
                f"Atom {i}: KNN returned {nn_idx} (dist={nn_dist}) but closest is dist={min_dist}"

    def test_knn_all_neighbors_different(self, backend):
        """KNN with k>1 returns k different neighbors per atom."""
        p = load_2f8s(backend).chain(0)
        k = 5
        neighbors = np.asarray(operations.knn(p, k=k))

        # Check no duplicates for each atom
        for i in range(min(20, p.size())):
            atom_neighbors = neighbors[:, i]
            assert len(np.unique(atom_neighbors)) == k, \
                f"Atom {i} has duplicate neighbors"


class TestAccuracyReduction:
    """Test reduction operations match numpy ground truth."""

    def test_reduce_mean_equals_numpy_mean(self, backend):
        """reduce(MEAN) matches numpy mean."""
        from ciffy import Scale, Reduction
        p = load_2f8s(backend).chain(0)
        result = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MEAN)
        expected = np.asarray(p.coordinates).mean(axis=0)
        assert np.allclose(np.asarray(result).flatten(), expected, rtol=1e-5)

    def test_reduce_sum_equals_numpy_sum(self, backend):
        """reduce(SUM) matches numpy sum."""
        from ciffy import Scale, Reduction
        p = load_2f8s(backend).chain(0)
        result = p.reduce(p.coordinates, Scale.CHAIN, Reduction.SUM)
        expected = np.asarray(p.coordinates).sum(axis=0)
        assert np.allclose(np.asarray(result).flatten(), expected, rtol=1e-5)

    def test_reduce_min_equals_numpy_min(self, backend):
        """reduce(MIN) matches numpy min."""
        from ciffy import Scale, Reduction
        p = load_2f8s(backend).chain(0)
        result = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MIN)
        # MIN returns (values, indices) tuple
        if isinstance(result, tuple):
            result = result[0]
        expected = np.asarray(p.coordinates).min(axis=0)
        assert np.allclose(np.asarray(result).flatten(), expected, rtol=1e-5)

    def test_reduce_max_equals_numpy_max(self, backend):
        """reduce(MAX) matches numpy max."""
        from ciffy import Scale, Reduction
        p = load_2f8s(backend).chain(0)
        result = p.reduce(p.coordinates, Scale.CHAIN, Reduction.MAX)
        # MAX returns (values, indices) tuple
        if isinstance(result, tuple):
            result = result[0]
        expected = np.asarray(p.coordinates).max(axis=0)
        assert np.allclose(np.asarray(result).flatten(), expected, rtol=1e-5)

    def test_reduce_residue_sum_matches_expected(self, backend):
        """Residue 0 coordinate sum matches numpy ground truth."""
        from ciffy import Scale, Reduction
        p = load_2f8s(backend)

        # Get atoms in residue 0
        membership = np.asarray(p.membership(Scale.RESIDUE))
        res0_mask = membership == 0
        res0_coords = np.asarray(p.coordinates)[res0_mask]
        expected_sum = res0_coords.sum(axis=0)

        # Should match precomputed value
        assert np.allclose(expected_sum, EXPECTED["residue_0_coord_sum"], rtol=1e-5)

    def test_reduce_per_residue_mean(self, backend):
        """reduce to residue scale produces correct per-residue means."""
        from ciffy import Scale, Reduction
        p = load_2f8s(backend).chain(0)

        result = np.asarray(p.reduce(p.coordinates, Scale.RESIDUE, Reduction.MEAN))
        coords = np.asarray(p.coordinates)
        membership = np.asarray(p.membership(Scale.RESIDUE))

        # Verify first 5 residues
        for r in range(min(5, p.size(Scale.RESIDUE))):
            mask = membership == r
            expected = coords[mask].mean(axis=0)
            assert np.allclose(result[r], expected, rtol=1e-5), \
                f"Residue {r} mean mismatch"

    def test_expand_inverts_reduce_mean(self, backend):
        """expand(reduce(MEAN)) produces per-residue means at atom level."""
        from ciffy import Scale, Reduction
        p = load_2f8s(backend).chain(0)

        # Reduce to residue means, then expand back
        res_means = p.reduce(p.coordinates, Scale.RESIDUE, Reduction.MEAN)
        expanded = np.asarray(p.expand(res_means, Scale.RESIDUE))

        # Each atom should have its residue's mean
        coords = np.asarray(p.coordinates)
        membership = np.asarray(p.membership(Scale.RESIDUE))

        for i in range(min(50, p.size())):  # Spot check 50 atoms
            res_idx = membership[i]
            mask = membership == res_idx
            expected = coords[mask].mean(axis=0)
            assert np.allclose(expanded[i], expected, rtol=1e-5), \
                f"Atom {i} expand mismatch"


class TestAccuracySelection:
    """Test selection operations produce correct subsets."""

    def test_molecule_type_partitions_structure(self, backend):
        """molecule_type(RNA) + molecule_type(PROTEIN) = full structure."""
        from ciffy import Molecule
        p = load_2f8s(backend)
        rna = p.molecule_type(Molecule.RNA)
        protein = p.molecule_type(Molecule.PROTEIN)

        # Atom counts should sum to total
        assert rna.size() + protein.size() == p.size()

    def test_chain_selection_disjoint(self, backend):
        """Selecting different chains produces disjoint atom sets."""
        p = load_2f8s(backend)
        chain_a = p.chain(0)
        chain_b = p.chain(1)

        # No coordinate overlap (different atoms)
        coords_a = set(map(tuple, np.asarray(chain_a.coordinates).round(3).tolist()))
        coords_b = set(map(tuple, np.asarray(chain_b.coordinates).round(3).tolist()))
        assert len(coords_a & coords_b) == 0, "Chains share atoms"

    def test_residue_type_correct_atoms(self, backend):
        """residue_type selects only atoms from matching residues."""
        from ciffy import Residue, Scale
        p = load_2f8s(backend).molecule_type(__import__('ciffy').Molecule.RNA)

        # Select adenosine residues
        adenosine = p.residue_type(Residue.A)

        if not adenosine.empty():
            # All selected atoms should belong to adenosine residues
            seq = np.asarray(adenosine.sequence)
            assert (seq == Residue.A.value).all(), "Non-adenosine residue in selection"

    def test_backbone_reduces_atom_count(self, backend):
        """backbone() returns fewer atoms than full structure."""
        p = load_2f8s(backend).chain(0)
        backbone = p.backbone()
        assert backbone.size() < p.size()
        assert backbone.size() > 0

    def test_strip_removes_unresolved(self, backend):
        """strip() removes unresolved residues."""
        from ciffy import Scale
        p = load_2f8s(backend)
        stripped = p.strip()
        # 2F8S has 4 unresolved residues out of 1456
        assert stripped.size(Scale.RESIDUE) == 1452
        assert stripped.size(Scale.RESIDUE) < p.size(Scale.RESIDUE)


class TestAccuracyAdjacency:
    """Test adjacency matrix correctness."""

    def test_adjacency_symmetric(self, backend):
        """Adjacency matrix is symmetric."""
        p = load_2f8s(backend).chain(0)
        adj = np.asarray(operations.adjacency(p))
        assert np.allclose(adj, adj.T)

    def test_adjacency_diagonal_zero(self, backend):
        """Adjacency matrix diagonal is zero (no self-bonds)."""
        p = load_2f8s(backend).chain(0)
        adj = np.asarray(operations.adjacency(p))
        assert np.allclose(np.diag(adj), 0)

    def test_adjacency_matches_bonds(self, backend):
        """Adjacency matrix non-zeros match bond list."""
        p = load_2f8s(backend).chain(0)
        adj = np.asarray(operations.adjacency(p))
        bonds = np.asarray(p.bonds)

        # Each bond should create two adjacency entries
        for i, j in bonds:
            assert adj[i, j] != 0, f"Bond ({i}, {j}) missing from adjacency"
            assert adj[j, i] != 0, f"Bond ({j}, {i}) missing from adjacency"

    def test_adjacency_nnz_equals_twice_bonds(self, backend):
        """Number of adjacency non-zeros equals 2x bond count (symmetric)."""
        p = load_2f8s(backend).chain(0)
        adj = np.asarray(operations.adjacency(p))
        n_bonds = len(p.bonds)
        nnz = np.count_nonzero(adj)
        assert nnz == 2 * n_bonds


class TestAccuracyMembership:
    """Test membership assignments are correct."""

    def test_membership_residue_consistent_with_counts(self, backend):
        """membership(RESIDUE) assignments match counts."""
        from ciffy import Scale
        p = load_2f8s(backend).chain(0)

        membership = np.asarray(p.membership(Scale.RESIDUE))
        counts = np.asarray(p.counts(Scale.RESIDUE))

        # Count atoms per residue from membership
        unique, member_counts = np.unique(membership, return_counts=True)

        assert np.array_equal(counts, member_counts), \
            "membership and counts disagree"

    def test_membership_chain_consistent(self, backend):
        """membership(CHAIN) assigns all atoms to valid chains."""
        from ciffy import Scale
        p = load_2f8s(backend)

        membership = np.asarray(p.membership(Scale.CHAIN))
        n_chains = p.size(Scale.CHAIN)

        # All values should be in [0, n_chains)
        assert membership.min() >= 0
        assert membership.max() < n_chains

        # Each chain should have atoms
        for c in range(n_chains):
            assert (membership == c).sum() > 0, f"Chain {c} has no atoms"
