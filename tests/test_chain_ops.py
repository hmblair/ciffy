"""Tests for chain operations (join and extend)."""

import numpy as np
import pytest

import ciffy
from ciffy import Residue, Scale, join
from ciffy.template import from_sequence


class TestJoin:
    """Tests for ciffy.join() function."""

    def test_join_two_polymers(self):
        """Join two single-chain RNA polymers."""
        p1 = from_sequence("ac")
        p2 = from_sequence("gu")

        combined = join(p1, p2)

        # Should have 2 chains
        assert combined.size(Scale.CHAIN) == 2
        # Should have 4 residues total
        assert combined.size(Scale.RESIDUE) == 4
        # Should have all atoms from both
        assert combined.size() == p1.size() + p2.size()
        # Sequence should be concatenated
        assert combined.sequence_str() == "acgu"
        # Should have both chain names
        assert len(combined.names) == 2

    def test_join_three_polymers(self):
        """Join three polymers."""
        p1 = from_sequence("a")
        p2 = from_sequence("c")
        p3 = from_sequence("g")

        combined = join(p1, p2, p3)

        assert combined.size(Scale.CHAIN) == 3
        assert combined.size(Scale.RESIDUE) == 3
        assert combined.sequence_str() == "acg"

    def test_join_multichain_polymers(self):
        """Join polymers that already have multiple chains."""
        p1 = from_sequence(["ac", "gu"])  # 2 chains
        p2 = from_sequence("aa")  # 1 chain

        combined = join(p1, p2)

        assert combined.size(Scale.CHAIN) == 3
        assert combined.size(Scale.RESIDUE) == 6

    def test_join_single_polymer(self):
        """Join with single polymer returns a copy."""
        p = from_sequence("acgu")

        combined = join(p)

        assert combined.size(Scale.CHAIN) == 1
        assert combined.size(Scale.RESIDUE) == 4
        # Should be a copy, not the same object
        assert combined is not p
        # Modifying one shouldn't affect the other
        assert np.allclose(combined.coordinates, p.coordinates)

    def test_join_empty_polymers(self):
        """Join with empty polymers skips them."""
        p1 = from_sequence("ac")
        p_empty = from_sequence("")

        combined = join(p1, p_empty)

        assert combined.size(Scale.CHAIN) == 1
        assert combined.size(Scale.RESIDUE) == 2

    def test_join_all_empty(self):
        """Join all empty polymers returns empty."""
        p1 = from_sequence("")
        p2 = from_sequence("")

        combined = join(p1, p2)

        assert combined.empty()
        assert combined.size(Scale.CHAIN) == 0

    def test_join_no_polymers_error(self):
        """Join with no arguments raises error."""
        with pytest.raises(ValueError, match="at least one polymer"):
            join()

    def test_join_hetatm_error(self):
        """Join with HETATM atoms raises error."""
        p1 = from_sequence("ac")
        p2 = ciffy.load("tests/data/9MDS.cif")

        # Skip if test structure doesn't have HETATM
        if p2.nonpoly == 0:
            pytest.skip("Test structure has no HETATM atoms")

        # Should fail because p2 has HETATM
        with pytest.raises(ValueError, match="poly-only"):
            join(p1, p2)

    def test_join_preserves_pdb_id_when_same(self):
        """Join preserves PDB ID when all are the same."""
        p1 = from_sequence("ac", id="TEST")
        p2 = from_sequence("gu", id="TEST")

        combined = join(p1, p2)

        assert combined.pdb_id == "TEST"

    def test_join_different_pdb_ids_becomes_joined(self):
        """Join with different PDB IDs becomes 'joined'."""
        p1 = from_sequence("ac", id="TEST1")
        p2 = from_sequence("gu", id="TEST2")

        combined = join(p1, p2)

        assert combined.pdb_id == "joined"

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_join_backend_preserved(self, backend):
        """Join preserves backend."""
        p1 = from_sequence("ac", backend=backend)
        p2 = from_sequence("gu", backend=backend)

        combined = join(p1, p2)

        assert combined.backend == backend


class TestExtend:
    """Tests for Polymer.extend() method."""

    def test_extend_rna(self):
        """Extend RNA chain with a new residue."""
        p = from_sequence("ac")
        initial_size = p.size()
        initial_res = p.size(Scale.RESIDUE)

        # Extend with guanine (simple API)
        extended = p.extend(Residue.G)

        # Should have one more residue
        assert extended.size(Scale.RESIDUE) == initial_res + 1
        # Should have more atoms
        assert extended.size() > initial_size
        # Sequence should be updated
        assert extended.sequence_str() == "acg"
        # Still single chain
        assert extended.size(Scale.CHAIN) == 1

    def test_extend_preserves_original(self):
        """Extend returns new polymer, original unchanged."""
        p = from_sequence("ac")
        original_size = p.size()
        original_seq = p.sequence_str()

        extended = p.extend(Residue.G)

        # Original should be unchanged
        assert p.size() == original_size
        assert p.sequence_str() == original_seq

        # Extended should be different
        assert extended is not p
        assert extended.size() > original_size

    def test_extend_chain_multiple(self):
        """Extend chain multiple times."""
        p = from_sequence("a")

        # Extend with c, g, u (simple API)
        for residue in [Residue.C, Residue.G, Residue.U]:
            p = p.extend(residue)

        assert p.size(Scale.RESIDUE) == 4
        assert p.sequence_str() == "acgu"

    def test_extend_multichain_error(self):
        """Extend fails on multi-chain polymer."""
        p = from_sequence(["ac", "gu"])

        with pytest.raises(ValueError, match="single-chain"):
            p.extend(Residue.A)

    def test_extend_hetatm_error(self):
        """Extend fails on polymer with HETATM."""
        p = ciffy.load("tests/data/9MDS.cif")

        # Skip if test structure doesn't have HETATM
        if p.nonpoly == 0:
            pytest.skip("Test structure has no HETATM atoms")

        with pytest.raises(ValueError, match="poly-only"):
            p.extend(Residue.A)

    def test_extend_with_custom_coords(self):
        """Extend with explicit coordinates."""
        p = from_sequence("ac")

        # Use custom coordinates (same as ideal for test)
        custom_coords = Residue.G.ideal.copy()
        extended = p.extend(Residue.G, coords=custom_coords)

        assert extended.size(Scale.RESIDUE) == 3
        assert extended.sequence_str() == "acg"

    def test_extend_wrong_atom_count_error(self):
        """Extend fails if coord shape doesn't match residue."""
        p = from_sequence("ac")

        # Wrong number of atoms (only 3 instead of full residue)
        wrong_coords = np.zeros((3, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Coordinate shape"):
            p.extend(Residue.G, coords=wrong_coords)

    def test_extend_residue_spacing(self):
        """Extended residues are spaced correctly along Z-axis."""
        p = from_sequence("a")
        extended = p.extend(Residue.C)

        # Get centroids of each residue
        coords = extended.coordinates
        res_sizes = extended._sizes[Scale.RESIDUE]
        first_res_atoms = res_sizes[0].item()

        first_centroid = coords[:first_res_atoms].mean(axis=0)
        second_centroid = coords[first_res_atoms:].mean(axis=0)

        # Residues should be spaced appropriately along Z-axis (not overlapping)
        z_spacing = second_centroid[2] - first_centroid[2]
        assert z_spacing > 5.0, f"Z spacing {z_spacing:.2f}Å too small (residues may clash)"

        # X and Y should be roughly the same (extending along Z)
        xy_drift = np.sqrt((second_centroid[0] - first_centroid[0])**2 +
                          (second_centroid[1] - first_centroid[1])**2)
        assert xy_drift < 1.0, f"XY drift {xy_drift:.2f}Å too large (not linear)"

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_extend_backend_preserved(self, backend):
        """Extend preserves backend."""
        pytest.importorskip("torch")

        p = from_sequence("ac", backend=backend)

        # Simple API should handle backend conversion automatically
        extended = p.extend(Residue.G)

        assert extended.backend == backend


class TestJoinAndExtendIntegration:
    """Integration tests combining join and extend."""

    def test_extend_then_join(self):
        """Extend chain then join with another."""
        p1 = from_sequence("a")
        p1 = p1.extend(Residue.C)

        p2 = from_sequence("gu")

        combined = join(p1, p2)

        assert combined.size(Scale.CHAIN) == 2
        assert combined.size(Scale.RESIDUE) == 4
        assert combined.sequence_str() == "acgu"

    def test_join_then_iterate_chains(self):
        """Join polymers then iterate over chains."""
        p1 = from_sequence("ac")
        p2 = from_sequence("gu")

        combined = join(p1, p2)

        chains = list(combined.chains())
        assert len(chains) == 2
        assert chains[0].sequence_str() == "ac"
        assert chains[1].sequence_str() == "gu"
