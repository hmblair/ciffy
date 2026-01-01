"""Tests for connection (hydrogen bond) loading from _struct_conn block."""

import numpy as np
import pytest
import ciffy
from ciffy._c import _load


class TestConnectionLoading:
    """Tests for loading connections from CIF files."""

    def test_connections_not_loaded_by_default(self):
        """Connections should not be loaded unless explicitly requested."""
        data = _load("tests/data/9MDS.cif", skip="descriptions")
        assert "connections" not in data
        assert "connection_types" not in data

    def test_connections_loaded_when_requested(self):
        """Connections should be loaded when connections=True."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        assert "connections" in data
        assert "connection_types" in data

    def test_connections_shape(self):
        """Connections should be (N, 2) array of atom index pairs."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        connections = data["connections"]

        assert connections.ndim == 2
        assert connections.shape[1] == 2
        assert connections.dtype == np.int32

    def test_connection_types_shape(self):
        """Connection types should be (N,) array matching connections."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        connections = data["connections"]
        conn_types = data["connection_types"]

        assert conn_types.ndim == 1
        assert len(conn_types) == len(connections)
        assert conn_types.dtype == np.int32

    def test_connection_indices_valid(self):
        """All connection atom indices should be within valid range."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        connections = data["connections"]
        n_atoms = data["atoms_per_chain"].sum()

        assert connections.min() >= 0
        assert connections.max() < n_atoms

    def test_connection_types_valid(self):
        """Connection types should be valid enum values (1-4)."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        conn_types = data["connection_types"]

        # ConnType enum: HYDROG=1, COVALE=2, METALC=3, DISULF=4, UNKNOWN=0
        assert conn_types.min() >= 0
        assert conn_types.max() <= 4

    def test_9mds_has_expected_connections(self):
        """9MDS should have ~4404 hydrogen bond connections."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        connections = data["connections"]

        # 9MDS is an RNA structure with many base pair hydrogen bonds
        assert len(connections) == 4404

    def test_connections_are_hydrogen_bonds(self):
        """9MDS connections should be primarily hydrogen bonds (type 1)."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        conn_types = data["connection_types"]

        # All connections in 9MDS should be hydrogen bonds
        assert np.all(conn_types == 1)

    def test_connection_pairs_are_distinct(self):
        """Connection pairs should have distinct atom indices."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        connections = data["connections"]

        # No self-connections
        assert np.all(connections[:, 0] != connections[:, 1])

    def test_no_connections_when_block_missing(self):
        """Files without _struct_conn should return empty connections."""
        # 1ZEW may not have _struct_conn block or have fewer connections
        data = _load("tests/data/1ZEW.cif", skip="descriptions", connections=True)

        # Should not crash, may have 0 or some connections
        if "connections" in data:
            assert data["connections"].shape[1] == 2


class TestConnectionCorrectness:
    """Tests verifying connections point to correct atoms."""

    def test_hydrogen_bond_distances(self):
        """Hydrogen bond connections should have distances in valid range (2-4 Å)."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        coords = data["coordinates"]
        connections = data["connections"]

        # Compute distances for all connections
        atom1_coords = coords[connections[:, 0]]
        atom2_coords = coords[connections[:, 1]]
        distances = np.linalg.norm(atom1_coords - atom2_coords, axis=1)

        # Hydrogen bonds should be 2.5-3.5 Å, allow 2-4 Å for flexibility
        assert distances.min() > 2.0, f"Min distance {distances.min():.2f} Å too short"
        assert distances.max() < 4.0, f"Max distance {distances.max():.2f} Å too long"

        # Most should be in the 2.7-3.2 Å range
        typical_range = (distances > 2.5) & (distances < 3.5)
        pct_typical = typical_range.sum() / len(distances) * 100
        assert pct_typical > 90, f"Only {pct_typical:.0f}% in typical H-bond range"

    def test_connections_between_different_residues(self):
        """Base pair H-bonds should connect atoms from different residues."""
        polymer = ciffy.load("tests/data/9MDS.cif", connections=True)
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        connections = data["connections"]

        # Get residue membership for each atom
        residue_idx = polymer.membership(ciffy.Scale.RESIDUE)

        # Check that connected atoms are in different residues
        res1 = residue_idx[connections[:, 0]]
        res2 = residue_idx[connections[:, 1]]

        # All H-bonds should be between different residues
        assert np.all(res1 != res2), "Found H-bond within same residue"

    def test_connections_involve_expected_elements(self):
        """H-bond atoms should be N, O, or sometimes C (for C-H...O bonds)."""
        data = _load("tests/data/9MDS.cif", skip="descriptions", connections=True)
        elements = data["elements"]
        connections = data["connections"]

        # Element indices: N=7, O=8, C=6
        N, O, C = 7, 8, 6
        valid_elements = {N, O, C}

        elem1 = elements[connections[:, 0]]
        elem2 = elements[connections[:, 1]]

        # At least one atom in each pair should be N or O (H-bond donor/acceptor)
        has_n_or_o = ((elem1 == N) | (elem1 == O) | (elem2 == N) | (elem2 == O))
        pct_valid = has_n_or_o.sum() / len(connections) * 100

        assert pct_valid > 99, f"Only {pct_valid:.1f}% involve N or O"
