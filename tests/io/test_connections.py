"""Tests for connection (hydrogen bond) loading from _struct_conn block."""

import numpy as np
import pytest
import ciffy
from tests.utils import get_test_cif


class TestConnectionLoading:
    """Tests for loading connections from CIF files."""

    def test_connections_not_loaded_by_default(self, cif_9mds):
        """Connections should not be loaded unless explicitly requested."""
        polymer = ciffy.load(cif_9mds)
        assert polymer.connections is None
        assert polymer.connection_types is None

    def test_connections_loaded_when_requested(self, cif_9mds):
        """Connections should be loaded when not in skip list."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        assert polymer.connections is not None
        assert polymer.connection_types is not None

    def test_connections_shape(self, cif_9mds):
        """Connections should be (N, 2) array of atom index pairs."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        connections = polymer.connections

        assert connections.ndim == 2
        assert connections.shape[1] == 2
        assert connections.dtype == np.int32

    def test_connection_types_shape(self, cif_9mds):
        """Connection types should be (N,) array matching connections."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        connections = polymer.connections
        conn_types = polymer.connection_types

        assert conn_types.ndim == 1
        assert len(conn_types) == len(connections)
        assert conn_types.dtype == np.int32

    def test_connection_indices_valid(self, cif_9mds):
        """All connection atom indices should be within valid range."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        connections = polymer.connections
        n_atoms = polymer.size()

        assert connections.min() >= 0
        assert connections.max() < n_atoms

    def test_connection_types_valid(self, cif_9mds):
        """Connection types should be valid enum values (1-4)."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        conn_types = polymer.connection_types

        # ConnType enum: HYDROG=1, COVALE=2, METALC=3, DISULF=4, UNKNOWN=0
        assert conn_types.min() >= 0
        assert conn_types.max() <= 4

    def test_9mds_has_expected_connections(self, cif_9mds):
        """9MDS should have ~4404 hydrogen bond connections."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        connections = polymer.connections

        # 9MDS is an RNA structure with many base pair hydrogen bonds
        assert len(connections) == 4404

    def test_connections_are_hydrogen_bonds(self, cif_9mds):
        """9MDS connections should be primarily hydrogen bonds (type 1)."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        conn_types = polymer.connection_types

        # All connections in 9MDS should be hydrogen bonds
        assert np.all(conn_types == 1)

    def test_connection_pairs_are_distinct(self, cif_9mds):
        """Connection pairs should have distinct atom indices."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        connections = polymer.connections

        # No self-connections
        assert np.all(connections[:, 0] != connections[:, 1])

    def test_no_connections_when_block_missing(self):
        """Files without _struct_conn should return empty or None connections."""
        # 1ZEW may not have _struct_conn block or have fewer connections
        cif_1zew = get_test_cif("1ZEW")
        polymer = ciffy.load(cif_1zew, skip=["descriptions"])

        # Should not crash, may have 0 or some connections
        if polymer.connections is not None:
            assert polymer.connections.shape[1] == 2


class TestConnectionCorrectness:
    """Tests verifying connections point to correct atoms."""

    def test_hydrogen_bond_distances(self, cif_9mds):
        """Hydrogen bond connections should have distances in valid range (2-4 Å)."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        coords = polymer.coordinates
        connections = polymer.connections

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

    def test_connections_between_different_residues(self, cif_9mds):
        """Base pair H-bonds should connect atoms from different residues."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        connections = polymer.connections

        # Get residue membership for each atom
        residue_idx = polymer.membership(ciffy.Scale.RESIDUE)

        # Check that connected atoms are in different residues
        res1 = residue_idx[connections[:, 0]]
        res2 = residue_idx[connections[:, 1]]

        # All H-bonds should be between different residues
        assert np.all(res1 != res2), "Found H-bond within same residue"

    def test_connections_involve_expected_elements(self, cif_9mds):
        """H-bond atoms should be N, O, or sometimes C (for C-H...O bonds)."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        elements = polymer.elements
        connections = polymer.connections

        # Element indices: N=7, O=8, C=6
        N, O, C = 7, 8, 6

        elem1 = elements[connections[:, 0]]
        elem2 = elements[connections[:, 1]]

        # At least one atom in each pair should be N or O (H-bond donor/acceptor)
        has_n_or_o = ((elem1 == N) | (elem1 == O) | (elem2 == N) | (elem2 == O))
        pct_valid = has_n_or_o.sum() / len(connections) * 100

        assert pct_valid > 99, f"Only {pct_valid:.1f}% involve N or O"


class TestConnectionsAfterSelection:
    """Tests verifying connections are correctly remapped after selection."""

    def test_chain_selection_remaps_connections(self, cif_9mds):
        """Connections should be filtered and remapped after chain selection."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        original_n_atoms = polymer.size()

        # Select first chain
        chain0 = polymer.chain(0)

        # Connections should exist (chain has internal H-bonds)
        if chain0.connections is not None:
            # All indices should be valid for new atom count
            assert chain0.connections.max() < chain0.size()
            assert chain0.connections.min() >= 0

            # Connection types should match connection count
            if chain0.connection_types is not None:
                assert len(chain0.connection_types) == len(chain0.connections)

    def test_residue_selection_remaps_connections(self, cif_9mds):
        """Connections should be filtered and remapped after residue selection."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])

        # Select first 10 residues
        selected = polymer.residue(list(range(10)))

        # After selection, any remaining connections should have valid indices
        if selected.connections is not None:
            n_atoms = selected.size()
            assert selected.connections.min() >= 0
            assert selected.connections.max() < n_atoms

    def test_atom_selection_remaps_connections(self, cif_9mds):
        """Connections should be filtered and remapped after atom selection."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        n_atoms = polymer.size()

        # Select first half of atoms
        mask = np.zeros(n_atoms, dtype=bool)
        mask[:n_atoms // 2] = True
        selected = polymer[mask]

        # After selection, any remaining connections should have valid indices
        if selected.connections is not None:
            assert selected.connections.min() >= 0
            assert selected.connections.max() < selected.size()

    def test_connections_filtered_when_atoms_removed(self, cif_9mds):
        """Connections involving removed atoms should be excluded."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])
        original_connections = polymer.connections
        original_count = len(original_connections)

        # Select only a subset of atoms (first chain)
        chain0 = polymer.chain(0)

        # Should have fewer connections (or none)
        if chain0.connections is not None:
            assert len(chain0.connections) <= original_count
        # If None, that's also valid (no connections in selected region)

    def test_connection_distances_preserved_after_selection(self, cif_9mds):
        """H-bond distances should remain valid after selection."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])

        # Select first chain
        chain0 = polymer.chain(0)

        if chain0.connections is not None and len(chain0.connections) > 0:
            coords = chain0.coordinates
            connections = chain0.connections

            # Compute distances
            atom1_coords = coords[connections[:, 0]]
            atom2_coords = coords[connections[:, 1]]
            distances = np.linalg.norm(atom1_coords - atom2_coords, axis=1)

            # Should still be valid H-bond distances
            assert distances.min() > 2.0
            assert distances.max() < 4.0

    def test_multiple_selections_remap_correctly(self, cif_9mds):
        """Chained selections should maintain valid connections."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])

        # Chain selections: chain(0) -> residue(first 5)
        chain0 = polymer.chain(0)
        subset = chain0.residue(list(range(min(5, chain0.size(ciffy.Scale.RESIDUE)))))

        if subset.connections is not None:
            # All indices valid
            assert subset.connections.min() >= 0
            assert subset.connections.max() < subset.size()

    def test_connections_none_when_no_skip(self, cif_9mds):
        """Without skip=[], connections should be None after selection too."""
        polymer = ciffy.load(cif_9mds)  # Default skip, no connections
        assert polymer.connections is None

        chain0 = polymer.chain(0)
        assert chain0.connections is None  # Should remain None

    def test_contiguous_selection_remaps_connections(self, cif_9mds):
        """Single-index chain/residue selection (contiguous path) should remap."""
        polymer = ciffy.load(cif_9mds, skip=["descriptions"])

        # Single chain (uses _select_contiguous fast path)
        chain0 = polymer.chain(0)
        if chain0.connections is not None:
            assert chain0.connections.max() < chain0.size()

        # Single residue
        res0 = polymer.residue(0)
        # Single residue unlikely to have H-bonds, but should not crash
        if res0.connections is not None:
            assert res0.connections.max() < res0.size()
