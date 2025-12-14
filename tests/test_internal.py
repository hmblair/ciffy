"""Tests for internal coordinates (Z-matrix) representation."""

import pytest
import numpy as np

from tests.utils import get_test_cif


class TestInternalCoordinatesBasic:
    """Basic tests for internal coordinate conversion."""

    def test_single_nucleotide_roundtrip(self):
        """Test round-trip for single nucleotide."""
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("a")
        orig_coords = polymer.coordinates.copy()

        # Access internal coordinates, then modify to trigger reconstruction
        dihedrals = polymer.dihedrals.copy()
        polymer.dihedrals = dihedrals

        # Coordinates are now reconstructed
        aligned, _, _ = kabsch_align(polymer.coordinates, orig_coords)
        rmsd = np.sqrt(((aligned - orig_coords) ** 2).sum(axis=1).mean())

        assert rmsd < 1e-5, f"RMSD {rmsd} exceeds threshold"

    def test_rna_tetramer_roundtrip(self):
        """Test round-trip for RNA tetramer."""
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu")
        orig_coords = polymer.coordinates.copy()

        # Access internal coordinates, then modify to trigger reconstruction
        dihedrals = polymer.dihedrals.copy()
        polymer.dihedrals = dihedrals

        # Coordinates are now reconstructed
        aligned, _, _ = kabsch_align(polymer.coordinates, orig_coords)
        rmsd = np.sqrt(((aligned - orig_coords) ** 2).sum(axis=1).mean())

        assert rmsd < 1e-4, f"RMSD {rmsd} exceeds threshold"

    def test_internal_polymer_properties(self):
        """Test Polymer has correct internal coordinate properties."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access internal coordinates
        distances = polymer.distances
        angles = polymer.angles
        dihedrals = polymer.dihedrals

        # Check array shapes
        n_zmatrix = len(polymer._coord_manager._zmatrix)
        assert distances.shape == (n_zmatrix,)
        assert angles.shape == (n_zmatrix,)
        assert dihedrals.shape == (n_zmatrix,)

    def test_distances_are_positive(self):
        """Test all bond distances are positive."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        distances = polymer.distances
        zmatrix = polymer._coord_manager._zmatrix

        # First entry has no distance (root atom)
        for i, entry in enumerate(zmatrix):
            if entry.distance_ref >= 0:
                assert distances[i] > 0, f"Distance at {i} is not positive"

    def test_angles_in_valid_range(self):
        """Test all bond angles are in [0, pi]."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        angles = polymer.angles
        zmatrix = polymer._coord_manager._zmatrix

        for i, entry in enumerate(zmatrix):
            if entry.angle_ref >= 0:
                angle = angles[i]
                assert 0 <= angle <= np.pi, f"Angle at {i} is {angle}, not in [0, pi]"

    def test_dihedrals_in_valid_range(self):
        """Test all dihedral angles are in [-pi, pi]."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        dihedrals = polymer.dihedrals
        zmatrix = polymer._coord_manager._zmatrix

        for i, entry in enumerate(zmatrix):
            if entry.dihedral_ref >= 0:
                dih = dihedrals[i]
                assert -np.pi <= dih <= np.pi, f"Dihedral at {i} is {dih}, not in [-pi, pi]"


class TestInternalCoordinatesPDB:
    """Tests using real PDB structures."""

    def test_multichain_relative_orientation(self):
        """Test multi-chain reconstruction preserves relative chain positions and orientations."""
        from ciffy import load, rmsd

        polymer = load(get_test_cif("1ZEW")).poly()

        # Verify this is actually a multi-chain structure
        n_chains = len(polymer.lengths)
        assert n_chains > 1, "Test requires multi-chain structure"

        # Save original polymer
        orig_polymer = polymer.with_coordinates(polymer.coordinates.copy())

        # Access internal coordinates to trigger computation
        dihedrals = polymer.dihedrals.copy()

        # Modify dihedrals to trigger reconstruction (set back to same values)
        polymer.dihedrals = dihedrals

        # Test 1: Per-chain RMSD should be good (each chain's internal structure preserved)
        for chain_idx, chain in enumerate(polymer.chains()):
            orig_chain = list(orig_polymer.chains())[chain_idx]

            # Per-chain alignment - should work because internal structure is preserved
            chain_rmsd = float(rmsd(orig_chain, chain))
            assert chain_rmsd < 1e-4, \
                f"Chain {chain_idx} internal structure RMSD {chain_rmsd:.6f} exceeds threshold"

        # Test 2: Global RMSD should fail (relative chain positions/orientations not preserved)
        global_rmsd = rmsd(orig_polymer, polymer)
        global_rmsd_val = float(global_rmsd)
        assert global_rmsd_val < 1e-4, \
            f"Global RMSD {global_rmsd_val:.6f} exceeds threshold - relative chain orientations not preserved"

    def test_rna_structure_per_chain(self):
        """Test round-trip for RNA structure (per-chain RMSD)."""
        from ciffy import load, rmsd

        polymer = load(get_test_cif("1ZEW")).poly()
        orig_polymer = polymer.with_coordinates(polymer.coordinates.copy())

        # Trigger reconstruction
        dihedrals = polymer.dihedrals.copy()
        polymer.dihedrals = dihedrals

        # Test per-chain RMSD
        for chain_idx, chain in enumerate(polymer.chains()):
            orig_chain = list(orig_polymer.chains())[chain_idx]
            chain_rmsd = float(rmsd(orig_chain, chain))

            assert chain_rmsd < 1e-4, f"Chain {chain_idx} RMSD {chain_rmsd} exceeds threshold"


class TestInternalCoordinatesTorchBackend:
    """Tests for PyTorch backend."""

    def test_torch_roundtrip(self):
        """Test round-trip with torch backend."""
        import torch
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu", backend="torch")
        orig_coords = polymer.coordinates.clone()

        # Trigger reconstruction
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals

        assert isinstance(polymer.coordinates, torch.Tensor)

        aligned, _, _ = kabsch_align(polymer.coordinates, orig_coords)
        rmsd = torch.sqrt(((aligned - orig_coords) ** 2).sum(dim=1).mean())

        assert rmsd.item() < 1e-4, f"RMSD {rmsd.item()} exceeds threshold"

    def test_torch_roundtrip_preserves_device_and_dtype(self):
        """Ensure internal coordinates preserve device/dtype."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        coords = polymer.coordinates

        # Access internal coordinates
        distances = polymer.distances
        assert isinstance(distances, torch.Tensor)
        assert distances.device == coords.device
        assert distances.dtype == coords.dtype

        # Trigger reconstruction
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals

        assert isinstance(polymer.coordinates, torch.Tensor)
        assert polymer.coordinates.device == coords.device
        assert polymer.coordinates.dtype == coords.dtype

    def test_torch_backend_property(self):
        """Test Polymer uses torch backend."""
        from ciffy import from_sequence
        import torch

        polymer = from_sequence("acgu", backend="torch")
        assert isinstance(polymer.coordinates, torch.Tensor)
        assert isinstance(polymer.dihedrals, torch.Tensor)

    def test_torch_to_numpy_conversion(self):
        """Test torch to numpy conversion."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        polymer_np = polymer.numpy()

        assert isinstance(polymer_np.coordinates, np.ndarray)
        assert isinstance(polymer_np.distances, np.ndarray)

    def test_numpy_to_torch_conversion(self):
        """Test numpy to torch conversion."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="numpy")
        polymer_torch = polymer.torch()

        assert isinstance(polymer_torch.coordinates, torch.Tensor)
        assert isinstance(polymer_torch.distances, torch.Tensor)

    def test_differentiability(self):
        """Test gradients flow through reconstruction."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")

        # Enable gradients on dihedrals
        dihedrals = polymer.dihedrals.clone()
        dihedrals.requires_grad_(True)
        polymer.dihedrals = dihedrals

        # Access coordinates (triggers reconstruction)
        coords = polymer.coordinates

        # Compute loss
        loss = coords.pow(2).mean()

        # Should not raise
        loss.backward()

        # Gradients should exist
        assert dihedrals.grad is not None
        assert not torch.all(dihedrals.grad == 0)


class TestNamedDihedrals:
    """Tests for named dihedral accessors."""

    def test_rna_backbone_dihedrals(self):
        """Test RNA backbone dihedral names from real structure."""
        from ciffy import load, DihedralType

        # Use real structure for proper dihedral detection
        polymer = load(get_test_cif("1ZEW")).poly()

        # Access some backbone dihedrals
        alpha = polymer.dihedral(DihedralType.ALPHA)
        beta = polymer.dihedral(DihedralType.BETA)

        # Should return arrays (may be empty if not found)
        assert alpha is not None
        assert beta is not None

    def test_unknown_dihedral_raises(self):
        """Test unsupported dihedral types are handled."""
        from ciffy import from_sequence, DihedralType

        polymer = from_sequence("acgu")

        # PHI is for proteins, should return empty or handle gracefully
        # (The actual behavior depends on implementation)
        result = polymer.dihedral(DihedralType.PHI)
        assert result is not None  # Should not crash


class TestSetMethods:
    """Tests for setting internal coordinates."""

    def test_set_dihedrals(self):
        """Test setting dihedrals modifies polymer in-place."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        orig_dihedrals = polymer.dihedrals.copy()

        new_dihedrals = polymer.dihedrals.copy()
        new_dihedrals[5] = 1.5

        polymer.dihedrals = new_dihedrals

        # Should be modified
        assert polymer.dihedrals[5] == 1.5
        assert polymer.dihedrals[5] != orig_dihedrals[5]

    def test_set_angles(self):
        """Test setting angles modifies polymer in-place."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        new_angles = polymer.angles.copy()
        new_angles[5] = 2.0

        polymer.angles = new_angles

        assert polymer.angles[5] == 2.0

    def test_set_distances(self):
        """Test setting distances modifies polymer in-place."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        new_distances = polymer.distances.copy()
        new_distances[5] = 2.0

        polymer.distances = new_distances

        assert polymer.distances[5] == 2.0


class TestOrphanAtoms:
    """Tests for orphan atom handling (single-atom connected components)."""

    def test_waters_become_orphans(self):
        """Test that water molecules become single-atom connected components."""
        from ciffy import load

        # Load structure with waters
        polymer = load(get_test_cif("1ZEW"))  # Don't call .poly()

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals

        # Should have single-atom components (waters, ions)
        components = polymer._coord_manager._connected_components
        single_atom_components = [c for c in components if len(c[0]) == 1]
        assert len(single_atom_components) > 0

    def test_orphan_coords_restored(self):
        """Test orphan coordinates are restored after round-trip."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW"))
        orig_coords = polymer.coordinates.copy()

        # Trigger reconstruction
        dihedrals = polymer.dihedrals.copy()
        polymer.dihedrals = dihedrals

        # Find single-atom components
        components = polymer._coord_manager._connected_components
        single_atom_components = [c for c in components if len(c[0]) == 1]

        # Orphan atoms should have original coordinates
        for atom_indices, _ in single_atom_components:
            atom_idx = atom_indices[0]
            orig_coord = orig_coords[atom_idx]
            rec_coord = polymer.coordinates[atom_idx]
            assert np.allclose(orig_coord, rec_coord), \
                f"Orphan atom {atom_idx} coords not restored"

    def test_no_orphans_for_clean_polymer(self):
        """Test no orphans for polymer-only structure."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW")).poly()

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals

        # Should have no single-atom components
        components = polymer._coord_manager._connected_components
        single_atom_components = [c for c in components if len(c[0]) == 1]
        assert len(single_atom_components) == 0


class TestZMatrix:
    """Tests for Z-matrix construction."""

    def test_zmatrix_references_valid(self):
        """Test all Z-matrix references point to valid atoms."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals
        zmatrix = polymer._coord_manager._zmatrix

        placed = set()
        for i, entry in enumerate(zmatrix):
            # Distance ref should be already placed (or -1)
            if entry.distance_ref >= 0:
                assert entry.distance_ref in placed, \
                    f"Entry {i}: distance_ref {entry.distance_ref} not yet placed"

            # Angle ref should be already placed (or -1)
            if entry.angle_ref >= 0:
                assert entry.angle_ref in placed, \
                    f"Entry {i}: angle_ref {entry.angle_ref} not yet placed"

            # Dihedral ref should be already placed (or -1)
            if entry.dihedral_ref >= 0:
                assert entry.dihedral_ref in placed, \
                    f"Entry {i}: dihedral_ref {entry.dihedral_ref} not yet placed"

            placed.add(entry.atom_idx)

    def test_zmatrix_distinct_references(self):
        """Test Z-matrix references are distinct for full entries."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals
        zmatrix = polymer._coord_manager._zmatrix

        for i, entry in enumerate(zmatrix):
            if entry.dihedral_ref >= 0:
                # All four atoms should be distinct
                atoms = {entry.atom_idx, entry.distance_ref, entry.angle_ref, entry.dihedral_ref}
                assert len(atoms) == 4, \
                    f"Entry {i}: atoms not all distinct: {entry}"

    def test_first_atom_at_origin(self):
        """Test first atom has no references (placed at origin)."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals
        zmatrix = polymer._coord_manager._zmatrix

        first = zmatrix[0]
        assert first.distance_ref == -1
        assert first.angle_ref == -1
        assert first.dihedral_ref == -1


class TestBondGraph:
    """Tests for bond graph construction."""

    def test_bond_graph_symmetric(self):
        """Test bond graph is symmetric (undirected)."""
        from ciffy import from_sequence
        from ciffy.internal import build_bond_graph

        polymer = from_sequence("acgu")
        graph = build_bond_graph(polymer)

        for atom, neighbors in graph.items():
            for neighbor in neighbors:
                assert atom in graph[neighbor], \
                    f"Bond {atom}-{neighbor} not symmetric"

    def test_bond_graph_has_expected_bonds(self):
        """Test bond graph has reasonable number of bonds."""
        from ciffy import from_sequence
        from ciffy.internal import build_bond_graph

        polymer = from_sequence("acgu")
        graph = build_bond_graph(polymer)

        # Count total bonds
        total_bonds = sum(len(neighbors) for neighbors in graph.values()) // 2

        # Should have roughly 1 bond per atom (organic molecules)
        n_atoms = polymer.size()
        assert total_bonds >= n_atoms - 4  # At least n-4 bonds (tree + some cycles)
        assert total_bonds <= n_atoms * 2  # At most 2 bonds per atom on average
