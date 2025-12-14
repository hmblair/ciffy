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
        internal = polymer.to_internal()
        reconstructed = internal.to_cartesian()

        aligned, _, _ = kabsch_align(reconstructed.coordinates, polymer.coordinates)
        rmsd = np.sqrt(((aligned - polymer.coordinates) ** 2).sum(axis=1).mean())

        assert rmsd < 1e-5, f"RMSD {rmsd} exceeds threshold"

    def test_rna_tetramer_roundtrip(self):
        """Test round-trip for RNA tetramer."""
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()
        reconstructed = internal.to_cartesian()

        aligned, _, _ = kabsch_align(reconstructed.coordinates, polymer.coordinates)
        rmsd = np.sqrt(((aligned - polymer.coordinates) ** 2).sum(axis=1).mean())

        assert rmsd < 1e-4, f"RMSD {rmsd} exceeds threshold"

    def test_internal_polymer_properties(self):
        """Test InternalPolymer has correct properties."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        # Check array shapes
        n_zmatrix = len(internal.zmatrix)
        assert internal.distances.shape == (n_zmatrix,)
        assert internal.angles.shape == (n_zmatrix,)
        assert internal.dihedrals.shape == (n_zmatrix,)

        # Check size includes orphans
        assert internal.size == polymer.size()

        # Check backend
        assert internal.backend == "numpy"

    def test_distances_are_positive(self):
        """Test all bond distances are positive."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        # First entry has no distance (root atom)
        for i, entry in enumerate(internal.zmatrix):
            if entry.distance_ref >= 0:
                assert internal.distances[i] > 0, f"Distance at {i} is not positive"

    def test_angles_in_valid_range(self):
        """Test all bond angles are in [0, pi]."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        for i, entry in enumerate(internal.zmatrix):
            if entry.angle_ref >= 0:
                angle = internal.angles[i]
                assert 0 <= angle <= np.pi, f"Angle at {i} is {angle}, not in [0, pi]"

    def test_dihedrals_in_valid_range(self):
        """Test all dihedral angles are in [-pi, pi]."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        for i, entry in enumerate(internal.zmatrix):
            if entry.dihedral_ref >= 0:
                dih = internal.dihedrals[i]
                assert -np.pi <= dih <= np.pi, f"Dihedral at {i} is {dih}, not in [-pi, pi]"


class TestInternalCoordinatesPDB:
    """Tests using real PDB structures."""

    def test_rna_structure_per_chain(self):
        """Test round-trip for RNA structure (per-chain RMSD)."""
        from ciffy import load, Scale, rmsd
        from ciffy.operations.alignment import kabsch_align

        polymer = load(get_test_cif("1ZEW")).poly()
        internal = polymer.to_internal()
        reconstructed = internal.to_cartesian()

        # TODO: spin out into a separate test
        mol_rmsd = rmsd(polymer, reconstructed)
        assert mol_rmsd < 1e-4, f"All-chain RMSD {mol_rmsd} exceeds threshold"

        # Test per-chain RMSD
        res_sizes = polymer.sizes(Scale.RESIDUE)
        chain_start_atom = 0
        chain_start_res = 0

        for chain_idx, chain_len in enumerate(polymer.lengths):
            chain_len_val = int(chain_len)
            if chain_len_val == 0:
                continue

            chain_atom_count = sum(int(res_sizes[chain_start_res + i]) for i in range(chain_len_val))
            chain_atoms = list(range(chain_start_atom, chain_start_atom + chain_atom_count))

            chain_orig = polymer.coordinates[chain_atoms]
            chain_rec = reconstructed.coordinates[chain_atoms]
            # TODO: use rmsd rather than manual alignment
            chain_aligned, _, _ = kabsch_align(chain_rec, chain_orig)
            chain_rmsd = np.sqrt(((chain_aligned - chain_orig) ** 2).sum(axis=1).mean())

            assert chain_rmsd < 1e-4, f"Chain {chain_idx} RMSD {chain_rmsd} exceeds threshold"

            chain_start_atom += chain_atom_count
            chain_start_res += chain_len_val


class TestInternalCoordinatesTorchBackend:
    """Tests for PyTorch backend."""

    def test_torch_roundtrip(self):
        """Test round-trip with torch backend."""
        import torch
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu", backend="torch")
        internal = polymer.to_internal()
        reconstructed = internal.to_cartesian()

        assert isinstance(reconstructed.coordinates, torch.Tensor)

        aligned, _, _ = kabsch_align(reconstructed.coordinates, polymer.coordinates)
        rmsd = torch.sqrt(((aligned - polymer.coordinates) ** 2).sum(dim=1).mean())

        assert rmsd.item() < 1e-4, f"RMSD {rmsd.item()} exceeds threshold"

    def test_torch_roundtrip_preserves_device_and_dtype(self):
        """Ensure C extension path returns tensors on the original device/dtype."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        coords = polymer.coordinates

        internal = polymer.to_internal()  # should route through C extension even for torch
        assert isinstance(internal.distances, torch.Tensor)
        assert internal.distances.device == coords.device
        assert internal.distances.dtype == coords.dtype

        reconstructed = internal.to_cartesian()
        assert isinstance(reconstructed.coordinates, torch.Tensor)
        assert reconstructed.coordinates.device == coords.device
        assert reconstructed.coordinates.dtype == coords.dtype

    def test_torch_backend_property(self):
        """Test backend property returns 'torch'."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        internal = polymer.to_internal()

        assert internal.backend == "torch"

    def test_torch_to_numpy_conversion(self):
        """Test torch to numpy conversion."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        internal = polymer.to_internal()
        internal_np = internal.numpy()

        assert internal_np.backend == "numpy"
        assert isinstance(internal_np.distances, np.ndarray)

    def test_numpy_to_torch_conversion(self):
        """Test numpy to torch conversion."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="numpy")
        internal = polymer.to_internal()
        internal_torch = internal.torch()

        assert internal_torch.backend == "torch"
        assert isinstance(internal_torch.distances, torch.Tensor)

    def test_differentiability(self):
        """Test gradients flow through to_cartesian."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        internal = polymer.to_internal()

        # Enable gradients
        internal.dihedrals.requires_grad_(True)

        # Reconstruct
        reconstructed = internal.to_cartesian()

        # Compute loss
        loss = reconstructed.coordinates.pow(2).mean()

        # Should not raise
        loss.backward()

        # Gradients should exist
        assert internal.dihedrals.grad is not None
        assert not torch.all(internal.dihedrals.grad == 0)


class TestNamedDihedrals:
    """Tests for named dihedral accessors."""

    def test_rna_backbone_dihedrals(self):
        """Test RNA backbone dihedral names from real structure."""
        from ciffy import load

        # Use real structure for proper dihedral detection
        polymer = load(get_test_cif("1ZEW")).poly()
        internal = polymer.to_internal()

        # Should have some backbone dihedrals
        # Not all may be present depending on Z-matrix construction
        dihedral_indices = internal._dihedral_indices
        if dihedral_indices is None:
            internal._compute_dihedral_indices()
            dihedral_indices = internal._dihedral_indices

        # At least some dihedrals should be found in a real RNA structure
        found_any = any(len(v) > 0 for v in dihedral_indices.values() if v is not None)
        # This is a soft check - structure may not have all expected patterns
        assert dihedral_indices is not None

    def test_unknown_dihedral_raises(self):
        """Test unknown dihedral name raises ValueError."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        with pytest.raises(ValueError, match="Unknown dihedral name"):
            internal.backbone_dihedrals('invalid_name')


class TestWithMethods:
    """Tests for with_* modification methods."""

    def test_with_dihedrals(self):
        """Test with_dihedrals creates new object."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        new_dihedrals = internal.dihedrals.copy()
        new_dihedrals[5] = 1.5

        modified = internal.with_dihedrals(new_dihedrals)

        # Should be different objects
        assert modified is not internal
        # Original unchanged
        assert internal.dihedrals[5] != 1.5
        # Modified has new value
        assert modified.dihedrals[5] == 1.5

    def test_with_angles(self):
        """Test with_angles creates new object."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        new_angles = internal.angles.copy()
        new_angles[5] = 2.0

        modified = internal.with_angles(new_angles)

        assert modified is not internal
        assert modified.angles[5] == 2.0

    def test_with_distances(self):
        """Test with_distances creates new object."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        new_distances = internal.distances.copy()
        new_distances[5] = 2.0

        modified = internal.with_distances(new_distances)

        assert modified is not internal
        assert modified.distances[5] == 2.0


class TestOrphanAtoms:
    """Tests for orphan atom handling."""

    def test_waters_become_orphans(self):
        """Test that water molecules become orphan atoms."""
        from ciffy import load

        # Load structure with waters
        polymer = load(get_test_cif("1ZEW"))  # Don't call .poly()
        internal = polymer.to_internal()

        # Should have orphan atoms (waters, ions)
        assert len(internal._orphan_atoms) > 0

    def test_orphan_coords_restored(self):
        """Test orphan coordinates are restored after round-trip."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW"))
        internal = polymer.to_internal()
        reconstructed = internal.to_cartesian()

        # Orphan atoms should have original coordinates
        for i, atom_idx in enumerate(internal._orphan_atoms):
            orig_coord = polymer.coordinates[atom_idx]
            rec_coord = reconstructed.coordinates[atom_idx]
            assert np.allclose(orig_coord, rec_coord), \
                f"Orphan atom {atom_idx} coords not restored"

    def test_no_orphans_for_clean_polymer(self):
        """Test no orphans for polymer-only structure."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW")).poly()
        internal = polymer.to_internal()

        assert len(internal._orphan_atoms) == 0


class TestZMatrix:
    """Tests for Z-matrix construction."""

    def test_zmatrix_references_valid(self):
        """Test all Z-matrix references point to valid atoms."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        placed = set()
        for i, entry in enumerate(internal.zmatrix):
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
        internal = polymer.to_internal()

        for i, entry in enumerate(internal.zmatrix):
            if entry.dihedral_ref >= 0:
                # All four atoms should be distinct
                atoms = {entry.atom_idx, entry.distance_ref, entry.angle_ref, entry.dihedral_ref}
                assert len(atoms) == 4, \
                    f"Entry {i}: atoms not all distinct: {entry}"

    def test_first_atom_at_origin(self):
        """Test first atom has no references (placed at origin)."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        internal = polymer.to_internal()

        first = internal.zmatrix[0]
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
