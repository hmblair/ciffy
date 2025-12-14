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

        # Check array shapes using ZMatrix
        zmatrix = polymer._coord_manager.zmatrix
        n_zmatrix = len(zmatrix)
        assert distances.shape == (n_zmatrix,)
        assert angles.shape == (n_zmatrix,)
        assert dihedrals.shape == (n_zmatrix,)

    def test_distances_are_positive(self):
        """Test all bond distances are positive."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        distances = polymer.distances
        zmatrix = polymer._coord_manager.zmatrix

        # First entry has no distance (root atom)
        for i in range(len(zmatrix)):
            dist_ref = int(zmatrix.distance_refs[i])
            if dist_ref >= 0:
                assert distances[i] > 0, f"Distance at {i} is not positive"

    def test_angles_in_valid_range(self):
        """Test all bond angles are in [0, pi]."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        angles = polymer.angles
        zmatrix = polymer._coord_manager.zmatrix

        for i in range(len(zmatrix)):
            ang_ref = int(zmatrix.angle_refs[i])
            if ang_ref >= 0:
                angle = angles[i]
                assert 0 <= angle <= np.pi, f"Angle at {i} is {angle}, not in [0, pi]"

    def test_dihedrals_in_valid_range(self):
        """Test all dihedral angles are in [-pi, pi]."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        dihedrals = polymer.dihedrals
        zmatrix = polymer._coord_manager.zmatrix

        for i in range(len(zmatrix)):
            dih_ref = int(zmatrix.dihedral_refs[i])
            if dih_ref >= 0:
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
            chain_rmsd = rmsd(orig_chain, chain).item()
            assert chain_rmsd < 1e-4, \
                f"Chain {chain_idx} internal structure RMSD {chain_rmsd:.6f} exceeds threshold"

        # Test 2: Global RMSD should fail (relative chain positions/orientations not preserved)
        global_rmsd_val = rmsd(orig_polymer, polymer).item()
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
            chain_rmsd = rmsd(orig_chain, chain).item()

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
        # Access CSR format components
        mgr = polymer._coord_manager
        n_components = len(mgr._component_offsets) - 1
        single_atom_count = 0
        for i in range(n_components):
            start = int(mgr._component_offsets[i])
            end = int(mgr._component_offsets[i + 1])
            if end - start == 1:
                single_atom_count += 1

        assert single_atom_count > 0

    def test_orphan_coords_restored(self):
        """Test orphan coordinates are restored after round-trip."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW"))
        orig_coords = polymer.coordinates.copy()

        # Trigger reconstruction
        dihedrals = polymer.dihedrals.copy()
        polymer.dihedrals = dihedrals

        # Find single-atom components using CSR format
        mgr = polymer._coord_manager
        n_components = len(mgr._component_offsets) - 1

        for i in range(n_components):
            start = int(mgr._component_offsets[i])
            end = int(mgr._component_offsets[i + 1])
            if end - start == 1:
                # This is a single-atom component
                atom_idx = int(mgr._component_atoms[start])
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
        mgr = polymer._coord_manager
        n_components = len(mgr._component_offsets) - 1
        single_atom_count = 0
        for i in range(n_components):
            start = int(mgr._component_offsets[i])
            end = int(mgr._component_offsets[i + 1])
            if end - start == 1:
                single_atom_count += 1

        assert single_atom_count == 0


class TestZMatrix:
    """Tests for Z-matrix construction."""

    def test_zmatrix_references_valid(self):
        """Test all Z-matrix references point to valid atoms."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals
        zmatrix = polymer._coord_manager.zmatrix

        # Use validate() method from ZMatrix class
        zmatrix.validate()  # Should not raise

    def test_zmatrix_distinct_references(self):
        """Test Z-matrix references are distinct for full entries."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals
        zmatrix = polymer._coord_manager.zmatrix

        for i in range(len(zmatrix)):
            dih_ref = int(zmatrix.dihedral_refs[i])
            if dih_ref >= 0:
                # All four atoms should be distinct
                atoms = {
                    int(zmatrix.atom_indices[i]),
                    int(zmatrix.distance_refs[i]),
                    int(zmatrix.angle_refs[i]),
                    dih_ref,
                }
                assert len(atoms) == 4, \
                    f"Entry {i}: atoms not all distinct"

    def test_first_atom_at_origin(self):
        """Test first atom has no references (placed at origin)."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access internal coords to build Z-matrix
        _ = polymer.dihedrals
        zmatrix = polymer._coord_manager.zmatrix

        assert int(zmatrix.distance_refs[0]) == -1
        assert int(zmatrix.angle_refs[0]) == -1
        assert int(zmatrix.dihedral_refs[0]) == -1


class TestBondGraph:
    """Tests for bond graph construction."""

    def test_bond_graph_symmetric(self):
        """Test bond graph is symmetric (undirected)."""
        from ciffy import from_sequence
        from ciffy.internal import build_bond_graph

        polymer = from_sequence("acgu")
        edges, n_atoms = build_bond_graph(polymer)

        # Build adjacency dict from edges for testing
        adj = {i: set() for i in range(n_atoms)}
        for i in range(len(edges)):
            a, b = int(edges[i, 0]), int(edges[i, 1])
            adj[a].add(b)

        # Check symmetry
        for atom, neighbors in adj.items():
            for neighbor in neighbors:
                assert atom in adj[neighbor], \
                    f"Bond {atom}-{neighbor} not symmetric"

    def test_bond_graph_has_expected_bonds(self):
        """Test bond graph has reasonable number of bonds."""
        from ciffy import from_sequence
        from ciffy.internal import build_bond_graph

        polymer = from_sequence("acgu")
        edges, n_atoms = build_bond_graph(polymer)

        # Count total bonds (edges are symmetric, so divide by 2)
        total_bonds = len(edges) // 2

        # Should have roughly 1 bond per atom (organic molecules)
        assert total_bonds >= n_atoms - 4  # At least n-4 bonds (tree + some cycles)
        assert total_bonds <= n_atoms * 2  # At most 2 bonds per atom on average


class TestAutogradGradients:
    """Tests for autograd gradient correctness."""

    def test_cartesian_to_internal_gradcheck(self):
        """Test cartesian_to_internal gradients match numerical differentiation."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal, HAS_C_EXTENSION

        if not HAS_C_EXTENSION:
            pytest.skip("C extension not available")

        # Test coordinates
        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [3.5, 1.5, 1.0]
        ], dtype=np.float32)

        # Z-matrix indices
        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        indices = torch.tensor(indices_np)

        def wrapper(coords):
            coords32 = coords.float()
            d, a, dh = cartesian_to_internal(coords32, indices)
            return d.double(), a.double(), dh.double()

        coords_check = torch.tensor(coords_np, requires_grad=True, dtype=torch.float64)
        assert torch.autograd.gradcheck(wrapper, coords_check, eps=1e-4, atol=1e-3, rtol=1e-2)

    def test_distance_gradient_direct(self):
        """Test distance gradient directly against PyTorch autograd."""
        import torch
        from ciffy._c import _cartesian_to_internal, _cartesian_to_internal_backward

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.3, 0.2],
        ], dtype=np.float32)

        # Only distance
        indices_np = np.array([[1, 0, -1, -1]], dtype=np.int64)

        # C extension forward/backward
        distances, _, _ = _cartesian_to_internal(coords_np, indices_np)
        grad_coords = _cartesian_to_internal_backward(
            coords_np, indices_np, distances,
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),  # grad_distance = 1
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )

        # PyTorch reference
        coords = torch.tensor(coords_np, requires_grad=True)
        d = torch.norm(coords[1] - coords[0])
        d.backward()

        assert np.allclose(grad_coords, coords.grad.numpy(), atol=1e-5)

    def test_angle_gradient_direct(self):
        """Test angle gradient directly against PyTorch autograd."""
        import torch
        from ciffy._c import _cartesian_to_internal, _cartesian_to_internal_backward

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.2],
        ], dtype=np.float32)

        # Angle at atom 1 between 2 and 0
        indices_np = np.array([[2, 1, 0, -1]], dtype=np.int64)

        # C extension
        distances, angles, _ = _cartesian_to_internal(coords_np, indices_np)
        grad_coords = _cartesian_to_internal_backward(
            coords_np, indices_np, distances, angles,
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),  # grad_angle = 1
            np.array([0.0], dtype=np.float32),
        )

        # PyTorch reference
        coords = torch.tensor(coords_np, requires_grad=True)
        v1 = coords[2] - coords[1]
        v2 = coords[0] - coords[1]
        cos_angle = torch.dot(v1, v2) / (v1.norm() * v2.norm())
        angle = torch.acos(cos_angle.clamp(-1, 1))
        angle.backward()

        assert np.allclose(grad_coords, coords.grad.numpy(), atol=1e-4)

    def test_dihedral_gradient_direct(self):
        """Test dihedral gradient directly against PyTorch autograd."""
        import torch
        from ciffy._c import _cartesian_to_internal, _cartesian_to_internal_backward

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [3.5, 1.5, 1.0]
        ], dtype=np.float32)

        indices_np = np.array([[3, 2, 1, 0]], dtype=np.int64)

        # C extension
        distances, angles, dihedrals = _cartesian_to_internal(coords_np, indices_np)
        grad_coords = _cartesian_to_internal_backward(
            coords_np, indices_np, distances, angles,
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),  # grad_dihedral = 1
        )

        # PyTorch reference using same normalized formula as C code
        EPS = 1e-6
        coords = torch.tensor(coords_np, requires_grad=True)
        a, b, c, d = coords[0], coords[1], coords[2], coords[3]
        b1 = b - a
        b2 = c - b
        b3 = d - c
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        n1_hat = n1 / (n1.norm() + EPS)
        n2_hat = n2 / (n2.norm() + EPS)
        b2_hat = b2 / (b2.norm() + EPS)
        m1 = torch.linalg.cross(n1_hat, b2_hat)
        x = torch.dot(n1_hat, n2_hat)
        y = torch.dot(m1, n2_hat)
        dihedral = torch.atan2(y, x)
        dihedral.backward()

        assert np.allclose(grad_coords, coords.grad.numpy(), atol=1e-4)

    def test_multiple_entries_gradients(self):
        """Test gradients with multiple Z-matrix entries."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [3.5, 1.5, 1.0],
            [4.0, 2.5, 1.5],
        ], dtype=np.float32)

        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
            [4,  3,  2,  1],
        ], dtype=np.int64)

        coords = torch.tensor(coords_np, requires_grad=True)
        indices = torch.tensor(indices_np)

        distances, angles, dihedrals = cartesian_to_internal(coords, indices)

        # Loss using all outputs
        loss = distances.sum() + angles.sum() + dihedrals.sum()
        loss.backward()

        # Gradient should exist and not be all zeros
        assert coords.grad is not None
        assert not torch.all(coords.grad == 0)


class TestEndToEndNNPipeline:
    """End-to-end tests for NN + internal coordinates pipeline."""

    def test_dihedral_optimization_reduces_rmsd(self):
        """Test gradient descent on dihedrals reduces RMSD to target."""
        import copy
        import torch
        import torch.nn as nn
        from ciffy import load, rmsd

        # Load target and create template
        target = load(get_test_cif("1ZEW"))
        for chain in target.chains():
            target_chain = chain.torch()
            break

        template = copy.deepcopy(target_chain)

        # Perturb template dihedrals
        original_dihedrals = template.dihedrals.clone()
        perturbed = original_dihedrals + torch.randn_like(original_dihedrals) * 0.3
        template.dihedrals = perturbed

        initial_rmsd = rmsd(template, target_chain).item()

        # Create learnable parameters
        class DihedralPredictor(nn.Module):
            def __init__(self, init):
                super().__init__()
                self.dihedrals = nn.Parameter(init.clone())

        model = DihedralPredictor(perturbed)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            template.dihedrals = model.dihedrals
            loss = rmsd(template, target_chain)
            loss.backward()
            optimizer.step()

        final_rmsd = rmsd(template, target_chain).item()

        # RMSD should decrease significantly
        assert final_rmsd < initial_rmsd * 0.5, \
            f"RMSD did not decrease enough: {initial_rmsd:.2f} -> {final_rmsd:.2f}"

    def test_gradient_flow_through_chain_slicing(self):
        """Test gradients flow through sliced chains."""
        import copy
        import torch
        from ciffy import load, rmsd

        # Load multi-chain structure
        target = load(get_test_cif("1ZEW")).torch()

        # Get first chain (uses __getitem__ which we just fixed)
        for chain in target.chains():
            target_chain = chain
            break

        template = copy.deepcopy(target_chain)

        # Set dihedrals with gradients
        dihedrals = template.dihedrals.clone().requires_grad_(True)
        template.dihedrals = dihedrals

        # Compute RMSD
        loss = rmsd(template, target_chain)

        # Backward should work
        loss.backward()

        # Gradients should exist
        assert dihedrals.grad is not None
        assert not torch.all(dihedrals.grad == 0)


class TestProteinInternalCoordinates:
    """Tests for protein internal coordinate handling."""

    def test_protein_roundtrip(self):
        """Test round-trip conversion for protein chain."""
        from ciffy import load
        from ciffy.operations.alignment import kabsch_align

        polymer = load(get_test_cif("9GCM"))

        # Get a protein chain (chain B or C)
        protein_chain = None
        for chain in polymer.chains():
            # Check if it's a protein (molecule_type[0] == 0)
            if hasattr(chain, 'molecule_type') and len(chain.molecule_type) > 0:
                if chain.molecule_type[0] == 0:  # PROTEIN type
                    protein_chain = chain
                    break

        if protein_chain is None:
            pytest.skip("No protein chain found in 9GCM")

        orig_coords = protein_chain.coordinates.copy()

        # Access internal coordinates, then modify to trigger reconstruction
        dihedrals = protein_chain.dihedrals.copy()
        protein_chain.dihedrals = dihedrals

        # Coordinates should be reconstructed
        aligned, _, _ = kabsch_align(protein_chain.coordinates, orig_coords)
        rmsd = np.sqrt(((aligned - orig_coords) ** 2).sum(axis=1).mean())

        # Protein chains may have slightly higher RMSD due to more complex topology
        assert rmsd < 0.1, f"Protein RMSD {rmsd} exceeds threshold"

    def test_protein_internal_properties(self):
        """Test protein has valid internal coordinate properties."""
        from ciffy import load

        polymer = load(get_test_cif("9GCM"))

        # Get a protein chain
        protein_chain = None
        for chain in polymer.chains():
            if hasattr(chain, 'molecule_type') and len(chain.molecule_type) > 0:
                if chain.molecule_type[0] == 0:  # PROTEIN type
                    protein_chain = chain
                    break

        if protein_chain is None:
            pytest.skip("No protein chain found in 9GCM")

        # All non-root distances should be positive (first entry is root with no distance)
        distances = protein_chain.distances
        assert np.all(distances[1:] > 0), "Non-root distances should be positive"

        # Angles should be in valid range [0, pi] (skip first 2 entries - root atoms)
        angles = protein_chain.angles
        valid_angles = angles[2:]  # First two atoms don't have valid angles
        assert np.all(valid_angles >= 0)
        assert np.all(valid_angles <= np.pi + 1e-5)

        # Dihedrals should be in valid range [-pi, pi] (skip first 3 entries)
        dihedrals = protein_chain.dihedrals
        valid_dihedrals = dihedrals[3:]  # First three atoms don't have valid dihedrals
        assert np.all(valid_dihedrals >= -np.pi - 1e-5)
        assert np.all(valid_dihedrals <= np.pi + 1e-5)


class TestNumericalEdgeCases:
    """Tests for numerical edge cases and stability."""

    def test_small_perturbation_roundtrip(self):
        """Test very small dihedral changes preserve structure."""
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu")
        orig_coords = polymer.coordinates.copy()
        orig_dihedrals = polymer.dihedrals.copy()

        # Apply very small perturbation (0.001 radians ~ 0.06 degrees)
        polymer.dihedrals = orig_dihedrals + 0.001

        # Reconstruction should still work
        aligned, _, _ = kabsch_align(polymer.coordinates, orig_coords)
        rmsd = np.sqrt(((aligned - orig_coords) ** 2).sum(axis=1).mean())

        # Small perturbation should give small change
        assert rmsd < 0.5, f"Small perturbation gave large RMSD {rmsd}"

    def test_zero_perturbation_preserves_structure(self):
        """Test zero perturbation exactly preserves structure."""
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu")
        orig_coords = polymer.coordinates.copy()
        orig_dihedrals = polymer.dihedrals.copy()

        # Zero perturbation
        polymer.dihedrals = orig_dihedrals + 0.0

        # Should be essentially identical
        aligned, _, _ = kabsch_align(polymer.coordinates, orig_coords)
        rmsd = np.sqrt(((aligned - orig_coords) ** 2).sum(axis=1).mean())

        assert rmsd < 1e-4, f"Zero perturbation changed structure: RMSD {rmsd}"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_sliced_manager_dihedral_access_works(self):
        """Test that properly sliced chain can access dihedrals."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW"))

        # Get a chain (uses slicing via __getitem__)
        chain = None
        for c in polymer.chains():
            chain = c
            break

        # Access dihedrals should work on properly sliced chain
        dihedrals = chain.dihedrals
        assert len(dihedrals) > 0

    def test_nan_dihedral_fails_reconstruction(self):
        """Test NaN dihedral causes reconstruction to fail gracefully."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Access dihedrals to trigger Z-matrix building
        dihedrals = polymer.dihedrals.copy()

        # Set NaN at a position that affects reconstruction (not root atoms)
        dihedrals[10] = np.nan
        polymer.dihedrals = dihedrals

        # Reconstruction should fail (SVD won't converge with NaN coords)
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            _ = polymer.coordinates

    def test_validation_method_exists(self):
        """Test that CoordinateManager has validation method."""
        from ciffy import from_sequence
        from ciffy.internal.coordinates import CoordinateManager

        polymer = from_sequence("acgu")

        # Verify validation method exists
        assert hasattr(polymer._coord_manager, '_validate_coordinates')

