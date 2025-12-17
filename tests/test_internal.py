"""Tests for internal coordinates (Z-matrix) representation."""

import pytest
import numpy as np

from tests.utils import (
    get_test_cif,
    GPU_DEVICES,
    skip_if_no_device,
    requires_cuda,
    requires_cuda_extension,
)


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

    def test_protein_dihedrals_match_biopython(self):
        """Test protein backbone dihedrals match Bio.PDB values.

        Compares PHI and PSI angles against Bio.PDB.internal_coords.
        Note: ciffy uses opposite sign convention from Bio.PDB.
        """
        pytest.importorskip("Bio")
        from Bio.PDB import MMCIFParser
        from ciffy import load, DihedralType

        # Load with Bio.PDB
        parser = MMCIFParser(QUIET=True)
        bio_struct = parser.get_structure("8CAM", get_test_cif("8CAM"))
        bio_chain = list(bio_struct.get_models())[0]["0"]  # First protein chain
        bio_chain.atom_to_internal_coordinates()

        # Extract Bio.PDB dihedrals (degrees -> radians)
        bio_phi, bio_psi = [], []
        for res in bio_chain.get_residues():
            if hasattr(res, "internal_coord") and res.internal_coord:
                ic = res.internal_coord
                phi = ic.get_angle("phi")
                psi = ic.get_angle("psi")
                bio_phi.append(np.deg2rad(phi) if phi else np.nan)
                bio_psi.append(np.deg2rad(psi) if psi else np.nan)
        bio_phi = np.array(bio_phi)
        bio_psi = np.array(bio_psi)

        # Load with ciffy - get first protein chain
        ciffy_polymer = load(get_test_cif("8CAM")).poly().by_index(0)
        ciffy_phi = ciffy_polymer.dihedral(DihedralType.PHI)
        ciffy_psi = ciffy_polymer.dihedral(DihedralType.PSI)

        # Bio.PDB includes NaN for missing dihedrals, ciffy omits them
        # Bio.PDB: first phi is NaN (no preceding residue)
        # Bio.PDB: last psi is NaN (no following residue)
        bio_phi_valid = bio_phi[1:]  # Skip first NaN
        bio_psi_valid = bio_psi[:-1]  # Skip last NaN

        # Compare with sign flip (convention difference)
        # ciffy_dihedral = -bio_dihedral
        n_phi = min(len(bio_phi_valid), len(ciffy_phi))
        n_psi = min(len(bio_psi_valid), len(ciffy_psi))

        phi_diff = np.abs(bio_phi_valid[:n_phi] + ciffy_phi[:n_phi])
        psi_diff = np.abs(bio_psi_valid[:n_psi] + ciffy_psi[:n_psi])

        # Handle wrap-around at ±π
        phi_diff = np.minimum(phi_diff, 2 * np.pi - phi_diff)
        psi_diff = np.minimum(psi_diff, 2 * np.pi - psi_diff)

        # Should match within numerical precision
        assert np.nanmax(phi_diff) < 1e-5, f"PHI max diff: {np.nanmax(phi_diff)}"
        assert np.nanmax(psi_diff) < 1e-5, f"PSI max diff: {np.nanmax(psi_diff)}"

    def test_sidechain_chi1_match_biopython(self):
        """Test sidechain CHI1 dihedrals match Bio.PDB values.

        Compares CHI1 angles against Bio.PDB.internal_coords by mapping
        chi1 Z-matrix entries to their owning residues.
        Note: ciffy uses opposite sign convention from Bio.PDB.
        """
        pytest.importorskip("Bio")
        from Bio.PDB import MMCIFParser
        from ciffy import load, DihedralType
        from ciffy.biochemistry import Residue
        from ciffy.types import Scale

        # Load with Bio.PDB
        parser = MMCIFParser(QUIET=True)
        bio_struct = parser.get_structure("8CAM", get_test_cif("8CAM"))
        bio_chain = list(bio_struct.get_models())[0]["0"]
        bio_chain.atom_to_internal_coordinates()

        # Extract Bio.PDB CHI1 indexed by position
        bio_chi1_by_idx = {}
        for idx, res in enumerate(bio_chain.get_residues()):
            if hasattr(res, "internal_coord") and res.internal_coord:
                ic = res.internal_coord
                chi1_deg = ic.get_angle("chi1")
                if chi1_deg is not None:
                    bio_chi1_by_idx[idx] = (res.get_resname(), np.deg2rad(chi1_deg))

        # Load with ciffy
        ciffy_polymer = load(get_test_cif("8CAM")).poly().by_index(0)
        ciffy_chi1 = ciffy_polymer.dihedral(DihedralType.CHI1)

        # Map chi1 values to residue indices using Z-matrix
        cm = ciffy_polymer._coord_manager
        zmat = cm.zmatrix
        dihedral_types = zmat.dihedral_types
        chi1_zmat_indices = np.where(dihedral_types == DihedralType.CHI1.value)[0]

        # Build residue offset array
        res_sizes = ciffy_polymer.sizes(Scale.RESIDUE)
        res_offsets = np.concatenate([[0], np.cumsum(res_sizes)[:-1]])

        # Get ciffy sequence for alignment
        ciffy_seq = [Residue(int(r)).name for r in ciffy_polymer.sequence]
        bio_seq = [res.get_resname() for res in bio_chain.get_residues()]

        # Find offset: ciffy may have extra N-terminal residues
        offset = 0
        for start in range(min(10, len(ciffy_seq))):
            if ciffy_seq[start : start + 5] == bio_seq[:5]:
                offset = start
                break

        # Compare chi1 values
        matched = 0
        total_compared = 0
        max_diff = 0.0

        for chi1_idx, zmat_idx in enumerate(chi1_zmat_indices):
            # Find which ciffy residue owns this chi1
            atom_idx = int(zmat.atom_indices[zmat_idx])
            ciffy_res_idx = int(np.searchsorted(res_offsets, atom_idx, side="right") - 1)

            # Map to Bio.PDB index
            bio_res_idx = ciffy_res_idx - offset
            if bio_res_idx < 0 or bio_res_idx not in bio_chi1_by_idx:
                continue

            bio_name, bio_val = bio_chi1_by_idx[bio_res_idx]
            ciffy_name = ciffy_seq[ciffy_res_idx]
            ciffy_val = ciffy_chi1[chi1_idx]

            # Skip if residue names don't match
            if bio_name != ciffy_name:
                continue

            total_compared += 1

            # Sign flip: ciffy = -bio
            diff = abs(bio_val + ciffy_val)
            if diff > np.pi:
                diff = 2 * np.pi - diff

            max_diff = max(max_diff, diff)
            if diff < 1e-4:
                matched += 1

        # Should have many matching residues
        assert total_compared >= 30, f"Only compared {total_compared} residues"
        assert matched >= 30, f"Only {matched}/{total_compared} CHI1 values matched"
        assert max_diff < 1e-4, f"CHI1 max diff: {max_diff:.6f} rad"


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
        # Access ConnectedComponents object
        mgr = polymer._coord_manager
        components = mgr._components
        n_components = components.n_components
        single_atom_count = 0
        for i in range(n_components):
            if components.get_component_size(i) == 1:
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

        # Find single-atom components using ConnectedComponents
        mgr = polymer._coord_manager
        components = mgr._components
        n_components = components.n_components

        for i in range(n_components):
            if components.get_component_size(i) == 1:
                # This is a single-atom component
                atom_idx = int(components.get_component_atoms(i)[0])
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
        components = mgr._components
        n_components = components.n_components
        single_atom_count = 0
        for i in range(n_components):
            if components.get_component_size(i) == 1:
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
        from ciffy.backend.dispatch import build_bond_graph

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
        from ciffy.backend.dispatch import build_bond_graph

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

        # Set dihedrals with gradients - add perturbation so RMSD > 0
        # (If dihedrals are identical, RMSD=0 and gradients are correctly zero)
        dihedrals = (template.dihedrals.clone() + 0.1).requires_grad_(True)
        template.dihedrals = dihedrals

        # Compute RMSD (should be non-zero due to perturbation)
        loss = rmsd(template, target_chain)
        assert loss > 0, "RMSD should be non-zero after perturbation"

        # Backward should work
        loss.backward()

        # Gradients should exist and be non-zero
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


class TestRingPreservation:
    """Tests for ring geometry preservation during backbone manipulation.

    The canonical Z-matrix construction uses ring-internal dihedrals that
    preserve ring geometry when only backbone dihedrals are modified.
    """

    def _measure_ring_distances(self, polymer, ring_atom_names, residue_idx=0):
        """Measure all pairwise distances between ring atoms.

        Args:
            polymer: Polymer structure
            ring_atom_names: List of ring atom names (e.g., ["N1", "C2", "N3", ...])
            residue_idx: Which residue to measure

        Returns:
            dict mapping atom pair tuples to distances
        """
        from ciffy import Scale
        from ciffy.biochemistry import Residue

        coords = polymer.coordinates
        atoms = polymer.atoms

        # Get residue boundaries
        res_sizes = polymer.sizes(Scale.RESIDUE)
        residue_starts = np.concatenate([[0], np.cumsum(res_sizes.numpy() if hasattr(res_sizes, 'numpy') else res_sizes)])
        start = int(residue_starts[residue_idx])
        end = int(residue_starts[residue_idx + 1])

        # Get residue type
        res_type = int(polymer.sequence[residue_idx])
        res = Residue(res_type)

        # Map atom names to local indices
        atom_name_to_local = {}
        for local_idx, atom in enumerate(res.atoms):
            py_name = atom.name.replace("'", "p").replace('"', "pp")
            atom_name_to_local[py_name] = local_idx

        # Get global indices for ring atoms
        ring_global_indices = {}
        for name in ring_atom_names:
            local_idx = atom_name_to_local.get(name)
            if local_idx is not None:
                ring_global_indices[name] = start + local_idx

        # Measure all pairwise distances
        distances = {}
        for i, name1 in enumerate(ring_atom_names):
            for name2 in ring_atom_names[i + 1:]:
                idx1 = ring_global_indices.get(name1)
                idx2 = ring_global_indices.get(name2)
                if idx1 is not None and idx2 is not None:
                    dist = np.linalg.norm(coords[idx1] - coords[idx2])
                    distances[(name1, name2)] = float(dist)

        return distances

    def test_pyrimidine_ring_preserved_on_backbone_rotation(self):
        """Test pyrimidine ring geometry preserved when backbone changes."""
        from ciffy import from_sequence, DihedralType

        # Create uracil (pyrimidine)
        polymer = from_sequence("u")

        # Pyrimidine ring: N1, C2, N3, C4, C5, C6
        ring_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Get initial ring distances
        initial_distances = self._measure_ring_distances(polymer, ring_atoms)

        # Rotate backbone by changing ALPHA dihedral
        alpha = polymer.dihedral(DihedralType.ALPHA)
        if len(alpha) > 0 and not np.isnan(alpha[0]):
            new_alpha = alpha.copy()
            new_alpha[0] = alpha[0] + 1.0  # Rotate by ~57 degrees
            polymer.set_dihedral(DihedralType.ALPHA, new_alpha)

            # Get ring distances after backbone rotation
            final_distances = self._measure_ring_distances(polymer, ring_atoms)

            # Ring distances should be unchanged
            for pair, initial_dist in initial_distances.items():
                final_dist = final_distances.get(pair)
                if final_dist is not None:
                    np.testing.assert_allclose(
                        initial_dist,
                        final_dist,
                        atol=1e-5,
                        err_msg=f"Ring bond {pair} changed from {initial_dist:.4f} to {final_dist:.4f}"
                    )

    def test_purine_ring_preserved_on_backbone_rotation(self):
        """Test purine ring geometry preserved when backbone changes."""
        from ciffy import from_sequence, DihedralType

        # Create adenine (purine)
        polymer = from_sequence("a")

        # Purine rings: 5-membered (N9, C8, N7, C5, C4) + 6-membered (C4, C5, C6, N1, C2, N3)
        ring_atoms = ["N9", "C8", "N7", "C5", "C4", "C6", "N1", "C2", "N3"]

        # Get initial ring distances
        initial_distances = self._measure_ring_distances(polymer, ring_atoms)

        # Rotate backbone by changing ALPHA dihedral
        alpha = polymer.dihedral(DihedralType.ALPHA)
        if len(alpha) > 0 and not np.isnan(alpha[0]):
            new_alpha = alpha.copy()
            new_alpha[0] = alpha[0] + 1.0  # Rotate by ~57 degrees
            polymer.set_dihedral(DihedralType.ALPHA, new_alpha)

            # Get ring distances after backbone rotation
            final_distances = self._measure_ring_distances(polymer, ring_atoms)

            # Ring distances should be unchanged
            for pair, initial_dist in initial_distances.items():
                final_dist = final_distances.get(pair)
                if final_dist is not None:
                    np.testing.assert_allclose(
                        initial_dist,
                        final_dist,
                        atol=1e-5,
                        err_msg=f"Ring bond {pair} changed from {initial_dist:.4f} to {final_dist:.4f}"
                    )

    def test_multi_residue_backbone_rotation_preserves_rings(self):
        """Test backbone rotations in multi-residue structures preserve rings.

        When rotating ALPHA (O3'(i-1)-P-O5'-C5'), the downstream nucleotide
        should rotate as a rigid body, preserving all internal distances.
        """
        from ciffy import from_sequence, DihedralType

        # Create 4-mer to test multiple backbone rotations
        # Sequence: A(0), C(1), G(2), U(3)
        polymer = from_sequence("acgu")

        # Pyrimidine ring atoms for U at position 3
        ring_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Get initial ring distances for residue 3 (U, a pyrimidine)
        initial_distances = self._measure_ring_distances(polymer, ring_atoms, residue_idx=3)

        # Rotate ALPHA dihedral for residue 3 (index 3)
        alpha = polymer.dihedral(DihedralType.ALPHA)
        if len(alpha) > 3 and not np.isnan(alpha[3]):
            new_alpha = alpha.copy()
            new_alpha[3] = alpha[3] + 0.5
            polymer.set_dihedral(DihedralType.ALPHA, new_alpha)

        # Ring distances should be unchanged
        final_distances = self._measure_ring_distances(polymer, ring_atoms, residue_idx=3)

        for pair, initial_dist in initial_distances.items():
            final_dist = final_distances.get(pair)
            if final_dist is not None:
                np.testing.assert_allclose(
                    initial_dist,
                    final_dist,
                    atol=1e-5,
                    err_msg=f"Ring bond {pair} changed from {initial_dist:.4f} to {final_dist:.4f}"
                )

    def test_base_ring_preserved_on_alpha_rotation(self):
        """Test base ring geometry is preserved when ALPHA (phosphate) rotates.

        ALPHA = O3'(i-1)-P-O5'-C5' is upstream of the sugar, so rotating it
        should move the entire nucleotide as a rigid body without affecting
        internal distances. Unlike GAMMA (which is within the sugar ring),
        ALPHA is in the phosphate backbone and doesn't break ring constraints.
        """
        from ciffy import from_sequence, DihedralType

        # Create di-nucleotide (need 2 residues for ALPHA to exist)
        polymer = from_sequence("cc")

        # Pyrimidine ring atoms
        ring_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Get initial ring distances for second residue
        initial_distances = self._measure_ring_distances(polymer, ring_atoms, residue_idx=1)

        # Rotate ALPHA (affects residue 1's position relative to residue 0)
        alpha = polymer.dihedral(DihedralType.ALPHA)
        if len(alpha) > 1 and not np.isnan(alpha[1]):
            new_alpha = alpha.copy()
            new_alpha[1] = alpha[1] + 0.8  # Rotate second residue's ALPHA
            polymer.set_dihedral(DihedralType.ALPHA, new_alpha)

            # Ring distances should be unchanged (rigid body rotation)
            final_distances = self._measure_ring_distances(polymer, ring_atoms, residue_idx=1)

            for pair, initial_dist in initial_distances.items():
                final_dist = final_distances.get(pair)
                if final_dist is not None:
                    np.testing.assert_allclose(
                        initial_dist,
                        final_dist,
                        atol=1e-5,
                        err_msg=f"Ring bond {pair} changed from {initial_dist:.4f} to {final_dist:.4f}"
                    )

    @pytest.mark.xfail(reason="Ring deformation during gradient-based backbone optimization - architectural limitation")
    def test_ring_torsion_during_backbone_optimization(self):
        """Test that rings remain planar during gradient-based backbone optimization.

        This test optimizes backbone dihedrals using gradient descent with a simple
        energy function and measures ring dihedral angles throughout. Currently fails
        because the Z-matrix structure causes ring atoms to move inconsistently when
        backbone dihedrals change.

        The root cause is that ring atoms reference a mix of backbone and ring atoms
        in the Z-matrix. When backbone atoms move, ring atoms that reference them
        move differently than ring atoms that reference other ring atoms, breaking
        ring planarity.
        """
        import torch
        from ciffy import load, DihedralType

        def compute_dihedral_angle(p1, p2, p3, p4):
            """Compute dihedral angle in degrees from 4 points."""
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3
            n1 = torch.linalg.cross(b1, b2)
            n2 = torch.linalg.cross(b2, b3)
            n1 = n1 / (torch.linalg.norm(n1) + 1e-10)
            n2 = n2 / (torch.linalg.norm(n2) + 1e-10)
            m1 = torch.linalg.cross(n1, b2 / (torch.linalg.norm(b2) + 1e-10))
            x = torch.dot(n1, n2)
            y = torch.dot(m1, n2)
            return torch.atan2(y, x) * 180 / np.pi

        def get_ring_dihedrals_for_residue(coords, atoms, res_start, res_end, res_type, is_purine):
            """Get ring dihedral angles for a single residue."""
            from ciffy.biochemistry import Residue

            residue = Residue(res_type)
            atom_enum = residue.atoms

            # Map atom types to global indices
            atom_idx = {}
            for i in range(res_start, res_end):
                atom_type = atoms[i]
                for name in dir(atom_enum):
                    if name.startswith('_'):
                        continue
                    try:
                        member = getattr(atom_enum, name)
                        if hasattr(member, 'value') and member.value == atom_type:
                            atom_idx[name] = i
                            break
                    except:
                        pass

            # Define ring dihedral patterns
            if is_purine:
                patterns = [
                    ["C8", "N9", "C4", "C5"],
                    ["N9", "C4", "C5", "N7"],
                    ["C4", "C5", "N7", "C8"],
                    ["C4", "C5", "C6", "N1"],
                ]
            else:
                patterns = [
                    ["C6", "N1", "C2", "N3"],
                    ["N1", "C2", "N3", "C4"],
                    ["C2", "N3", "C4", "C5"],
                ]

            dihedrals = []
            for atom_names in patterns:
                try:
                    indices = [atom_idx[n] for n in atom_names]
                    angle = compute_dihedral_angle(
                        coords[indices[0]], coords[indices[1]],
                        coords[indices[2]], coords[indices[3]]
                    )
                    dihedrals.append(angle.item())
                except KeyError:
                    pass
            return dihedrals

        def get_all_ring_dihedrals(polymer, coords):
            """Get all ring dihedrals for RNA."""
            from ciffy.biochemistry import Residue

            atoms = polymer._coord_manager._topology.atoms
            sequence = polymer._coord_manager._topology.sequence
            res_sizes = polymer._coord_manager._topology.residue_sizes

            res_starts = np.zeros(len(res_sizes) + 1, dtype=np.int64)
            res_starts[1:] = np.cumsum(res_sizes)

            purine_types = {Residue.A.value, Residue.G.value}
            pyrimidine_types = {Residue.C.value, Residue.U.value}

            all_dihedrals = []
            for res_idx in range(len(sequence)):
                res_type = sequence[res_idx]
                start, end = int(res_starts[res_idx]), int(res_starts[res_idx + 1])

                if res_type in purine_types:
                    dihedrals = get_ring_dihedrals_for_residue(
                        coords, atoms, start, end, res_type, is_purine=True)
                    all_dihedrals.extend(dihedrals)
                elif res_type in pyrimidine_types:
                    dihedrals = get_ring_dihedrals_for_residue(
                        coords, atoms, start, end, res_type, is_purine=False)
                    all_dihedrals.extend(dihedrals)
            return all_dihedrals

        # Load RNA structure
        rna = load(get_test_cif("9GCM")).by_index(0).torch()

        # Get initial ring dihedrals
        coords_initial = rna.coordinates.clone()
        ring_dihedrals_initial = get_all_ring_dihedrals(rna, coords_initial)

        # Setup backbone optimization
        cm = rna._coord_manager
        zm = cm.zmatrix
        dihedral_types = zm.dihedral_types

        # Include CHI (glycosidic) dihedrals - these connect base to sugar
        # and cause ring deformation when optimized
        backbone_types = [
            DihedralType.ALPHA, DihedralType.BETA, DihedralType.GAMMA,
            DihedralType.DELTA, DihedralType.EPSILON, DihedralType.ZETA,
            DihedralType.CHI_PURINE, DihedralType.CHI_PYRIMIDINE,
        ]
        backbone_mask = torch.zeros(len(dihedral_types), dtype=torch.bool)
        for dt in backbone_types:
            backbone_mask |= torch.from_numpy(dihedral_types == dt.value)

        dihedrals = cm.dihedrals.clone()
        dihedrals.requires_grad_(True)

        # Run optimization steps
        n_steps = 20
        lr = 0.01
        max_deviation = 0.0

        for step in range(n_steps):
            cm._dihedrals = dihedrals
            cm._invalidate_cartesian()
            coords = cm.coordinates

            # Simple energy: sum of squared coordinates
            E = (coords ** 2).sum()

            # Measure ring deviation
            ring_dihedrals_current = get_all_ring_dihedrals(rna, coords.detach())
            for init, curr in zip(ring_dihedrals_initial, ring_dihedrals_current):
                diff = abs(init - curr)
                if diff > 180:
                    diff = 360 - diff
                max_deviation = max(max_deviation, diff)

            # Update backbone dihedrals
            E.backward()
            with torch.no_grad():
                grad = dihedrals.grad
                if grad is not None:
                    update = torch.zeros_like(dihedrals)
                    update[backbone_mask] = grad[backbone_mask]
                    dihedrals -= lr * update
                    dihedrals.grad.zero_()

        # Rings should remain planar (< 5 degree deviation)
        assert max_deviation < 5.0, (
            f"Ring deformation detected: max deviation {max_deviation:.1f}° >= 5°. "
            "This is a known architectural limitation where ring atoms reference "
            "a mix of backbone and ring atoms in the Z-matrix."
        )


# =============================================================================
# GPU Device Tests (parameterized across CUDA/MPS)
# =============================================================================

class TestInternalCoordinatesGPU:
    """Tests for internal coordinate operations on GPU devices.

    These tests run on all available GPU devices (CUDA, MPS) and verify
    that coordinate conversions work correctly on accelerator hardware.
    """

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_roundtrip_on_gpu(self, device):
        """Test round-trip conversion on GPU device."""
        skip_if_no_device(device)
        import torch
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu", backend="torch").to(device)
        orig_coords = polymer.coordinates.clone()

        # Trigger reconstruction by setting dihedrals
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals

        new_coords = polymer.coordinates
        assert new_coords.device.type == device

        # Compare using Kabsch alignment
        aligned, _, _ = kabsch_align(new_coords.cpu(), orig_coords.cpu())
        rmsd = torch.sqrt(((aligned - orig_coords.cpu()) ** 2).sum(dim=1).mean())

        assert rmsd.item() < 1e-4, f"GPU round-trip RMSD {rmsd.item()} exceeds threshold"

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_internal_coords_on_gpu(self, device):
        """Test accessing internal coordinates on GPU device."""
        skip_if_no_device(device)
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch").to(device)

        distances = polymer.distances
        angles = polymer.angles
        dihedrals = polymer.dihedrals

        assert distances.device.type == device
        assert angles.device.type == device
        assert dihedrals.device.type == device

        # Values should be reasonable
        assert torch.all(distances >= 0)
        assert torch.all(angles >= 0) and torch.all(angles <= np.pi + 1e-5)
        assert torch.all(dihedrals >= -np.pi - 1e-5) and torch.all(dihedrals <= np.pi + 1e-5)

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_pdb_roundtrip_on_gpu(self, device):
        """Test round-trip on real PDB structure on GPU."""
        skip_if_no_device(device)
        import torch
        from ciffy import load
        from ciffy.operations.alignment import kabsch_align

        polymer = load(get_test_cif("1ZEW")).poly().torch().to(device)
        orig_coords = polymer.coordinates.clone()

        # Trigger reconstruction
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals

        new_coords = polymer.coordinates

        aligned, _, _ = kabsch_align(new_coords.cpu(), orig_coords.cpu())
        rmsd = torch.sqrt(((aligned - orig_coords.cpu()) ** 2).sum(dim=1).mean())

        assert rmsd.item() < 1e-3, f"PDB round-trip RMSD {rmsd.item()} exceeds threshold"

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_gpu_cpu_transfer(self, device):
        """Test moving between CPU and GPU preserves internal coordinates."""
        skip_if_no_device(device)
        import torch
        from ciffy import from_sequence

        polymer_cpu = from_sequence("acgu", backend="torch")
        dihedrals_cpu = polymer_cpu.dihedrals.clone()

        polymer_gpu = polymer_cpu.to(device)
        dihedrals_gpu = polymer_gpu.dihedrals

        assert torch.allclose(dihedrals_cpu, dihedrals_gpu.cpu(), atol=1e-5)

        # Move back to CPU
        polymer_back = polymer_gpu.to("cpu")
        dihedrals_back = polymer_back.dihedrals

        assert torch.allclose(dihedrals_cpu, dihedrals_back, atol=1e-5)

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_differentiability_on_gpu(self, device):
        """Test gradient flow through reconstruction on GPU."""
        skip_if_no_device(device)
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch").to(device)

        # Enable gradients on dihedrals
        dihedrals = polymer.dihedrals.clone()
        dihedrals.requires_grad_(True)
        polymer.dihedrals = dihedrals

        # Access coordinates (triggers reconstruction)
        coords = polymer.coordinates

        # Compute loss and backpropagate
        loss = coords.pow(2).mean()
        loss.backward()

        # Gradients should exist and be on GPU
        assert dihedrals.grad is not None
        assert dihedrals.grad.device.type == device
        assert not torch.all(dihedrals.grad == 0)


class TestAutogradGradientsGPU:
    """Tests for autograd gradient correctness on GPU devices.

    These tests verify that gradients computed on GPU match those from CPU,
    ensuring numerical consistency across devices.
    """

    @requires_cuda
    @requires_cuda_extension
    def test_cartesian_to_internal_cuda_matches_cpu(self):
        """Test CUDA cartesian_to_internal matches CPU results."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [3.5, 1.5, 1.0]
        ], dtype=np.float32)

        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        coords_cpu = torch.tensor(coords_np, dtype=torch.float32)
        indices = torch.tensor(indices_np, dtype=torch.int64)

        # CPU computation
        d_cpu, a_cpu, dh_cpu = cartesian_to_internal(coords_cpu, indices)

        # CUDA computation
        d_cuda, a_cuda, dh_cuda = cartesian_to_internal(
            coords_cpu.cuda(), indices.cuda()
        )

        assert torch.allclose(d_cpu, d_cuda.cpu(), atol=1e-5)
        assert torch.allclose(a_cpu, a_cuda.cpu(), atol=1e-5)
        assert torch.allclose(dh_cpu, dh_cuda.cpu(), atol=1e-5)

    @requires_cuda
    @requires_cuda_extension
    def test_nerf_reconstruct_cuda_matches_cpu(self):
        """Test CUDA nerf_reconstruct matches CPU results."""
        import torch
        from ciffy.backend.autograd import nerf_reconstruct

        n_atoms = 4
        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        distances = torch.tensor([0.0, 1.5, 1.5, 1.5], dtype=torch.float32)
        angles = torch.tensor([0.0, 0.0, 1.91, 1.91], dtype=torch.float32)
        dihedrals = torch.tensor([0.0, 0.0, 0.0, 1.57], dtype=torch.float32)
        indices = torch.tensor(indices_np, dtype=torch.int64)

        # CPU reconstruction
        coords_cpu = nerf_reconstruct(indices, distances, angles, dihedrals, n_atoms)

        # CUDA reconstruction
        coords_cuda = nerf_reconstruct(
            indices.cuda(), distances.cuda(), angles.cuda(), dihedrals.cuda(), n_atoms
        )

        assert torch.allclose(coords_cpu, coords_cuda.cpu(), atol=1e-5)

    @requires_cuda
    @requires_cuda_extension
    def test_cartesian_to_internal_grad_matches_cpu(self):
        """Test CUDA gradients match CPU gradients for cartesian_to_internal."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.3, 0.2],
            [2.0, 1.5, 0.1],
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

        indices = torch.tensor(indices_np, dtype=torch.int64)

        # CPU gradient
        coords_cpu = torch.tensor(coords_np, requires_grad=True)
        d, a, dh = cartesian_to_internal(coords_cpu, indices)
        loss = d.sum() + a.sum() + dh.sum()
        loss.backward()
        grad_cpu = coords_cpu.grad.clone()

        # CUDA gradient
        coords_cuda = torch.tensor(coords_np, device="cuda", requires_grad=True)
        d, a, dh = cartesian_to_internal(coords_cuda, indices.cuda())
        loss = d.sum() + a.sum() + dh.sum()
        loss.backward()
        grad_cuda = coords_cuda.grad.cpu()

        assert torch.allclose(grad_cpu, grad_cuda, atol=1e-4)

    @requires_cuda
    @requires_cuda_extension
    def test_nerf_reconstruct_grad_matches_cpu(self):
        """Test CUDA gradients match CPU gradients for nerf_reconstruct."""
        import torch
        from ciffy.backend.autograd import nerf_reconstruct

        n_atoms = 4
        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        distances_np = np.array([0.0, 1.5, 1.5, 1.5], dtype=np.float32)
        angles_np = np.array([0.0, 0.0, 1.91, 1.91], dtype=np.float32)
        dihedrals_np = np.array([0.0, 0.0, 0.0, 1.57], dtype=np.float32)

        # CPU gradient
        indices = torch.tensor(indices_np)
        distances_cpu = torch.tensor(distances_np, requires_grad=True)
        angles_cpu = torch.tensor(angles_np, requires_grad=True)
        dihedrals_cpu = torch.tensor(dihedrals_np, requires_grad=True)

        coords = nerf_reconstruct(indices, distances_cpu, angles_cpu, dihedrals_cpu, n_atoms)
        coords.sum().backward()

        grad_d_cpu = distances_cpu.grad.clone()
        grad_a_cpu = angles_cpu.grad.clone()
        grad_dh_cpu = dihedrals_cpu.grad.clone()

        # CUDA gradient
        distances_cuda = torch.tensor(distances_np, device="cuda", requires_grad=True)
        angles_cuda = torch.tensor(angles_np, device="cuda", requires_grad=True)
        dihedrals_cuda = torch.tensor(dihedrals_np, device="cuda", requires_grad=True)

        coords = nerf_reconstruct(
            indices.cuda(), distances_cuda, angles_cuda, dihedrals_cuda, n_atoms
        )
        coords.sum().backward()

        assert torch.allclose(grad_d_cpu, distances_cuda.grad.cpu(), atol=1e-4)
        assert torch.allclose(grad_a_cpu, angles_cuda.grad.cpu(), atol=1e-4)
        assert torch.allclose(grad_dh_cpu, dihedrals_cuda.grad.cpu(), atol=1e-4)

    @requires_cuda
    @requires_cuda_extension
    def test_cuda_gradcheck(self):
        """Test CUDA gradient correctness via gradcheck."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [3.5, 1.5, 1.0]
        ], dtype=np.float32)

        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        indices = torch.tensor(indices_np, dtype=torch.int64, device="cuda")

        def wrapper(coords):
            coords32 = coords.float()
            d, a, dh = cartesian_to_internal(coords32, indices)
            return d.double(), a.double(), dh.double()

        coords_check = torch.tensor(
            coords_np, requires_grad=True, dtype=torch.float64, device="cuda"
        )
        assert torch.autograd.gradcheck(
            wrapper, coords_check, eps=1e-4, atol=1e-3, rtol=1e-2
        )


class TestInternalCoordsEdgeCasesGPU:
    """Edge case tests for internal coordinates on GPU."""

    @requires_cuda
    @requires_cuda_extension
    def test_single_atom_cuda(self):
        """Test handling of single-atom structure on CUDA."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
        indices = torch.tensor([[0, -1, -1, -1]], dtype=torch.int64, device="cuda")

        d, a, dh = cartesian_to_internal(coords, indices)

        assert len(d) == 1
        assert len(a) == 1
        assert len(dh) == 1

    @requires_cuda
    @requires_cuda_extension
    def test_two_atoms_cuda(self):
        """Test handling of two-atom structure on CUDA."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ], dtype=torch.float32, device="cuda")
        indices = torch.tensor([
            [0, -1, -1, -1],
            [1,  0, -1, -1]
        ], dtype=torch.int64, device="cuda")

        d, a, dh = cartesian_to_internal(coords, indices)

        assert len(d) == 2
        assert d[1].item() == pytest.approx(1.5, abs=1e-5)

    @requires_cuda
    @requires_cuda_extension
    def test_large_structure_cuda(self):
        """Test CUDA handles large structures."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal, nerf_reconstruct

        n_atoms = 1000
        coords_np = np.random.randn(n_atoms, 3).astype(np.float32) * 10

        # Create linear Z-matrix
        indices_np = np.zeros((n_atoms, 4), dtype=np.int64)
        indices_np[:, 0] = np.arange(n_atoms)
        indices_np[1:, 1] = np.arange(n_atoms - 1)
        indices_np[2:, 2] = np.arange(n_atoms - 2)
        indices_np[3:, 3] = np.arange(n_atoms - 3)
        indices_np[0, 1:] = -1
        indices_np[1, 2:] = -1
        indices_np[2, 3] = -1

        coords_cuda = torch.tensor(coords_np, device="cuda")
        indices_cuda = torch.tensor(indices_np, device="cuda")

        d, a, dh = cartesian_to_internal(coords_cuda, indices_cuda)
        assert len(d) == n_atoms

        coords_recon = nerf_reconstruct(indices_cuda, d, a, dh, n_atoms)
        assert coords_recon.shape == (n_atoms, 3)


class TestBFSLevels:
    """Tests for BFS level computation in Z-matrix construction."""

    def test_zmatrix_has_levels(self):
        """Test that ZMatrix stores BFS levels."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        _ = polymer.dihedrals  # Trigger Z-matrix build

        zm = polymer._coord_manager.zmatrix
        assert zm.levels is not None
        assert len(zm.levels) == len(zm)

    def test_levels_parent_child_relationship(self):
        """Test that child level = parent level + 1."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW")).poly()
        _ = polymer.dihedrals

        zm = polymer._coord_manager.zmatrix
        indices = zm.indices
        levels = zm.levels

        # Build atom_idx -> level map from Z-matrix
        atom_to_level = {}
        for i in range(len(indices)):
            atom_idx = int(indices[i, 0])
            atom_to_level[atom_idx] = int(levels[i])

        # Verify parent-child relationships
        violations = 0
        for i in range(len(indices)):
            dist_ref = int(indices[i, 1])
            if dist_ref >= 0:  # Has a parent
                parent_level = atom_to_level.get(dist_ref, -1)
                child_level = int(levels[i])
                if parent_level >= 0 and child_level != parent_level + 1:
                    violations += 1

        assert violations == 0, f"Found {violations} parent-child level violations"

    def test_level_offsets_partition_entries(self):
        """Test that level_offsets CSR format covers all entries."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW")).poly()
        _ = polymer.dihedrals

        zm = polymer._coord_manager.zmatrix
        offsets = zm.level_offsets

        assert offsets is not None
        # CSR format: offsets[i]:offsets[i+1] defines entries at level i
        assert offsets[0] == 0, "Offsets should start at 0"
        assert offsets[-1] == len(zm), f"Offsets should end at {len(zm)}, got {offsets[-1]}"

        # Offsets should be monotonically non-decreasing
        for i in range(len(offsets) - 1):
            assert offsets[i] <= offsets[i + 1], f"Offsets not monotonic at {i}"

    def test_level_offsets_consistent_with_levels(self):
        """Test that level_offsets matches per-entry levels."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW")).poly()
        _ = polymer.dihedrals

        zm = polymer._coord_manager.zmatrix
        levels = zm.levels
        offsets = zm.level_offsets

        n_levels = int(levels.max()) + 1

        # Count entries per level from levels array
        level_counts = np.bincount(levels.astype(np.int64), minlength=n_levels)

        # Counts from offsets
        offset_counts = np.diff(offsets)

        assert len(level_counts) == len(offset_counts), \
            f"Level count mismatch: {len(level_counts)} vs {len(offset_counts)}"
        assert np.array_equal(level_counts, offset_counts), \
            "Level offsets don't match level counts"

    def test_multichain_levels(self):
        """Test levels are computed correctly for multi-chain structures."""
        from ciffy import load

        polymer = load(get_test_cif("1ZEW")).poly()
        n_chains = len(polymer.lengths)
        assert n_chains > 1, "Test requires multi-chain structure"

        _ = polymer.dihedrals

        zm = polymer._coord_manager.zmatrix
        levels = zm.levels

        # Each chain should have level 0 entries (chain roots)
        n_roots = np.sum(levels == 0)
        assert n_roots >= n_chains, f"Expected at least {n_chains} roots, got {n_roots}"

    def test_levels_reasonable_depth(self):
        """Test BFS depth is reasonable for molecular structures."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        _ = polymer.dihedrals

        zm = polymer._coord_manager.zmatrix
        levels = zm.levels

        max_level = int(levels.max())
        n_atoms = len(zm)

        # BFS depth should be much less than number of atoms for realistic molecules
        # (typical depth is sqrt(N) to log(N) depending on branching)
        assert max_level < n_atoms, f"BFS depth {max_level} >= n_atoms {n_atoms}"

        # For a tetramer (~130 atoms), depth should be < 100
        assert max_level < 100, f"BFS depth {max_level} seems too large"


class TestLeveledNERFCUDA:
    """Tests for level-parallel NERF reconstruction on CUDA."""

    @requires_cuda
    @requires_cuda_extension
    def test_leveled_nerf_available(self):
        """Test that leveled NERF is available on CUDA."""
        from ciffy.backend.cuda_ops import HAS_LEVELED_NERF
        # This may be False if cooperative groups not supported
        # Just check the import works
        assert isinstance(HAS_LEVELED_NERF, bool)

    @requires_cuda
    @requires_cuda_extension
    def test_level_offsets_passed_to_nerf(self):
        """Test that level_offsets flows through to CUDA NERF."""
        import torch
        from ciffy import load
        from ciffy.backend.cuda_ops import HAS_LEVELED_NERF

        if not HAS_LEVELED_NERF:
            pytest.skip("Leveled NERF CUDA kernel not available")

        polymer = load(get_test_cif("1ZEW")).poly().torch().cuda()
        _ = polymer.dihedrals

        zm = polymer._coord_manager.zmatrix
        assert zm.level_offsets is not None

        # Trigger reconstruction
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals
        coords = polymer.coordinates

        # Should still be on CUDA
        assert coords.is_cuda

    @requires_cuda
    @requires_cuda_extension
    def test_leveled_cuda_matches_cpu(self):
        """Test leveled CUDA NERF matches CPU results."""
        import torch
        from ciffy import load
        from ciffy.backend.cuda_ops import HAS_LEVELED_NERF

        if not HAS_LEVELED_NERF:
            pytest.skip("Leveled NERF CUDA kernel not available")

        # Load on CPU
        polymer_cpu = load(get_test_cif("1ZEW")).poly().torch()
        orig_coords = polymer_cpu.coordinates.clone()

        # Trigger CPU reconstruction
        dihedrals = polymer_cpu.dihedrals.clone()
        polymer_cpu.dihedrals = dihedrals
        coords_cpu = polymer_cpu.coordinates

        # Load on CUDA
        polymer_cuda = load(get_test_cif("1ZEW")).poly().torch().cuda()
        dihedrals_cuda = polymer_cuda.dihedrals.clone()
        polymer_cuda.dihedrals = dihedrals_cuda
        coords_cuda = polymer_cuda.coordinates

        # Compare (1e-4 tolerance needed for accumulated float32 precision differences)
        assert torch.allclose(coords_cpu, coords_cuda.cpu(), atol=1e-4)
