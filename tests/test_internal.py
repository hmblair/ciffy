"""Tests for internal coordinates (Z-matrix) representation."""

import pytest
import numpy as np

from tests.utils import (
    get_test_cif,
    GPU_DEVICES,
    skip_if_no_device,
)
from tests.testing import (
    assert_roundtrip_preserves_structure,
    assert_gradient_flows,
    assert_valid_angles,
    assert_valid_dihedrals,
    assert_positive_distances,
    get_tolerances,
)


class TestInternalCoordinatesBasic:
    """Basic tests for internal coordinate conversion."""

    @pytest.mark.parametrize("sequence,tolerance_key", [
        ("a", "roundtrip_single_residue"),
        ("acgu", "roundtrip_small"),
    ])
    def test_roundtrip(self, sequence, tolerance_key):
        """Test round-trip conversion preserves structure."""
        from ciffy import from_sequence

        polymer = from_sequence(sequence)
        assert_roundtrip_preserves_structure(polymer, tolerance_key=tolerance_key)


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
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        assert_roundtrip_preserves_structure(polymer, tolerance_key="roundtrip_small")

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
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        assert_gradient_flows(polymer)


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

class TestSetMethods:
    """Tests for setting internal coordinates."""

    @pytest.mark.parametrize("coord_type,index,new_value", [
        ("dihedrals", 5, 1.5),
        ("angles", 5, 2.0),
        ("distances", 5, 2.0),
    ])
    def test_set_internal_coords(self, coord_type, index, new_value):
        """Test setting internal coordinates modifies polymer in-place."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")

        # Get, modify, set
        coords = getattr(polymer, coord_type).copy()
        coords[index] = new_value
        setattr(polymer, coord_type, coords)

        # Verify modification
        assert getattr(polymer, coord_type)[index] == new_value


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
        from ciffy import load, kabsch_align

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

        # Use testing infrastructure assertions
        assert_positive_distances(protein_chain.distances, skip_first=1)
        assert_valid_angles(protein_chain.angles, skip_first=2)
        assert_valid_dihedrals(protein_chain.dihedrals, skip_first=3)


class TestNumericalEdgeCases:
    """Tests for numerical edge cases and stability."""

    def test_small_perturbation_roundtrip(self):
        """Test very small dihedral changes preserve structure."""
        from ciffy import from_sequence, kabsch_align

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

        polymer = from_sequence("acgu")
        assert_roundtrip_preserves_structure(polymer, tolerance_key="roundtrip_small")


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



class TestRingPreservation:
    """Tests for ring geometry preservation during backbone manipulation.

    The canonical Z-matrix construction uses ring-internal dihedrals that
    preserve ring geometry when only backbone dihedrals are modified.
    """

    def _measure_ring_distances(self, polymer, ring_atom_names, residue_idx=0):
        """Measure all pairwise distances between ring atoms.

        Uses the atom type array for correct indexing, since terminal atoms
        are filtered differently for first/middle/last residues.

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
        if hasattr(coords, 'numpy'):
            coords = coords.numpy()

        atoms_array = polymer.atoms
        if hasattr(atoms_array, 'numpy'):
            atoms_array = atoms_array.numpy()

        # Get residue boundaries
        res_sizes = polymer.sizes(Scale.RESIDUE)
        if hasattr(res_sizes, 'numpy'):
            res_sizes = res_sizes.numpy()
        residue_starts = np.concatenate([[0], np.cumsum(res_sizes)])
        start = int(residue_starts[residue_idx])
        end = int(residue_starts[residue_idx + 1]) if residue_idx + 1 < len(residue_starts) else len(coords)

        # Get residue type and build atom type -> name mapping
        res_type = int(polymer.sequence[residue_idx])
        res = Residue(res_type)

        atom_type_to_name = {}
        for atom in res.atoms:
            atom_type_to_name[atom.value] = atom.name.replace("'", "p").replace('"', "pp")

        # Find ring atoms by looking up actual atom types in this residue
        ring_global_indices = {}
        for i in range(start, end):
            atom_type = int(atoms_array[i])
            atom_name = atom_type_to_name.get(atom_type, '')
            if atom_name in ring_atom_names:
                ring_global_indices[atom_name] = i

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
        """Test pyrimidine ring geometry preserved when backbone changes.

        GAMMA is upstream of the nucleobase in the Z-matrix, so rotating
        it should not affect ring geometry - the ring moves as a rigid body.
        """
        from ciffy import from_sequence, DihedralType

        # Create uracil (pyrimidine)
        polymer = from_sequence("u")

        # Pyrimidine ring: N1, C2, N3, C4, C5, C6
        ring_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Get initial ring distances
        initial_distances = self._measure_ring_distances(polymer, ring_atoms)

        # Rotate backbone by changing GAMMA dihedral (exists on single residues)
        gamma = polymer.dihedral(DihedralType.GAMMA)
        assert len(gamma) > 0, "Expected GAMMA dihedral to exist"
        assert not np.isnan(gamma[0]), "GAMMA should not be NaN"

        new_gamma = gamma.copy()
        new_gamma[0] = gamma[0] + 1.0  # Rotate by ~57 degrees
        polymer.set_dihedral(DihedralType.GAMMA, new_gamma)

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
        """Test purine ring geometry preserved when backbone changes.

        GAMMA is upstream of the nucleobase in the Z-matrix, so rotating
        it should not affect ring geometry - the ring moves as a rigid body.
        """
        from ciffy import from_sequence, DihedralType

        # Create adenine (purine)
        polymer = from_sequence("a")

        # Purine rings: 5-membered (N9, C8, N7, C5, C4) + 6-membered (C4, C5, C6, N1, C2, N3)
        ring_atoms = ["N9", "C8", "N7", "C5", "C4", "C6", "N1", "C2", "N3"]

        # Get initial ring distances
        initial_distances = self._measure_ring_distances(polymer, ring_atoms)

        # Rotate backbone by changing GAMMA dihedral
        gamma = polymer.dihedral(DihedralType.GAMMA)
        assert len(gamma) > 0, "Expected GAMMA dihedral to exist"
        assert not np.isnan(gamma[0]), "GAMMA should not be NaN"

        new_gamma = gamma.copy()
        new_gamma[0] = gamma[0] + 1.0  # Rotate by ~57 degrees
        polymer.set_dihedral(DihedralType.GAMMA, new_gamma)

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

        ALPHA is upstream of the nucleobase in the Z-matrix, so rotating
        it should move the entire nucleotide as a rigid body without
        affecting ring geometry.
        """
        from ciffy import from_sequence, DihedralType

        # Create 4-mer to test multiple backbone rotations
        # Sequence: A(0), C(1), G(2), U(3)
        polymer = from_sequence("acgu")

        # Pyrimidine ring atoms for U at position 3
        ring_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Get initial ring distances for residue 3 (U, a pyrimidine)
        initial_distances = self._measure_ring_distances(polymer, ring_atoms, residue_idx=3)

        # Rotate ALPHA dihedral - for 4-mer, alpha has 3 values (for residues 1, 2, 3)
        # alpha[2] corresponds to residue 3
        alpha = polymer.dihedral(DihedralType.ALPHA)
        assert len(alpha) >= 3, f"Expected at least 3 ALPHA dihedrals, got {len(alpha)}"
        assert not np.isnan(alpha[2]), "ALPHA[2] should not be NaN"

        new_alpha = alpha.copy()
        new_alpha[2] = alpha[2] + 0.5
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
        internal distances.

        Note: For a 2-residue polymer, ALPHA only exists for residue 1 (index 0
        in the dihedral array), since residue 0 has no previous O3'.
        """
        from ciffy import from_sequence, DihedralType

        # Create di-nucleotide (need 2 residues for ALPHA to exist)
        polymer = from_sequence("cc")

        # Pyrimidine ring atoms
        ring_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Get initial ring distances for second residue (which has ALPHA)
        initial_distances = self._measure_ring_distances(polymer, ring_atoms, residue_idx=1)

        # Rotate ALPHA - for cc, alpha[0] is for residue 1
        alpha = polymer.dihedral(DihedralType.ALPHA)
        assert len(alpha) >= 1, "Expected at least one ALPHA dihedral"
        assert not np.isnan(alpha[0]), "ALPHA[0] should not be NaN"

        new_alpha = alpha.copy()
        new_alpha[0] = alpha[0] + 0.8
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

    def _measure_ring_planarity(self, polymer, ring_atom_names, residue_idx=0):
        """Measure how planar a ring is by computing RMS deviation from best-fit plane.

        Uses the atom type array for correct indexing, since terminal atoms
        are filtered differently for first/middle/last residues.

        Args:
            polymer: Polymer structure
            ring_atom_names: List of ring atom names (e.g., ['N9', 'C8', ...])
            residue_idx: Which residue to measure

        Returns:
            RMS deviation from best-fit plane in Angstroms
        """
        from ciffy import Scale
        from ciffy.biochemistry import Residue

        coords = polymer.coordinates
        if hasattr(coords, 'numpy'):
            coords = coords.numpy()

        atoms_array = polymer.atoms
        if hasattr(atoms_array, 'numpy'):
            atoms_array = atoms_array.numpy()

        # Get residue boundaries
        res_sizes = polymer.sizes(Scale.RESIDUE)
        if hasattr(res_sizes, 'numpy'):
            res_sizes = res_sizes.numpy()
        residue_starts = np.concatenate([[0], np.cumsum(res_sizes)])
        start = int(residue_starts[residue_idx])
        end = int(residue_starts[residue_idx + 1]) if residue_idx + 1 < len(residue_starts) else len(coords)

        # Get residue type and build atom type -> name mapping
        res_type = int(polymer.sequence[residue_idx])
        res = Residue(res_type)

        atom_type_to_name = {}
        for atom in res.atoms:
            atom_type_to_name[atom.value] = atom.name.replace("'", "p").replace('"', "pp")

        # Find ring atoms by looking up actual atom types in this residue
        ring_coords = []
        for i in range(start, end):
            atom_type = int(atoms_array[i])
            atom_name = atom_type_to_name.get(atom_type, '')
            if atom_name in ring_atom_names:
                ring_coords.append(coords[i])

        if len(ring_coords) < 3:
            return 0.0  # Can't measure planarity with < 3 atoms

        ring_coords = np.array(ring_coords)

        # Fit plane using SVD: find normal vector to plane
        centroid = ring_coords.mean(axis=0)
        centered = ring_coords - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # Last row is normal to best-fit plane

        # Compute signed distances from plane
        distances = centered @ normal

        # Return RMS deviation
        return float(np.sqrt(np.mean(distances ** 2)))

    def test_initial_ring_planarity(self):
        """Test that template-generated structures have planar nucleobase rings.

        This verifies that from_sequence() produces structures with correct
        ring geometry. All nucleobase rings should have RMS planarity < 0.01 Å.
        """
        from ciffy import from_sequence

        purine_atoms = ["N9", "C8", "N7", "C5", "C4", "C6", "N1", "C2", "N3"]
        pyrimidine_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Test multi-residue polymer - all residues should have planar rings
        polymer = from_sequence("acgu")

        # A (purine) at position 0
        planarity_a = self._measure_ring_planarity(polymer, purine_atoms, residue_idx=0)
        assert planarity_a < 0.01, f"Adenine ring not planar: {planarity_a:.4f} Å"

        # C (pyrimidine) at position 1
        planarity_c = self._measure_ring_planarity(polymer, pyrimidine_atoms, residue_idx=1)
        assert planarity_c < 0.01, f"Cytosine ring not planar: {planarity_c:.4f} Å"

        # G (purine) at position 2
        planarity_g = self._measure_ring_planarity(polymer, purine_atoms, residue_idx=2)
        assert planarity_g < 0.01, f"Guanine ring not planar: {planarity_g:.4f} Å"

        # U (pyrimidine) at position 3
        planarity_u = self._measure_ring_planarity(polymer, pyrimidine_atoms, residue_idx=3)
        assert planarity_u < 0.01, f"Uracil ring not planar: {planarity_u:.4f} Å"

    @pytest.mark.xfail(reason="Ring deformation during CHI modification - Z-matrix ring dihedrals not yet implemented")
    def test_pyrimidine_ring_preserved_on_chi_rotation(self):
        """Test pyrimidine ring geometry preserved when CHI (glycosidic) angle changes.

        CHI_PYRIMIDINE = O4' - C1' - N1 - C2 defines rotation around the glycosidic bond.
        When CHI rotates, the entire nucleobase should rotate as a rigid body.
        Ring internal distances must remain constant.

        This is a critical test: if ring dihedrals are not properly included in the
        Z-matrix, modifying CHI will implicitly change them, deforming the ring.
        """
        from ciffy import from_sequence, DihedralType

        # Create uracil (pyrimidine)
        polymer = from_sequence("u")

        # Pyrimidine ring: N1, C2, N3, C4, C5, C6
        ring_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Get initial ring distances
        initial_distances = self._measure_ring_distances(polymer, ring_atoms)
        initial_planarity = self._measure_ring_planarity(polymer, ring_atoms)

        # Rotate CHI by a significant amount (90 degrees)
        chi = polymer.dihedral(DihedralType.CHI_PYRIMIDINE)
        assert len(chi) > 0, "No CHI_PYRIMIDINE dihedrals found"
        assert not np.isnan(chi[0]), "CHI_PYRIMIDINE is NaN"

        new_chi = chi.copy()
        new_chi[0] = chi[0] + np.pi / 2  # Rotate by 90 degrees
        polymer.set_dihedral(DihedralType.CHI_PYRIMIDINE, new_chi)

        # Get ring distances after CHI rotation
        final_distances = self._measure_ring_distances(polymer, ring_atoms)
        final_planarity = self._measure_ring_planarity(polymer, ring_atoms)

        # Ring distances should be unchanged (rigid body rotation)
        for pair, initial_dist in initial_distances.items():
            final_dist = final_distances.get(pair)
            if final_dist is not None:
                np.testing.assert_allclose(
                    initial_dist,
                    final_dist,
                    atol=1e-4,
                    err_msg=f"Ring bond {pair} changed from {initial_dist:.4f} to {final_dist:.4f} after CHI rotation"
                )

        # Ring planarity should be preserved
        np.testing.assert_allclose(
            initial_planarity,
            final_planarity,
            atol=1e-4,
            err_msg=f"Ring planarity changed from {initial_planarity:.4f} to {final_planarity:.4f} after CHI rotation"
        )

    @pytest.mark.xfail(reason="Ring deformation during CHI modification - Z-matrix ring dihedrals not yet implemented")
    def test_purine_ring_preserved_on_chi_rotation(self):
        """Test purine ring geometry preserved when CHI (glycosidic) angle changes.

        CHI_PURINE = O4' - C1' - N9 - C4 defines rotation around the glycosidic bond.
        Purines have fused 5+6 membered rings that must remain planar.
        """
        from ciffy import from_sequence, DihedralType

        # Create adenine (purine)
        polymer = from_sequence("a")

        # Purine has fused rings: 5-membered (N9, C8, N7, C5, C4) + 6-membered (C4, C5, C6, N1, C2, N3)
        ring_atoms = ["N9", "C8", "N7", "C5", "C4", "C6", "N1", "C2", "N3"]

        # Get initial ring distances
        initial_distances = self._measure_ring_distances(polymer, ring_atoms)
        initial_planarity = self._measure_ring_planarity(polymer, ring_atoms)

        # Rotate CHI by a significant amount (90 degrees)
        chi = polymer.dihedral(DihedralType.CHI_PURINE)
        assert len(chi) > 0, "No CHI_PURINE dihedrals found"
        assert not np.isnan(chi[0]), "CHI_PURINE is NaN"

        new_chi = chi.copy()
        new_chi[0] = chi[0] + np.pi / 2  # Rotate by 90 degrees
        polymer.set_dihedral(DihedralType.CHI_PURINE, new_chi)

        # Get ring distances after CHI rotation
        final_distances = self._measure_ring_distances(polymer, ring_atoms)
        final_planarity = self._measure_ring_planarity(polymer, ring_atoms)

        # Ring distances should be unchanged
        for pair, initial_dist in initial_distances.items():
            final_dist = final_distances.get(pair)
            if final_dist is not None:
                np.testing.assert_allclose(
                    initial_dist,
                    final_dist,
                    atol=1e-4,
                    err_msg=f"Ring bond {pair} changed from {initial_dist:.4f} to {final_dist:.4f} after CHI rotation"
                )

        # Ring planarity should be preserved
        np.testing.assert_allclose(
            initial_planarity,
            final_planarity,
            atol=1e-4,
            err_msg=f"Ring planarity changed from {initial_planarity:.4f} to {final_planarity:.4f} after CHI rotation"

        )

    @pytest.mark.xfail(reason="Ring deformation during CHI modification - Z-matrix ring dihedrals not yet implemented")
    def test_multi_residue_chi_rotation_preserves_rings(self):
        """Test CHI rotation in multi-residue RNA preserves all ring geometries.

        This tests that modifying CHI for one residue doesn't affect rings
        in other residues (which could happen if Z-matrix dependencies are wrong).
        """
        from ciffy import from_sequence, DihedralType

        # Create 4-mer with both purines and pyrimidines
        polymer = from_sequence("acgu")

        # Measure initial ring distances for all residues
        purine_atoms = ["N9", "C8", "N7", "C5", "C4", "C6", "N1", "C2", "N3"]
        pyrimidine_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        initial_distances = {
            0: self._measure_ring_distances(polymer, purine_atoms, residue_idx=0),      # A
            1: self._measure_ring_distances(polymer, pyrimidine_atoms, residue_idx=1),  # C
            2: self._measure_ring_distances(polymer, purine_atoms, residue_idx=2),      # G
            3: self._measure_ring_distances(polymer, pyrimidine_atoms, residue_idx=3),  # U
        }

        # Rotate CHI for the first residue (A, purine)
        chi_purine = polymer.dihedral(DihedralType.CHI_PURINE)
        if len(chi_purine) > 0 and not np.isnan(chi_purine[0]):
            new_chi = chi_purine.copy()
            new_chi[0] = chi_purine[0] + np.pi / 3  # Rotate by 60 degrees
            polymer.set_dihedral(DihedralType.CHI_PURINE, new_chi)

        # All ring distances should be unchanged
        final_distances = {
            0: self._measure_ring_distances(polymer, purine_atoms, residue_idx=0),
            1: self._measure_ring_distances(polymer, pyrimidine_atoms, residue_idx=1),
            2: self._measure_ring_distances(polymer, purine_atoms, residue_idx=2),
            3: self._measure_ring_distances(polymer, pyrimidine_atoms, residue_idx=3),
        }

        for res_idx in range(4):
            for pair, initial_dist in initial_distances[res_idx].items():
                final_dist = final_distances[res_idx].get(pair)
                if final_dist is not None:
                    np.testing.assert_allclose(
                        initial_dist,
                        final_dist,
                        atol=1e-4,
                        err_msg=f"Residue {res_idx} ring bond {pair} changed from {initial_dist:.4f} to {final_dist:.4f}"
                    )

    def test_sugar_ring_preserved_on_chi_rotation(self):
        """Test that the ribose sugar ring is preserved when CHI rotates.

        The sugar ring (C1'-C2'-C3'-C4'-O4') should not be affected by
        CHI rotation since CHI only rotates atoms attached to C1' (the base).
        This test passes because sugar atoms are upstream of CHI in the Z-matrix.
        """
        from ciffy import from_sequence, DihedralType

        polymer = from_sequence("a")

        # Ribose sugar ring atoms
        sugar_atoms = ["C1p", "C2p", "C3p", "C4p", "O4p"]

        initial_distances = self._measure_ring_distances(polymer, sugar_atoms)
        initial_planarity = self._measure_ring_planarity(polymer, sugar_atoms)

        # Rotate CHI
        chi = polymer.dihedral(DihedralType.CHI_PURINE)
        if len(chi) > 0 and not np.isnan(chi[0]):
            new_chi = chi.copy()
            new_chi[0] = chi[0] + np.pi / 2
            polymer.set_dihedral(DihedralType.CHI_PURINE, new_chi)

        final_distances = self._measure_ring_distances(polymer, sugar_atoms)
        final_planarity = self._measure_ring_planarity(polymer, sugar_atoms)

        # Sugar ring distances should be unchanged
        for pair, initial_dist in initial_distances.items():
            final_dist = final_distances.get(pair)
            if final_dist is not None:
                np.testing.assert_allclose(
                    initial_dist,
                    final_dist,
                    atol=1e-4,
                    err_msg=f"Sugar ring bond {pair} changed after CHI rotation"
                )

        # Sugar ring planarity should be preserved (note: sugar is puckered, not planar)
        # We just check it doesn't change, not that it's flat
        np.testing.assert_allclose(
            initial_planarity,
            final_planarity,
            atol=1e-4,
            err_msg=f"Sugar ring geometry changed after CHI rotation"
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
        from ciffy import from_sequence

        tol = get_tolerances(device)
        polymer = from_sequence("acgu", backend="torch").to(device)
        assert_roundtrip_preserves_structure(
            polymer, threshold=tol.roundtrip_small,
            message=f"on {device}"
        )

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

        # Use centralized tolerances
        tol = get_tolerances(device)
        assert torch.all(distances >= 0)
        assert torch.all(angles >= -tol.angle_range_epsilon)
        assert torch.all(angles <= np.pi + tol.angle_range_epsilon)
        assert torch.all(dihedrals >= -np.pi - tol.angle_range_epsilon)
        assert torch.all(dihedrals <= np.pi + tol.angle_range_epsilon)

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_pdb_roundtrip_on_gpu(self, device):
        """Test round-trip on real PDB structure on GPU."""
        skip_if_no_device(device)
        from ciffy import load

        tol = get_tolerances(device)
        polymer = load(get_test_cif("1ZEW")).poly().torch().to(device)
        assert_roundtrip_preserves_structure(
            polymer, threshold=tol.roundtrip_real_structure,
            message=f"on {device}"
        )

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_gpu_cpu_transfer(self, device):
        """Test moving between CPU and GPU preserves internal coordinates."""
        skip_if_no_device(device)
        import torch
        from ciffy import from_sequence

        tol = get_tolerances()
        polymer_cpu = from_sequence("acgu", backend="torch")
        dihedrals_cpu = polymer_cpu.dihedrals.clone()

        polymer_gpu = polymer_cpu.to(device)
        dihedrals_gpu = polymer_gpu.dihedrals

        assert torch.allclose(dihedrals_cpu, dihedrals_gpu.cpu(), atol=tol.allclose_atol)

        # Move back to CPU
        polymer_back = polymer_gpu.to("cpu")
        dihedrals_back = polymer_back.dihedrals

        assert torch.allclose(dihedrals_cpu, dihedrals_back, atol=tol.allclose_atol)

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_differentiability_on_gpu(self, device):
        """Test gradient flow through reconstruction on GPU."""
        skip_if_no_device(device)
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch").to(device)
        assert_gradient_flows(polymer)
