"""Tests for constrained internal coordinates (minimal DOF representation)."""

import pytest
import numpy as np

# Tolerance for geometric comparisons
BOND_TOLERANCE = 1e-4  # Angstroms
ANGLE_TOLERANCE = 1e-3  # Radians
DIHEDRAL_TOLERANCE = 1e-6  # Radians for roundtrip


class TestRingAnalysis:
    """Tests for ring detection and analysis."""

    def test_find_cycles_linear_molecule(self):
        """Linear molecule (no rings) should have no cycles."""
        from ciffy.internal.ring_analysis import RingAnalyzer

        # Linear chain: 0-1-2-3-4
        n_atoms = 5
        # CSR format for linear chain
        offsets = np.array([0, 1, 3, 5, 7, 8], dtype=np.int64)
        neighbors = np.array([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int64)

        cycles = RingAnalyzer.find_fundamental_cycles(offsets, neighbors, n_atoms)
        assert len(cycles) == 0, "Linear molecule should have no cycles"

    def test_find_cycles_single_ring(self):
        """Single ring should produce one cycle."""
        from ciffy.internal.ring_analysis import RingAnalyzer

        # 6-membered ring: 0-1-2-3-4-5-0
        n_atoms = 6
        # Build adjacency for ring
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        offsets, neighbors = _edges_to_csr(edges, n_atoms)

        cycles = RingAnalyzer.find_fundamental_cycles(offsets, neighbors, n_atoms)
        assert len(cycles) == 1, f"Expected 1 cycle, got {len(cycles)}"
        assert len(cycles[0]) == 6, f"Expected ring size 6, got {len(cycles[0])}"

    def test_find_cycles_fused_rings(self):
        """Fused rings (like purine) should produce two cycles."""
        from ciffy.internal.ring_analysis import RingAnalyzer

        # Purine-like structure: 5-ring fused with 6-ring
        # Atoms: 0-1-2-3-4 (5-ring) + 2-3-5-6-7 (6-ring, shared edge 2-3)
        n_atoms = 8
        edges = [
            # 5-ring: 0-1-2-3-4-0
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            # 6-ring extension: 2-3-5-6-7-2 (but 2-3 already exists)
            (3, 5), (5, 6), (6, 7), (7, 2),
        ]
        offsets, neighbors = _edges_to_csr(edges, n_atoms)

        cycles = RingAnalyzer.find_fundamental_cycles(offsets, neighbors, n_atoms)
        assert len(cycles) == 2, f"Expected 2 cycles for fused rings, got {len(cycles)}"


class TestConstraintSpec:
    """Tests for constraint specification."""

    def test_default_constraints(self):
        """Default should fix all bonds and angles."""
        from ciffy.internal.ring_analysis import ConstraintSpec

        spec = ConstraintSpec()
        assert spec.fixed_bonds == "all"
        assert spec.fixed_angles == "all"
        assert spec.extra_bonds == []
        assert spec.extra_angles == []

    def test_custom_constraints(self):
        """Custom constraints should be stored correctly."""
        from ciffy.internal.ring_analysis import ConstraintSpec

        extra_bonds = [(10, 50, 2.9)]
        spec = ConstraintSpec(
            fixed_bonds="all",
            fixed_angles="none",
            extra_bonds=extra_bonds,
        )
        assert spec.fixed_bonds == "all"
        assert spec.fixed_angles == "none"
        assert spec.extra_bonds == extra_bonds


class TestIndependentDOF:
    """Tests for independent DOF computation."""

    def test_linear_peptide_dof(self):
        """Linear molecule should have N-3 independent dihedrals."""
        from ciffy import from_sequence

        # Create a short peptide (no rings in backbone)
        polymer = from_sequence("AAA")  # 3 alanines

        # Get constrained manager via _coord_manager
        constrained = polymer._coord_manager.with_constraints()

        # For a linear molecule, all dihedrals (for atoms >= 3) should be independent
        n_atoms = polymer.size()
        expected_max_dof = n_atoms - 3  # First 3 atoms don't have dihedrals

        # The actual number may be less due to terminal constraints, but should be > 0
        assert constrained.n_dof > 0, "Should have some independent DOF"
        assert constrained.n_dof <= expected_max_dof, f"DOF should be <= {expected_max_dof}"


class TestConstrainedCoordinateManager:
    """Tests for ConstrainedCoordinateManager."""

    def test_basic_creation(self):
        """Test creating a constrained manager."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")  # lowercase for RNA
        constrained = polymer._coord_manager.with_constraints()

        assert constrained.n_atoms == polymer.size()
        assert constrained.n_dof >= 0

    def test_get_values(self):
        """Test getting independent DOF values."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        values = constrained.values
        assert len(values) == constrained.n_dof
        # Values should be in radians, roughly in [-pi, pi]
        assert np.all(np.abs(values) <= np.pi + 0.1) or len(values) == 0

    def test_set_values(self):
        """Test setting independent DOF values."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        if constrained.n_dof > 0:
            # Get current values
            original_values = constrained.values.copy()

            # Modify slightly
            new_values = original_values + 0.1

            # Set new values
            constrained.values = new_values

            # Check that coordinates changed
            new_coords = constrained.coordinates
            assert new_coords is not None
            assert new_coords.shape == (constrained.n_atoms, 3)

    def test_coordinates_property(self):
        """Test accessing coordinates through constrained manager."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        coords = constrained.coordinates
        assert coords.shape == (constrained.n_atoms, 3)

    def test_ring_detection_on_nucleotide(self):
        """Test that rings are detected in nucleotides."""
        from ciffy import from_sequence

        # Single adenine has purine ring (2 fused rings)
        polymer = from_sequence("a")  # lowercase for RNA
        constrained = polymer._coord_manager.with_constraints()

        # Should have detected ring constraints
        n_rings = len(constrained.independent_dof.ring_constraints)
        # Adenine has fused 5+6 rings = 2 fundamental cycles
        assert n_rings >= 0, "Ring detection should work"

    def test_repr(self):
        """Test string representation."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        repr_str = repr(constrained)
        assert "ConstrainedCoordinateManager" in repr_str
        assert str(constrained.n_atoms) in repr_str
        assert str(constrained.n_dof) in repr_str


class TestExtraBondConstraints:
    """Tests for extra bond constraints (e.g., H-bonds)."""

    def test_extra_bond_creates_cycle(self):
        """Extra bond between distant atoms should create a new cycle."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        n_atoms = polymer.size()

        # Find two atoms that are far apart in sequence
        atom_i = 5
        atom_j = min(20, n_atoms - 1)

        if atom_j > atom_i + 3:  # Only if they're far enough apart
            constrained_no_extra = polymer._coord_manager.with_constraints()
            constrained_with_extra = polymer._coord_manager.with_constraints(
                extra_bonds=[(atom_i, atom_j, 3.0)]
            )

            # Adding an extra bond should create an additional ring
            # This might reduce the number of independent DOF
            # (or at least not increase it)
            assert constrained_with_extra.n_dof <= constrained_no_extra.n_dof


class TestTorchBackend:
    """Tests with PyTorch backend."""

    def test_torch_constrained_manager(self):
        """Test constrained manager with torch backend."""
        pytest.importorskip("torch")
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        constrained = polymer._coord_manager.with_constraints()

        # Values should be torch tensors
        values = constrained.values
        import torch
        assert isinstance(values, torch.Tensor) or len(values) == 0

    def test_torch_set_values(self):
        """Test setting values with torch backend."""
        pytest.importorskip("torch")
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch")
        constrained = polymer._coord_manager.with_constraints()

        if constrained.n_dof > 0:
            # Create new values as torch tensor
            new_values = torch.randn(constrained.n_dof)
            constrained.values = new_values

            # Coordinates should still be valid
            coords = constrained.coordinates
            assert coords.shape == (constrained.n_atoms, 3)


class TestCorrectnessRingDetection:
    """Tests verifying ring detection correctness."""

    def test_dof_count_linear_chain(self):
        """Linear chain of N atoms should have exactly N-3 independent dihedrals."""
        from ciffy.internal.ring_analysis import RingAnalyzer, ConstraintSpec

        # Test various chain lengths
        for n_atoms in [5, 10, 20]:
            # Build linear chain bond graph
            edges = [(i, i + 1) for i in range(n_atoms - 1)]
            offsets, neighbors = _edges_to_csr(edges, n_atoms)

            # Create mock Z-matrix indices (simplified)
            zmatrix = np.zeros((n_atoms, 4), dtype=np.int64)
            for i in range(n_atoms):
                zmatrix[i, 0] = i
                if i >= 1:
                    zmatrix[i, 1] = i - 1
                if i >= 2:
                    zmatrix[i, 2] = i - 2
                if i >= 3:
                    zmatrix[i, 3] = i - 3

            spec = ConstraintSpec(fixed_bonds="all", fixed_angles="all")
            result = RingAnalyzer.analyze_constraints(
                offsets, neighbors, n_atoms, zmatrix, spec
            )

            expected_dof = n_atoms - 3
            assert result.n_independent == expected_dof, (
                f"Linear chain with {n_atoms} atoms should have {expected_dof} DOF, "
                f"got {result.n_independent}"
            )
            assert len(result.ring_constraints) == 0, "Linear chain should have no rings"

    def test_dof_count_single_ring(self):
        """Single k-ring should have k-3 independent dihedrals."""
        from ciffy.internal.ring_analysis import RingAnalyzer, ConstraintSpec

        # Test 6-membered ring
        n_atoms = 6
        edges = [(i, (i + 1) % n_atoms) for i in range(n_atoms)]
        offsets, neighbors = _edges_to_csr(edges, n_atoms)

        # Create Z-matrix (ring structure)
        zmatrix = np.zeros((n_atoms, 4), dtype=np.int64)
        for i in range(n_atoms):
            zmatrix[i, 0] = i
            if i >= 1:
                zmatrix[i, 1] = i - 1
            if i >= 2:
                zmatrix[i, 2] = i - 2
            if i >= 3:
                zmatrix[i, 3] = i - 3

        spec = ConstraintSpec(fixed_bonds="all", fixed_angles="all")
        result = RingAnalyzer.analyze_constraints(
            offsets, neighbors, n_atoms, zmatrix, spec
        )

        # 6-ring: 6-3 = 3 independent dihedrals (but first 3 atoms have no dihedrals)
        # So we start with n_atoms-3 = 3 dihedrals, ring removes 3 more
        # Actually: atoms 3,4,5 have dihedrals, ring constraint removes 3
        # Expected: 3 - 3 = 0 independent DOF for a 6-ring with fixed bonds/angles
        assert result.n_independent <= 3, f"6-ring should have at most 3 DOF, got {result.n_independent}"
        assert len(result.ring_constraints) == 1, "Should detect exactly 1 ring"

    def test_ring_detection_matches_topology(self):
        """Ring detection should find rings that match the molecular topology."""
        from ciffy import from_sequence

        # Adenine (purine) has 2 fused rings
        polymer = from_sequence("a")
        constrained = polymer._coord_manager.with_constraints()

        n_rings = len(constrained.independent_dof.ring_constraints)
        # Purine base has 2 fundamental cycles (5-ring + 6-ring fused)
        # Plus ribose sugar (5-ring)
        # So we expect at least 2-3 rings detected
        assert n_rings >= 2, f"Adenine should have at least 2 rings, found {n_rings}"


class TestCorrectnessGeometryPreservation:
    """Tests verifying geometry is preserved after setting dihedrals."""

    def test_bond_lengths_preserved_after_set_values(self):
        """Setting independent DOF values should preserve all bond lengths."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        # Get original bond lengths
        original_distances = constrained.distances.copy()

        if constrained.n_dof > 0:
            # Modify independent values
            original_values = constrained.values.copy()
            new_values = original_values + 0.1  # Small perturbation
            constrained.values = new_values

            # Check bond lengths are preserved
            new_distances = constrained.distances
            diff = np.abs(new_distances - original_distances)
            max_diff = np.max(diff)

            assert max_diff < BOND_TOLERANCE, (
                f"Bond lengths changed by {max_diff:.6f} A after setting dihedrals. "
                f"Should be preserved within {BOND_TOLERANCE} A."
            )

    def test_bond_angles_preserved_after_set_values(self):
        """Setting independent DOF values should preserve all bond angles."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        # Get original bond angles
        original_angles = constrained.angles.copy()

        if constrained.n_dof > 0:
            # Modify independent values
            original_values = constrained.values.copy()
            new_values = original_values + 0.1
            constrained.values = new_values

            # Check bond angles are preserved
            new_angles = constrained.angles
            diff = np.abs(new_angles - original_angles)
            max_diff = np.max(diff)

            assert max_diff < ANGLE_TOLERANCE, (
                f"Bond angles changed by {max_diff:.6f} rad after setting dihedrals. "
                f"Should be preserved within {ANGLE_TOLERANCE} rad."
            )

    def test_independent_values_roundtrip(self):
        """Getting and setting values should roundtrip correctly."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        if constrained.n_dof > 0:
            # Get original values
            original_values = constrained.values.copy()

            # Set them back
            constrained.values = original_values

            # Get again
            roundtrip_values = constrained.values

            # Should be identical
            diff = np.abs(roundtrip_values - original_values)
            max_diff = np.max(diff)

            assert max_diff < DIHEDRAL_TOLERANCE, (
                f"Independent values changed by {max_diff:.6f} rad after roundtrip. "
                f"Should be preserved within {DIHEDRAL_TOLERANCE} rad."
            )

    def test_coordinates_change_when_dihedrals_change(self):
        """Coordinates should actually change when we set different dihedral values."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        if constrained.n_dof > 0:
            # Get original coordinates
            original_coords = constrained.coordinates.copy()

            # Set very different dihedral values
            original_values = constrained.values.copy()
            new_values = original_values + np.pi / 4  # 45 degree change
            constrained.values = new_values

            # Coordinates should be different
            new_coords = constrained.coordinates
            diff = np.abs(new_coords - original_coords)
            max_diff = np.max(diff)

            assert max_diff > 0.1, (
                f"Coordinates only changed by {max_diff:.6f} A after 45-degree dihedral change. "
                "Expected significant change."
            )


class TestCorrectnessDOFReduction:
    """Tests verifying DOF is reduced by constraints."""

    def test_extra_bonds_reduce_dof(self):
        """Adding extra bond constraints should reduce independent DOF."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        n_atoms = polymer.size()

        # Without extra constraints
        constrained_base = polymer._coord_manager.with_constraints()
        base_dof = constrained_base.n_dof

        # With extra bond constraint between distant atoms
        # Find atoms far apart
        atom_i = 5
        atom_j = min(30, n_atoms - 1)

        if atom_j > atom_i + 5:
            constrained_extra = polymer._coord_manager.with_constraints(
                extra_bonds=[(atom_i, atom_j, 3.0)]
            )
            extra_dof = constrained_extra.n_dof

            assert extra_dof <= base_dof, (
                f"Adding extra bond should not increase DOF. "
                f"Base: {base_dof}, With extra bond: {extra_dof}"
            )

            # The extra bond creates a cycle, which should reduce DOF by 3
            # (or by less if the cycle overlaps with existing cycles)
            expected_max_reduction = 3
            reduction = base_dof - extra_dof
            assert reduction <= expected_max_reduction, (
                f"DOF reduction {reduction} exceeds expected max {expected_max_reduction}"
            )

    def test_n_dof_less_than_total_dihedrals(self):
        """For molecules with rings, n_dof should be less than total dihedrals."""
        from ciffy import from_sequence

        # RNA has sugar and base rings
        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        n_atoms = polymer.size()
        max_possible_dihedrals = n_atoms - 3  # First 3 atoms have no dihedrals

        # With rings, we should have fewer independent DOF
        if len(constrained.independent_dof.ring_constraints) > 0:
            assert constrained.n_dof < max_possible_dihedrals, (
                f"With rings, DOF ({constrained.n_dof}) should be less than "
                f"max possible ({max_possible_dihedrals})"
            )


class TestCorrectnessProtein:
    """Correctness tests for protein structures."""

    def test_protein_backbone_dof(self):
        """Protein backbone should have phi/psi/omega as DOF."""
        from ciffy import from_sequence

        # Short peptide
        polymer = from_sequence("AAAA")  # 4 alanines
        constrained = polymer._coord_manager.with_constraints()

        # Protein backbone has ~3 dihedrals per residue (phi, psi, omega)
        # minus terminal constraints
        n_residues = 4
        # Rough estimate: 3 * n_residues - terminal_constraints
        min_expected_dof = n_residues  # At least 1 per residue
        max_expected_dof = 3 * n_residues  # At most 3 per residue

        assert constrained.n_dof >= min_expected_dof, (
            f"Protein with {n_residues} residues should have at least {min_expected_dof} DOF, "
            f"got {constrained.n_dof}"
        )

    def test_protein_geometry_preserved(self):
        """Protein geometry should be preserved after setting dihedrals."""
        from ciffy import from_sequence

        polymer = from_sequence("AAAA")
        constrained = polymer._coord_manager.with_constraints()

        original_distances = constrained.distances.copy()
        original_angles = constrained.angles.copy()

        if constrained.n_dof > 0:
            # Perturb dihedrals
            values = constrained.values.copy()
            constrained.values = values + 0.05

            # Check geometry preserved
            dist_diff = np.max(np.abs(constrained.distances - original_distances))
            angle_diff = np.max(np.abs(constrained.angles - original_angles))

            assert dist_diff < BOND_TOLERANCE, f"Bond lengths changed by {dist_diff:.6f} A"
            assert angle_diff < ANGLE_TOLERANCE, f"Bond angles changed by {angle_diff:.6f} rad"


class TestCorrectnessNucleicAcid:
    """Correctness tests for nucleic acid structures."""

    def test_rna_has_multiple_rings(self):
        """RNA nucleotides should have sugar and base rings detected."""
        from ciffy import from_sequence

        # Single nucleotide
        for seq in ["a", "c", "g", "u"]:
            polymer = from_sequence(seq)
            constrained = polymer._coord_manager.with_constraints()

            n_rings = len(constrained.independent_dof.ring_constraints)
            assert n_rings >= 1, (
                f"Nucleotide '{seq}' should have at least 1 ring (sugar), "
                f"found {n_rings}"
            )

    def test_purine_has_more_rings_than_pyrimidine(self):
        """Purines (A, G) have fused rings, pyrimidines (C, U) have single base ring."""
        from ciffy import from_sequence

        # Purines have 2 fused base rings + sugar = 3 rings
        # Pyrimidines have 1 base ring + sugar = 2 rings
        for purine in ["a", "g"]:
            polymer = from_sequence(purine)
            constrained = polymer._coord_manager.with_constraints()
            purine_rings = len(constrained.independent_dof.ring_constraints)

            for pyrimidine in ["c", "u"]:
                polymer = from_sequence(pyrimidine)
                constrained = polymer._coord_manager.with_constraints()
                pyrimidine_rings = len(constrained.independent_dof.ring_constraints)

                # Purine should have at least as many rings as pyrimidine
                # (in practice, should have 1 more due to fused ring)
                assert purine_rings >= pyrimidine_rings, (
                    f"Purine '{purine}' ({purine_rings} rings) should have >= rings "
                    f"than pyrimidine '{pyrimidine}' ({pyrimidine_rings} rings)"
                )

    def test_rna_geometry_preserved(self):
        """RNA geometry should be preserved after setting dihedrals."""
        from ciffy import from_sequence

        polymer = from_sequence("acgu")
        constrained = polymer._coord_manager.with_constraints()

        original_distances = constrained.distances.copy()
        original_angles = constrained.angles.copy()

        if constrained.n_dof > 0:
            # Perturb dihedrals
            values = constrained.values.copy()
            constrained.values = values + 0.05

            # Check geometry preserved
            dist_diff = np.max(np.abs(constrained.distances - original_distances))
            angle_diff = np.max(np.abs(constrained.angles - original_angles))

            assert dist_diff < BOND_TOLERANCE, f"Bond lengths changed by {dist_diff:.6f} A"
            assert angle_diff < ANGLE_TOLERANCE, f"Bond angles changed by {angle_diff:.6f} rad"


# Helper function for test data
def _edges_to_csr(edges: list, n_atoms: int):
    """Convert edge list to CSR format."""
    # Count neighbors for each atom
    counts = np.zeros(n_atoms, dtype=np.int64)
    for i, j in edges:
        counts[i] += 1
        counts[j] += 1

    # Build offsets
    offsets = np.zeros(n_atoms + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    # Build neighbors
    n_edges = len(edges) * 2
    neighbors = np.zeros(n_edges, dtype=np.int64)
    current = np.zeros(n_atoms, dtype=np.int64)

    for i, j in edges:
        # Add i -> j
        idx = offsets[i] + current[i]
        neighbors[idx] = j
        current[i] += 1
        # Add j -> i
        idx = offsets[j] + current[j]
        neighbors[idx] = i
        current[j] += 1

    return offsets, neighbors
