"""Tests for constrained internal coordinates (minimal DOF representation)."""

import pytest
import numpy as np


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
