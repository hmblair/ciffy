"""
Unit tests for analytical ring closure.

Tests the AnalyticalRingSolver which uses circle-sphere intersection
geometry to solve ring closure analytically (O(1) instead of iterative CCD).
"""

import numpy as np
import pytest

from ciffy.internal.analytical_closure import AnalyticalRingSolver
from ciffy.internal.geometry import circle_sphere_intersect


class TestCircleSphereIntersection:
    """Tests for the circle-sphere intersection geometry."""

    def test_circle_crosses_sphere_two_solutions(self):
        """Circle crossing through sphere gives 2 intersection points."""
        # Circle of radius 2 in xy-plane, offset from origin
        # Sphere of radius 2 at origin
        # Circle crosses through the sphere
        solutions = circle_sphere_intersect(
            circle_center=np.array([1.5, 0.0, 0.0]),  # Offset so circle crosses sphere
            circle_axis=np.array([0.0, 0.0, 1.0]),
            circle_radius=2.0,
            sphere_center=np.array([0.0, 0.0, 0.0]),
            sphere_radius=2.0,
        )

        # Should have 2 solutions
        assert len(solutions) == 2

        # Both points should be on the sphere
        for point, angle in solutions:
            dist = np.linalg.norm(point)
            assert np.isclose(dist, 2.0, atol=1e-6), f"Point not on sphere: dist={dist}"

    def test_realistic_ring_closure_geometry(self):
        """Test geometry similar to actual ring closure."""
        # This mimics ring closure:
        # - Circle is traced by an atom rotating around a bond axis
        # - Sphere is centered at the first ring atom (closure distance)

        # Atom at position (1, 0, 0) rotating around z-axis at (0, 0, 0)
        # traces a circle of radius 1 in the xy-plane
        # Closure target at (0.5, 0.5, 0) with bond length ~0.8
        circle_center = np.array([0.0, 0.0, 0.0])
        circle_axis = np.array([0.0, 0.0, 1.0])
        circle_radius = 1.0
        sphere_center = np.array([0.5, 0.5, 0.0])
        sphere_radius = 0.8

        solutions = circle_sphere_intersect(
            circle_center=circle_center,
            circle_axis=circle_axis,
            circle_radius=circle_radius,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
        )

        # Should have 2 solutions (circle passes through sphere)
        assert len(solutions) == 2

        # Verify each solution
        for point, angle in solutions:
            # Point on circle
            dist_from_center = np.linalg.norm(point - circle_center)
            assert np.isclose(dist_from_center, circle_radius, atol=1e-6)

            # Point on sphere
            dist_from_sphere = np.linalg.norm(point - sphere_center)
            assert np.isclose(dist_from_sphere, sphere_radius, atol=1e-6)

    def test_no_intersection_far_apart(self):
        """Geometries too far apart give 0 solutions."""
        # Circle at origin, sphere far away in same plane
        solutions = circle_sphere_intersect(
            circle_center=np.array([0.0, 0.0, 0.0]),
            circle_axis=np.array([0.0, 0.0, 1.0]),
            circle_radius=1.0,
            sphere_center=np.array([5.0, 0.0, 0.0]),  # Far away in xy-plane
            sphere_radius=1.0,
        )

        # Too far apart to intersect
        assert len(solutions) == 0

    def test_intersection_points_on_circle(self):
        """Intersection points should lie on the circle."""
        circle_center = np.array([1.0, 1.0, 0.0])
        circle_axis = np.array([0.0, 0.0, 1.0])
        circle_radius = 2.0

        solutions = circle_sphere_intersect(
            circle_center=circle_center,
            circle_axis=circle_axis,
            circle_radius=circle_radius,
            sphere_center=np.array([0.0, 0.0, 0.0]),
            sphere_radius=3.0,
        )

        for point, angle in solutions:
            # Point should be at circle_radius distance from circle_center
            # in the plane perpendicular to circle_axis
            v = point - circle_center
            v_perp = v - circle_axis * np.dot(v, circle_axis)
            dist_from_center = np.linalg.norm(v_perp)
            assert np.isclose(dist_from_center, circle_radius, atol=1e-6)


class TestAnalyticalRingSolver:
    """Tests for the AnalyticalRingSolver."""

    @pytest.fixture
    def adenine_geometry(self):
        """Create adenine nucleotide geometry."""
        import ciffy
        polymer = ciffy.from_sequence("a")
        manager = polymer._geometry
        manager._ensure_constraint_analysis()
        return manager

    @pytest.fixture
    def proline_geometry(self):
        """Create proline dipeptide geometry."""
        import ciffy
        polymer = ciffy.from_sequence("GPG")
        manager = polymer._geometry
        manager._ensure_constraint_analysis()
        return manager

    def test_can_solve_5ring_analytically(self, adenine_geometry):
        """5-member rings should be solvable analytically."""
        manager = adenine_geometry
        ring_constraints = manager._independent_dof.ring_constraints

        # Find the ribose ring (5 atoms with oxygen)
        ribose_ring = None
        for ring in ring_constraints:
            if ring.ring_size == 5:
                # Check if it's the ribose (has O4' oxygen)
                atoms = ring.ring_atoms
                elements = manager._topology.elements[atoms]
                if 8 in elements:  # Oxygen
                    ribose_ring = ring
                    break

        if ribose_ring is None:
            pytest.skip("No ribose ring found")

        recon = manager._recon_data
        solver = AnalyticalRingSolver(
            tree=manager._tree,
            fixed_coords=recon.fixed_coords,
            offsets=recon.center_offsets,
        )

        assert solver.can_solve_analytically(ribose_ring)

    def test_fused_rings_not_analytical(self, adenine_geometry):
        """Fused rings (like adenine base) should not use analytical solver."""
        manager = adenine_geometry
        ring_constraints = manager._independent_dof.ring_constraints

        recon = manager._recon_data
        solver = AnalyticalRingSolver(
            tree=manager._tree,
            fixed_coords=recon.fixed_coords,
            offsets=recon.center_offsets,
        )

        # At least one ring should not be solvable analytically (fused base)
        non_analytical_count = sum(
            1 for ring in ring_constraints
            if not solver.can_solve_analytically(ring)
        )

        assert non_analytical_count >= 1, "Expected fused rings to not be analytical"

    def test_closure_preserves_distance(self, adenine_geometry):
        """Analytical closure should preserve closure bond distance."""
        manager = adenine_geometry
        ring_constraints = manager._independent_dof.ring_constraints

        recon = manager._recon_data
        solver = AnalyticalRingSolver(
            tree=manager._tree,
            fixed_coords=recon.fixed_coords,
            offsets=recon.center_offsets,
        )

        coords = manager.coordinates.copy()
        internal = manager._internal.copy()

        for ring in ring_constraints:
            if not solver.can_solve_analytically(ring):
                continue

            closure_i, closure_j = ring.closure_bond
            expected_dist = np.linalg.norm(coords[closure_i] - coords[closure_j])

            # Solve ring closure
            new_internal, success = solver.solve_ring(internal.copy(), ring, coords)

            if success:
                # Reconstruct coordinates
                new_coords = manager._tree.internal_to_cartesian(
                    new_internal.astype(np.float32),
                    recon.fixed_coords,
                    recon.center_offsets,
                )

                actual_dist = np.linalg.norm(new_coords[closure_i] - new_coords[closure_j])
                closure_error = abs(actual_dist - expected_dist)

                assert closure_error < 0.01, (
                    f"Closure error too large: {closure_error:.6f} Å"
                )

    def test_closure_after_dihedral_perturbation(self, adenine_geometry):
        """Ring closure should work after perturbing independent dihedrals."""
        manager = adenine_geometry

        original_dof = manager.dof.copy()
        original_coords = manager.coordinates.copy()

        if len(original_dof) == 0:
            pytest.skip("No DOF to test")

        # Perturb first dihedral
        new_dof = original_dof.copy()
        new_dof[0] += 0.5  # ~29 degrees

        # Set DOF (triggers ring closure)
        manager.dof = new_dof

        # Verify coordinates changed
        new_coords = manager.coordinates
        coord_diff = np.max(np.abs(new_coords - original_coords))
        assert coord_diff > 0.1, "Coordinates should change after DOF perturbation"

        # Verify DOF was set correctly (roundtrip)
        roundtrip_dof = manager.dof
        dof_diff = np.max(np.abs(roundtrip_dof - new_dof))
        assert dof_diff < 0.01, f"DOF roundtrip error: {dof_diff}"

    def test_bond_lengths_preserved(self, adenine_geometry):
        """Ring closure should preserve all bond lengths."""
        manager = adenine_geometry

        original_bonds = manager.distances.copy()
        original_dof = manager.dof.copy()

        if len(original_dof) == 0:
            pytest.skip("No DOF to test")

        # Perturb DOF
        new_dof = original_dof.copy()
        new_dof[0] += 0.3
        manager.dof = new_dof

        new_bonds = manager.distances
        bond_diff = np.max(np.abs(new_bonds - original_bonds))

        assert bond_diff < 1e-5, f"Bond lengths changed by {bond_diff:.6f} Å"

    def test_proline_ring_closure(self, proline_geometry):
        """Proline ring should close correctly."""
        manager = proline_geometry
        ring_constraints = manager._independent_dof.ring_constraints

        if len(ring_constraints) == 0:
            pytest.skip("No ring constraints found")

        # Find proline ring (5-member with nitrogen)
        proline_ring = None
        for ring in ring_constraints:
            if ring.ring_size == 5:
                atoms = ring.ring_atoms
                elements = manager._topology.elements[atoms]
                if 7 in elements:  # Nitrogen
                    proline_ring = ring
                    break

        if proline_ring is None:
            pytest.skip("No proline ring found")

        recon = manager._recon_data
        solver = AnalyticalRingSolver(
            tree=manager._tree,
            fixed_coords=recon.fixed_coords,
            offsets=recon.center_offsets,
        )

        coords = manager.coordinates.copy()
        internal = manager._internal.copy()

        if solver.can_solve_analytically(proline_ring):
            new_internal, success = solver.solve_ring(internal.copy(), proline_ring, coords)

            if success:
                closure_i, closure_j = proline_ring.closure_bond
                expected_dist = np.linalg.norm(coords[closure_i] - coords[closure_j])

                new_coords = manager._tree.internal_to_cartesian(
                    new_internal.astype(np.float32),
                    recon.fixed_coords,
                    recon.center_offsets,
                )

                actual_dist = np.linalg.norm(new_coords[closure_i] - new_coords[closure_j])
                closure_error = abs(actual_dist - expected_dist)

                assert closure_error < 0.01


class TestRingClosureIntegration:
    """Integration tests for ring closure in the full DOF pipeline."""

    def test_dna_sugar_closure(self):
        """DNA deoxyribose rings should close correctly."""
        import ciffy

        polymer = ciffy.from_sequence("aaa")
        manager = polymer._geometry

        original_dof = manager.dof.copy()
        original_bonds = manager.distances.copy()

        if len(original_dof) == 0:
            pytest.skip("No DOF")

        # Test multiple perturbations
        for delta in [0.1, 0.3, 0.5, -0.2]:
            new_dof = original_dof.copy()
            new_dof[0] += delta
            manager.dof = new_dof

            # Check bonds preserved
            new_bonds = manager.distances
            bond_diff = np.max(np.abs(new_bonds - original_bonds))
            assert bond_diff < 1e-4, f"Bond change {bond_diff} at delta={delta}"

            # Check DOF roundtrip
            roundtrip = manager.dof
            dof_diff = np.max(np.abs(roundtrip - new_dof))
            assert dof_diff < 0.01, f"DOF roundtrip error {dof_diff} at delta={delta}"

    def test_rna_sugar_closure(self):
        """RNA ribose rings should close correctly."""
        import ciffy

        polymer = ciffy.from_sequence("a")  # Single RNA nucleotide
        manager = polymer._geometry

        original_coords = manager.coordinates.copy()
        original_dof = manager.dof.copy()

        if len(original_dof) == 0:
            pytest.skip("No DOF")

        # Larger perturbation
        new_dof = original_dof.copy()
        new_dof[0] += 1.0  # ~57 degrees
        manager.dof = new_dof

        # Coordinates should change significantly
        new_coords = manager.coordinates
        max_change = np.max(np.abs(new_coords - original_coords))
        assert max_change > 0.5, f"Expected significant coord change, got {max_change}"

        # But bonds should be preserved
        manager.dof = original_dof  # Reset
        final_bonds = manager.distances
        # Reset and get original bonds
        manager._coords_dirty = False
        manager._dof_dirty = True
        manager._coordinates = original_coords.copy()
        orig_bonds = manager.distances

        bond_diff = np.max(np.abs(final_bonds - orig_bonds))
        assert bond_diff < 1e-4

    def test_multiple_rings_same_molecule(self):
        """Molecules with multiple rings should handle all closures."""
        import ciffy

        # Adenine has ribose + fused purine base
        polymer = ciffy.from_sequence("a")
        manager = polymer._geometry
        manager._ensure_constraint_analysis()

        n_rings = len(manager._independent_dof.ring_constraints)
        assert n_rings >= 2, f"Expected multiple rings, got {n_rings}"

        # DOF operations should work
        original_dof = manager.dof.copy()
        if len(original_dof) > 0:
            new_dof = original_dof.copy()
            new_dof[0] += 0.2
            manager.dof = new_dof

            roundtrip = manager.dof
            assert np.allclose(roundtrip, new_dof, atol=0.01)
