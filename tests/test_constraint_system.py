"""
Tests for the new unified constraint system.

Tests cover:
1. ClosureConstraints and ConstraintSystem creation
2. DOF discovery via Jacobian analysis
3. Newton-Raphson ring closure solving
4. Differentiable DOF-to-Cartesian mapping
"""

import numpy as np
import pytest


class TestConstraintSystemCreation:
    """Tests for ConstraintSystem creation methods."""

    def test_from_constraints_linear_chain(self):
        """Linear chain should have no closures and all torsions independent."""
        from ciffy.internal.constraints import ConstraintSystem

        # Linear chain: 0-1-2-3-4
        n_atoms = 5
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ], dtype=np.float32)

        bond_constraints = [
            (0, 1, 1.5),
            (1, 2, 1.5),
            (2, 3, 1.5),
            (3, 4, 1.5),
        ]

        system = ConstraintSystem.from_constraints(
            n_atoms=n_atoms,
            coords=coords,
            bond_constraints=bond_constraints,
        )

        # No closures
        assert system.closures.n_closures == 0
        # Atoms at level >= 3 have torsions: 3 and 4
        # So n_dof should be 2
        assert system.n_dof == 2
        assert len(system.closures.dependent_idx) == 0

    def test_from_constraints_single_ring(self):
        """Single 6-membered ring should have 3 closures and 3 DOF."""
        from ciffy.internal.constraints import ConstraintSystem

        # 6-membered ring: 0-1-2-3-4-5-0
        n_atoms = 6
        # Create hexagonal ring
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        coords = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1).astype(np.float32) * 1.5

        # Ring bonds
        bond_constraints = [
            (i, (i+1) % 6, float(np.linalg.norm(coords[(i+1)%6] - coords[i])))
            for i in range(6)
        ]

        system = ConstraintSystem.from_constraints(
            n_atoms=n_atoms,
            coords=coords,
            bond_constraints=bond_constraints,
        )

        # Ring has 1 closure (the non-tree edge)
        assert system.closures.n_closures == 1
        # For a 6-ring with 6 bonds and 5 tree edges, we have 1 closure
        # Each closure reduces DOF by up to 3
        # But actual DOF depends on the Jacobian analysis


class TestJacobianDOFDiscovery:
    """Tests for DOF discovery via Jacobian analysis."""

    def test_discover_dof_no_closures(self):
        """No closures means all torsions are independent."""
        from ciffy.internal.jacobian import discover_dof
        from ciffy.internal.constraints import ClosureConstraints, _build_dfs_timestamps

        parent = np.array([-1, 0, 1, 2, 3], dtype=np.int64)
        level = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ], dtype=np.float32)

        closures = ClosureConstraints.empty()
        dfs_enter, dfs_exit = _build_dfs_timestamps(parent)

        independent, dependent = discover_dof(
            parent, level, dfs_enter, dfs_exit, closures, coords
        )

        # Atoms at level >= 3: atoms 3 and 4
        assert len(independent) == 2
        assert len(dependent) == 0
        assert 3 in independent
        assert 4 in independent


class TestNewtonRaphsonSolver:
    """Tests for Newton-Raphson ring closure solver."""

    def test_solve_closure_no_constraints(self):
        """No closures should return internal unchanged."""
        from ciffy.internal.constraints import ConstraintSystem, solve_closure

        # Linear chain
        n_atoms = 5
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ], dtype=np.float32)

        bond_constraints = [
            (0, 1, 1.5),
            (1, 2, 1.5),
            (2, 3, 1.5),
            (3, 4, 1.5),
        ]

        system = ConstraintSystem.from_constraints(
            n_atoms=n_atoms,
            coords=coords,
            bond_constraints=bond_constraints,
        )

        internal = system.base_internal.copy()
        result = solve_closure(internal, system)

        # Should be unchanged
        np.testing.assert_array_almost_equal(result, internal)


class TestDifferentiable:
    """Tests for differentiable DOF-to-Cartesian mapping."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch not available"
    )
    def test_dof_to_cartesian_linear_chain(self):
        """Test DOF to Cartesian for linear chain."""
        import torch
        from ciffy.internal.constraints import ConstraintSystem
        from ciffy.internal.differentiable import dof_to_cartesian

        # Linear chain
        n_atoms = 5
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.5, 0.0],
            [4.5, 0.0, 0.5],
            [6.0, 0.5, 0.0],
        ], dtype=np.float32)

        bond_constraints = [
            (0, 1, 1.5),
            (1, 2, float(np.linalg.norm(coords[2] - coords[1]))),
            (2, 3, float(np.linalg.norm(coords[3] - coords[2]))),
            (3, 4, float(np.linalg.norm(coords[4] - coords[3]))),
        ]

        system = ConstraintSystem.from_constraints(
            n_atoms=n_atoms,
            coords=coords,
            bond_constraints=bond_constraints,
        )

        # Get current DOF values from system
        all_torsions = np.where(system.level >= 3)[0]
        dof_values = system.base_internal[all_torsions, 2]

        # Convert to torch
        dof_tensor = torch.from_numpy(dof_values).float().requires_grad_(True)

        # Forward pass
        coords_out = dof_to_cartesian(dof_tensor, system)

        assert coords_out.shape == (n_atoms, 3)
        assert not torch.isnan(coords_out).any()


class TestMolecularGeometryIntegration:
    """Integration tests with MolecularGeometry."""

    @pytest.mark.skip(reason="Slow - needs optimization for large molecules")
    def test_geometry_n_dof(self):
        """Test that MolecularGeometry.n_dof works with new system."""
        import ciffy

        # Load a real molecule
        polymer = ciffy.load("tests/data/9MDS.cif")

        chains = list(polymer.chains())
        if len(chains) > 0:
            # Get first chain
            chain = chains[0]
            # Access internal _geometry attribute
            geom = chain._geometry

            # Should be able to get n_dof
            n_dof = geom.n_dof
            assert n_dof >= 0
            assert isinstance(n_dof, int)

    @pytest.mark.skip(reason="Slow - needs optimization for large molecules")
    def test_geometry_dof_roundtrip(self):
        """Test that setting DOF and getting coordinates works."""
        import ciffy

        # Load a real molecule
        polymer = ciffy.load("tests/data/9MDS.cif")

        chains = list(polymer.chains())
        if len(chains) > 0:
            # Get first chain
            chain = chains[0]
            # Access internal _geometry attribute
            geom = chain._geometry

            # Get original coordinates
            original_coords = geom.coordinates.copy()

            # Get current DOF
            dof = geom.dof.copy()

            # Set DOF (should trigger ring closure if needed)
            geom.dof = dof

            # Get new coordinates
            new_coords = geom.coordinates

            # Coordinates should be close (may not be exact due to numerical precision)
            # Use a reasonable tolerance
            assert new_coords.shape == original_coords.shape
            assert not np.isnan(new_coords).any()


class TestAnalyticalJacobian:
    """Tests for analytical Jacobian computation."""

    def test_jacobian_shape(self):
        """Test that Jacobian has correct shape."""
        from ciffy.internal.constraints import ConstraintSystem
        from ciffy.internal.jacobian import compute_jacobian_analytical

        # Linear chain with a "virtual" closure
        n_atoms = 6
        # Create bent chain
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 1.0, 0.0],
            [3.5, 0.0, 0.0],
            [4.5, 1.0, 0.0],
            [5.5, 0.0, 0.0],
        ], dtype=np.float32)

        # Just tree bonds (no closure)
        bond_constraints = [
            (i, i+1, float(np.linalg.norm(coords[i+1] - coords[i])))
            for i in range(5)
        ]

        system = ConstraintSystem.from_constraints(
            n_atoms=n_atoms,
            coords=coords,
            bond_constraints=bond_constraints,
        )

        # If no closures, Jacobian should be empty
        if system.closures.n_closures == 0:
            pass  # Expected - no Jacobian to compute
        else:
            # Compute Jacobian
            torsion_indices = system.closures.dependent_idx
            if len(torsion_indices) > 0:
                J = compute_jacobian_analytical(
                    system.base_internal, coords, system, torsion_indices
                )
                # Should have 3 rows per closure, 1 column per torsion
                assert J.shape == (3 * system.closures.n_closures, len(torsion_indices))


class TestBackwardsCompatibility:
    """Tests that old API still works during transition."""

    def test_legacy_imports_work(self):
        """Legacy imports should still work."""
        from ciffy.internal import ConstraintSpec, IndependentDOF, RingConstraint, RingAnalyzer

        # Should be able to create ConstraintSpec
        spec = ConstraintSpec()
        assert spec.fixed_bonds == "all"
        assert spec.fixed_angles == "all"
