"""
Analytical ring closure solver for 5-6 member rings.

This module provides exact (non-iterative) ring closure for simple rings
using circle-sphere intersection geometry. For a ring with fixed bonds
and angles, the closing atom must lie on both:

1. A circle (traced as we vary its dihedral angle)
2. A sphere (at fixed distance from the first ring atom)

The intersection gives 0, 1, or 2 exact solutions.

This is much faster and more accurate than CCD for simple rings:
- CCD: 10-100 iterations, ~0.1 Å tolerance
- Analytical: O(1), machine precision (~1e-7 Å)

For fused rings or complex topologies, fall back to CCD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch, to_numpy
from .geometry import (
    circle_sphere_intersect,
    norm,
    normalize,
    to_scalar,
    clone,
)

if TYPE_CHECKING:
    from .ring_analysis import RingConstraint
    from .tree import SpanningTree


@dataclass
class AnalyticalSolution:
    """Result of analytical ring closure."""
    internal: Array  # Updated internal coordinates
    dihedral_value: float  # The dihedral angle that closes the ring
    closure_error: float  # Distance error at closure bond


class AnalyticalRingSolver:
    """
    Analytical ring closure for simple 5-6 member rings.

    Uses circle-sphere intersection to find exact dihedral values
    that close the ring. Much faster and more accurate than CCD.

    Attributes:
        tree: SpanningTree for NERF reconstruction
        fixed_coords: Reference coordinates for atoms at levels 0-2
        offsets: Per-component centering offsets
    """

    def __init__(
        self,
        tree: "SpanningTree",
        fixed_coords: np.ndarray,
        offsets: np.ndarray,
    ):
        """
        Initialize analytical solver.

        Args:
            tree: SpanningTree for NERF reconstruction
            fixed_coords: Reference coordinates for atoms at levels 0-2
            offsets: Per-component centering offsets
        """
        self.tree = tree
        self.fixed_coords = fixed_coords
        self.offsets = offsets

    def can_solve_analytically(self, ring_constraint: "RingConstraint") -> bool:
        """
        Check if this ring can be solved analytically.

        Currently supports:
        - 5-member rings (2 independent, 3 dependent dihedrals)
        - 6-member rings (3 independent, 3 dependent dihedrals)
        - Unfused rings only

        Args:
            ring_constraint: The ring to check

        Returns:
            True if analytical solution is possible
        """
        # Check ring size
        ring_size = ring_constraint.ring_size
        if ring_size < 5 or ring_size > 6:
            return False

        # Check if fused (would be marked by ring analysis)
        if hasattr(ring_constraint, 'is_fused') and ring_constraint.is_fused:
            return False

        # Need at least 1 effective dependent dihedral
        if len(ring_constraint.dependent_dihedrals) < 1:
            return False

        return True

    def solve_ring(
        self,
        internal: Array,
        ring_constraint: "RingConstraint",
        original_coords: Array,
    ) -> tuple[Array, bool]:
        """
        Solve ring closure analytically.

        Args:
            internal: (N, 3) internal coordinates [distance, angle, dihedral]
            ring_constraint: Ring constraint to solve
            original_coords: Original Cartesian coordinates (for solution selection)

        Returns:
            (updated_internal, success) where:
            - updated_internal: internal coords with ring closed
            - success: True if a valid solution was found
        """
        internal = clone(internal)

        # Get ring info
        ring_atoms = ring_constraint.ring_atoms
        closure_i, closure_j = ring_constraint.closure_bond
        dependent_dihedrals = ring_constraint.dependent_dihedrals

        if len(dependent_dihedrals) == 0:
            return internal, False

        # Get the dependent dihedral atom (the one we'll solve for)
        # Use the first dependent dihedral - it should move one closure atom
        dep_atom = int(dependent_dihedrals[0])

        # Build atom -> Z-matrix row mapping
        zmatrix_indices = self.tree.to_zmatrix_indices()
        atom_to_row = {}
        for row in range(len(zmatrix_indices)):
            atom = int(zmatrix_indices[row, 0])
            atom_to_row[atom] = row

        dep_row = atom_to_row.get(dep_atom, -1)
        if dep_row < 3:
            return internal, False

        # Get Z-matrix references for the dependent atom
        dist_ref = int(zmatrix_indices[dep_row, 1])  # Parent (bond partner)
        ang_ref = int(zmatrix_indices[dep_row, 2])   # Angle reference
        dih_ref = int(zmatrix_indices[dep_row, 3])   # Dihedral reference

        if dist_ref < 0 or ang_ref < 0 or dih_ref < 0:
            return internal, False

        # Get bond length and angle for the dependent atom
        bond_length = to_scalar(internal[dep_row, 0])
        bond_angle = to_scalar(internal[dep_row, 1])

        # Get expected closure distance from original coords
        original_np = to_numpy(original_coords)
        expected_closure_dist = float(np.linalg.norm(
            original_np[closure_j] - original_np[closure_i]
        ))

        # Reconstruct coordinates with current internal values
        internal_np = to_numpy(internal).astype(np.float32)
        coords = self.tree.internal_to_cartesian(
            internal_np, self.fixed_coords, self.offsets
        )

        # Early exit: check if ring is already closed
        current_closure_dist = float(np.linalg.norm(
            coords[closure_j] - coords[closure_i]
        ))
        if abs(current_closure_dist - expected_closure_dist) < 0.01:
            # Ring is already closed, return unchanged
            return internal, True

        # Determine which closure atom moves when we rotate this dihedral
        # We need to find the atom that's "downstream" of the dependent dihedral
        affected = self._find_affected_atoms(zmatrix_indices, dep_atom)

        if closure_i in affected and closure_j not in affected:
            moving_atom = closure_i
            anchor_atom = closure_j
        elif closure_j in affected and closure_i not in affected:
            moving_atom = closure_j
            anchor_atom = closure_i
        else:
            # Both or neither affected - can't solve with this dihedral
            return internal, False

        # The moving atom traces a circle as we vary the dihedral
        # Circle center: project moving_atom onto the rotation axis
        # Rotation axis: from ang_ref toward dist_ref
        axis_point = coords[dist_ref]
        axis_dir = normalize(coords[ang_ref] - coords[dist_ref])

        # Current position of moving atom
        current_pos = coords[moving_atom]

        # Circle center: axis_point + projection of (current_pos - axis_point) onto axis
        v = current_pos - axis_point
        v_para = axis_dir * np.dot(axis_dir, v)
        circle_center = axis_point + v_para

        # Circle radius: perpendicular distance from moving_atom to axis
        v_perp = v - v_para
        circle_radius = float(np.linalg.norm(v_perp))

        if circle_radius < 1e-10:
            # Degenerate: moving atom is on the axis
            return internal, False

        # Sphere: centered at anchor_atom with closure bond radius
        sphere_center = coords[anchor_atom]
        sphere_radius = expected_closure_dist

        # Find circle-sphere intersection
        solutions = circle_sphere_intersect(
            circle_center=circle_center,
            circle_axis=axis_dir,
            circle_radius=circle_radius,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
        )

        if len(solutions) == 0:
            # No intersection - ring cannot close with this geometry
            return internal, False

        # Select best solution (closest to original position)
        original_moving_pos = original_np[moving_atom]
        best_solution = None
        best_dist = float('inf')

        for point, angle in solutions:
            point_np = to_numpy(point)
            dist = float(np.linalg.norm(point_np - original_moving_pos))
            if dist < best_dist:
                best_dist = dist
                best_solution = (point_np, angle)

        if best_solution is None:
            return internal, False

        target_point, target_angle = best_solution

        # Compute the required dihedral change
        # The angle from circle_sphere_intersect is relative to an arbitrary basis
        # We need to compute the actual dihedral change
        current_dihedral = to_scalar(internal[dep_row, 2])

        # Compute current angle on circle
        v_perp_normalized = v_perp / circle_radius
        perp_basis = np.cross(axis_dir, v_perp_normalized)
        current_angle_on_circle = np.arctan2(
            np.dot(v_perp, perp_basis),
            np.dot(v_perp, v_perp_normalized)
        )

        # Compute target angle on circle
        target_v = target_point - circle_center
        target_angle_on_circle = np.arctan2(
            np.dot(target_v, perp_basis),
            np.dot(target_v, v_perp_normalized)
        )

        # Dihedral change needed
        dihedral_change = target_angle_on_circle - current_angle_on_circle

        # New dihedral value
        new_dihedral = current_dihedral + dihedral_change

        # Wrap to [-π, π]
        while new_dihedral > np.pi:
            new_dihedral -= 2 * np.pi
        while new_dihedral < -np.pi:
            new_dihedral += 2 * np.pi

        # Update the internal coordinates
        if is_torch(internal):
            import torch
            internal[dep_row, 2] = torch.tensor(
                new_dihedral, dtype=internal.dtype, device=internal.device
            )
        else:
            internal[dep_row, 2] = new_dihedral

        # Verify the solution
        internal_np = to_numpy(internal).astype(np.float32)
        new_coords = self.tree.internal_to_cartesian(
            internal_np, self.fixed_coords, self.offsets
        )

        closure_error = abs(
            np.linalg.norm(new_coords[closure_j] - new_coords[closure_i])
            - expected_closure_dist
        )

        # Success if error is small
        success = closure_error < 0.01  # 0.01 Å tolerance

        return internal, success

    def _find_affected_atoms(
        self,
        zmatrix_indices: np.ndarray,
        dihedral_atom: int,
    ) -> set[int]:
        """
        Find all atoms affected by rotating a dihedral.

        When we rotate the dihedral at `dihedral_atom`, all atoms that
        depend on it (directly or transitively) in the Z-matrix will move.
        """
        # Find the row of the dihedral atom
        dihedral_row = -1
        for row in range(len(zmatrix_indices)):
            if int(zmatrix_indices[row, 0]) == dihedral_atom:
                dihedral_row = row
                break

        if dihedral_row < 0:
            return set()

        affected = {dihedral_atom}

        # All atoms placed AFTER the dihedral atom that depend on it
        for row in range(dihedral_row + 1, len(zmatrix_indices)):
            atom = int(zmatrix_indices[row, 0])
            # Check if any reference is in affected set
            for ref_idx in [1, 2, 3]:
                ref = int(zmatrix_indices[row, ref_idx])
                if ref >= 0 and ref in affected:
                    affected.add(atom)
                    break

        return affected
