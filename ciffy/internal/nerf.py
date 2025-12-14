"""
NERF (Natural Extension Reference Frame) algorithm for coordinate reconstruction.

Reconstructs Cartesian coordinates from internal coordinates (bond lengths,
bond angles, dihedral angles) in a differentiable manner suitable for
gradient-based optimization and machine learning.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch

# Try to import C extension
try:
    from .._c import _nerf_reconstruct as _c_nerf_reconstruct
    _HAS_C_EXTENSION = True
except ImportError:
    _HAS_C_EXTENSION = False


def nerf_reconstruct(
    zmatrix_indices: Array,
    distances: Array,
    angles: Array,
    dihedrals: Array,
    n_atoms: int | None = None,
) -> Array:
    """
    Reconstruct Cartesian coordinates using NERF algorithm.

    The Natural Extension Reference Frame algorithm places each atom
    by constructing a local coordinate system from three previously
    placed atoms, then positioning the new atom using spherical-like
    coordinates (distance, angle, dihedral).

    This implementation is fully differentiable for PyTorch tensors.

    Args:
        zmatrix_indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]
        distances: (M,) bond lengths in Angstroms (in BFS order).
        angles: (M,) bond angles in radians (in BFS order).
        dihedrals: (M,) dihedral angles in radians (in BFS order).
        n_atoms: Total number of atoms (including orphans). If None,
            inferred from max Z-matrix index.

    Returns:
        (N, 3) array of Cartesian coordinates in original atom order.
    """
    n_entries = len(zmatrix_indices)

    # Find max atom index to allocate coords array
    if n_atoms is None:
        if n_entries > 0:
            max_idx = int(zmatrix_indices[:, 0].max())
            n_atoms = max_idx + 1
        else:
            n_atoms = 0

    # Use C extension if available (works for NumPy and Torch tensors without grad)
    use_c = _HAS_C_EXTENSION
    if is_torch(distances):
        # If any input tensor tracks gradients, stay on the pure torch path.
        if distances.requires_grad or angles.requires_grad or dihedrals.requires_grad:
            use_c = False

    if use_c:
        # Ensure indices are numpy int64
        if is_torch(zmatrix_indices):
            indices_np = zmatrix_indices.cpu().numpy()
        else:
            indices_np = np.asarray(zmatrix_indices)

        if is_torch(distances):
            import torch
            device = distances.device
            dtype = distances.dtype
            dist_f32 = distances.detach().cpu().to(torch.float32).numpy()
            ang_f32 = angles.detach().cpu().to(torch.float32).numpy()
            dih_f32 = dihedrals.detach().cpu().to(torch.float32).numpy()
        else:
            dist_f32 = np.ascontiguousarray(distances, dtype=np.float32)
            ang_f32 = np.ascontiguousarray(angles, dtype=np.float32)
            dih_f32 = np.ascontiguousarray(dihedrals, dtype=np.float32)

        # Call C extension
        coords_np = _c_nerf_reconstruct(indices_np, dist_f32, ang_f32, dih_f32, n_atoms)

        if is_torch(distances):
            import torch
            coords = torch.from_numpy(coords_np).to(device=device, dtype=dtype)
        else:
            coords = coords_np
    else:
        # Python fallback (also used for PyTorch)
        if is_torch(distances):
            import torch
            coords = torch.zeros(n_atoms, 3, dtype=distances.dtype, device=distances.device)
        else:
            coords = np.zeros((n_atoms, 3), dtype=np.float32)

        # Process entries in BFS order
        # Each entry places the atom at coords[entry.atom_idx]
        # References point to original atom indices which are already placed
        for i in range(n_entries):
            atom_idx = int(zmatrix_indices[i, 0])
            dist_ref = int(zmatrix_indices[i, 1])
            ang_ref = int(zmatrix_indices[i, 2])
            dih_ref = int(zmatrix_indices[i, 3])

            if dist_ref < 0:
                # First atom: place at origin
                coords[atom_idx] = _zeros_3(distances)

            elif ang_ref < 0:
                # Second atom: place along +X axis from first
                coords[atom_idx] = _place_along_x(
                    coords[dist_ref],
                    distances[i],
                    distances,
                )

            elif dih_ref < 0:
                # Third atom: place in plane with first two
                coords[atom_idx] = _place_in_xy_plane(
                    coords[dist_ref],
                    coords[ang_ref],
                    distances[i],
                    angles[i],
                    distances,
                )

            else:
                # Full NERF placement
                coords[atom_idx] = _nerf_place_atom(
                    coords[dih_ref],
                    coords[ang_ref],
                    coords[dist_ref],
                    distances[i],
                    angles[i],
                    dihedrals[i],
                )

    return coords


def _zeros_3(like: Array) -> Array:
    """Create (3,) zeros array matching backend."""
    if is_torch(like):
        import torch
        return torch.zeros(3, dtype=like.dtype, device=like.device)
    return np.zeros(3, dtype=np.float32)


def _place_along_x(ref_pos: Array, distance: Array, like: Array) -> Array:
    """
    Place second atom at given distance from reference, along +X axis.

    The second atom in a chain is placed along the +X axis from the
    reference atom (which was placed at origin).
    """
    if is_torch(like):
        import torch
        result = ref_pos.clone()
        result[0] = result[0] + distance
        return result
    result = ref_pos.copy()
    result[0] = result[0] + float(distance)
    return result


def _place_in_xy_plane(
    ref1: Array,  # Distance reference (parent)
    ref2: Array,  # Angle reference
    distance: Array,
    angle: Array,
    like: Array,
) -> Array:
    """
    Place third atom using distance and angle.

    The atom is placed at the given distance from ref1, with the given
    bond angle at ref1 (angle between ref2-ref1-new_atom).

    Args:
        ref1: Position of distance reference atom (the atom we're bonding to).
        ref2: Position of angle reference atom.
        distance: Bond length to ref1.
        angle: Bond angle at ref1 (angle ref2-ref1-new).
        like: Template for backend detection.

    Returns:
        Position (3,) of new atom.
    """
    if is_torch(like):
        import torch
        sin, cos = torch.sin, torch.cos
        norm = torch.norm
    else:
        sin, cos = np.sin, np.cos
        norm = np.linalg.norm

    # Direction from ref1 to ref2 (this is our reference direction for the angle)
    u = ref2 - ref1
    u_norm = norm(u) + 1e-8
    u = u / u_norm

    # Create perpendicular direction in XY plane
    # We need a vector perpendicular to u
    if is_torch(like):
        import torch
        # For simplicity, use a fixed perpendicular (works when u is in XY plane)
        perp = torch.tensor([0.0, 0.0, 1.0], dtype=like.dtype, device=like.device)
        perp = torch.cross(perp, u, dim=0)
        perp_norm = torch.norm(perp) + 1e-8
        perp = perp / perp_norm
    else:
        # Use z-axis cross product to get perpendicular in XY plane
        perp = np.cross([0.0, 0.0, 1.0], u)
        perp_norm = np.linalg.norm(perp) + 1e-8
        perp = perp / perp_norm

    # Place new atom at distance from ref1
    # The angle is measured from the ref1->ref2 direction
    # new_pos = ref1 + distance * (cos(angle) * u + sin(angle) * perp)
    new_pos = ref1 + distance * (cos(angle) * u + sin(angle) * perp)

    return new_pos


def _nerf_place_atom(
    p1: Array,  # Dihedral reference
    p2: Array,  # Angle reference
    p3: Array,  # Distance reference (parent)
    distance: Array,
    angle: Array,
    dihedral: Array,
) -> Array:
    """
    Place atom using NERF algorithm.

    The new atom D is placed such that:
    - |D - p3| = distance
    - angle(p2, p3, D) = angle  (angle at p3 between p2-p3-D)
    - dihedral(p1, p2, p3, D) = dihedral

    Args:
        p1: Dihedral reference position.
        p2: Angle reference position.
        p3: Distance reference position (new atom is bonded to this).
        distance: Bond length from p3 to new atom.
        angle: Bond angle at p3 (p2-p3-new_atom).
        dihedral: Dihedral angle (p1-p2-p3-new_atom).

    Returns:
        Position (3,) of the new atom D.
    """
    if is_torch(p1):
        import torch
        sin, cos = torch.sin, torch.cos
        norm = torch.norm
        cross = lambda a, b: torch.cross(a, b, dim=0)
    else:
        sin, cos = np.sin, np.cos
        norm = np.linalg.norm
        cross = np.cross

    # Build local coordinate system at p3:
    # - z-axis: direction from p3 towards p2 (the reference direction for angle)
    # - x-axis: in the p1-p2-p3 plane, perpendicular to z
    # - y-axis: perpendicular to both (normal to p1-p2-p3 plane)

    # z = direction from p3 to p2 (normalized)
    z = p2 - p3
    z_len = norm(z) + 1e-8
    z = z / z_len

    # Build x and y from the p1-p2-p3 plane
    # Vector in the plane: p1 - p3
    v = p1 - p3
    v_len = norm(v) + 1e-8
    v = v / v_len

    # y = z cross v (normal to plane, right-handed)
    y = cross(z, v)
    y_len = norm(y) + 1e-8
    y = y / y_len

    # x = y cross z (in plane, perpendicular to z)
    x = cross(y, z)

    # Place new atom D at distance from p3:
    # - The angle between (D - p3) and z should equal `angle`
    # - The dihedral is the rotation around z
    #
    # D - p3 = distance * (cos(angle) * z + sin(angle) * (cos(dihedral) * x + sin(dihedral) * y))

    # Component along z (towards p2)
    d_z = distance * cos(angle)
    # Component perpendicular to z (in the x-y plane)
    d_perp = distance * sin(angle)
    d_x = d_perp * cos(dihedral)
    d_y = d_perp * sin(dihedral)

    # New position in global coordinates
    new_pos = p3 + d_z * z + d_x * x + d_y * y

    return new_pos
