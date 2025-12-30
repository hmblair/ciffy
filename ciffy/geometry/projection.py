"""
Geometry projection onto bond length constraints.

This module provides differentiable projection of coordinates onto
bond length constraints using Gauss-Newton optimization.

The main function is `project_bond_lengths`, which takes coordinates,
bond pairs, and target lengths, and returns coordinates that satisfy
the bond length constraints.

Note: This module is torch-only because it uses torch.autograd.Function
for differentiable implicit differentiation.
"""

from __future__ import annotations

import torch


def project_bond_lengths(
    coords: torch.Tensor,
    bonds: torch.Tensor,
    ideal_lengths: torch.Tensor,
    n_steps: int = 2,
    differentiable: bool = True,
) -> torch.Tensor:
    """
    Project coordinates onto bond length constraints using Gauss-Newton.

    Uses iterative Gauss-Newton optimization to adjust coordinates so that
    specified bonds match their target lengths. Typically 2-5 steps are
    sufficient for sub-0.01Å accuracy.

    Args:
        coords: (N, n_atoms, 3) or (n_atoms, 3) coordinates.
        bonds: (n_bonds, 2) atom index pairs (int64).
        ideal_lengths: (n_bonds,) target bond lengths (float32).
        n_steps: Number of Newton iterations (default 2).
        differentiable: If True, use implicit differentiation for backward pass.
            This makes gradients project onto the constraint manifold's tangent
            space, which is useful for latent space optimization.

    Returns:
        Projected coordinates with same shape as input.

    Example:
        >>> coords = torch.randn(10, 23, 3)  # 10 samples, 23 atoms
        >>> bonds = torch.tensor([[0, 1], [1, 2], [2, 3]])  # 3 bonds
        >>> lengths = torch.tensor([1.5, 1.4, 1.5])  # target lengths
        >>> projected = project_bond_lengths(coords, bonds, lengths)
    """
    if bonds.shape[0] == 0:
        return coords  # No constraints

    single = coords.dim() == 2
    if single:
        coords = coords.unsqueeze(0)

    # Build projector functions
    newton_step, compute_jacobian = _build_projector(bonds, ideal_lengths, coords.device)

    if differentiable:
        projected = _ImplicitBondProjection.apply(
            coords, newton_step, compute_jacobian, n_steps
        )
    else:
        # Explicit iteration (faster, but gradients depend on n_steps)
        projected = []
        for i in range(coords.shape[0]):
            c = coords[i]
            for _ in range(n_steps):
                c = newton_step(c)
            projected.append(c)
        projected = torch.stack(projected)

    return projected[0] if single else projected


def _build_projector(
    bonds: torch.Tensor,
    ideal_lengths: torch.Tensor,
    device: torch.device,
) -> tuple:
    """
    Build Newton step and Jacobian functions for bond constraints.

    Args:
        bonds: (n_bonds, 2) atom index pairs.
        ideal_lengths: (n_bonds,) target bond lengths.
        device: Target device.

    Returns:
        Tuple of (newton_step, compute_jacobian) functions.
    """
    n_bonds = bonds.shape[0]
    bonds = bonds.to(device)
    ideal_lengths = ideal_lengths.to(device)

    # Pre-compute index tensors
    bond_idx = torch.arange(n_bonds, device=device)
    dims = torch.arange(3, device=device)

    def compute_jacobian(coords: torch.Tensor) -> torch.Tensor:
        """Compute constraint Jacobian. Returns (n_bonds, n_atoms*3)."""
        n_atoms = coords.shape[0]
        J = torch.zeros(n_bonds, n_atoms * 3, device=coords.device, dtype=coords.dtype)

        a1_idx = bonds[:, 0]
        a2_idx = bonds[:, 1]
        diff = coords[a2_idx] - coords[a1_idx]
        lengths = torch.norm(diff, dim=1)
        units = diff / (lengths.unsqueeze(1) + 1e-8)

        row_idx = bond_idx.unsqueeze(1).expand(-1, 3).reshape(-1)
        col_a1 = (a1_idx.unsqueeze(1) * 3 + dims).reshape(-1)
        col_a2 = (a2_idx.unsqueeze(1) * 3 + dims).reshape(-1)
        J[row_idx, col_a1] = -units.reshape(-1)
        J[row_idx, col_a2] = units.reshape(-1)

        return J

    def newton_step(coords: torch.Tensor) -> torch.Tensor:
        """Single vectorized Newton step. coords: (n_atoms, 3)."""
        n_atoms = coords.shape[0]

        # Compute residuals
        a1_idx = bonds[:, 0]
        a2_idx = bonds[:, 1]
        diff = coords[a2_idx] - coords[a1_idx]
        lengths = torch.norm(diff, dim=1)
        residuals = lengths - ideal_lengths

        # Get Jacobian
        J = compute_jacobian(coords)

        # Gauss-Newton: dx = -J^T @ (J @ J^T)^{-1} @ residuals
        JJT = J @ J.T
        y = torch.linalg.solve(JJT, residuals)
        dx = -J.T @ y

        return coords + dx.reshape(n_atoms, 3)

    return newton_step, compute_jacobian


class _ImplicitBondProjection(torch.autograd.Function):
    """
    Implicit differentiation for bond length projection.

    At convergence, the gradient is projected onto the constraint manifold's
    tangent space:

        grad_input = P_tangent @ grad_output

    where P_tangent = I - J^T @ (J @ J^T)^{-1} @ J is the projection onto the
    null space of the constraint Jacobian J.

    This makes gradients independent of iteration count - at convergence,
    the gradient reflects the geometry of the constraint manifold rather
    than the optimization path.
    """

    @staticmethod
    def forward(ctx, coords, newton_step, compute_jacobian, n_steps):
        """Run Newton projection to convergence."""
        with torch.no_grad():
            projected = []
            for i in range(coords.shape[0]):
                c = coords[i].clone()
                for _ in range(n_steps):
                    c = newton_step(c)
                projected.append(c)
            result = torch.stack(projected)

        ctx.compute_jacobian = compute_jacobian
        ctx.save_for_backward(result)

        return result.requires_grad_(coords.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        """Project gradient onto constraint tangent space."""
        (converged_coords,) = ctx.saved_tensors
        compute_jacobian = ctx.compute_jacobian

        grad_input = []
        for i in range(converged_coords.shape[0]):
            coords_i = converged_coords[i]
            grad_i = grad_output[i]

            J = compute_jacobian(coords_i)

            if J.shape[0] == 0:
                grad_input.append(grad_i)
                continue

            g_flat = grad_i.reshape(-1)

            # Project onto tangent space:
            # g_proj = g - J^T @ (J @ J^T)^{-1} @ (J @ g)
            Jg = J @ g_flat
            JJT = J @ J.T
            v = torch.linalg.solve(JJT, Jg)
            correction = J.T @ v

            g_proj = g_flat - correction
            grad_input.append(g_proj.reshape_as(grad_i))

        grad_input = torch.stack(grad_input)

        return grad_input, None, None, None
