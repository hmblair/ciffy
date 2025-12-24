"""
Differentiable DOF to Cartesian mapping with implicit differentiation.

This module provides PyTorch autograd functions for mapping independent
degrees of freedom (torsions) to Cartesian coordinates, with exact gradients
via implicit differentiation.

Key insight: at the Newton solution, constraints F(φ_dep, φ_ind) = 0 are satisfied.
By the Implicit Function Theorem:
    ∂φ_dep/∂φ_ind = -J_dep⁻¹ @ J_ind

This gives exact gradients without unrolling Newton iterations, using O(D²)
memory instead of O(iters × D²).

Classes:
    DOFToCartesian: Autograd function for DOF → Cartesian with closure solving

Functions:
    dof_to_cartesian: Functional API for DOFToCartesian
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

__all__ = [
    "DOFToCartesian",
    "dof_to_cartesian",
    "TORCH_AVAILABLE",
]

try:
    import torch
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Function = object  # Dummy for type hints

if TYPE_CHECKING:
    from .constraints import ConstraintSystem


class DOFToCartesian(Function):
    """
    Autograd function for DOF → Cartesian with ring closure solving.

    Forward pass:
        1. Set independent torsions from DOF values
        2. Newton-Raphson solve for dependent torsions
        3. NERF reconstruction to Cartesian coordinates

    Backward pass:
        Uses implicit differentiation (Implicit Function Theorem) for exact
        gradients through the Newton solve, without unrolling iterations.

        At solution: F(φ_dep, φ_ind) = 0
        By IFT: ∂φ_dep/∂φ_ind = -J_dep⁻¹ @ J_ind

        Chain rule for gradients:
        ∂L/∂φ_ind = ∂L/∂φ_ind_direct - J_ind.T @ J_dep^{-T} @ ∂L/∂φ_dep

    Usage:
        >>> system = ConstraintSystem.from_topology(topology, coords)
        >>> dof = torch.randn(system.n_dof, requires_grad=True)
        >>> coords = DOFToCartesian.apply(dof, system)
        >>> loss = coords.sum()
        >>> loss.backward()  # Exact gradients via IFT
    """

    @staticmethod
    def forward(
        ctx: Any,
        dof_values: "torch.Tensor",
        system: "ConstraintSystem",
    ) -> "torch.Tensor":
        """
        Map DOF values to Cartesian coordinates.

        Args:
            ctx: Autograd context for saving tensors.
            dof_values: (K,) float32 tensor of independent torsion values (radians).
            system: ConstraintSystem with topology and constraints.

        Returns:
            (N, 3) float32 tensor of Cartesian coordinates.
        """
        from .constraints import solve_closure
        from .tree import SpanningTree, derive_zmatrix_from_parent

        device = dof_values.device
        dtype = dof_values.dtype

        # Get numpy values for computation
        dof_np = dof_values.detach().cpu().numpy().astype(np.float32)

        # Set independent torsions in internal coordinates
        internal = system.base_internal.copy()
        independent_idx = system.closures.independent_idx
        dependent_set = set(system.closures.dependent_idx.tolist())

        # Map DOF to torsions: all torsions that are not dependent
        all_torsions = np.where(system.level >= 3)[0]
        dof_i = 0
        for torsion_atom in all_torsions:
            if torsion_atom not in dependent_set and dof_i < len(dof_np):
                internal[torsion_atom, 2] = dof_np[dof_i]
                dof_i += 1

        # Newton-Raphson solve for dependent torsions
        internal = solve_closure(internal, system)

        # Build reconstruction data
        tree = SpanningTree(
            parent=system.parent,
            level=system.level,
            component_id=np.zeros(system.n_atoms, dtype=np.int32),
            n_components=1,
        )

        # NERF reconstruction
        coords_np = tree.internal_to_cartesian(
            internal, system.fixed_coords, system.center_offsets
        )

        # Convert to tensor
        coords = torch.from_numpy(coords_np.astype(np.float32)).to(device)

        # Save for backward
        ctx.save_for_backward(
            torch.from_numpy(internal.astype(np.float32)).to(device),
            torch.from_numpy(coords_np.astype(np.float32)).to(device),
        )
        ctx.system = system
        ctx.device = device
        ctx.dtype = dtype

        return coords

    @staticmethod
    def backward(
        ctx: Any,
        grad_coords: "torch.Tensor",
    ) -> tuple["torch.Tensor", None]:
        """
        Backward pass using implicit differentiation.

        Uses the Implicit Function Theorem to compute exact gradients
        through the Newton solve without unrolling.

        Args:
            ctx: Autograd context with saved tensors.
            grad_coords: (N, 3) upstream gradients for coordinates.

        Returns:
            Tuple of (grad_dof, None) - None for system (not differentiable).
        """
        from .jacobian import compute_jacobian_for_backward
        from .tree import SpanningTree, derive_zmatrix_from_parent

        internal_t, coords_t = ctx.saved_tensors
        system = ctx.system
        device = ctx.device

        internal = internal_t.detach().cpu().numpy()
        coords = coords_t.detach().cpu().numpy()
        grad_coords_np = grad_coords.detach().cpu().numpy()

        # Step 1: NERF backward to get grad_internal
        # Use C extension if available, otherwise approximate
        grad_internal = _nerf_backward_numpy(
            internal, coords, grad_coords_np, system
        )

        # Step 2: Implicit differentiation for closure solve
        # At solution: F(φ_dep, φ_ind) = 0
        # By IFT: ∂φ_dep/∂φ_ind = -J_dep⁻¹ @ J_ind
        # Chain rule: ∂L/∂φ_ind = ∂L/∂φ_ind_direct - J_ind.T @ J_dep^{-T} @ ∂L/∂φ_dep

        if system.closures.n_dependent > 0:
            # Get Jacobians
            J_dep, J_ind = compute_jacobian_for_backward(internal, coords, system)

            # Get gradients for dependent torsions
            dep_idx = system.closures.dependent_idx
            grad_dep = grad_internal[dep_idx, 2]  # (D,)

            # Solve adjoint system: J_dep.T @ v = grad_dep
            # This gives v = J_dep^{-T} @ grad_dep
            if J_dep.shape[0] > 0 and J_dep.shape[1] > 0:
                try:
                    v, *_ = np.linalg.lstsq(J_dep.T, grad_dep, rcond=1e-10)

                    # Contribution from implicit differentiation
                    # ∂L/∂φ_ind -= J_ind.T @ v
                    grad_implicit = J_ind.T @ v  # (K,)

                    # Apply to independent torsions
                    ind_idx = system.closures.independent_idx
                    for i, idx in enumerate(ind_idx):
                        if i < len(grad_implicit):
                            grad_internal[idx, 2] -= grad_implicit[i]
                except np.linalg.LinAlgError:
                    # Singular Jacobian - skip implicit diff contribution
                    pass

        # Step 3: Collect gradients for DOF
        # DOF maps to all independent torsions (level >= 3, not dependent)
        all_torsions = np.where(system.level >= 3)[0]
        n_dof = len(all_torsions) - len(system.closures.dependent_idx)

        grad_dof = np.zeros(n_dof, dtype=np.float32)
        dof_idx = 0
        for torsion_atom in all_torsions:
            if torsion_atom not in system.closures.dependent_idx:
                if dof_idx < n_dof:
                    grad_dof[dof_idx] = grad_internal[torsion_atom, 2]
                    dof_idx += 1

        grad_dof_t = torch.from_numpy(grad_dof).to(device)
        return grad_dof_t, None


def _nerf_backward_numpy(
    internal: np.ndarray,
    coords: np.ndarray,
    grad_coords: np.ndarray,
    system: "ConstraintSystem",
) -> np.ndarray:
    """
    Compute NERF backward pass in NumPy using DFS timestamps.

    For each atom at level >= 3, the gradient w.r.t. dihedral is:
    ∂L/∂θ = Σ_d (∂L/∂x_d) · (∂x_d/∂θ)

    where d ranges over descendants of the torsion atom.

    Args:
        internal: (N, 3) internal coordinates
        coords: (N, 3) Cartesian coordinates
        grad_coords: (N, 3) upstream gradients
        system: ConstraintSystem

    Returns:
        (N, 3) gradients for internal coordinates
    """
    n_atoms = system.n_atoms
    grad_internal = np.zeros_like(internal)

    # Get torsion atoms (level >= 3)
    torsion_atoms = np.where(system.level >= 3)[0]
    if len(torsion_atoms) == 0:
        return grad_internal

    # For each torsion, compute rotation axis and accumulate gradients
    parents = system.parent[torsion_atoms]
    grandparents = system.parent[parents]

    # Filter valid torsions (have grandparent)
    valid = (parents >= 0) & (grandparents >= 0)
    valid_torsions = torsion_atoms[valid]
    valid_parents = parents[valid]
    valid_grandparents = grandparents[valid]

    if len(valid_torsions) == 0:
        return grad_internal

    # Compute rotation axes for all torsions
    axis_origins = coords[valid_parents]  # (T, 3)
    axis_dirs = coords[valid_grandparents] - axis_origins  # (T, 3)
    axis_norms = np.linalg.norm(axis_dirs, axis=1, keepdims=True)
    axis_norms = np.maximum(axis_norms, 1e-10)
    axis_dirs = axis_dirs / axis_norms  # (T, 3)

    # For each torsion, find descendants using DFS timestamps
    dfs_enter = system.dfs_enter
    dfs_exit = system.dfs_exit

    # All atom indices
    all_atoms = np.arange(n_atoms)

    for i, torsion_atom in enumerate(valid_torsions):
        # Descendants: atoms where enter[torsion] <= enter[atom] <= exit[torsion]
        enter_t = dfs_enter[torsion_atom]
        exit_t = dfs_exit[torsion_atom]
        is_descendant = (dfs_enter >= enter_t) & (dfs_enter <= exit_t)

        # Exclude the torsion atom itself
        is_descendant[torsion_atom] = False

        # Get descendant positions and gradients
        desc_mask = np.where(is_descendant)[0]
        if len(desc_mask) == 0:
            continue

        # r vectors: position relative to axis origin
        r = coords[desc_mask] - axis_origins[i]  # (D, 3)

        # ∂x_d/∂θ = axis × r
        dpos_dtheta = np.cross(axis_dirs[i], r)  # (D, 3)

        # ∂L/∂θ = Σ (∂L/∂x_d) · (∂x_d/∂θ)
        grad_internal[torsion_atom, 2] = np.sum(
            grad_coords[desc_mask] * dpos_dtheta
        )

    return grad_internal


def dof_to_cartesian(
    dof_values: "torch.Tensor",
    system: "ConstraintSystem",
) -> "torch.Tensor":
    """
    Convert DOF values to Cartesian coordinates with autograd support.

    Functional wrapper around DOFToCartesian.apply().

    Args:
        dof_values: (K,) tensor of independent torsion values in radians.
        system: ConstraintSystem with topology and constraints.

    Returns:
        (N, 3) tensor of Cartesian coordinates.

    Example:
        >>> system = ConstraintSystem.from_topology(topology, coords)
        >>> dof = torch.randn(system.n_dof, requires_grad=True)
        >>> coords = dof_to_cartesian(dof, system)
        >>> loss = compute_loss(coords)
        >>> loss.backward()  # Gradients flow through closure solving
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for dof_to_cartesian")

    return DOFToCartesian.apply(dof_values, system)


# =============================================================================
# Batch Operations (for ML training)
# =============================================================================


def batch_dof_to_cartesian(
    dof_batch: "torch.Tensor",
    system: "ConstraintSystem",
) -> "torch.Tensor":
    """
    Convert batch of DOF values to Cartesian coordinates.

    Args:
        dof_batch: (B, K) tensor of DOF values for B samples.
        system: ConstraintSystem (shared across batch).

    Returns:
        (B, N, 3) tensor of Cartesian coordinates.

    Note:
        Currently loops over batch dimension. Future optimization could
        parallelize Newton solves across batch.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for batch_dof_to_cartesian")

    batch_size = dof_batch.shape[0]
    n_atoms = system.n_atoms

    coords_list = []
    for b in range(batch_size):
        coords_b = dof_to_cartesian(dof_batch[b], system)
        coords_list.append(coords_b)

    return torch.stack(coords_list, dim=0)


# =============================================================================
# Utility Functions
# =============================================================================


def get_dof_from_internal(
    internal: np.ndarray,
    system: "ConstraintSystem",
) -> np.ndarray:
    """
    Extract DOF values from internal coordinates.

    Args:
        internal: (N, 3) internal coordinates
        system: ConstraintSystem

    Returns:
        (K,) array of DOF values (independent torsions)
    """
    all_torsions = np.where(system.level >= 3)[0]
    dependent_set = set(system.closures.dependent_idx.tolist())

    dof = []
    for atom in all_torsions:
        if atom not in dependent_set:
            dof.append(internal[atom, 2])

    return np.array(dof, dtype=np.float32)


def set_dof_in_internal(
    internal: np.ndarray,
    dof: np.ndarray,
    system: "ConstraintSystem",
) -> np.ndarray:
    """
    Set DOF values in internal coordinates.

    Args:
        internal: (N, 3) internal coordinates (modified in place)
        dof: (K,) DOF values
        system: ConstraintSystem

    Returns:
        (N, 3) modified internal coordinates
    """
    all_torsions = np.where(system.level >= 3)[0]
    dependent_set = set(system.closures.dependent_idx.tolist())

    dof_idx = 0
    for atom in all_torsions:
        if atom not in dependent_set:
            if dof_idx < len(dof):
                internal[atom, 2] = dof[dof_idx]
                dof_idx += 1

    return internal
