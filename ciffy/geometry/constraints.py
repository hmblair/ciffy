"""
Geometry constraints for generative model training.

This module provides the GeometryConstraints class which:
1. Extracts bond and angle constraints from ciffy biochemistry
2. Remaps constraints to match model atom ordering
3. Computes differentiable losses for training

All operations are fully vectorized for efficient GPU computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ciffy.biochemistry.atom import AtomGroup


@dataclass
class GeometryConstraints:
    """
    Geometry constraints remapped to model atom ordering.

    Stores bond and angle constraints as tensors ready for loss computation.
    Created once per model, then reused for all training batches.

    Attributes:
        bond_indices: (n_bonds, 2) column indices in model's atom ordering.
        bond_targets: (n_bonds,) ideal bond lengths in Angstroms.
        angle_indices: (n_angles, 3) column indices [A, B, C] where B is vertex.
        angle_targets: (n_angles,) ideal angles in radians.
        inter_bond_target: Target inter-residue bond length (1.60Å for RNA).
    """

    bond_indices: torch.Tensor
    bond_targets: torch.Tensor
    angle_indices: torch.Tensor
    angle_targets: torch.Tensor
    inter_bond_target: float = 1.60

    @classmethod
    def from_residue(
        cls,
        residue: "AtomGroup",
        model_atoms: list[int],
        inter_bond_target: float = 1.60,
        device: str | torch.device = "cpu",
    ) -> "GeometryConstraints":
        """
        Create constraints from a residue, remapped to model's atom ordering.

        Constraints involving atoms not in model_atoms are automatically filtered out.
        This handles structures without hydrogens gracefully.

        Args:
            residue: AtomGroup (e.g., Residue.A) with bonds, ideal coords.
            model_atoms: List of atom values in model's column order.
            inter_bond_target: Target inter-residue bond length.
            device: Target device for tensors.

        Returns:
            GeometryConstraints ready for loss computation.
        """
        # Build lookup tables for vectorized remapping
        # local_idx -> atom_value
        max_local = max(atom.local for atom in residue) + 1
        local_to_value = np.full(max_local, -1, dtype=np.int64)
        for atom in residue:
            local_to_value[atom.local] = int(atom)

        # atom_value -> model_column
        max_value = max(model_atoms) + 1
        value_to_col = np.full(max_value, -1, dtype=np.int64)
        for col, val in enumerate(model_atoms):
            value_to_col[val] = col

        # Remap bonds
        bond_indices, bond_targets = cls._remap_bonds(
            residue, local_to_value, value_to_col
        )

        # Remap angles
        angle_indices, angle_targets = cls._remap_angles(
            residue, local_to_value, value_to_col
        )

        return cls(
            bond_indices=torch.tensor(bond_indices, dtype=torch.long, device=device),
            bond_targets=torch.tensor(bond_targets, dtype=torch.float32, device=device),
            angle_indices=torch.tensor(angle_indices, dtype=torch.long, device=device),
            angle_targets=torch.tensor(angle_targets, dtype=torch.float32, device=device),
            inter_bond_target=inter_bond_target,
        )

    @staticmethod
    def _remap_bonds(
        residue: "AtomGroup",
        local_to_value: np.ndarray,
        value_to_col: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remap bond indices and filter missing atoms (vectorized)."""
        if residue.bonds is None or len(residue.bonds) == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float64)

        bonds = residue.bonds  # (n_bonds, 2) local indices
        lengths = residue.bond_lengths  # (n_bonds,)

        # local_idx -> atom_value -> model_col (vectorized)
        atom_values = local_to_value[bonds]  # (n_bonds, 2)

        # Handle out-of-bounds for value_to_col
        max_val = len(value_to_col)
        safe_values = np.clip(atom_values, 0, max_val - 1)
        model_cols = value_to_col[safe_values]  # (n_bonds, 2)

        # Mark out-of-range values as invalid
        model_cols = np.where(atom_values >= max_val, -1, model_cols)
        model_cols = np.where(atom_values < 0, -1, model_cols)

        # Filter: keep only rows where all atoms are present
        valid_mask = (model_cols >= 0).all(axis=1)

        return model_cols[valid_mask], lengths[valid_mask]

    @staticmethod
    def _remap_angles(
        residue: "AtomGroup",
        local_to_value: np.ndarray,
        value_to_col: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remap angle indices and filter missing atoms (vectorized)."""
        if residue.angles is None or len(residue.angles) == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.float64)

        angles = residue.angles  # (n_angles, 3) local indices
        values = residue.angle_values  # (n_angles,)

        # local_idx -> atom_value -> model_col (vectorized)
        atom_values = local_to_value[angles]  # (n_angles, 3)

        # Handle out-of-bounds for value_to_col
        max_val = len(value_to_col)
        safe_values = np.clip(atom_values, 0, max_val - 1)
        model_cols = value_to_col[safe_values]  # (n_angles, 3)

        # Mark out-of-range values as invalid
        model_cols = np.where(atom_values >= max_val, -1, model_cols)
        model_cols = np.where(atom_values < 0, -1, model_cols)

        # Filter: keep only rows where all atoms are present
        valid_mask = (model_cols >= 0).all(axis=1)

        return model_cols[valid_mask], values[valid_mask]

    def to(self, device: str | torch.device) -> "GeometryConstraints":
        """Move constraints to a device."""
        return GeometryConstraints(
            bond_indices=self.bond_indices.to(device),
            bond_targets=self.bond_targets.to(device),
            angle_indices=self.angle_indices.to(device),
            angle_targets=self.angle_targets.to(device),
            inter_bond_target=self.inter_bond_target,
        )

    @property
    def n_bonds(self) -> int:
        """Number of bond constraints."""
        return len(self.bond_indices)

    @property
    def n_angles(self) -> int:
        """Number of angle constraints."""
        return len(self.angle_indices)

    def bond_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss on bond lengths.

        Args:
            coords: (batch, n_atoms, 3) or (n_atoms, 3) coordinates.

        Returns:
            Scalar MSE loss on bond lengths vs ideal.
        """
        if self.n_bonds == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        # Gather atom positions for all bonds
        a1 = coords[:, self.bond_indices[:, 0]]  # (batch, n_bonds, 3)
        a2 = coords[:, self.bond_indices[:, 1]]  # (batch, n_bonds, 3)
        lengths = torch.norm(a2 - a1, dim=-1)    # (batch, n_bonds)

        return ((lengths - self.bond_targets) ** 2).mean()

    def angle_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss on valence angles.

        Args:
            coords: (batch, n_atoms, 3) or (n_atoms, 3) coordinates.

        Returns:
            Scalar MSE loss on angles vs ideal (in radians).
        """
        if self.n_angles == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        # Gather all three atoms for each angle
        a = coords[:, self.angle_indices[:, 0]]  # (batch, n_angles, 3)
        b = coords[:, self.angle_indices[:, 1]]  # (batch, n_angles, 3) - vertex
        c = coords[:, self.angle_indices[:, 2]]  # (batch, n_angles, 3)

        v1 = a - b  # B -> A
        v2 = c - b  # B -> C

        # Compute angles
        cos_angles = (v1 * v2).sum(-1) / (
            torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8
        )
        angles = torch.acos(cos_angles.clamp(-0.999, 0.999))

        return ((angles - self.angle_targets) ** 2).mean()

    def inter_bond_loss(self, transforms: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss on inter-residue bond length.

        The translation component of the transform encodes the O3'-P distance.

        Args:
            transforms: (batch, 6) SE(3) transforms [axis_angle, translation].

        Returns:
            Scalar MSE loss on inter-residue bond length.
        """
        lengths = torch.norm(transforms[:, 3:], dim=-1)
        return ((lengths - self.inter_bond_target) ** 2).mean()

    def total_loss(
        self,
        coords: torch.Tensor,
        transforms: torch.Tensor,
        weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """
        Compute combined geometry loss.

        Args:
            coords: (batch, n_atoms, 3) coordinates.
            transforms: (batch, 6) SE(3) transforms.
            weights: Optional weights for each component. Defaults to
                {"bond": 1.0, "angle": 1.0, "inter": 1.0}.

        Returns:
            Weighted sum of bond, angle, and inter-residue losses.
        """
        if weights is None:
            weights = {"bond": 1.0, "angle": 1.0, "inter": 1.0}

        loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        if weights.get("bond", 0) > 0:
            loss = loss + weights["bond"] * self.bond_loss(coords)

        if weights.get("angle", 0) > 0:
            loss = loss + weights["angle"] * self.angle_loss(coords)

        if weights.get("inter", 0) > 0:
            loss = loss + weights["inter"] * self.inter_bond_loss(transforms)

        return loss

    def __repr__(self) -> str:
        return (
            f"GeometryConstraints("
            f"n_bonds={self.n_bonds}, "
            f"n_angles={self.n_angles}, "
            f"inter_bond={self.inter_bond_target:.2f}Å)"
        )


__all__ = ["GeometryConstraints"]
