"""
Geometry constraints for generative model training.

This module provides the GeometryConstraints class which:
1. Extracts bond and angle constraints from ciffy biochemistry
2. Remaps constraints to match model atom ordering
3. Computes differentiable losses for training
4. Supports inter-residue angle constraints from MonomerLibrary

All operations are fully vectorized for efficient GPU computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ciffy.biochemistry.linking import NUCLEIC_ACID_LINK, BACKBONE_ATOM_VALUES
from ciffy.biochemistry._generated_linking import (
    LinkGeometry,
    NUCLEIC_ACID_LINK_GEOMETRY,
)

if TYPE_CHECKING:
    from ciffy.biochemistry.atom import AtomGroup


def _mmcif_to_python_name(name: str) -> str:
    """Convert mmCIF atom name to Python name (apostrophe -> p)."""
    return name.replace("'", "p")


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
        inter_bond_target: Target inter-residue bond length (from MonomerLibrary).
        inter_angle_indices: (n_inter, 3) atom column indices for inter-residue angles.
        inter_angle_offsets: (n_inter, 3) residue offset for each atom (0=current, 1=next).
        inter_angle_targets: (n_inter,) target angles in radians.
        inter_angle_weights: (n_inter,) weights = 1/esd^2 for weighted MSE.
    """

    bond_indices: torch.Tensor
    bond_targets: torch.Tensor
    angle_indices: torch.Tensor
    angle_targets: torch.Tensor
    inter_bond_target: float = NUCLEIC_ACID_LINK.bond_length

    # Inter-residue angle constraints (optional, None if not set)
    inter_angle_indices: torch.Tensor | None = None
    inter_angle_offsets: torch.Tensor | None = None
    inter_angle_targets: torch.Tensor | None = None
    inter_angle_weights: torch.Tensor | None = None

    @classmethod
    def from_residue(
        cls,
        residue: "AtomGroup",
        model_atoms: list[int],
        inter_bond_target: float = NUCLEIC_ACID_LINK.bond_length,
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

    @classmethod
    def from_residue_with_linking(
        cls,
        residue: "AtomGroup",
        model_atoms: list[int],
        link_geometry: LinkGeometry = NUCLEIC_ACID_LINK_GEOMETRY,
        device: str | torch.device = "cpu",
    ) -> "GeometryConstraints":
        """
        Create constraints including inter-residue angles from MonomerLibrary.

        This extends from_residue() by adding inter-residue angle constraints
        that span consecutive residues. These angles enforce proper backbone
        geometry at the linkage point (e.g., C3'-O3'-P for nucleic acids).

        Args:
            residue: AtomGroup (e.g., Residue.A) with bonds, ideal coords.
            model_atoms: List of atom values in model's column order.
            link_geometry: LinkGeometry from _generated_linking.py.
            device: Target device for tensors.

        Returns:
            GeometryConstraints with both intra- and inter-residue constraints.
        """
        # Get base constraints (intra-residue bonds and angles)
        base = cls.from_residue(
            residue,
            model_atoms,
            inter_bond_target=link_geometry.bond_length,
            device=device,
        )

        # Build atom value -> model column mapping
        max_value = max(model_atoms) + 1
        value_to_col = np.full(max_value, -1, dtype=np.int64)
        for col, val in enumerate(model_atoms):
            value_to_col[val] = col

        # Build inter-residue angle constraints
        inter_indices = []
        inter_offsets = []
        inter_targets = []
        inter_weights = []

        for angle_data in link_geometry.angles:
            # Convert mmCIF names to Python names, then to backbone values
            atom_names = [_mmcif_to_python_name(a) for a in angle_data.atoms]

            # Check if all atoms are in BACKBONE_ATOM_VALUES
            try:
                atom_values = [BACKBONE_ATOM_VALUES[name] for name in atom_names]
            except KeyError:
                # Skip angles with non-backbone atoms (e.g., H in peptide)
                continue

            # Check if all atoms are in model's atom ordering
            atom_cols = []
            valid = True
            for val in atom_values:
                if val >= max_value or value_to_col[val] < 0:
                    valid = False
                    break
                atom_cols.append(value_to_col[val])

            if not valid:
                continue

            # Convert comps (1=prev, 2=next) to offsets (0=current, 1=next)
            offsets = [c - 1 for c in angle_data.comps]

            inter_indices.append(atom_cols)
            inter_offsets.append(offsets)
            inter_targets.append(math.radians(angle_data.value))
            inter_weights.append(1.0 / (angle_data.esd ** 2))

        # Create tensors for inter-residue angles
        if inter_indices:
            base.inter_angle_indices = torch.tensor(
                inter_indices, dtype=torch.long, device=device
            )
            base.inter_angle_offsets = torch.tensor(
                inter_offsets, dtype=torch.long, device=device
            )
            base.inter_angle_targets = torch.tensor(
                inter_targets, dtype=torch.float32, device=device
            )
            base.inter_angle_weights = torch.tensor(
                inter_weights, dtype=torch.float32, device=device
            )

        return base

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
            inter_angle_indices=(
                self.inter_angle_indices.to(device)
                if self.inter_angle_indices is not None else None
            ),
            inter_angle_offsets=(
                self.inter_angle_offsets.to(device)
                if self.inter_angle_offsets is not None else None
            ),
            inter_angle_targets=(
                self.inter_angle_targets.to(device)
                if self.inter_angle_targets is not None else None
            ),
            inter_angle_weights=(
                self.inter_angle_weights.to(device)
                if self.inter_angle_weights is not None else None
            ),
        )

    @property
    def n_bonds(self) -> int:
        """Number of bond constraints."""
        return len(self.bond_indices)

    @property
    def n_angles(self) -> int:
        """Number of angle constraints."""
        return len(self.angle_indices)

    @property
    def n_inter_angles(self) -> int:
        """Number of inter-residue angle constraints."""
        if self.inter_angle_indices is None:
            return 0
        return len(self.inter_angle_indices)

    def bond_errors(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute absolute bond length errors.

        Args:
            coords: (batch, n_atoms, 3) or (n_atoms, 3) coordinates.

        Returns:
            (batch, n_bonds) absolute errors in Angstroms.
            Empty tensor if no bonds.
        """
        if self.n_bonds == 0:
            return torch.zeros(0, device=coords.device, dtype=coords.dtype)

        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        a1 = coords[:, self.bond_indices[:, 0]]
        a2 = coords[:, self.bond_indices[:, 1]]
        lengths = torch.norm(a2 - a1, dim=-1)

        return (lengths - self.bond_targets).abs()

    def bond_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss on bond lengths.

        Args:
            coords: (batch, n_atoms, 3) or (n_atoms, 3) coordinates.

        Returns:
            Scalar MSE loss on bond lengths vs ideal.
        """
        errors = self.bond_errors(coords)
        if errors.numel() == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        return (errors ** 2).mean()

    def angle_errors(self, coords: torch.Tensor, degrees: bool = True) -> torch.Tensor:
        """
        Compute absolute angle errors.

        Args:
            coords: (batch, n_atoms, 3) or (n_atoms, 3) coordinates.
            degrees: If True, return errors in degrees. Otherwise radians.

        Returns:
            (batch, n_angles) absolute errors.
            Empty tensor if no angles.
        """
        if self.n_angles == 0:
            return torch.zeros(0, device=coords.device, dtype=coords.dtype)

        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        a = coords[:, self.angle_indices[:, 0]]
        b = coords[:, self.angle_indices[:, 1]]  # vertex
        c = coords[:, self.angle_indices[:, 2]]

        v1 = a - b
        v2 = c - b

        cos_angles = (v1 * v2).sum(-1) / (
            torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8
        )
        angles = torch.acos(cos_angles.clamp(-0.999, 0.999))

        errors = (angles - self.angle_targets).abs()
        if degrees:
            errors = errors * (180.0 / math.pi)
        return errors

    def angle_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss on valence angles.

        Args:
            coords: (batch, n_atoms, 3) or (n_atoms, 3) coordinates.

        Returns:
            Scalar MSE loss on angles vs ideal (in radians).
        """
        errors = self.angle_errors(coords, degrees=False)
        if errors.numel() == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        return (errors ** 2).mean()

    def inter_bond_errors(self, transforms: torch.Tensor) -> torch.Tensor:
        """
        Compute absolute inter-residue bond length errors.

        The translation component of the transform encodes the O3'-P distance.

        Args:
            transforms: (batch, 6) SE(3) transforms [axis_angle, translation].

        Returns:
            (batch,) absolute errors in Angstroms.
        """
        lengths = torch.norm(transforms[:, 3:], dim=-1)
        return (lengths - self.inter_bond_target).abs()

    def inter_bond_loss(self, transforms: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss on inter-residue bond length.

        The translation component of the transform encodes the O3'-P distance.

        Args:
            transforms: (batch, 6) SE(3) transforms [axis_angle, translation].

        Returns:
            Scalar MSE loss on inter-residue bond length.
        """
        errors = self.inter_bond_errors(transforms)
        return (errors ** 2).mean()

    def inter_angle_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss on inter-residue angles.

        Inter-residue angles span consecutive residues (e.g., C3'-O3'-P in RNA).
        This loss enforces proper geometry at the linkage point.

        Args:
            coords: (batch, n_residues, n_atoms, 3) chain coordinates.
                Each residue has n_atoms in model's column ordering.

        Returns:
            Scalar weighted MSE loss on inter-residue angles.
        """
        if self.n_inter_angles == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        # coords: (B, R, N, 3) where B=batch, R=residues, N=atoms per residue
        B, R, N, _ = coords.shape

        if R < 2:
            # Need at least 2 residues for inter-residue angles
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        # For each angle type, gather atoms from consecutive residue pairs
        # inter_angle_indices: (A, 3) atom columns
        # inter_angle_offsets: (A, 3) residue offset (0=current, 1=next)
        A = self.n_inter_angles
        indices = self.inter_angle_indices  # (A, 3)
        offsets = self.inter_angle_offsets  # (A, 3)

        # For residue pairs (0,1), (1,2), ..., (R-2, R-1)
        # Total of R-1 pairs, each with A angle types
        n_pairs = R - 1

        # Build gather indices for all pairs and all angle types
        # Shape: (B, n_pairs, A, 3, 3) -> positions for each atom
        all_angles = []

        for pair_idx in range(n_pairs):
            # Current residue is pair_idx, next is pair_idx + 1
            # Gather three atoms for each angle type
            positions = []
            for atom_idx in range(3):
                # Residue index for this atom
                res_idx = pair_idx + offsets[:, atom_idx]  # (A,)
                col_idx = indices[:, atom_idx]  # (A,)

                # Gather: coords[:, res_idx, col_idx, :]
                # Need advanced indexing for batch dimension
                pos = coords[:, res_idx, col_idx, :]  # (B, A, 3)
                positions.append(pos)

            # positions: list of 3 tensors, each (B, A, 3)
            a, b, c = positions  # a, b (vertex), c

            # Compute angle at vertex b
            v1 = a - b  # (B, A, 3)
            v2 = c - b  # (B, A, 3)

            cos_angles = (v1 * v2).sum(-1) / (
                torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8
            )
            angles = torch.acos(cos_angles.clamp(-0.999, 0.999))  # (B, A)
            all_angles.append(angles)

        # Stack all pairs: (n_pairs, B, A) -> (B, n_pairs, A)
        all_angles = torch.stack(all_angles, dim=1)  # (B, n_pairs, A)

        # Compute weighted MSE
        # targets: (A,) broadcast to (B, n_pairs, A)
        # weights: (A,) broadcast to (B, n_pairs, A)
        diff = all_angles - self.inter_angle_targets  # (B, n_pairs, A)
        weighted_sq_error = self.inter_angle_weights * (diff ** 2)

        return weighted_sq_error.mean()

    def total_loss(
        self,
        coords: torch.Tensor,
        transforms: torch.Tensor,
        chain_coords: torch.Tensor | None = None,
        weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """
        Compute combined geometry loss.

        Args:
            coords: (batch, n_atoms, 3) per-residue coordinates.
            transforms: (batch, 6) SE(3) inter-residue transforms.
            chain_coords: Optional (batch, n_residues, n_atoms, 3) chain coordinates
                for inter-residue angle loss. If None, inter_angle loss is skipped.
            weights: Optional weights for each component. Defaults to
                {"bond": 1.0, "angle": 1.0, "inter_bond": 1.0, "inter_angle": 1.0}.

        Returns:
            Weighted sum of bond, angle, and inter-residue losses.
        """
        if weights is None:
            weights = {"bond": 1.0, "angle": 1.0, "inter_bond": 1.0, "inter_angle": 1.0}

        # Handle legacy "inter" key for backwards compatibility
        if "inter" in weights and "inter_bond" not in weights:
            weights["inter_bond"] = weights["inter"]

        loss = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

        if weights.get("bond", 0) > 0:
            loss = loss + weights["bond"] * self.bond_loss(coords)

        if weights.get("angle", 0) > 0:
            loss = loss + weights["angle"] * self.angle_loss(coords)

        if weights.get("inter_bond", 0) > 0:
            loss = loss + weights["inter_bond"] * self.inter_bond_loss(transforms)

        if chain_coords is not None and weights.get("inter_angle", 0) > 0:
            loss = loss + weights["inter_angle"] * self.inter_angle_loss(chain_coords)

        return loss

    def compute_error_metrics(
        self,
        coords: torch.Tensor,
        transforms: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute geometry error metrics for logging/validation.

        Args:
            coords: (batch, n_atoms, 3) per-residue coordinates.
            transforms: (batch, 6) SE(3) inter-residue transforms.

        Returns:
            Dict with keys: bond_mae, bond_max, angle_mae, angle_max,
            inter_bond_mae, inter_bond_max. Values in Angstroms (bonds)
            or degrees (angles).
        """
        metrics = {}

        # Bond errors
        bond_errs = self.bond_errors(coords)
        if bond_errs.numel() > 0:
            metrics["bond_mae"] = bond_errs.mean().item()
            metrics["bond_max"] = bond_errs.max().item()

        # Angle errors (already in degrees)
        angle_errs = self.angle_errors(coords, degrees=True)
        if angle_errs.numel() > 0:
            metrics["angle_mae"] = angle_errs.mean().item()
            metrics["angle_max"] = angle_errs.max().item()

        # Inter-residue bond errors
        inter_errs = self.inter_bond_errors(transforms)
        if inter_errs.numel() > 0:
            metrics["inter_bond_mae"] = inter_errs.mean().item()
            metrics["inter_bond_max"] = inter_errs.max().item()

        return metrics

    def __repr__(self) -> str:
        inter_angles = f", n_inter_angles={self.n_inter_angles}" if self.n_inter_angles > 0 else ""
        return (
            f"GeometryConstraints("
            f"n_bonds={self.n_bonds}, "
            f"n_angles={self.n_angles}, "
            f"inter_bond={self.inter_bond_target:.2f}Å{inter_angles})"
        )


__all__ = ["GeometryConstraints"]
