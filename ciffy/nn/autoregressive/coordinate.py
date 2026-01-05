"""
Coordinate autoregressive model.

This module provides CoordinateAR, which directly predicts atom coordinates
and SE(3) transforms conditioned on the global assembled structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, TYPE_CHECKING

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from ..layers import CausalTransformer

if TYPE_CHECKING:
    from ...biochemistry import Residue


@dataclass
class CoordinateARConfig:
    """Configuration for CoordinateAR model.

    Args:
        d_model: Transformer hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length.
        num_residue_types: Number of distinct residue types (default 32).
    """
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048
    num_residue_types: int = 32


class CoordinateAR(nn.Module if TORCH_AVAILABLE else object):
    """
    Autoregressive model conditioned on global assembled structure.

    This model receives the global positions and orientations of all previously
    generated residues, enabling learning of long-range interactions.

    Input at position i:
    - Residue types for positions 0..i
    - Global centroids for positions 0..i-1 (assembled chain)
    - Global orientations for positions 0..i-1 (as axis-angle)

    Output at position i:
    - Local coordinates (in glycosidic frame)
    - SE(3) transform to position this residue

    Args:
        residue_atoms: Dict mapping Residue to number of atoms.
        config: Model configuration.

    Example:
        >>> model = CoordinateAR(
        ...     residue_atoms={Residue.A: 22, Residue.C: 20, Residue.G: 23, Residue.U: 20},
        ...     d_model=256,
        ... )
        >>> # Generation assembles the chain incrementally
        >>> coords, transforms = model.generate(sequence)
    """

    def __init__(
        self,
        residue_atoms: Dict["Residue", int],
        config: Optional[CoordinateARConfig] = None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if config is None:
            config = CoordinateARConfig(**kwargs)
        self.config = config
        self.d_model = config.d_model

        # Store residue info
        self.residue_atoms = {
            (r.value if hasattr(r, 'value') else r): n
            for r, n in residue_atoms.items()
        }
        self.max_atoms = max(self.residue_atoms.values())

        # Residue type embedding
        self.residue_embed = nn.Embedding(config.num_residue_types, config.d_model)

        # Global structure encoding
        # Input: centroid (3) + orientation as axis-angle (3) = 6
        self.global_proj = nn.Linear(6, config.d_model)

        # Learned start token (for first position with no previous structure)
        self.start_token = nn.Parameter(torch.randn(config.d_model))

        # Causal transformer
        self.transformer = CausalTransformer(
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Per-residue output heads
        self.output_heads = nn.ModuleDict()
        for res_val, n_atoms in self.residue_atoms.items():
            output_dim = n_atoms * 3 + 6  # coords + transform
            self.output_heads[str(res_val)] = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, output_dim),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.residue_embed.weight, std=0.02)
        nn.init.normal_(self.global_proj.weight, std=0.02)
        nn.init.zeros_(self.global_proj.bias)

        for head in self.output_heads.values():
            for layer in head:
                if hasattr(layer, 'weight'):
                    nn.init.normal_(layer.weight, std=0.02)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _rotation_to_axis_angle(self, R: "torch.Tensor") -> "torch.Tensor":
        """Convert rotation matrix to axis-angle representation."""
        # Trace gives angle
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

        # Axis from skew-symmetric part
        axis = torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ], dim=-1)

        # Normalize and scale by angle
        norm = axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis_angle = axis / norm * angle.unsqueeze(-1)

        return axis_angle

    def _axis_angle_to_rotation(self, axis_angle: "torch.Tensor") -> "torch.Tensor":
        """Convert axis-angle to rotation matrix."""
        angle = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis = axis_angle / angle

        # Rodrigues formula
        K = torch.zeros(*axis_angle.shape[:-1], 3, 3, device=axis_angle.device)
        K[..., 0, 1] = -axis[..., 2]
        K[..., 0, 2] = axis[..., 1]
        K[..., 1, 0] = axis[..., 2]
        K[..., 1, 2] = -axis[..., 0]
        K[..., 2, 0] = -axis[..., 1]
        K[..., 2, 1] = axis[..., 0]

        I = torch.eye(3, device=axis_angle.device).expand(*axis_angle.shape[:-1], 3, 3)
        angle = angle.unsqueeze(-1)

        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    def forward(
        self,
        sequence: "torch.Tensor",
        global_centroids: "torch.Tensor",
        global_orientations: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Forward pass with teacher forcing.

        Args:
            sequence: Residue type indices (batch, seq_len).
            global_centroids: Global centroid positions (batch, seq_len, 3).
            global_orientations: Global orientations as axis-angle (batch, seq_len, 3).
            padding_mask: Optional mask (batch, seq_len) where True = padded.

        Returns:
            Dict with outputs per residue type.
        """
        B, L = sequence.shape

        # Embed residue types
        residue_emb = self.residue_embed(sequence)  # (B, L, d_model)

        # Encode global structure and shift right (causal)
        global_feat = torch.cat([global_centroids, global_orientations], dim=-1)  # (B, L, 6)
        global_emb = self.global_proj(global_feat)  # (B, L, d_model)

        # Shift right: [start, g0, g1, ..., g_{L-2}]
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        shifted_global = torch.cat([start, global_emb[:, :-1]], dim=1)

        # Combine embeddings
        h = residue_emb + shifted_global

        # Transformer
        hidden = self.transformer(h, padding_mask=padding_mask)

        # Apply per-residue output heads
        outputs = {}
        for res_val_str, head in self.output_heads.items():
            res_val = int(res_val_str)
            mask = (sequence == res_val)
            if mask.any():
                out = head(hidden)
                outputs[res_val] = out

        return {"hidden": hidden, "outputs": outputs}

    def compute_loss(
        self,
        sequence: "torch.Tensor",
        local_coords: "torch.Tensor",
        transforms: "torch.Tensor",
        global_centroids: "torch.Tensor",
        global_orientations: "torch.Tensor",
        n_atoms: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Compute MSE loss on local coordinates and transforms.

        Args:
            sequence: Residue type indices (batch, seq_len).
            local_coords: Ground truth local coordinates (batch, seq_len, max_atoms, 3).
            transforms: Ground truth transforms (batch, seq_len, 6).
            global_centroids: Global centroid positions (batch, seq_len, 3).
            global_orientations: Global orientations as axis-angle (batch, seq_len, 3).
            n_atoms: Actual atom counts per position (batch, seq_len).
            padding_mask: Optional mask (batch, seq_len) where True = padded.

        Returns:
            Loss tensor.
        """
        B, L = sequence.shape
        device = sequence.device

        result = self.forward(sequence, global_centroids, global_orientations, padding_mask)
        outputs = result["outputs"]

        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        for res_val, pred in outputs.items():
            n_res_atoms = self.residue_atoms[res_val]
            mask = (sequence == res_val)

            if padding_mask is not None:
                mask = mask & ~padding_mask

            if not mask.any():
                continue

            # Extract predictions
            pred_flat = pred[mask]  # (N, n_atoms*3 + 6)

            # Extract ground truth
            gt_coords = local_coords[mask][:, :n_res_atoms]  # (N, n_atoms, 3)
            gt_transforms = transforms[mask]  # (N, 6)
            N = gt_coords.shape[0]
            gt_flat = torch.cat([
                gt_coords.reshape(N, n_res_atoms * 3),
                gt_transforms
            ], dim=-1)

            # MSE loss
            loss = F.mse_loss(pred_flat, gt_flat, reduction='sum')
            total_loss = total_loss + loss
            n_valid += mask.sum().item() * (n_res_atoms * 3 + 6)

        if n_valid > 0:
            return total_loss / n_valid
        return total_loss

    def _apply_transform(
        self,
        prev_centroid: "torch.Tensor",
        prev_R: "torch.Tensor",
        transform: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Apply SE(3) transform to get new global position."""
        rot_aa = transform[:3]
        trans = transform[3:]

        # Relative rotation
        rel_R = self._axis_angle_to_rotation(rot_aa.unsqueeze(0)).squeeze(0)

        # New orientation: prev_R @ rel_R
        new_R = prev_R @ rel_R

        # New position: prev_centroid + prev_R @ trans
        new_centroid = prev_centroid + (prev_R @ trans)

        return new_centroid, new_R

    @torch.no_grad()
    def generate(
        self,
        sequence: "torch.Tensor",
        temperature: float = 0.0,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Generate coordinates and transforms autoregressively.

        Assembles the global structure incrementally and conditions
        each prediction on the full assembled chain so far.

        Args:
            sequence: Residue type indices (batch, seq_len) or (seq_len,).
            temperature: Noise scale (0 = deterministic).

        Returns:
            local_coords: (batch, seq_len, max_atoms, 3)
            transforms: (batch, seq_len, 6)
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        B, L = sequence.shape
        device = sequence.device

        # Initialize outputs
        local_coords = torch.zeros(B, L, self.max_atoms, 3, device=device)
        transforms = torch.zeros(B, L, 6, device=device)

        # Initialize global state
        global_centroids = torch.zeros(B, L, 3, device=device)
        global_orientations = torch.zeros(B, L, 3, device=device)

        # Track current global frame for assembly
        current_centroid = torch.zeros(B, 3, device=device)
        current_R = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        # Generate autoregressively
        for i in range(L):
            # Forward pass up to position i
            result = self.forward(
                sequence[:, :i+1],
                global_centroids[:, :i+1],
                global_orientations[:, :i+1],
            )

            # Get prediction for position i
            res_val = sequence[0, i].item()
            if res_val in result["outputs"]:
                pred = result["outputs"][res_val][:, i]  # (B, output_dim)
                n_res_atoms = self.residue_atoms[res_val]

                # Add noise if temperature > 0
                if temperature > 0:
                    pred = pred + temperature * torch.randn_like(pred)

                # Split into coords and transform
                pred_coords = pred[:, :n_res_atoms * 3].reshape(B, n_res_atoms, 3)
                pred_transform = pred[:, n_res_atoms * 3:]

                local_coords[:, i, :n_res_atoms] = pred_coords
                transforms[:, i] = pred_transform

                # Assemble: compute global position for this residue
                if i == 0:
                    # First residue at origin
                    centroid = pred_coords.mean(dim=1)  # (B, 3)
                    global_centroids[:, i] = centroid
                    global_orientations[:, i] = 0  # Identity
                    current_centroid = centroid
                else:
                    # Apply transform to get global position
                    for b in range(B):
                        new_cent, new_R = self._apply_transform(
                            current_centroid[b],
                            current_R[b],
                            pred_transform[b],
                        )
                        global_centroids[b, i] = new_cent
                        global_orientations[b, i] = self._rotation_to_axis_angle(new_R.unsqueeze(0)).squeeze(0)
                        current_centroid[b] = new_cent
                        current_R[b] = new_R

        return local_coords, transforms

    def save(self, path: str) -> None:
        """Save model to disk."""
        import json
        from pathlib import Path
        from dataclasses import asdict

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        with open(path / "residue_atoms.json", "w") as f:
            json.dump(self.residue_atoms, f, indent=2)

        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CoordinateAR":
        """Load model from disk."""
        import json
        from pathlib import Path

        path = Path(path)

        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = CoordinateARConfig(**config_dict)

        with open(path / "residue_atoms.json") as f:
            residue_atoms = {int(k): v for k, v in json.load(f).items()}

        model = cls(residue_atoms, config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.to(device)

        return model
