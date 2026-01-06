"""
Coordinate autoregressive model.

This module provides CoordinateARModel, which directly predicts atom coordinates
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
    from ...polymer import Polymer


@dataclass
class CoordinateARModelConfig:
    """Configuration for CoordinateARModel model.

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


class CoordinateARModel(nn.Module if TORCH_AVAILABLE else object):
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
        >>> model = CoordinateARModel(
        ...     residue_atoms={Residue.A: 22, Residue.C: 20, Residue.G: 23, Residue.U: 20},
        ...     d_model=256,
        ... )
        >>> # Generation assembles the chain incrementally
        >>> coords, transforms = model.generate(sequence)
    """

    def __init__(
        self,
        residue_atoms: Dict["Residue", int],
        config: Optional[CoordinateARModelConfig] = None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if config is None:
            config = CoordinateARModelConfig(**kwargs)
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

    def compute_loss(self, polymer: "Polymer") -> "torch.Tensor":
        """
        Compute training loss from a Polymer using ciffy.rmsd.

        Uses RMSD for local coordinates (per-residue) and MSE for SE(3) transforms.

        Args:
            polymer: Input polymer structure.

        Returns:
            Loss tensor.
        """
        import ciffy
        from ...biochemistry import Scale
        from ...biochemistry.linking import GLYCOSIDIC_FRAME
        from ...geometry.transforms import (
            rotation_to_axis_angle,
            compute_relative_transform,
        )

        device = next(self.parameters()).device

        # Convert to torch if needed
        if polymer.backend != "torch":
            polymer = polymer.torch()
        polymer = polymer.to(device)

        n_residues = polymer.size(Scale.RESIDUE)
        sequence = polymer.sequence  # (R,)
        counts = polymer.counts(Scale.RESIDUE)  # (R,)

        # Align to local frames
        aligned, Rs = polymer.align()  # Rs: (R, 3, 3)

        # Get frame origins (C1' positions for glycosidic frame)
        origin_atom = GLYCOSIDIC_FRAME.origin
        origins = polymer.gather([origin_atom])[:, 0, :]  # (R, 3)

        # Convert rotation matrices to axis-angle (loop since function handles single matrices)
        global_orientations = torch.stack([
            rotation_to_axis_angle(Rs[i]) for i in range(n_residues)
        ])

        # Compute ground truth transforms using geometry utility
        gt_transforms = torch.zeros(n_residues, 6, device=device)
        for i in range(1, n_residues):
            gt_transforms[i] = compute_relative_transform(
                origins[i - 1], Rs[i - 1], origins[i], Rs[i]
            )

        # Forward pass - add batch dimension
        result = self.forward(
            sequence.unsqueeze(0),
            origins.unsqueeze(0),
            global_orientations.unsqueeze(0),
        )
        outputs = result["outputs"]

        # Compute loss per residue
        total_rmsd = torch.tensor(0.0, device=device)
        total_transform_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        offset = 0
        for i in range(n_residues):
            res_val = sequence[i].item()
            n_atoms = counts[i].item()

            if res_val not in outputs:
                offset += n_atoms
                continue

            pred = outputs[res_val]  # (1, L, output_dim)
            n_res_atoms = self.residue_atoms.get(res_val, n_atoms)

            # Get prediction for this residue
            pred_i = pred[0, i]
            pred_coords = pred_i[:n_res_atoms * 3].reshape(n_res_atoms, 3)
            pred_transform = pred_i[n_res_atoms * 3:]

            # Ground truth
            gt_coords = aligned.coordinates[offset:offset + n_atoms]
            gt_transform = gt_transforms[i]

            # RMSD for coordinates
            rmsd_loss = ciffy.rmsd(pred_coords, gt_coords, eps=1e-8)
            total_rmsd = total_rmsd + rmsd_loss

            # MSE for transforms (6D SE(3) parameters)
            transform_loss = F.mse_loss(pred_transform, gt_transform)
            total_transform_loss = total_transform_loss + transform_loss

            n_valid += 1
            offset += n_atoms

        if n_valid > 0:
            return (total_rmsd + total_transform_loss) / n_valid
        return torch.tensor(0.0, device=device)

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
        from ...geometry.transforms import (
            rotation_to_axis_angle,
            apply_relative_transform,
        )

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
                        new_cent, new_R = apply_relative_transform(
                            current_centroid[b],
                            current_R[b],
                            pred_transform[b],
                        )
                        global_centroids[b, i] = new_cent
                        global_orientations[b, i] = rotation_to_axis_angle(new_R)
                        current_centroid[b] = new_cent
                        current_R[b] = new_R

        return local_coords, transforms

    def sample(
        self,
        template: "Polymer",
        n_samples: int = 1,
        temperature: float = 0.0,
        **kwargs,
    ) -> list["Polymer"]:
        """
        Generate polymer conformations from a template.

        Implements the PolymerGenerativeModel protocol.

        Args:
            template: Template Polymer with sequence and atom topology.
                Must have atoms that match what the model was trained on.
            n_samples: Number of samples to generate.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            List of Polymers with generated coordinates.
        """
        from ...biochemistry import Scale, Residue
        from ...polymer import Polymer

        device = next(self.parameters()).device

        # Get sequence from template
        if template.backend != "torch":
            template = template.torch()
        template = template.to(device)

        sequence = template.sequence  # (n_residues,)
        n_residues = len(sequence)

        # Expand sequence for n_samples
        seq_batch = sequence.unsqueeze(0).expand(n_samples, -1)

        # Generate local coordinates and transforms
        local_coords, transforms = self.generate(seq_batch, temperature=temperature)
        # local_coords: (n_samples, n_residues, max_atoms, 3)
        # transforms: (n_samples, n_residues, 6)

        # Get per-residue info from template
        counts = template.counts(Scale.RESIDUE).cpu().numpy()
        template_np = template.numpy()

        results = []
        for s in range(n_samples):
            # Build polymer residue by residue using extend_new
            poly = Polymer()
            offset = 0

            for i in range(n_residues):
                n_atoms = counts[i]
                res_val = sequence[i].item()
                res = Residue(res_val)

                # Get atom group from template residue
                template_res = template_np.residue(i)
                atom_group = res.subset(set(template_res.atoms.tolist()))

                # Get generated coords for this residue
                coords_i = local_coords[s, i, :n_atoms].cpu().numpy()

                if i == 0:
                    # First residue: absolute coordinates
                    poly = poly.append(atom_group, coords_i, residue=res)
                else:
                    # Subsequent residues: relative transform
                    from ...geometry import LocalCoordinates
                    transform_i = transforms[s, i].cpu().numpy()
                    poly = poly.append(atom_group, LocalCoordinates(coords_i, transform_i), residue=res)

                offset += n_atoms

            results.append(poly)

        return results

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
    def load(cls, path: str, device: str = "cpu") -> "CoordinateARModel":
        """Load model from disk."""
        import json
        from pathlib import Path

        path = Path(path)

        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = CoordinateARModelConfig(**config_dict)

        with open(path / "residue_atoms.json") as f:
            residue_atoms = {int(k): v for k, v in json.load(f).items()}

        model = cls(residue_atoms, config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.to(device)

        return model
