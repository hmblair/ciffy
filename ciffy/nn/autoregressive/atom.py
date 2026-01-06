"""
All-atom conditioned autoregressive model for structure generation.

This module provides AtomARModel, which predicts residue coordinates one residue
at a time, conditioned on all atoms from previously placed residues.

Key design:
- **Atom-level conditioning**: Full atomic detail from residues 0..i-1
- **Residue-level prediction**: Output all atoms for residue i at once
- This avoids arbitrary atom ordering within residues while maintaining
  rich spatial context from the growing chain.

Architecture:
1. Encode all previous atoms (positions + types) with a bidirectional transformer
2. Pool to get a context representation
3. Decode current residue's atoms conditioned on this context
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from ..layers import CausalTransformer, RMSNorm, SwiGLU, Transformer
from ..blocks import RBFDistanceEncoder, build_mlp_stack, ResidualBlock
from ...biochemistry import NUM_ATOMS, NUM_RESIDUES, NUM_ELEMENTS

if TYPE_CHECKING:
    pass


@dataclass
class AtomARModelConfig:
    """Configuration for AtomARModel model.

    Args:
        d_model: Transformer hidden dimension.
        num_encoder_layers: Number of transformer layers for encoding context.
        num_decoder_layers: Number of transformer layers for residue decoding.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        max_atoms: Maximum number of atoms in context.
        max_residues: Maximum number of residues in a chain.

        # Embedding dimensions
        atom_embed_dim: Dimension for atom type embeddings.
        residue_embed_dim: Dimension for residue type embeddings.
        element_embed_dim: Dimension for element type embeddings.

        # Spatial encoding
        n_rbf: Number of RBF basis functions for distance encoding.
        rbf_cutoff: Cutoff distance for RBF encoding.
        coord_freq: Number of frequencies for coordinate sinusoidal encoding.

        # Output
        max_atoms_per_residue: Maximum atoms in any residue type.
        output_hidden_dims: Hidden dimensions for coordinate output head.
    """
    d_model: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_atoms: int = 4096
    max_residues: int = 512

    # Embedding dimensions
    atom_embed_dim: int = 64
    residue_embed_dim: int = 64
    element_embed_dim: int = 32

    # Spatial encoding
    n_rbf: int = 32
    rbf_cutoff: float = 20.0
    coord_freq: int = 32

    # Output
    max_atoms_per_residue: int = 32
    output_hidden_dims: tuple[int, ...] = (256, 256)


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal encoding for 3D coordinates."""

    def __init__(self, d_out: int, n_freq: int = 32, max_freq: float = 10.0):
        super().__init__()
        self.n_freq = n_freq
        freqs = torch.linspace(0, max_freq, n_freq)
        self.register_buffer("freqs", freqs)
        # 3 dims * 2 (sin+cos) * n_freq
        self.proj = nn.Linear(3 * 2 * n_freq, d_out)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode 3D coordinates to features."""
        # coords: (..., 3)
        x = coords.unsqueeze(-1) * self.freqs  # (..., 3, n_freq)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (..., 3, 2*n_freq)
        enc = enc.flatten(-2)  # (..., 6*n_freq)
        return self.proj(enc)


class AtomContextEncoder(nn.Module):
    """
    Encode all atoms from previous residues into a context representation.

    Uses bidirectional self-attention over atoms, then pools to get
    a fixed-size context vector.
    """

    def __init__(self, config: AtomARModelConfig):
        super().__init__()
        self.d_model = config.d_model

        # Type embeddings
        self.atom_embed = nn.Embedding(NUM_ATOMS, config.atom_embed_dim)
        self.residue_embed = nn.Embedding(NUM_RESIDUES, config.residue_embed_dim)
        self.element_embed = nn.Embedding(NUM_ELEMENTS, config.element_embed_dim)

        embed_dim = config.atom_embed_dim + config.residue_embed_dim + config.element_embed_dim

        # Coordinate encoding
        self.coord_encoder = SinusoidalPositionEncoding(
            d_out=config.d_model,
            n_freq=config.coord_freq,
        )

        # Combine type embeddings + coordinate encoding
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim + config.d_model, config.d_model),
            RMSNorm(config.d_model),
        )

        # Bidirectional transformer for atom context
        self.transformer = Transformer(
            d_model=config.d_model,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_atoms,
        )

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.d_model))

    def forward(
        self,
        atoms: torch.Tensor,
        residues: torch.Tensor,
        elements: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode atom context.

        Args:
            atoms: (batch, n_atoms) atom type indices.
            residues: (batch, n_atoms) residue type indices.
            elements: (batch, n_atoms) element type indices.
            coords: (batch, n_atoms, 3) atom coordinates.
            mask: (batch, n_atoms) boolean, True = valid atom.

        Returns:
            (batch, d_model) pooled context representation.
        """
        B, N = atoms.shape
        device = atoms.device

        if N == 0:
            # No context atoms - return zeros
            return torch.zeros(B, self.d_model, device=device)

        # Check if any batch has all atoms masked
        valid_per_batch = mask.sum(dim=1)  # (B,)
        if (valid_per_batch == 0).any():
            # Some batches have no valid atoms - return zeros for those
            result = torch.zeros(B, self.d_model, device=device)
            valid_batches = valid_per_batch > 0
            if valid_batches.any():
                # Process only valid batches
                valid_result = self._forward_valid(
                    atoms[valid_batches], residues[valid_batches],
                    elements[valid_batches], coords[valid_batches],
                    mask[valid_batches]
                )
                result[valid_batches] = valid_result
            return result

        return self._forward_valid(atoms, residues, elements, coords, mask)

    def _forward_valid(
        self,
        atoms: torch.Tensor,
        residues: torch.Tensor,
        elements: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for batches with at least one valid atom."""
        B, N = atoms.shape
        device = atoms.device

        # Embed types
        atom_emb = self.atom_embed(atoms)
        res_emb = self.residue_embed(residues)
        elem_emb = self.element_embed(elements)
        type_emb = torch.cat([atom_emb, res_emb, elem_emb], dim=-1)

        # Encode coordinates
        coord_emb = self.coord_encoder(coords)

        # Combine and project
        h = self.input_proj(torch.cat([type_emb, coord_emb], dim=-1))

        # Transformer (with padding mask)
        padding_mask = ~mask
        h = self.transformer(h, mask=padding_mask)

        # Pool via attention with learnable query
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, d_model)

        # Simple attention pooling
        scores = torch.bmm(query, h.transpose(1, 2)) / (self.d_model ** 0.5)  # (B, 1, N)
        scores = scores.masked_fill(padding_mask.unsqueeze(1), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, h).squeeze(1)  # (B, d_model)

        return pooled


class ResidueDecoder(nn.Module):
    """
    Decode atom coordinates for a single residue given context.

    Takes:
    - Context vector from previous atoms
    - Residue type being predicted
    - Atom types within this residue

    Outputs coordinates for all atoms in the residue.
    """

    def __init__(self, config: AtomARModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_atoms = config.max_atoms_per_residue

        # Residue type embedding (what residue we're generating)
        self.residue_embed = nn.Embedding(NUM_RESIDUES, config.d_model)

        # Per-atom embeddings for atoms within the residue
        self.atom_embed = nn.Embedding(NUM_ATOMS, config.atom_embed_dim)
        self.element_embed = nn.Embedding(NUM_ELEMENTS, config.element_embed_dim)

        atom_feat_dim = config.atom_embed_dim + config.element_embed_dim

        # Combine context + residue + atom features
        self.input_proj = nn.Sequential(
            nn.Linear(config.d_model * 2 + atom_feat_dim, config.d_model),
            RMSNorm(config.d_model),
        )

        # Self-attention over atoms within residue (they can see each other)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.num_decoder_layers)
        ])

        # Output head per atom
        self.coord_head = build_mlp_stack(
            config.d_model,
            list(config.output_hidden_dims),
            output_dim=3,
            dropout=config.dropout,
            zero_init_final=True,
        )

    def forward(
        self,
        context: torch.Tensor,
        residue_type: torch.Tensor,
        atom_types: torch.Tensor,
        element_types: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode coordinates for a residue.

        Args:
            context: (batch, d_model) context from previous atoms.
            residue_type: (batch,) residue type index.
            atom_types: (batch, max_atoms) atom type indices for this residue.
            element_types: (batch, max_atoms) element type indices.
            atom_mask: (batch, max_atoms) boolean, True = valid atom.

        Returns:
            (batch, max_atoms, 3) predicted coordinates.
        """
        B = context.shape[0]
        device = context.device
        M = atom_types.shape[1]  # max atoms in this batch

        # Embed residue type
        res_emb = self.residue_embed(residue_type)  # (B, d_model)

        # Embed atom types
        atom_emb = self.atom_embed(atom_types)  # (B, M, atom_dim)
        elem_emb = self.element_embed(element_types)  # (B, M, elem_dim)
        atom_feat = torch.cat([atom_emb, elem_emb], dim=-1)  # (B, M, atom_feat_dim)

        # Expand context and residue embedding to all atoms
        context_exp = context.unsqueeze(1).expand(-1, M, -1)  # (B, M, d_model)
        res_emb_exp = res_emb.unsqueeze(1).expand(-1, M, -1)  # (B, M, d_model)

        # Combine features
        h = self.input_proj(torch.cat([context_exp, res_emb_exp, atom_feat], dim=-1))

        # Self-attention over atoms (bidirectional - atoms can see each other)
        padding_mask = ~atom_mask
        for layer in self.self_attn_layers:
            h = layer(h, src_key_padding_mask=padding_mask)

        # Predict coordinates
        coords = self.coord_head(h)  # (B, M, 3)

        return coords


class AtomARModel(nn.Module if TORCH_AVAILABLE else object):
    """
    All-atom conditioned autoregressive model.

    Generates polymer structures residue-by-residue, where each residue's
    atom coordinates are predicted conditioned on all atoms from previous
    residues.

    This design:
    - Provides rich atomic context (not just residue centroids)
    - Avoids arbitrary atom ordering within residues
    - Enables learning precise local geometry from atomic neighbors

    Example:
        >>> model = AtomARModel(AtomARModelConfig(d_model=256))
        >>>
        >>> # Training: provide all atoms, get predictions for each residue
        >>> loss = model.compute_loss(polymer_data)
        >>>
        >>> # Generation: build chain residue by residue
        >>> coords = model.generate(sequence, atom_types_per_residue)
    """

    def __init__(self, config: Optional[AtomARModelConfig] = None, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if config is None:
            config = AtomARModelConfig(**kwargs)
        self.config = config

        # Context encoder (all previous atoms → context vector)
        self.context_encoder = AtomContextEncoder(config)

        # Residue decoder (context → atom coordinates)
        self.residue_decoder = ResidueDecoder(config)

        # Causal transformer over residue-level representations
        # This captures sequence-level patterns
        self.residue_embed = nn.Embedding(NUM_RESIDUES, config.d_model)
        self.residue_transformer = CausalTransformer(
            d_model=config.d_model,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_residues,
        )

        # Combine context encoder output with residue transformer
        self.context_combine = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            RMSNorm(config.d_model),
        )

    def _get_context_for_residue(
        self,
        all_atoms: torch.Tensor,
        all_residues: torch.Tensor,
        all_elements: torch.Tensor,
        all_coords: torch.Tensor,
        all_mask: torch.Tensor,
        residue_boundaries: torch.Tensor,
        residue_idx: int,
    ) -> torch.Tensor:
        """Get context from atoms of residues 0..residue_idx-1."""
        B = all_atoms.shape[0]
        device = all_atoms.device

        if residue_idx == 0:
            # No previous residues
            return torch.zeros(B, self.config.d_model, device=device)

        # Get atom indices for residues 0..residue_idx-1
        end_idx = residue_boundaries[:, residue_idx]  # (B,) end of previous residue

        # Create mask for context atoms
        max_atoms = all_atoms.shape[1]
        atom_indices = torch.arange(max_atoms, device=device).unsqueeze(0)
        context_mask = (atom_indices < end_idx.unsqueeze(1)) & all_mask

        # Encode context
        context = self.context_encoder(
            all_atoms, all_residues, all_elements, all_coords, context_mask
        )

        return context

    def forward_residue(
        self,
        context: torch.Tensor,
        residue_hidden: torch.Tensor,
        residue_type: torch.Tensor,
        atom_types: torch.Tensor,
        element_types: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict coordinates for a single residue.

        Args:
            context: (batch, d_model) atomic context from previous residues.
            residue_hidden: (batch, d_model) residue-level hidden state.
            residue_type: (batch,) residue type index.
            atom_types: (batch, n_atoms) atom types in this residue.
            element_types: (batch, n_atoms) element types.
            atom_mask: (batch, n_atoms) valid atom mask.

        Returns:
            (batch, n_atoms, 3) predicted coordinates.
        """
        # Combine atomic context with residue-level hidden state
        combined = self.context_combine(torch.cat([context, residue_hidden], dim=-1))

        # Decode residue
        coords = self.residue_decoder(
            combined, residue_type, atom_types, element_types, atom_mask
        )

        return coords

    def compute_loss(
        self,
        atoms: torch.Tensor,
        residues_per_atom: torch.Tensor,
        elements: torch.Tensor,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_types: torch.Tensor,
        residue_boundaries: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss over all residues with teacher forcing.

        Args:
            atoms: (batch, max_atoms) atom type indices.
            residues_per_atom: (batch, max_atoms) residue type for each atom.
            elements: (batch, max_atoms) element type indices.
            coords: (batch, max_atoms, 3) ground truth coordinates.
            atom_mask: (batch, max_atoms) True = valid atom.
            residue_types: (batch, max_residues) residue type sequence.
            residue_boundaries: (batch, max_residues+1) cumulative atom counts.
            residue_mask: (batch, max_residues) True = valid residue.

        Returns:
            Scalar loss.
        """
        B = atoms.shape[0]
        device = atoms.device
        n_residues = residue_types.shape[1]

        # Get residue-level hidden states from causal transformer
        res_emb = self.residue_embed(residue_types)  # (B, n_res, d_model)
        res_hidden = self.residue_transformer(res_emb, padding_mask=~residue_mask)

        total_loss = 0.0
        n_atoms_total = 0

        for i in range(n_residues):
            # Skip if this residue is padded in all batches
            if not residue_mask[:, i].any():
                continue

            # Get context from previous atoms
            context = self._get_context_for_residue(
                atoms, residues_per_atom, elements, coords, atom_mask,
                residue_boundaries, i
            )

            # Get atoms for this residue
            start_idx = residue_boundaries[:, i]
            end_idx = residue_boundaries[:, i + 1]
            max_res_atoms = (end_idx - start_idx).max().item()

            if max_res_atoms == 0:
                continue

            # Gather atom data for this residue
            res_atoms = torch.zeros(B, max_res_atoms, dtype=torch.long, device=device)
            res_elements = torch.zeros(B, max_res_atoms, dtype=torch.long, device=device)
            res_coords = torch.zeros(B, max_res_atoms, 3, device=device)
            res_atom_mask = torch.zeros(B, max_res_atoms, dtype=torch.bool, device=device)

            for b in range(B):
                s, e = start_idx[b].item(), end_idx[b].item()
                n = e - s
                if n > 0:
                    res_atoms[b, :n] = atoms[b, s:e]
                    res_elements[b, :n] = elements[b, s:e]
                    res_coords[b, :n] = coords[b, s:e]
                    res_atom_mask[b, :n] = True

            # Predict coordinates
            pred_coords = self.forward_residue(
                context,
                res_hidden[:, i],
                residue_types[:, i],
                res_atoms,
                res_elements,
                res_atom_mask,
            )

            # Compute loss (only for valid atoms)
            loss = F.mse_loss(pred_coords, res_coords, reduction='none').sum(dim=-1)
            loss = (loss * res_atom_mask.float()).sum()

            total_loss += loss
            n_atoms_total += res_atom_mask.sum().item()

        if n_atoms_total > 0:
            return total_loss / n_atoms_total
        return torch.tensor(0.0, device=device)

    @torch.no_grad()
    def generate(
        self,
        residue_types: torch.Tensor,
        atoms_per_residue: list[torch.Tensor],
        elements_per_residue: list[torch.Tensor],
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate structure autoregressively.

        Args:
            residue_types: (batch, n_residues) or (n_residues,) residue sequence.
            atoms_per_residue: List of (batch, n_atoms) atom types per residue.
            elements_per_residue: List of (batch, n_atoms) element types per residue.
            temperature: Noise scale (0 = deterministic).

        Returns:
            List of (batch, n_atoms, 3) coordinates per residue.
        """
        if residue_types.dim() == 1:
            residue_types = residue_types.unsqueeze(0)
            atoms_per_residue = [a.unsqueeze(0) if a.dim() == 1 else a for a in atoms_per_residue]
            elements_per_residue = [e.unsqueeze(0) if e.dim() == 1 else e for e in elements_per_residue]

        B, n_residues = residue_types.shape
        device = residue_types.device

        # Get residue-level hidden states
        res_emb = self.residue_embed(residue_types)
        res_hidden = self.residue_transformer(res_emb)

        # Accumulate generated atoms
        all_atoms = []
        all_residues = []
        all_elements = []
        all_coords = []
        generated_coords = []

        for i in range(n_residues):
            # Build context from previously generated atoms
            if i == 0:
                context = torch.zeros(B, self.config.d_model, device=device)
            else:
                # Stack all previous atoms
                ctx_atoms = torch.cat(all_atoms, dim=1)
                ctx_residues = torch.cat(all_residues, dim=1)
                ctx_elements = torch.cat(all_elements, dim=1)
                ctx_coords = torch.cat(all_coords, dim=1)
                ctx_mask = torch.ones(B, ctx_atoms.shape[1], dtype=torch.bool, device=device)

                context = self.context_encoder(
                    ctx_atoms, ctx_residues, ctx_elements, ctx_coords, ctx_mask
                )

            # Get atom info for this residue
            res_atoms = atoms_per_residue[i]
            res_elements = elements_per_residue[i]
            n_atoms = res_atoms.shape[1]
            res_atom_mask = torch.ones(B, n_atoms, dtype=torch.bool, device=device)

            # Predict coordinates
            pred_coords = self.forward_residue(
                context,
                res_hidden[:, i],
                residue_types[:, i],
                res_atoms,
                res_elements,
                res_atom_mask,
            )

            # Add noise if temperature > 0
            if temperature > 0:
                pred_coords = pred_coords + temperature * torch.randn_like(pred_coords)

            generated_coords.append(pred_coords)

            # Accumulate for next iteration's context
            all_atoms.append(res_atoms)
            all_residues.append(residue_types[:, i:i+1].expand(-1, n_atoms))
            all_elements.append(res_elements)
            all_coords.append(pred_coords)

        return generated_coords

    def save(self, path: str) -> None:
        """Save model to disk."""
        import json
        from pathlib import Path
        from dataclasses import asdict

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self.config)
        config_dict['output_hidden_dims'] = list(config_dict['output_hidden_dims'])

        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AtomARModel":
        """Load model from disk."""
        import json
        from pathlib import Path

        path = Path(path)

        with open(path / "config.json") as f:
            config_dict = json.load(f)

        config_dict['output_hidden_dims'] = tuple(config_dict['output_hidden_dims'])
        config = AtomARModelConfig(**config_dict)

        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        model.to(device)

        return model
