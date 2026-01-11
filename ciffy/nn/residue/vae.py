"""Residue-level VAE for polymer generative modeling."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

import ciffy
from ciffy import Scale
from ciffy.geometry import rotation_6d_to_axis_angle
from ciffy.polymer import Polymer

from .encoder import ResidueEncoder
from .decoder import ResidueDecoder


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence from N(mu, sigma) to N(0, 1).

    Supports "free bits" regularization which sets a minimum KL per latent
    dimension, preventing posterior collapse where some dimensions encode
    no information.

    Args:
        mu: (*, latent_dim) latent means.
        logvar: (*, latent_dim) latent log-variances.
        free_bits: Minimum KL per dimension in nats. Dimensions with KL below
            this threshold are clamped. 0.0 disables (default). Typical values:
            0.1-0.5 nats.
        reduction: How to reduce the loss:
            - "mean": Mean over all elements (default)
            - "sum": Sum over all elements
            - "none": No reduction, returns per-element KL

    Returns:
        KL divergence loss.

    Example:
        >>> mu, logvar = encoder(x)
        >>> kl_loss = kl_divergence(mu, logvar, free_bits=0.25)
        >>> loss = recon_loss + beta * kl_loss
    """
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    if reduction == "mean":
        return kl_per_dim.mean()
    elif reduction == "sum":
        return kl_per_dim.sum()
    else:
        return kl_per_dim


class ResidueVAE(nn.Module):
    """
    Complete VAE for residue-level polymer modeling.

    Combines ResidueEncoder and ResidueDecoder into a full VAE that can:
    - Encode polymers to per-residue latent distributions
    - Decode latent samples to local coordinates and inter-residue transforms
    - Generate new polymer conformations via sampling

    Args:
        latent_dim: Dimension of the latent space per residue.
        d_model: Hidden dimension for transformer/MLP layers.
        n_heads: Number of attention heads in encoder.
        encoder_layers: Number of transformer layers in encoder.
        decoder_layers: Number of MLP layers in decoder.
        atom_dim: Dimension for atom type embeddings. Defaults to d_model // 2.
        residue_dim: Dimension for residue type embeddings. Defaults to d_model // 2.
        dropout: Dropout probability.
        rotation_repr: Rotation representation: "axis_angle" (6D output) or
            "rotation_6d" (9D output with continuous 6D rotation).

    Example:
        >>> model = ResidueVAE(latent_dim=32, d_model=128)
        >>> coords, transforms, mu, logvar = model(polymer)
        >>>
        >>> # Reconstruction
        >>> z = model.encode(polymer)
        >>> coords, transforms = model.decode(z, polymer)
        >>>
        >>> # Sampling
        >>> z = torch.randn(n_residues, 32)
        >>> coords, transforms = model.decode(z, template)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        atom_dim: int | None = None,
        residue_dim: int | None = None,
        dropout: float = 0.1,
        rotation_repr: Literal["axis_angle", "rotation_6d"] = "axis_angle",
    ):
        super().__init__()
        self.rotation_repr = rotation_repr

        self.encoder = ResidueEncoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=encoder_layers,
            atom_dim=atom_dim,
            residue_dim=residue_dim,
            dropout=dropout,
        )

        self.decoder = ResidueDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=decoder_layers,
            n_heads=n_heads,
            atom_dim=atom_dim,
            residue_dim=residue_dim,
            dropout=dropout,
            rotation_repr=rotation_repr,
        )

    @property
    def latent_dim(self) -> int:
        """Latent dimension per residue."""
        return self.encoder.output_dim

    def encode(
        self,
        polymer: Polymer,
        transforms: torch.Tensor | None = None,
        return_distribution: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode polymer to per-residue latent vectors.

        Args:
            polymer: Input polymer (torch backend).
            transforms: Optional (n_residues, 6) inter-residue transforms.
            return_distribution: If True, return (z, mu, logvar).

        Returns:
            z: (n_residues, latent_dim) latent vectors.
            If return_distribution: (z, mu, logvar) tuple.
        """
        return self.encoder(polymer, transforms, return_distribution)

    def decode(
        self,
        z: torch.Tensor,
        polymer: Polymer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latents to local coordinates and transforms.

        Args:
            z: (n_residues, latent_dim) latent vectors.
            polymer: Template polymer for atom structure.

        Returns:
            coords: (n_atoms, 3) local coordinates.
            transforms: (n_residues, 6) inter-residue SE(3) transforms.
        """
        return self.decoder(z, polymer)

    def forward(
        self,
        polymer: Polymer,
        transforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode and decode.

        Args:
            polymer: Input polymer (torch backend).
            transforms: Optional (n_residues, 6) inter-residue transforms.

        Returns:
            coords: (n_atoms, 3) predicted local coordinates.
            pred_transforms: (n_residues, 6) predicted inter-residue transforms.
            mu: (n_residues, latent_dim) latent means.
            logvar: (n_residues, latent_dim) latent log-variances.
        """
        z, mu, logvar = self.encode(polymer, transforms, return_distribution=True)
        coords, pred_transforms = self.decode(z, polymer)
        return coords, pred_transforms, mu, logvar

    @torch.no_grad()
    def sample(
        self,
        sequence: str,
        temperature: float = 1.0,
    ) -> Polymer:
        """
        Sample a polymer conformation from the prior.

        Creates a template from the sequence, samples latent vectors from N(0,1),
        decodes to local coordinates and transforms, then chains residues together.

        Args:
            sequence: Polymer sequence (e.g., "acguacgu" for RNA).
            temperature: Sampling temperature. Higher = more diverse.

        Returns:
            Polymer with sampled coordinates (on model's device).

        Example:
            >>> model = ResidueVAE(latent_dim=32)
            >>> sampled = model.sample("acguacguacgu", temperature=0.8)
            >>> sampled.write("sampled.cif")
        """
        self.eval()

        device = next(self.parameters()).device
        template = ciffy.template(sequence).torch().heavy().to(device)
        n_residues = template.size(Scale.RESIDUE)

        z = torch.randn(n_residues, self.latent_dim, device=device) * temperature
        local_coords, transforms = self.decode(z, template)

        # Convert 6D rotation to axis-angle for apply_local_transforms
        if self.rotation_repr == "rotation_6d":
            rot_6d = transforms[:, :6]
            trans = transforms[:, 6:]
            rot_aa = rotation_6d_to_axis_angle(rot_6d)
            transforms = torch.cat([rot_aa, trans], dim=-1)

        from ciffy.biochemistry.linking import O3P_FRAME, P_FRAME

        return (
            template
            .copy(coordinates=local_coords)
            .apply_local_transforms(transforms, O3P_FRAME, P_FRAME)
        )
