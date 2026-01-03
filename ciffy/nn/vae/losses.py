"""
Shared loss functions and utilities for VAE training.

This module provides common components used across different VAE architectures:
- KL divergence computation with free bits
- Beta scheduling (warmup)
- ELBO loss computation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute KL divergence between q(z|x) and N(0,1) prior.

    KL(q||p) = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))

    Supports "free bits" (Kingma et al., 2016) which allows minimum
    information per dimension before applying the KL penalty. This
    helps prevent posterior collapse in beta-VAEs.

    Args:
        mu: (batch, latent_dim) latent means.
        logvar: (batch, latent_dim) log variances.
        free_bits: Minimum nats per dimension before penalty (0 = disabled).
        reduction: 'mean' (default), 'sum', or 'none'.

    Returns:
        KL divergence. Shape depends on reduction:
        - 'mean': scalar (mean over batch)
        - 'sum': scalar (sum over batch)
        - 'none': (batch,) per-sample KL
    """
    # Per-dimension KL: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)

    if free_bits > 0:
        # Only penalize KL above free_bits threshold per dimension
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)

    # Sum over latent dimensions
    kl_per_sample = kl_per_dim.sum(dim=-1)

    if reduction == "mean":
        return kl_per_sample.mean()
    elif reduction == "sum":
        return kl_per_sample.sum()
    else:  # 'none'
        return kl_per_sample


def get_beta_with_warmup(
    current_epoch: int,
    target_beta: float,
    warmup_epochs: int,
) -> float:
    """Get beta value with linear warmup.

    Linearly increases beta from 0 to target_beta over warmup_epochs.
    After warmup, returns target_beta.

    Args:
        current_epoch: Current training epoch (0-indexed).
        target_beta: Target beta value after warmup.
        warmup_epochs: Number of epochs to warm up over.

    Returns:
        Current beta value.
    """
    if warmup_epochs <= 0:
        return target_beta

    warmup_progress = min(1.0, current_epoch / warmup_epochs)
    return target_beta * warmup_progress


def compute_elbo_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
    recon_reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ELBO loss for VAE training.

    ELBO = -E[log p(x|z)] + beta * KL(q(z|x) || p(z))
         ≈ recon_loss + beta * kl_loss

    Args:
        recon: Reconstructed output.
        target: Target output.
        mu: (batch, latent_dim) latent means.
        logvar: (batch, latent_dim) log variances.
        beta: Weight for KL term (beta-VAE). Default 1.0.
        free_bits: Min nats per latent dim before KL penalty.
        recon_reduction: Reduction for reconstruction loss.

    Returns:
        total_loss: Combined ELBO loss.
        recon_loss: Reconstruction loss (MSE).
        kl_loss: KL divergence.
    """
    recon_loss = F.mse_loss(recon, target, reduction=recon_reduction)
    kl_loss = compute_kl_divergence(mu, logvar, free_bits=free_bits)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


class VAELossTracker:
    """Track and log VAE training metrics.

    Provides consistent logging interface for different VAE architectures.
    """

    def __init__(self, log_fn, prefix: str = "train"):
        """Initialize loss tracker.

        Args:
            log_fn: Logging function (e.g., self.log from LightningModule).
            prefix: Prefix for metric names ('train' or 'val').
        """
        self.log_fn = log_fn
        self.prefix = prefix

    def log_losses(
        self,
        total_loss: torch.Tensor,
        recon_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        beta: float,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        step: bool = True,
        epoch: bool = True,
    ) -> None:
        """Log VAE training metrics.

        Args:
            total_loss: Combined ELBO loss.
            recon_loss: Reconstruction loss.
            kl_loss: KL divergence.
            beta: Current beta value.
            mu: Latent means (for statistics).
            logvar: Latent log-variances (for statistics).
            step: Whether to log per-step.
            epoch: Whether to log per-epoch.
        """
        p = self.prefix

        # Main losses
        self.log_fn(f"{p}/loss", total_loss, on_step=False, on_epoch=epoch)
        self.log_fn(f"{p}/recon", recon_loss, prog_bar=True, on_step=step, on_epoch=epoch)
        self.log_fn(f"{p}/kl", kl_loss, on_step=step, on_epoch=epoch)
        self.log_fn(f"{p}/beta", beta, on_step=False, on_epoch=epoch)

        # Latent statistics
        self.log_fn(f"{p}/z_mean", mu.mean(), on_step=False, on_epoch=epoch)
        self.log_fn(f"{p}/z_std", mu.std(), on_step=False, on_epoch=epoch)
        self.log_fn(f"{p}/logvar_mean", logvar.mean(), on_step=False, on_epoch=epoch)


class GeometrySamplingLoss:
    """Compute geometry loss on samples from the VAE prior.

    Samples latent vectors from N(0,1), decodes them, and penalizes
    invalid geometry (bond lengths, angles, inter-residue distances).
    This regularizes the latent space to produce valid structures.

    Works with both per-residue VAEs (single residue type) and
    consolidated VAEs (multiple residue types).

    Example (per-residue VAE):
        >>> from ciffy.geometry import GeometryConstraints
        >>> constraints = GeometryConstraints.from_residue(Residue.A, atom_indices)
        >>> geom_loss = GeometrySamplingLoss({Residue.A: constraints})
        >>>
        >>> # In training loop:
        >>> loss = geom_loss.compute(model, n_samples=16)

    Example (consolidated VAE):
        >>> constraints = {
        ...     Residue.A: GeometryConstraints.from_residue(Residue.A, atoms_a),
        ...     Residue.C: GeometryConstraints.from_residue(Residue.C, atoms_c),
        ...     ...
        ... }
        >>> geom_loss = GeometrySamplingLoss(constraints)
    """

    def __init__(
        self,
        constraints: "dict[Residue, GeometryConstraints]",
        weights: dict[str, float] | None = None,
    ):
        """Initialize geometry sampling loss.

        Args:
            constraints: Dict mapping Residue to GeometryConstraints.
            weights: Optional weights for loss components.
                Defaults to {"bond": 1.0, "angle": 1.0, "inter": 1.0}.
        """
        from ciffy.geometry import GeometryConstraints as GC  # Avoid circular import

        self.constraints = constraints
        self.weights = weights or {"bond": 1.0, "angle": 1.0, "inter": 1.0}
        self._residues = list(constraints.keys())

    def to(self, device: "str | torch.device") -> "GeometrySamplingLoss":
        """Move constraints to device."""
        return GeometrySamplingLoss(
            {res: c.to(device) for res, c in self.constraints.items()},
            self.weights,
        )

    def compute(
        self,
        model: "torch.nn.Module",
        n_samples: int = 16,
        device: "torch.device | None" = None,
    ) -> torch.Tensor:
        """Compute geometry loss on samples from prior.

        Supports two model interfaces:
        1. Per-residue: model.sample(n_samples) -> (coords, transforms)
        2. Consolidated: model.sample(residue, n_samples) -> (coords, transforms)

        Args:
            model: VAE model with sample() method.
            n_samples: Number of samples per residue type.
            device: Device for sampling. Defaults to model.device.

        Returns:
            Scalar geometry loss averaged over all residue types.
        """
        if device is None:
            device = model.device

        total_loss = torch.tensor(0.0, device=device)
        n_residues = 0

        for residue in self._residues:
            constraints = self.constraints[residue].to(device)

            # Try consolidated interface first, then per-residue
            try:
                coords, transforms = model.sample(residue, n_samples)
            except TypeError:
                # Per-residue model - sample() takes only n_samples
                coords, transforms = model.sample(n_samples)

            # Compute geometry loss for this residue type
            loss = constraints.total_loss(coords, transforms, self.weights)
            total_loss = total_loss + loss
            n_residues += 1

        if n_residues > 0:
            total_loss = total_loss / n_residues

        return total_loss

    @classmethod
    def from_residue(
        cls,
        residue: "Residue",
        atom_indices: list[int],
        device: str = "cpu",
        **kwargs,
    ) -> "GeometrySamplingLoss":
        """Create for a single residue type.

        Convenience factory for per-residue VAEs.

        Args:
            residue: Residue type.
            atom_indices: List of atom indices in model ordering.
            device: Device for constraints.
            **kwargs: Additional arguments for GeometryConstraints.

        Returns:
            GeometrySamplingLoss for the residue.
        """
        from ciffy.geometry import GeometryConstraints

        constraints = GeometryConstraints.from_residue(
            residue, atom_indices, device=device, **kwargs
        )
        return cls({residue: constraints})

    @classmethod
    def from_residue_atoms(
        cls,
        residue_atoms: "dict[Residue, list[int]]",
        device: str = "cpu",
        **kwargs,
    ) -> "GeometrySamplingLoss":
        """Create for multiple residue types.

        Convenience factory for consolidated VAEs.

        Args:
            residue_atoms: Dict mapping Residue to atom indices.
            device: Device for constraints.
            **kwargs: Additional arguments for GeometryConstraints.

        Returns:
            GeometrySamplingLoss for all residue types.
        """
        from ciffy.geometry import GeometryConstraints

        constraints = {
            res: GeometryConstraints.from_residue(res, atoms, device=device, **kwargs)
            for res, atoms in residue_atoms.items()
        }
        return cls(constraints)

    def __repr__(self) -> str:
        residue_names = [r.name for r in self._residues]
        return f"GeometrySamplingLoss(residues={residue_names})"


# Type hint imports at module level cause circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.geometry import GeometryConstraints


__all__ = [
    "compute_kl_divergence",
    "get_beta_with_warmup",
    "compute_elbo_loss",
    "VAELossTracker",
    "GeometrySamplingLoss",
]
