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


__all__ = [
    "compute_kl_divergence",
    "get_beta_with_warmup",
    "compute_elbo_loss",
    "VAELossTracker",
]
