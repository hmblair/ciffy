"""Variational Autoencoder utilities for ciffy.

Provides shared loss functions and utilities for VAE-based models.

Note:
    The residue-level VAE models have been replaced by ciffy.nn.residue.ResidueVAE.
    Old models are archived in archive/nn/vae/residue/.

Utilities:
    losses: Shared loss functions (KL divergence, ELBO, beta scheduling).
"""

from .losses import (
    compute_kl_divergence,
    get_beta_with_warmup,
    compute_elbo_loss,
    VAELossTracker,
)

__all__ = [
    # Loss utilities
    "compute_kl_divergence",
    "get_beta_with_warmup",
    "compute_elbo_loss",
    "VAELossTracker",
]
