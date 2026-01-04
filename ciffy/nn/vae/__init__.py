"""Variational Autoencoder models for ciffy.

Provides VAE-based generative models that share the same interface
as flow models, enabling use with PolymerModel for chain assembly.

Models:
    ConsolidatedResidueVAE: Shared encoder VAE for all residue types.

Utilities:
    losses: Shared loss functions (KL divergence, ELBO, beta scheduling).
"""

from .residue import (
    ConsolidatedResidueVAE,
    ConsolidatedVAEConfig,
)
from .losses import (
    compute_kl_divergence,
    get_beta_with_warmup,
    compute_elbo_loss,
    VAELossTracker,
)

__all__ = [
    # Models
    "ConsolidatedResidueVAE",
    "ConsolidatedVAEConfig",
    # Loss utilities
    "compute_kl_divergence",
    "get_beta_with_warmup",
    "compute_elbo_loss",
    "VAELossTracker",
]
