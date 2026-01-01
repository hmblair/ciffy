"""Variational Autoencoder models for ciffy.

Provides VAE-based generative models that share the same interface
as flow models, enabling use with PolymerModel for chain assembly.

Models:
    ResidueVAE: MLP-based VAE for residue conformations.
    AttentionResidueVAE: Attention-based VAE that handles missing atoms.

Utilities:
    losses: Shared loss functions (KL divergence, ELBO, beta scheduling).
"""

from .residue import (
    ResidueVAE,
    ResidueVAEConfig,
    AttentionResidueVAE,
    AttentionResidueVAEConfig,
)
from .losses import (
    compute_kl_divergence,
    get_beta_with_warmup,
    compute_elbo_loss,
    VAELossTracker,
)

__all__ = [
    # Models
    "ResidueVAE",
    "ResidueVAEConfig",
    "AttentionResidueVAE",
    "AttentionResidueVAEConfig",
    # Loss utilities
    "compute_kl_divergence",
    "get_beta_with_warmup",
    "compute_elbo_loss",
    "VAELossTracker",
]
