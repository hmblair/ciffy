"""
VAE loss functions for polymer dihedral reconstruction.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from .distributions import VonMisesNLL


class VAELoss(nn.Module if TORCH_AVAILABLE else object):
    """
    Combined VAE loss for dihedral reconstruction.

    Total loss = reconstruction_loss + beta * kl_divergence

    The reconstruction loss uses von Mises negative log-likelihood,
    appropriate for periodic angular quantities. The KL divergence
    regularizes the latent space to be close to a standard normal prior.

    Args:
        beta: Weight for KL divergence term (for beta-VAE training).
              beta=1 gives standard VAE, beta>1 encourages disentanglement.

    Example:
        >>> loss_fn = VAELoss(beta=1.0)
        >>> losses = loss_fn(mu_pred, kappa_pred, target, z_mu, z_logvar, mask)
        >>> losses['loss'].backward()
    """

    def __init__(self, beta: float = 1.0):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for VAELoss. Install with: pip install torch"
            )
        super().__init__()
        self.beta = beta
        self.von_mises_nll = VonMisesNLL(reduction="none")

    def forward(
        self,
        mu_pred: "torch.Tensor",
        kappa_pred: "torch.Tensor",
        target: "torch.Tensor",
        z_mu: "torch.Tensor",
        z_logvar: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
    ) -> dict[str, "torch.Tensor"]:
        """
        Compute VAE loss.

        Args:
            mu_pred: (L, D) predicted mean angles
            kappa_pred: (L, D) predicted concentration parameters
            target: (L, D) target angles
            z_mu: (latent_dim,) latent distribution mean
            z_logvar: (latent_dim,) latent distribution log-variance
            mask: (L, D) boolean mask for valid dihedrals (True = valid)

        Returns:
            Dictionary with keys:
                'loss': Total loss (scalar)
                'recon_loss': Reconstruction loss (scalar)
                'kl_loss': KL divergence (scalar)
        """
        # Reconstruction loss (von Mises NLL)
        nll = self.von_mises_nll(mu_pred, kappa_pred, target)

        if mask is not None:
            nll = nll * mask.float()
            recon_loss = nll.sum() / mask.float().sum().clamp(min=1)
        else:
            recon_loss = nll.mean()

        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        total_loss = recon_loss + self.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
