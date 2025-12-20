"""
Protocols for generative models in ciffy.

Defines interfaces for neural network models that generate polymer structures.
Uses Protocol for duck-typing flexibility while providing clear contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..polymer import Polymer
    import torch


@runtime_checkable
class PolymerGenerativeModel(Protocol):
    """
    Protocol for models that generate polymer conformations.

    Models implementing this protocol take a template Polymer (from sequence)
    and generate one or more Polymers with predicted coordinates/conformations.

    This is the main interface for any generative model in ciffy (VAE, diffusion,
    flow models, etc.). Models don't need to inherit from this class - if they
    implement the required methods, they satisfy the protocol.

    Example:
        >>> import ciffy
        >>> from ciffy.nn import load_vae
        >>>
        >>> # Load a trained VAE
        >>> vae = load_vae("checkpoint_best.pt", device="cuda")
        >>>
        >>> # Create template from sequence
        >>> template = ciffy.from_sequence("MGKLF", backend="torch").to("cuda")
        >>>
        >>> # Generate samples (model implements this method)
        >>> samples = vae.sample(template, n_samples=10, temperature=1.0)
        >>>
        >>> # Each sample is a Polymer with predicted coordinates
        >>> for i, polymer in enumerate(samples):
        ...     polymer.numpy().write(f"sample_{i}.cif")
    """

    def sample(
        self,
        template: Polymer,
        n_samples: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ) -> list[Polymer]:
        """
        Generate polymer conformations from a template.

        Args:
            template: Template Polymer with sequence and topology information.
                Coordinates may be provided (for reconstruction tasks) or ignored
                (for pure generative sampling).
            n_samples: Number of independent conformations to generate.
            temperature: Sampling temperature (higher = more diverse).
                For VAE: scales the diagonal of the latent covariance.
                For diffusion: affects denoising step size.
            **kwargs: Model-specific sampling parameters.

        Returns:
            List of n_samples Polymers with predicted coordinates.
            Each polymer has the same topology as the template but different
            dihedral angles / coordinates depending on the generative model.

        Raises:
            RuntimeError: If model is in training mode (should call .eval() first).
            ValueError: If template is incompatible with model.
        """
        ...


@runtime_checkable
class PolymerEncoder(Protocol):
    """
    Protocol for models that can encode polymers to latent representations.

    Useful for reconstruction, interpolation, and latent space analysis.
    Typically implemented by VAE-like models.

    Example:
        >>> vae = load_vae("checkpoint_best.pt")
        >>> polymer = ciffy.load("structure.cif", backend="torch")
        >>>
        >>> # Encode to latent distribution
        >>> z_mu, z_logvar = vae.encode(polymer)
        >>>
        >>> # Reconstruct through decoder
        >>> recon = vae.reconstruct(polymer, sample_latent=False)
        >>> recon.numpy().write("reconstructed.cif")
    """

    def encode(
        self,
        polymer: Polymer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode polymer to latent distribution parameters.

        Args:
            polymer: Input polymer to encode.

        Returns:
            Tuple of (mu, logvar) for the latent distribution.
            - mu: (batch_size, latent_dim) tensor of distribution means
            - logvar: (batch_size, latent_dim) tensor of distribution log-variances

        Note:
            For a single polymer, batch_size is 1.
        """
        ...

    def reconstruct(
        self,
        polymer: Polymer,
        sample_latent: bool = False,
    ) -> Polymer:
        """
        Reconstruct polymer through encoder-decoder.

        Args:
            polymer: Input polymer to reconstruct.
            sample_latent: If True, sample z from latent distribution.
                If False, use mean of distribution (deterministic).

        Returns:
            Reconstructed polymer with same topology as input.
        """
        ...


__all__ = [
    "PolymerGenerativeModel",
    "PolymerEncoder",
]
