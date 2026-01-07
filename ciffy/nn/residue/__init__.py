"""Residue-level neural network components.

Provides encoder/decoder architectures for per-residue latent modeling:

- :class:`ResidueEncoder`: Encodes polymer to per-residue latents using
  packed attention and inter-residue transform features.
- :class:`ResidueDecoder`: Decodes latents to local coordinates and
  inter-residue SE(3) transforms.
- :class:`ResidueVAE`: Complete VAE combining encoder and decoder.

Example:
    >>> from ciffy.nn.residue import ResidueVAE
    >>> import ciffy
    >>>
    >>> model = ResidueVAE(latent_dim=32, d_model=128)
    >>> polymer = ciffy.load("structure.cif").torch()
    >>>
    >>> # Encode
    >>> z = model.encode(polymer)
    >>>
    >>> # Decode
    >>> coords, transforms = model.decode(z, polymer)
    >>>
    >>> # Full forward pass
    >>> coords, transforms, mu, logvar = model(polymer)
"""

from .encoder import ResidueEncoder
from .decoder import ResidueDecoder
from .vae import ResidueVAE, kl_divergence

__all__ = [
    "ResidueEncoder",
    "ResidueDecoder",
    "ResidueVAE",
    "kl_divergence",
]
