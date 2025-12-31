"""Variational Autoencoder models for ciffy.

Provides VAE-based generative models that share the same interface
as flow models, enabling use with PolymerFlowModel for chain assembly.
"""

from .residue import ResidueVAE, ResidueVAEConfig

__all__ = ["ResidueVAE", "ResidueVAEConfig"]
