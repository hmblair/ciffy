"""
Variational Autoencoder for polymer backbone conformations.

Provides modules for encoding and decoding backbone dihedral angles
using a transformer-based architecture with von Mises output distributions.

Example:
    >>> from ciffy.nn.vae import PolymerVAE
    >>> import ciffy
    >>>
    >>> # Create and train a VAE
    >>> vae = PolymerVAE(latent_dim=64, hidden_dim=256)
    >>> polymer = ciffy.load("structure.cif", backend="torch")
    >>>
    >>> # Training loop
    >>> optimizer = torch.optim.Adam(vae.parameters())
    >>> for epoch in range(100):
    ...     losses = vae.compute_loss(polymer)
    ...     optimizer.zero_grad()
    ...     losses["loss"].backward()
    ...     optimizer.step()
    >>>
    >>> # Sample new conformations
    >>> samples = vae.sample(polymer, n_samples=10)
"""

from .vae import PolymerVAE
from .encoder import DihedralEncoder
from .decoder import DihedralDecoder
from .losses import VAELoss
from .distributions import (
    sincos_encode,
    sincos_decode,
    angular_distance,
    VonMisesNLL,
)
from .trainer import VAEConfig, VAETrainer, VAEModelConfig, DataConfig
from .inference import (
    reconstruct_polymer,
    interpolate_structures,
    sample_from_sequence,
)
from ...types.dihedral import (
    MAX_DIHEDRALS_PER_RESIDUE,
    NUM_PROTEIN_BACKBONE_DIHEDRALS,
    NUM_RNA_BACKBONE_DIHEDRALS,
)

__all__ = [
    # Main VAE class
    "PolymerVAE",
    # Training
    "VAETrainer",
    "VAEConfig",
    "VAEModelConfig",
    "DataConfig",
    # Components
    "DihedralEncoder",
    "DihedralDecoder",
    # Losses
    "VAELoss",
    "VonMisesNLL",
    # Inference
    "reconstruct_polymer",
    "interpolate_structures",
    "sample_from_sequence",
    # Utilities
    "sincos_encode",
    "sincos_decode",
    "angular_distance",
    # Constants
    "MAX_DIHEDRALS_PER_RESIDUE",
    "NUM_PROTEIN_BACKBONE_DIHEDRALS",
    "NUM_RNA_BACKBONE_DIHEDRALS",
]
