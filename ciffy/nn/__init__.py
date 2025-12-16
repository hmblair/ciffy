"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.

Modules:
    - dataset: PolymerDataset for loading CIF files
    - embedding: PolymerEmbedding for learnable embeddings
    - transformer: Modern transformer with Pre-LN, RoPE, SwiGLU
    - training: Reusable training utilities
    - vae: Variational autoencoder for polymer conformations
"""

from .dataset import PolymerDataset
from .embedding import PolymerEmbedding
from .transformer import (
    Transformer,
    TransformerBlock,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)
from .training import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    polymer_collate_fn,
    get_worker_init_fn,
    BetaScheduler,
)
from .vae import PolymerVAE, DihedralEncoder, DihedralDecoder

__all__ = [
    # Dataset
    "PolymerDataset",
    # Embedding
    "PolymerEmbedding",
    # Transformer components
    "Transformer",
    "TransformerBlock",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
    # Training utilities
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "train_epoch",
    "polymer_collate_fn",
    "get_worker_init_fn",
    "BetaScheduler",
    # VAE
    "PolymerVAE",
    "DihedralEncoder",
    "DihedralDecoder",
]
