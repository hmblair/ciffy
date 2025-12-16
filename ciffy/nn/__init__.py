"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.

Modules:
    - dataset: PolymerDataset for loading CIF files
    - embedding: PolymerEmbedding for learnable embeddings
    - transformer: Modern transformer with Pre-LN, RoPE, SwiGLU
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
    # VAE
    "PolymerVAE",
    "DihedralEncoder",
    "DihedralDecoder",
]
