"""Reusable neural network building blocks.

Provides foundational layers used across the ciffy.nn module:
- DenseNetwork: Multi-layer perceptron
- PolymerEmbedding: Embeddings for polymer features
- Transformer components: RMSNorm, RoPE, SwiGLU, etc.

Example:
    >>> from ciffy.nn.layers import DenseNetwork, Transformer
    >>>
    >>> mlp = DenseNetwork(64, 10, hidden_sizes=[128, 64])
    >>> transformer = Transformer(d_model=256, num_layers=4, num_heads=8)
"""

from .dense_network import DenseNetwork
from .embedding import PolymerEmbedding
from .transformer import (
    Transformer,
    TransformerBlock,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)

__all__ = [
    # Dense network
    "DenseNetwork",
    # Embedding
    "PolymerEmbedding",
    # Transformer components
    "Transformer",
    "TransformerBlock",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
]
