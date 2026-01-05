"""Reusable neural network building blocks.

Provides foundational layers used across the ciffy.nn module:
- DenseNetwork: Multi-layer perceptron
- PolymerEmbedding: Embeddings for polymer features
- Transformer components: RMSNorm, RoPE, SwiGLU, etc.
- Pairformer: AlphaFold3-style transformer for pair representations

Example:
    >>> from ciffy.nn.layers import DenseNetwork, Transformer, Pairformer
    >>>
    >>> mlp = DenseNetwork(64, 10, hidden_sizes=[128, 64])
    >>> transformer = Transformer(d_model=256, num_layers=4, num_heads=8)
    >>> pairformer = Pairformer(d_pair=128, num_layers=4, num_heads=8)
"""

from .dense_network import DenseNetwork
from .embedding import PolymerEmbedding
from .transformer import (
    Transformer,
    TransformerBlock,
    AdaLNTransformer,
    AdaLNTransformerBlock,
    AdaLN,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)
from .pairformer import (
    Pairformer,
    PairformerBlock,
    TriangularMultiplicativeUpdate,
    TriangularAttention,
    PairTransition,
    OuterProductMean,
    PairToSingleAttention,
)
from .causal import (
    CausalTransformer,
    CausalTransformerBlock,
    CausalMultiHeadAttention,
    create_causal_mask,
)

__all__ = [
    # Dense network
    "DenseNetwork",
    # Embedding
    "PolymerEmbedding",
    # Transformer components
    "Transformer",
    "TransformerBlock",
    "AdaLNTransformer",
    "AdaLNTransformerBlock",
    "AdaLN",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
    # Pairformer components
    "Pairformer",
    "PairformerBlock",
    "TriangularMultiplicativeUpdate",
    "TriangularAttention",
    "PairTransition",
    "OuterProductMean",
    "PairToSingleAttention",
    # Causal (autoregressive) components
    "CausalTransformer",
    "CausalTransformerBlock",
    "CausalMultiHeadAttention",
    "create_causal_mask",
]
