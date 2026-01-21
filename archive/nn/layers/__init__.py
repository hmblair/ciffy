"""Reusable neural network building blocks.

Provides foundational layers used across the ciffy.nn module:
- MLP: Unified multi-layer perceptron with optional residual connections
- Transformer components: RMSNorm, RoPE, SwiGLU, etc.
- Pairformer: AlphaFold3-style transformer for pair representations

Example:
    >>> from ciffy.nn.layers import MLP, Transformer, Pairformer
    >>>
    >>> mlp = MLP(64, 10, hidden_dims=[128, 64])
    >>> transformer = Transformer(d_model=256, num_layers=4, num_heads=8)
    >>> pairformer = Pairformer(d_pair=128, num_layers=4, num_heads=8)
"""

from .mlp import MLP
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
from .causal import CausalTransformer, create_causal_mask

__all__ = [
    # Transformer components
    "Transformer",
    "TransformerBlock",
    "AdaLNTransformer",
    "AdaLNTransformerBlock",
    "AdaLN",
    "MLP",
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
    "create_causal_mask",
]
