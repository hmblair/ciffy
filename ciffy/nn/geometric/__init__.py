"""Geometric deep learning with SO(3) equivariance.

This module provides equivariant neural network layers for processing 3D
molecular and point cloud data while maintaining rotation and translation
equivariance.

Requires the optional `sphericart` dependency for spherical harmonic computation.
Install with: pip install ciffy[geometric]

Key components:

Representations:
    - Irrep: Single irreducible representation of SO(3)
    - ProductIrrep: Tensor product decomposition of two irreps
    - Repr: Collection of irreps into a unified representation
    - ProductRepr: Tensor product of two representations

Equivariant Primitives:
    - RepNorm: Compute norms of spherical tensor components
    - SphericalHarmonic: Compute spherical harmonics features
    - RadialBasisFunctions: Learnable radial basis function expansion
    - SequencePositionEncoding: Sinusoidal encoding for sequence positions
    - EquivariantBasis: Compute equivariant basis matrices
    - EquivariantBases: Compute bases for multiple product representations

Layers:
    - MatrixOutput: Full-rank matrix generation from hidden features
    - LowRankMatrixOutput: Low-rank factorized matrix generation
    - RadialWeight: Neural network for tensor product weights
    - EquivariantLinear: Linear layer preserving spherical tensor structure
    - EquivariantGating: Norm-based gating for spherical tensors
    - EquivariantTransition: Feed-forward transition layer
    - EquivariantConvolution: SE(3)-equivariant convolution
    - EquivariantLayerNorm: Equivariant layer normalization

Attention:
    - Attention: k-NN based multi-head attention
    - EquivariantAttention: SE(3)-equivariant sparse attention
    - EquivariantTransformerBlock: Transformer block with sparse attention
    - EquivariantTransformer: Full equivariant transformer

Utilities:
    - build_knn_graph: Build k-nearest neighbor graph from coordinates

Example:
    >>> from ciffy.nn.geometric import Repr, EquivariantTransformer
    >>> import torch
    >>>
    >>> # Define representations
    >>> in_repr = Repr(lvals=[0, 1], mult=4)   # scalars + vectors
    >>> out_repr = Repr(lvals=[0, 1], mult=1)
    >>> hidden_repr = Repr(lvals=[0, 1], mult=16)
    >>>
    >>> # Create model
    >>> model = EquivariantTransformer(
    ...     in_repr, out_repr, hidden_repr,
    ...     hidden_layers=4,
    ...     edge_dim=16,
    ...     edge_hidden_dim=32,
    ...     k_neighbors=16,
    ... )
    >>>
    >>> # Forward pass
    >>> coords = torch.randn(100, 3)
    >>> features = torch.randn(100, 4, 4)  # (N, mult, dim)
    >>> output = model(coords, features)

References:
    - Thomas, N., et al. (2018). "Tensor field networks"
    - Weiler, M., et al. (2018). "3D Steerable CNNs"
"""

from .representations import Irrep, ProductIrrep, Repr, ProductRepr
from .equivariant import (
    FEATURE_DIM,
    REPR_DIM,
    RepNorm,
    SphericalHarmonic,
    RadialBasisFunctions,
    SequencePositionEncoding,
    EquivariantBasis,
    EquivariantBases,
)
from .layers import (
    build_knn_graph,
    MatrixOutput,
    LowRankMatrixOutput,
    RadialWeight,
    EquivariantLinear,
    EquivariantGating,
    EquivariantTransition,
    EquivariantConvolution,
    EquivariantLayerNorm,
    Attention,
    EquivariantAttention,
    EquivariantTransformerBlock,
    EquivariantTransformer,
)

__all__ = [
    # Constants
    "FEATURE_DIM",
    "REPR_DIM",
    # Representations
    "Irrep",
    "ProductIrrep",
    "Repr",
    "ProductRepr",
    # Equivariant primitives
    "RepNorm",
    "SphericalHarmonic",
    "RadialBasisFunctions",
    "SequencePositionEncoding",
    "EquivariantBasis",
    "EquivariantBases",
    # Layers
    "build_knn_graph",
    "MatrixOutput",
    "LowRankMatrixOutput",
    "RadialWeight",
    "EquivariantLinear",
    "EquivariantGating",
    "EquivariantTransition",
    "EquivariantConvolution",
    "EquivariantLayerNorm",
    # Attention
    "Attention",
    "EquivariantAttention",
    "EquivariantTransformerBlock",
    "EquivariantTransformer",
]
