"""
Neural network utilities for ciffy.

Provides PyTorch-compatible utilities for loading polymer structures:
- PolymerDataset: Dataset for loading polymer structures from disk
- PolymerEmbedding: Embedding layer for polymer atoms/residues

Requires PyTorch. Install with: pip install torch
"""

try:
    import torch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ciffy.nn requires PyTorch. Install with: pip install torch"
    ) from e

from .dataset import PolymerDataset
from .embedding import PolymerEmbedding

__all__ = ["PolymerDataset", "PolymerEmbedding"]
