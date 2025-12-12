"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.
"""

from .dataset import PolymerDataset
from .embedding import PolymerEmbedding

__all__ = [
    "PolymerDataset",
    "PolymerEmbedding",
]
