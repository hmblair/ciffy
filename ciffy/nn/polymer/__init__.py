"""Polymer-specific neural network utilities.

This module provides utilities for loading and embedding polymer structures:

- PolymerDataset: Dataset for loading polymer structures from disk
- PolymerEmbedding: Embedding layer for polymer atoms/residues
"""

from .dataset import PolymerDataset
from .embedding import PolymerEmbedding

__all__ = ["PolymerDataset", "PolymerEmbedding"]
