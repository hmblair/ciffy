"""
RNA-specific functionality for ciffy.

This module provides tools for working with RNA structures and data:
- ReactivityIndex: Match SHAPE/DMS reactivity data to structures by sequence
- ReactivityMatch: Result of a successful reactivity match
- dotbracket_to_pairs: Convert dot-bracket notation to base pair array
- pairs_to_dotbracket: Convert base pair array to dot-bracket notation
- secondary_structure: Extract secondary structure from polymer connections
"""

from .reactivity import ReactivityIndex, ReactivityMatch
from .secondary import dotbracket_to_pairs, pairs_to_dotbracket, secondary_structure

__all__ = [
    "ReactivityIndex",
    "ReactivityMatch",
    "dotbracket_to_pairs",
    "pairs_to_dotbracket",
    "secondary_structure",
]
