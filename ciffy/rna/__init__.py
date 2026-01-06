"""
RNA-specific functionality for ciffy.

This module provides tools for working with RNA structures and data:
- ReactivityIndex: Match SHAPE/DMS reactivity data to structures by sequence
- ReactivityMatch: Result of a successful reactivity match
"""

from .reactivity import ReactivityIndex, ReactivityMatch

__all__ = ["ReactivityIndex", "ReactivityMatch"]
