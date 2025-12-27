"""
Polymer module for molecular structure representation.

This module provides the Polymer class and factory functions for creating
molecular structures from sequences or CIF files.

Classes:
    Polymer: Main structure class representing molecular assemblies.
    Field: Descriptor for Polymer array fields with dtype information.
    Metadata: Descriptor for Polymer metadata (non-array values).

Factory Functions:
    from_sequence: Create Polymer from sequence string (e.g., "acgu").
    from_extract: Convert extracted coordinates back to Polymer.
"""

from .polymer import Polymer, Field, Metadata
from .template import from_sequence, from_extract
from .builder import ChainBuilder, ResidueData, expand_residue

__all__ = [
    "Polymer",
    "Field",
    "Metadata",
    "from_sequence",
    "from_extract",
    "ChainBuilder",
    "ResidueData",
    "expand_residue",
]
