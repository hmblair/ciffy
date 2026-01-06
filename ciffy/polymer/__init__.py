"""
Polymer module for molecular structure representation.

This module provides the Polymer and HeteroAtoms classes, along with
factory functions for creating molecular structures from sequences or CIF files.

Classes:
    Polymer: Main structure class representing polymer assemblies.
    HeteroAtoms: Lightweight container for non-polymer atoms (HETATM).
    AtomContainer: Base class for atom containers (internal use).
    Field: Descriptor for array fields with scale metadata.
    Metadata: Descriptor for non-array metadata values.

Factory Functions:
    from_sequence: Create Polymer from sequence string (e.g., "acgu").
    from_extract: Convert extracted coordinates back to Polymer.
"""

from .base import AtomContainer, Field, Metadata
from .polymer import Polymer
from .hetero import HeteroAtoms
from .template import from_sequence, from_extract

__all__ = [
    "AtomContainer",
    "Polymer",
    "HeteroAtoms",
    "Field",
    "Metadata",
    "from_sequence",
    "from_extract",
]
