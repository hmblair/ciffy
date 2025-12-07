"""
Backward-compatible re-exports from reorganized modules.

This module provides backward compatibility for code that imports
from ciffy.enum. New code should import directly from the appropriate
submodules:
- ciffy.utils: IndexEnum, PairEnum
- ciffy.biochemistry: Residue, Element, Adenosine, etc.
"""

# Re-export from utils
from .utils.enum_base import IndexEnum, PairEnum

# Re-export from biochemistry
from .biochemistry.elements import Element
from .biochemistry.residues import Residue, RES_ABBREV
from .biochemistry.nucleotides import (
    Adenosine,
    Cytosine,
    Guanosine,
    Uridine,
    RibonucleicAcid,
    RibonucleicAcidNoPrefix,
)
from .biochemistry.constants import (
    FRAMES,
    FRAME1,
    FRAME2,
    FRAME3,
    COARSE,
    Backbone,
    Nucleobase,
    Phosphate,
)

__all__ = [
    # Enum base classes
    "IndexEnum",
    "PairEnum",
    # Elements
    "Element",
    # Residues
    "Residue",
    "RES_ABBREV",
    # Nucleotides
    "Adenosine",
    "Cytosine",
    "Guanosine",
    "Uridine",
    "RibonucleicAcid",
    "RibonucleicAcidNoPrefix",
    # Constants
    "FRAMES",
    "FRAME1",
    "FRAME2",
    "FRAME3",
    "COARSE",
    "Backbone",
    "Nucleobase",
    "Phosphate",
]
