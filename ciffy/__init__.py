"""
ciffy - Fast CIF file parsing for molecular structures.

A Python package for loading and manipulating molecular structures from
CIF (Crystallographic Information File) format files.
"""

__version__ = "0.5.3"

# Core types
from .polymer import Polymer
from .types import Scale, Molecule

# Operations
from .operations.reduction import Reduction
from .operations.alignment import kabsch_distance as rmsd

# I/O
from .io.loader import load

# Convenience aliases
RESIDUE = Scale.RESIDUE
CHAIN = Scale.CHAIN
MOLECULE = Scale.MOLECULE

PROTEIN = Molecule.PROTEIN
RNA = Molecule.RNA
DNA = Molecule.DNA

__all__ = [
    # Version
    "__version__",
    # Core types
    "Polymer",
    "Scale",
    "Molecule",
    "Reduction",
    # Functions
    "load",
    "rmsd",
    # Convenience aliases
    "RESIDUE",
    "CHAIN",
    "MOLECULE",
    "PROTEIN",
    "RNA",
    "DNA",
]
