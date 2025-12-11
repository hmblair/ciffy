"""
ciffy - Fast CIF file parsing for molecular structures.

A Python package for loading and manipulating molecular structures from
CIF (Crystallographic Information File) format files.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ciffy")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without scm

# Core types
from .polymer import Polymer
from .types import Scale, Molecule

# Operations
from .operations.reduction import Reduction
from .operations.alignment import kabsch_distance as rmsd

# I/O
from .io.loader import load
from .io.writer import write_cif

# Template generation
from .template import from_sequence

# Vocabulary sizes (for embedding layers)
from .biochemistry import NUM_ELEMENTS, NUM_RESIDUES, NUM_ATOMS

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
    "write_cif",
    "from_sequence",
    "rmsd",
    # Vocabulary sizes
    "NUM_ELEMENTS",
    "NUM_RESIDUES",
    "NUM_ATOMS",
    # Convenience aliases
    "RESIDUE",
    "CHAIN",
    "MOLECULE",
    "PROTEIN",
    "RNA",
    "DNA",
]
