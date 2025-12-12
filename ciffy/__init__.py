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
from .operations.metrics import tm_score, lddt

# I/O
from .io.loader import load, load_metadata
from .io.writer import write_cif

# Template generation
from .template import from_sequence

# Vocabulary sizes (for embedding layers)
from .biochemistry import NUM_ELEMENTS, NUM_RESIDUES, NUM_ATOMS

# Neural network utilities (requires PyTorch)
from . import nn

# Expose profiling function if available (when built with CIFFY_PROFILE=1)
try:
    from ._c import _get_profile
except (ImportError, AttributeError):
    pass  # Profiling not enabled in this build

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
    "load_metadata",
    "write_cif",
    "from_sequence",
    "rmsd",
    "tm_score",
    "lddt",
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
    # Submodules
    "nn",
]
