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

# On macOS, import torch BEFORE loading the C extension to avoid OpenMP conflicts.
# Both ciffy and PyTorch bundle libomp, and loading them in the wrong order causes
# segfaults due to duplicate OpenMP runtime initialization.
# By importing torch first (if installed), we ensure its libomp is loaded first,
# and ciffy's @rpath linking will find and reuse it.
import sys
if sys.platform == 'darwin':
    try:
        import torch  # noqa: F401 - imported for side effect (loads libomp)
    except ImportError:
        pass  # torch not installed, no conflict possible

# Verify C extension is available (required for all operations)
try:
    from . import _c
except ImportError as e:
    raise ImportError(
        "ciffy requires the C extension. Reinstall with: pip install ciffy --force-reinstall"
    ) from e

# Core types
from .polymer import Polymer
from .types import (
    Scale, Molecule, DihedralType,
    PROTEIN_BACKBONE, RNA_BACKBONE, RNA_GLYCOSIDIC,
    DIHEDRAL_ATOMS, DIHEDRAL_NAME_TO_TYPE,
)

# Operations
from .operations.reduction import Reduction
from .operations.alignment import kabsch_distance as rmsd, kabsch_rotation, kabsch_align, align
from .operations.metrics import tm_score, lddt

# I/O
from .io.loader import load, load_metadata
from .io.writer import write_cif

# Template generation
from .template import from_sequence, from_extract

# Sampling utilities
from .sampling import randomize_backbone

# Ensemble for conformational analysis
from .ensemble import Ensemble

# Vocabulary sizes (for embedding layers)
from .biochemistry import NUM_ELEMENTS, NUM_RESIDUES, NUM_ATOMS

# Neural network utilities (requires PyTorch)
from . import nn

# Visualization utilities
from . import visualize
from .visualize import to_defattr, plot_profile, contact_map

# Expose profiling function if available (when built with CIFFY_PROFILE=1)
try:
    from ._c import _get_profile
except (ImportError, AttributeError):
    pass  # Profiling not enabled in this build

# Convenience aliases
ATOM = Scale.ATOM
RESIDUE = Scale.RESIDUE
CHAIN = Scale.CHAIN
MOLECULE = Scale.MOLECULE

PROTEIN = Molecule.PROTEIN
RNA = Molecule.RNA
DNA = Molecule.DNA
LIGAND = Molecule.LIGAND
ION = Molecule.ION
WATER = Molecule.WATER

__all__ = [
    # Version
    "__version__",
    # Core types
    "Polymer",
    "Scale",
    "Molecule",
    "DihedralType",
    "PROTEIN_BACKBONE",
    "RNA_BACKBONE",
    "RNA_GLYCOSIDIC",
    "DIHEDRAL_ATOMS",
    "DIHEDRAL_NAME_TO_TYPE",
    "Reduction",
    # Functions
    "load",
    "load_metadata",
    "write_cif",
    "from_sequence",
    "from_extract",
    "randomize_backbone",
    "Ensemble",
    "rmsd",
    "kabsch_rotation",
    "kabsch_align",
    "align",
    "tm_score",
    "lddt",
    # Vocabulary sizes
    "NUM_ELEMENTS",
    "NUM_RESIDUES",
    "NUM_ATOMS",
    # Convenience aliases
    "ATOM",
    "RESIDUE",
    "CHAIN",
    "MOLECULE",
    "PROTEIN",
    "RNA",
    "DNA",
    "LIGAND",
    "ION",
    "WATER",
    # Submodules
    "nn",
    "visualize",
    # Visualization functions
    "to_defattr",
    "plot_profile",
    "contact_map",
]
