"""
ciffy - Fast CIF file parsing for molecular structures.

A Python package for loading and manipulating molecular structures from
CIF (Crystallographic Information File) format files.

Primary API:
    load(path)              Load structure from CIF file
    Polymer                 Main structure class
    Scale                   Hierarchy levels (ATOM, RESIDUE, CHAIN, MOLECULE)
    Molecule                Molecule types (PROTEIN, RNA, DNA, ...)
    Residue                 Residue types with atom accessors

Operations:
    rmsd(p1, p2, scale)     Kabsch-aligned RMSD
    align(p1, p2, scale)    Align structures
    tm_score(pred, ref)     TM-score
    lddt(pred, ref)         lDDT score
    join(*polymers)         Combine polymers

Submodules:
    ciffy.biochemistry      Full biochemistry constants and types
    ciffy.nn                Neural network utilities (requires PyTorch)
    ciffy.flow              High-level flow model API
    ciffy.visualize         Visualization utilities
    ciffy.operations        All operations (alignment, metrics, reduction)
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
from .biochemistry import Scale, Molecule, Residue

# Primary I/O
from .io.loader import load, load_metadata
from .io.writer import write_cif

# Template generation
from .template import from_sequence, from_extract

# Ensemble for conformational analysis
from .ensemble import Ensemble

# Operations - commonly used, re-exported at top level
from .operations.alignment import kabsch_distance as rmsd, align
from .operations.metrics import tm_score, lddt
from .operations.chain import join
from .operations.reduction import Reduction

# Submodules (lazy-ish - imported but not used directly)
from . import biochemistry
from . import operations
from . import nn
from . import flow
from . import visualize

# Visualization convenience functions
from .visualize import to_defattr, plot_profile, contact_map

# Expose profiling function if available (when built with CIFFY_PROFILE=1)
try:
    from ._c import _get_profile
except (ImportError, AttributeError):
    pass  # Profiling not enabled in this build

# Convenience aliases - these are commonly used so we keep them
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
    "Residue",
    # I/O
    "load",
    "load_metadata",
    "write_cif",
    # Template
    "from_sequence",
    "from_extract",
    # Ensemble
    "Ensemble",
    # Operations
    "rmsd",
    "align",
    "tm_score",
    "lddt",
    "join",
    "Reduction",
    # Submodules
    "biochemistry",
    "operations",
    "nn",
    "flow",
    "visualize",
    # Visualization
    "to_defattr",
    "plot_profile",
    "contact_map",
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
]
