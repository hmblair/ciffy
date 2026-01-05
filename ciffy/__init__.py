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
    rg(p)                   Radius of gyration
    join(*polymers)         Combine polymers

Submodules:
    ciffy.biochemistry      Full biochemistry constants and types
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
from .polymer import Polymer, Field, Metadata, from_sequence, from_extract
from .hetero import HeteroAtoms
from .biochemistry import Scale, Molecule, Residue
from .backend import Dtype

# Primary I/O
from .io.loader import load, load_metadata
from .io.writer import write_cif

# Template generation (re-exported from polymer module above)

# Ensemble for conformational analysis
from .ensemble import Ensemble

# Operations - commonly used, re-exported at top level
from .operations.alignment import align, intersect
from .operations.metrics import rmsd, tm_score, lddt, rg, clashes
from .operations.chain import join
from .operations.reduction import Reduction

# Submodules (lazy-ish - imported but not used directly)
from . import biochemistry
from . import operations
from . import visualize

# Optional submodules (require additional dependencies)
# nn and flow require PyTorch and are not included in the PyPI distribution
try:
    from . import nn
    from . import flow
except ImportError:
    nn = None
    flow = None

# Visualization convenience functions
from .visualize import to_defattr, plot_profile, contact_map

# Reactivity data utilities
from .reactivity import ReactivityIndex, ReactivityMatch

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
    "HeteroAtoms",
    "Field",
    "Scale",
    "Molecule",
    "Residue",
    "Dtype",
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
    "rg",
    "clashes",
    "join",
    "Reduction",
    # Submodules
    "biochemistry",
    "operations",
    "visualize",
    # Visualization
    "to_defattr",
    "plot_profile",
    "contact_map",
    # Reactivity
    "ReactivityIndex",
    "ReactivityMatch",
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
