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

# Sampling utilities (DEPRECATED - use ciffy.nn.flow.PolymerFlowModel instead)
from .sampling import randomize_backbone

# Ensemble for conformational analysis
from .ensemble import Ensemble

# Vocabulary sizes (for embedding layers)
from .biochemistry import NUM_ELEMENTS, NUM_RESIDUES, NUM_ATOMS

# Re-export Residue for common use cases (reduce imports needed)
from .biochemistry import Residue

# Neural network utilities (requires PyTorch)
from . import nn

# High-level flow API
from . import flow


def load_flow_model(name: str = "rna", device: str = "cpu") -> "nn.flow.PolymerFlowModel":
    """
    Load a pre-trained PolymerFlowModel.

    This is a convenience function that provides easy access to pre-trained
    flow models for polymer conformation generation.

    Args:
        name: Name of the pre-trained model ('rna' for RNA residues A, C, G, U).
        device: Device to load model to ('cpu' or 'cuda').

    Returns:
        PolymerFlowModel ready for encoding, decoding, and sampling.

    Raises:
        ValueError: If the model name is not recognized.
        FileNotFoundError: If the model files are not found.

    Example:
        >>> import ciffy
        >>>
        >>> # Load pre-trained RNA model
        >>> model = ciffy.load_flow_model("rna", device="cuda")
        >>>
        >>> # Encode a polymer's coordinates
        >>> polymer = ciffy.load("structure.cif")
        >>> latents = model.encode(polymer.coordinates, polymer.sequence)
        >>>
        >>> # Sample new conformations
        >>> samples = model.sample(polymer.sequence, n_samples=10)
    """
    from .nn.flow import load_pretrained
    return load_pretrained(name, device=device)

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
    "load_flow_model",
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
    # Common biochemistry types
    "Residue",
    # High-level flow API
    "flow",
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
