"""
Shared test utilities for ciffy tests.

Downloads test CIF files from RCSB PDB on demand.
Provides common constants, helpers, and fixtures.
"""

from pathlib import Path

import numpy as np
import pytest

# =============================================================================
# PyTorch availability check (centralized)
# =============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# =============================================================================
# Backend configuration
# =============================================================================

# Available backends for parametrized tests
BACKENDS = ["numpy", "torch"]

# Test PDB IDs - add new structures here to include them in generic tests
TEST_PDBS = ["3SKW", "9GCM"]

# Large structures (excluded from parametrized tests by default for speed)
LARGE_PDBS = ["9MDS", "8CAM"]

DATA_DIR = Path(__file__).parent / "data"


def get_test_cif(pdb_id: str) -> str:
    """Get path to a test CIF file, downloading if necessary.

    Uses ciffy.datasets.pdb for downloading. Raises on failure.
    """
    from ciffy.datasets.pdb import download_structure

    pdb_id = pdb_id.upper()
    filepath = DATA_DIR / f"{pdb_id}.cif"

    # Return cached file if it exists
    if filepath.is_file() and filepath.stat().st_size > 0:
        return str(filepath)

    # Download using centralized function
    print(f"Downloading {pdb_id}.cif from RCSB PDB...", flush=True)
    path, _ = download_structure(pdb_id, DATA_DIR, overwrite=False)

    return str(path or filepath)


# =============================================================================
# Test helpers
# =============================================================================

def skip_if_no_torch(backend: str) -> None:
    """Skip test if backend is 'torch' but PyTorch is not available.

    Use at the start of parametrized test methods:
        def test_something(self, backend):
            skip_if_no_torch(backend)
            ...
    """
    if backend == "torch" and not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")


def random_coordinates(n: int, backend: str, scale: float = 10.0):
    """Generate random coordinates for testing.

    Args:
        n: Number of points (atoms)
        backend: "numpy" or "torch"
        scale: Scale factor for coordinates (default 10.0 angstroms)

    Returns:
        Array/tensor of shape (n, 3) with random coordinates
    """
    coords = np.random.randn(n, 3).astype(np.float32) * scale
    if backend == "torch":
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        return torch.from_numpy(coords)
    return coords


def set_random_coordinates(polymer, scale: float = 10.0) -> None:
    """Set random coordinates on a polymer (in-place).

    Args:
        polymer: A ciffy.Polymer instance
        scale: Scale factor for coordinates
    """
    coords = random_coordinates(polymer.size(), polymer.backend, scale)
    polymer.coordinates = coords


# =============================================================================
# Device configuration (centralized for all test files)
# =============================================================================

def cuda_available() -> bool:
    """Check if PyTorch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def mps_available() -> bool:
    """Check if PyTorch MPS is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


# Available devices for parametrized tests
DEVICES = ["cpu"]
if cuda_available():
    DEVICES.append("cuda")
if mps_available():
    DEVICES.append("mps")

# GPU devices only (for tests that require acceleration)
GPU_DEVICES = [d for d in DEVICES if d != "cpu"]

# Skip markers (centralized)
requires_cuda = pytest.mark.skipif(
    not cuda_available(), reason="CUDA not available"
)
requires_mps = pytest.mark.skipif(
    not mps_available(), reason="MPS not available"
)
requires_gpu = pytest.mark.skipif(
    len(GPU_DEVICES) == 0, reason="No GPU available"
)


def skip_if_no_device(device: str) -> None:
    """Skip test if specified device is not available.

    Use at the start of parametrized test methods:
        def test_something(self, device):
            skip_if_no_device(device)
            ...
    """
    if device == "cuda" and not cuda_available():
        pytest.skip("CUDA not available")
    elif device == "mps" and not mps_available():
        pytest.skip("MPS not available")


# =============================================================================
# Test polymer helpers
# =============================================================================

def get_single_chain_poly(backend: str = "numpy", sequence: str = "acgu"):
    """Get a small single-chain polymer with random coordinates.

    Use this for tests that need a polymer with coordinates but aren't
    specifically testing _template generation.

    Args:
        backend: "numpy" or "torch"
        sequence: Sequence string (default "acgu" for 4-residue RNA)

    Returns:
        Polymer with random coordinates, single chain.
    """
    import ciffy

    _template = ciffy.template(sequence, backend=backend)
    coords = random_coordinates(_template.size(), backend, scale=10.0)
    return _template.copy(coordinates=coords)
