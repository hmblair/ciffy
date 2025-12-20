"""
Shared test utilities for ciffy tests.

Downloads test CIF files from RCSB PDB on demand.
Provides common constants, helpers, and fixtures.
"""

import time
import urllib.request
import urllib.error
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
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

# Retry settings for transient network errors
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Track PDBs that failed to download (skip future tests for these)
_failed_downloads: set[str] = set()


def _download_cif(pdb_id: str) -> Path:
    """Download a CIF file from RCSB PDB if not already cached.

    Includes retry logic for transient network errors (502, 503, etc.).
    Skips test if server is unavailable after retries.
    """
    # Skip if we already failed to download this PDB
    if pdb_id in _failed_downloads:
        pytest.skip(f"RCSB PDB previously unavailable: {pdb_id}")

    DATA_DIR.mkdir(exist_ok=True)
    filepath = DATA_DIR / f"{pdb_id}.cif"

    # Check if file exists AND is a valid file (not empty)
    if filepath.is_file() and filepath.stat().st_size > 0:
        return filepath

    # Remove any invalid file (empty or corrupted from failed download)
    if filepath.exists():
        filepath.unlink()

    # Need to download
    url = PDB_URL.format(pdb_id=pdb_id)
    print(f"Downloading {pdb_id}.cif from RCSB PDB...", flush=True)

    for attempt in range(MAX_RETRIES):
        try:
            urllib.request.urlretrieve(url, filepath)
            # Verify download produced a valid file
            if filepath.is_file() and filepath.stat().st_size > 0:
                return filepath  # Success
            # Download produced empty/invalid file
            print(f"  Download produced invalid file, retrying...", flush=True)
            if filepath.exists():
                filepath.unlink()
            continue
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504) and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"  HTTP {e.code}, retrying in {delay}s...")
                time.sleep(delay)
                continue
            # Non-retryable error or last attempt - skip test
            _failed_downloads.add(pdb_id)
            pytest.skip(f"RCSB PDB unavailable (HTTP {e.code}): {pdb_id}")
        except urllib.error.URLError as e:
            _failed_downloads.add(pdb_id)
            pytest.skip(f"Network error downloading {pdb_id}: {e}")

    # All retries exhausted or download produced no file
    _failed_downloads.add(pdb_id)
    pytest.skip(f"RCSB PDB unavailable after {MAX_RETRIES} retries: {pdb_id}")


def get_test_cif(pdb_id: str) -> str:
    """Get path to a test CIF file, downloading if necessary."""
    return str(_download_cif(pdb_id))


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


def cuda_extension_available() -> bool:
    """Check if ciffy CUDA extension is built."""
    try:
        from ciffy.backend.cuda_ops import CUDA_EXTENSION_AVAILABLE
        return CUDA_EXTENSION_AVAILABLE
    except ImportError:
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
requires_cuda_extension = pytest.mark.skipif(
    not cuda_extension_available(), reason="CUDA extension not built"
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
