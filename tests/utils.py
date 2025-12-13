"""
Shared test utilities for ciffy tests.

Downloads test CIF files from RCSB PDB on demand.
"""

import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest

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
