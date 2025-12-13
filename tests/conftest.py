"""
Pytest configuration and fixtures for ciffy tests.

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

    if not filepath.exists():
        url = PDB_URL.format(pdb_id=pdb_id)
        print(f"Downloading {pdb_id}.cif from RCSB PDB...")

        for attempt in range(MAX_RETRIES):
            try:
                urllib.request.urlretrieve(url, filepath)
                return filepath  # Success - return immediately
            except urllib.error.HTTPError as e:
                if e.code in (502, 503, 504) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"  HTTP {e.code}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                # Non-retryable error or last attempt
                _failed_downloads.add(pdb_id)
                pytest.skip(f"RCSB PDB unavailable (HTTP {e.code}): {pdb_id}")
            except urllib.error.URLError as e:
                _failed_downloads.add(pdb_id)
                pytest.skip(f"Network error downloading {pdb_id}: {e}")

        # All retries exhausted
        _failed_downloads.add(pdb_id)
        pytest.skip(f"RCSB PDB unavailable after {MAX_RETRIES} retries: {pdb_id}")

    return filepath


def get_test_cif(pdb_id: str) -> str:
    """Get path to a test CIF file, downloading if necessary."""
    return str(_download_cif(pdb_id))


# =============================================================================
# Parametrized fixtures for generic tests
# =============================================================================

@pytest.fixture(scope="session", params=TEST_PDBS)
def any_cif(request) -> str:
    """Parametrized fixture that runs tests on all standard test PDBs."""
    return get_test_cif(request.param)


@pytest.fixture(scope="session", params=TEST_PDBS)
def any_polymer_numpy(request):
    """Parametrized fixture providing polymers with numpy backend."""
    from ciffy import load
    return load(get_test_cif(request.param), backend="numpy")


@pytest.fixture(scope="session", params=TEST_PDBS)
def any_polymer_torch(request):
    """Parametrized fixture providing polymers with torch backend."""
    from ciffy import load
    return load(get_test_cif(request.param), backend="torch")


# =============================================================================
# Named fixtures for specific structures
# =============================================================================

@pytest.fixture(scope="session")
def cif_3skw() -> str:
    """Path to 3SKW.cif (RNA + ligands + ions)."""
    return get_test_cif("3SKW")


@pytest.fixture(scope="session")
def cif_9gcm() -> str:
    """Path to 9GCM.cif (RNA-protein complex)."""
    return get_test_cif("9GCM")


@pytest.fixture(scope="session")
def cif_9mds() -> str:
    """Path to 9MDS.cif (large ribosome structure)."""
    return get_test_cif("9MDS")


# =============================================================================
# Synthetic polymer fixtures for edge case testing
# =============================================================================

@pytest.fixture(params=["numpy", "torch"])
def backend(request) -> str:
    """Parametrized backend fixture."""
    return request.param


@pytest.fixture
def empty_polymer(backend):
    """Polymer with 0 atoms (via impossible mask)."""
    from ciffy import from_sequence
    template = from_sequence("a", backend=backend)
    return template[template.atoms < 0]


@pytest.fixture
def single_atom_polymer(backend):
    """Polymer with exactly 1 atom."""
    from ciffy import from_sequence
    template = from_sequence("g", backend=backend)  # Glycine has few atoms
    return template[:1]


@pytest.fixture
def single_residue_polymer(backend):
    """Polymer with 1 residue (multiple atoms)."""
    from ciffy import from_sequence
    return from_sequence("a", backend=backend)


@pytest.fixture
def single_chain_polymer(backend):
    """Polymer with 1 chain, multiple residues."""
    from ciffy import from_sequence
    return from_sequence("acgu", backend=backend)


@pytest.fixture
def multi_chain_polymer(backend):
    """Polymer loaded from CIF with multiple chains."""
    from ciffy import load
    return load(get_test_cif("9GCM"), backend=backend)
