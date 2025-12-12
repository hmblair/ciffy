"""
Pytest configuration and fixtures for ciffy tests.

Downloads test CIF files from RCSB PDB on demand.
"""

import urllib.request
from pathlib import Path

import pytest

# Test PDB IDs - add new structures here to include them in generic tests
TEST_PDBS = ["3SKW", "9GCM", "8CAM"]

# Large structures (excluded from parametrized tests by default for speed)
LARGE_PDBS = ["9MDS"]

DATA_DIR = Path(__file__).parent / "data"
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def _download_cif(pdb_id: str) -> Path:
    """Download a CIF file from RCSB PDB if not already cached."""
    DATA_DIR.mkdir(exist_ok=True)
    filepath = DATA_DIR / f"{pdb_id}.cif"

    if not filepath.exists():
        url = PDB_URL.format(pdb_id=pdb_id)
        print(f"Downloading {pdb_id}.cif from RCSB PDB...")
        urllib.request.urlretrieve(url, filepath)

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
