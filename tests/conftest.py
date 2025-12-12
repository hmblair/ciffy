"""
Pytest configuration and fixtures for ciffy tests.

Downloads test CIF files from RCSB PDB on demand.
"""

import os
import urllib.request
from pathlib import Path

import pytest

# Test PDB IDs and their expected properties
TEST_PDBS = {
    "3SKW": {"atoms": 2874, "chains": 13},  # RNA + ligands + ions
    "9GCM": {"atoms": 4466, "chains": 4},   # RNA-protein complex
    "9MDS": {"atoms": 102216, "chains": 2}, # Large ribosome structure
}

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
    if pdb_id not in TEST_PDBS:
        raise ValueError(f"Unknown test PDB: {pdb_id}. Available: {list(TEST_PDBS.keys())}")
    return str(_download_cif(pdb_id))


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


@pytest.fixture(scope="session")
def all_test_cifs() -> dict[str, str]:
    """Dict of all test CIF paths, keyed by PDB ID."""
    return {pdb_id: get_test_cif(pdb_id) for pdb_id in TEST_PDBS}
