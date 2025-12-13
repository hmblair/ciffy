"""
Pytest configuration and fixtures for ciffy tests.

Downloads test CIF files from RCSB PDB on demand.
"""

import pytest

from tests.utils import get_test_cif, TEST_PDBS, LARGE_PDBS, DATA_DIR


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
