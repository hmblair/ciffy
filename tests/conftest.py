"""
Pytest configuration and fixtures for ciffy tests.

Downloads test CIF files from RCSB PDB on demand.

Auto-parametrization:
    Any test with a `backend` parameter (without explicit parametrize) will
    automatically run with both "numpy" and "torch" backends. Tests are skipped
    for torch if PyTorch is not available.

Polymer fixtures:
    Use `rna_polymer`, `protein_polymer`, `small_polymer`, `medium_polymer`
    for common test cases. These are parametrized by backend automatically.
"""

import inspect
import pytest

from tests.utils import (
    get_test_cif, TEST_PDBS, LARGE_PDBS, DATA_DIR,
    BACKENDS, TORCH_AVAILABLE, skip_if_no_torch,
)


# =============================================================================
# Auto-parametrization for backend argument
# =============================================================================

def pytest_generate_tests(metafunc):
    """Auto-parametrize tests that have a 'backend' argument.

    If a test function has a 'backend' parameter and doesn't already have
    an explicit parametrize marker for it, automatically parametrize with
    BACKENDS = ["numpy", "torch"].

    This eliminates the need for @pytest.mark.parametrize("backend", BACKENDS)
    on every test method.

    Note: Tests that use fixtures depending on `backend` (like `rna_polymer`)
    get parametrization from the `backend` fixture, not from this hook.
    """
    if "backend" in metafunc.fixturenames:
        # Check if backend is already parametrized via marker
        for marker in metafunc.definition.iter_markers("parametrize"):
            if marker.args and "backend" in str(marker.args[0]):
                return  # Already parametrized via marker, skip

        # Check if backend will be provided by a parametrized fixture
        # (the backend fixture has params=["numpy", "torch"])
        # We detect this by checking if backend is in the fixture manager
        try:
            fixturedef = metafunc._arg2fixturedefs.get("backend")
            if fixturedef and any(f.params for f in fixturedef):
                return  # Already parametrized via fixture, skip
        except (AttributeError, KeyError):
            pass  # Fallback: proceed with auto-parametrization

        # Auto-parametrize with backends
        metafunc.parametrize("backend", BACKENDS)


@pytest.fixture(autouse=True)
def skip_torch_if_unavailable(request):
    """Auto-skip torch backend tests if PyTorch is not available.

    This runs for every test. If the test has a 'backend' parameter
    set to 'torch' and torch is not available, skip the test.
    """
    # Check if this test has a backend parameter
    if "backend" in request.fixturenames:
        backend = request.getfixturevalue("backend")
        if backend == "torch" and not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")


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


# =============================================================================
# Testing infrastructure fixtures
# =============================================================================

@pytest.fixture
def tolerances():
    """Fixture providing tolerance profile for numerical comparisons."""
    from tests.testing import DEFAULT
    return DEFAULT


@pytest.fixture
def gpu_tolerances():
    """Fixture providing relaxed tolerances for GPU tests."""
    from tests.testing import GPU
    return GPU


@pytest.fixture
def strict_tolerances():
    """Fixture providing strict tolerances for precision tests."""
    from tests.testing import STRICT
    return STRICT


@pytest.fixture
def assert_roundtrip():
    """Fixture providing the roundtrip assertion function."""
    from tests.testing import assert_roundtrip_preserves_structure
    return assert_roundtrip_preserves_structure


@pytest.fixture
def assert_gradients():
    """Fixture providing the gradient flow assertion function."""
    from tests.testing import assert_gradient_flows
    return assert_gradient_flows


# =============================================================================
# Polymer factory fixtures
# =============================================================================
# These provide common test polymers without needing to call from_sequence
# directly. They're automatically parametrized by backend via the backend fixture.

@pytest.fixture
def rna_polymer(backend):
    """4-residue RNA polymer (acgu)."""
    from ciffy import from_sequence
    return from_sequence("acgu", backend=backend)


@pytest.fixture
def protein_polymer(backend):
    """5-residue protein polymer (MGKLF)."""
    from ciffy import from_sequence
    return from_sequence("MGKLF", backend=backend)


@pytest.fixture
def dna_polymer(backend):
    """4-residue DNA polymer (acgt)."""
    from ciffy import from_sequence
    return from_sequence("acgt", backend=backend)


@pytest.fixture
def small_rna(backend):
    """Single nucleotide RNA (a)."""
    from ciffy import from_sequence
    return from_sequence("a", backend=backend)


@pytest.fixture
def small_protein(backend):
    """Single amino acid protein (G - glycine)."""
    from ciffy import from_sequence
    return from_sequence("G", backend=backend)


@pytest.fixture
def medium_rna(backend):
    """8-residue RNA polymer."""
    from ciffy import from_sequence
    return from_sequence("acguacgu", backend=backend)


@pytest.fixture
def medium_protein(backend):
    """10-residue protein polymer."""
    from ciffy import from_sequence
    return from_sequence("MGKLFAGKLF", backend=backend)


# =============================================================================
# Polymer factory function (for custom sequences)
# =============================================================================

@pytest.fixture
def make_polymer(backend):
    """Factory fixture for creating polymers with custom sequences.

    Usage:
        def test_something(make_polymer):
            rna = make_polymer("acgu")
            protein = make_polymer("MGKLF")
    """
    from ciffy import from_sequence

    def _make(sequence: str):
        return from_sequence(sequence, backend=backend)

    return _make


@pytest.fixture
def load_polymer(backend):
    """Factory fixture for loading CIF files with specified backend.

    Usage:
        def test_something(load_polymer):
            polymer = load_polymer("3SKW")
    """
    from ciffy import load

    def _load(pdb_id: str):
        return load(get_test_cif(pdb_id), backend=backend)

    return _load


# =============================================================================
# Multi-chain and complex structure fixtures
# =============================================================================

@pytest.fixture
def multi_chain_rna(backend):
    """Multi-chain RNA (2 chains)."""
    from ciffy import from_sequence
    return from_sequence("acgu/acgu", backend=backend)


@pytest.fixture
def multi_chain_protein(backend):
    """Multi-chain protein (2 chains)."""
    from ciffy import from_sequence
    return from_sequence("MGKLF/ARNDCE", backend=backend)


@pytest.fixture
def rna_protein_complex(backend):
    """RNA + protein complex (9GCM)."""
    from ciffy import load
    return load(get_test_cif("9GCM"), backend=backend)


@pytest.fixture
def structure_with_ligands(backend):
    """Structure with ligands and ions (3SKW)."""
    from ciffy import load
    return load(get_test_cif("3SKW"), backend=backend)


# =============================================================================
# Larger structure fixtures for stress testing
# =============================================================================

@pytest.fixture
def large_rna(backend):
    """16-residue RNA for stress testing."""
    from ciffy import from_sequence
    return from_sequence("acguacguacguacgu", backend=backend)


@pytest.fixture
def large_protein(backend):
    """20-residue protein for stress testing."""
    from ciffy import from_sequence
    return from_sequence("MGKLFAGKLFMGKLFAGKLF", backend=backend)


# =============================================================================
# Specialized edge case fixtures
# =============================================================================

@pytest.fixture
def all_same_residue_rna(backend):
    """RNA with all same residues (edge case for reduction)."""
    from ciffy import from_sequence
    return from_sequence("aaaa", backend=backend)


@pytest.fixture
def all_same_residue_protein(backend):
    """Protein with all same residues (edge case for reduction)."""
    from ciffy import from_sequence
    return from_sequence("GGGG", backend=backend)
