# Contributing to ciffy

## Development Setup

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy
pip install -e ".[dev]"
```

**Prerequisites:** Python 3.9+, C compiler, gperf 3.1+

## Running Tests

```bash
pytest tests/                    # all tests
pytest tests/ -n auto            # parallel execution
pytest tests/io/                 # specific module
pytest tests/ -k "test_load"     # by name pattern
```

## Repository Structure

- **`ciffy/`** - Main Python package
  - `polymer/` - Core Polymer class and hierarchy operations
  - `biochemistry/` - Chemical definitions (auto-generated from CCD)
  - `backend/` - NumPy/PyTorch abstraction layer
  - `operations/` - Geometric operations (alignment, RMSD, etc.)
  - `io/` - CIF file reading and writing
  - `cli/` - Command-line interface
  - `geometry/` - Geometric primitives (frames, transforms)
  - `nn/` - Neural network utilities (PolymerDataset, PolymerEmbedding)
  - `rna/` - RNA-specific utilities (reactivity, secondary structure)
  - `utils/` - Internal helper utilities
  - `visualize/` - Visualization tools
  - `src/` - C source code for fast parsing
- **`codegen/`** - Code generation from PDB Chemical Component Dictionary
- **`tests/`** - Test suite (organized by module)

## Code Generation

Biochemistry definitions are auto-generated from the [PDB Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) during `pip install -e .` (pre-generated files are included in PyPI releases).

To regenerate manually:

```bash
python -m codegen.generate
```

## Environment Variables

- `CIFFY_NO_OPENMP=1` - Disable OpenMP (single-threaded builds)
- `CIFFY_CCD_PATH` - Custom path to CCD file

## Writing Tests

Tests live in `tests/`, organized by module (e.g. `tests/polymer/`, `tests/io/`).

### Backend parametrization

Any test with a `backend` parameter is automatically run with both `"numpy"` and `"torch"` backends (torch tests are skipped if PyTorch is unavailable). You do not need to add `@pytest.mark.parametrize` for this:

```python
def test_something(self, backend):
    p = ciffy.template("acgu", backend=backend)
    assert p.size() > 0
```

### Fixtures

Use the fixtures in `conftest.py` instead of constructing polymers inline:

- **`rna_polymer`**, **`protein_polymer`**, **`dna_polymer`** - Single-chain templates (auto-parametrized by backend)
- **`multi_chain_polymer`** - Multi-chain CIF load (auto-parametrized by backend)
- **`any_cif`** - Parametrized over all test PDB files, yields CIF file paths
- **`any_polymer_numpy`**, **`any_polymer_torch`** - Loaded polymers for each test PDB
- **`make_polymer(sequence)`** - Factory fixture for custom sequences
- **`load_polymer(path)`** - Factory fixture for loading CIF files
- **`small_rna`**, **`small_protein`**, **`medium_rna`**, **`medium_protein`**, **`large_rna`**, **`large_protein`** - Various sizes

### Conventions

- Use enum values (`Residue.A.value`) instead of hardcoded integers.
- Place new test files in the subdirectory matching the module under test.
- Prefer the `any_cif` fixture for tests that should run against all test structures.
