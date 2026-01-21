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
  - `utils/` - Clustering and splitting utilities
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

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes and run tests
4. Submit a pull request

For significant changes, please open an issue first to discuss.
