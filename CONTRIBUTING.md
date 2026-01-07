# Contributing to ciffy

## Development Setup

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy
pip install -e .
```

**Prerequisites:** Python 3.9+, C compiler, gperf 3.1+

## Running Tests

```bash
pytest tests/
pytest tests/ -v              # verbose
pytest tests/test_loader.py   # specific file
```

## Repository Structure

- **`ciffy/`** - Main Python package
  - `polymer/` - Core Polymer class and hierarchy operations
  - `biochemistry/` - Chemical definitions (auto-generated from CCD)
  - `backend/` - NumPy/PyTorch abstraction layer
  - `operations/` - Geometric operations (alignment, RMSD, etc.)
  - `io/` - CIF file reading and writing
  - `src/` - C source code for fast parsing
- **`codegen/`** - Code generation from PDB Chemical Component Dictionary
- **`tests/`** - Test suite

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
