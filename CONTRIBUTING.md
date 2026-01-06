# Contributing to ciffy

## Development Setup

### Prerequisites

- Python 3.9+
- C compiler (gcc, clang, or MSVC)
- [gperf](https://www.gnu.org/software/gperf/) 3.1+ (for code generation)
- NumPy

### Installing gperf

```bash
# macOS
brew install gperf

# Ubuntu/Debian
sudo apt install gperf

# Fedora/RHEL
sudo dnf install gperf

# Conda
conda install -c conda-forge gperf
```

### Development Install

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy

# Install base package in editable mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Skip profiling tests
pytest tests/ --ignore=tests/profile.py

# Run specific test file
pytest tests/test_loader.py
```

## Repository Structure

- **`ciffy/`** - Main Python package
  - `polymer/` - Core Polymer class and hierarchy operations
  - `biochemistry/` - Chemical definitions (mostly auto-generated from CCD)
  - `backend/` - NumPy/PyTorch abstraction layer
  - `operations/` - Geometric operations (alignment, RMSD, etc.)
  - `nn/` - Neural network utilities (datasets, embeddings, models)
  - `io/` - CIF file reading and writing
  - `visualize/` - ChimeraX visualization tools
  - `src/` - C source code for fast parsing
- **`codegen/`** - Code generation from PDB Chemical Component Dictionary
- **`tests/`** - Test suite
- **`docs/`** - Documentation (MkDocs)

## Code Generation

Most biochemistry definitions are auto-generated from the [PDB Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (CCD). This runs automatically during `pip install`.

### Manual Regeneration

```bash
# CCD is auto-downloaded to ~/.cache/ciffy/ on first run
python -m codegen.generate

# Use specific CCD file
python -m codegen.generate /path/to/components.cif

# Use custom gperf path
python -m codegen.generate --gperf-path /opt/homebrew/bin/gperf
```

### Adding New Residues

Edit `RESIDUE_WHITELIST` in `codegen/config.py`, then regenerate:

```python
RESIDUE_WHITELIST = {
    # Standard amino acids
    "ALA", "ARG", ...
    # Add your new residue
    "XYZ",
}
```

## Build System Overview

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, tool config |
| `setup.py` | C extension build (OpenMP detection, codegen integration) |

### Environment Variables

- `CIFFY_NO_OPENMP=1` - Disable OpenMP (single-threaded builds)
- `CIFFY_PROFILE=1` - Enable profiling instrumentation
- `CIFFY_CCD_PATH` - Custom path to CCD file

### Rebuilding After C Changes

When modifying C source files, reinstall to recompile:

```bash
pip install -e .
```

For faster iteration, you can use `ccache`:

```bash
# Install ccache
brew install ccache  # macOS
sudo apt install ccache  # Linux

# Set as compiler wrapper
export CC="ccache gcc"
pip install -e .
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where practical
- Keep C code consistent with existing style (K&R braces, 4-space indent)

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Submit a pull request

For significant changes, please open an issue first to discuss the approach.
