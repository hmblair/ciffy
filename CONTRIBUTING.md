# Contributing to ciffy

## Development Setup

### Prerequisites

- Python 3.9+
- C compiler (gcc, clang, or MSVC)
- [gperf](https://www.gnu.org/software/gperf/) 3.1+ (for code generation)
- NumPy

**Optional for CUDA support:**
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- CUDA toolkit

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

# Optional: Install CUDA extension
pip install -e ./cuda
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

```
ciffy/
├── ciffy/                  # Main Python package
│   ├── __init__.py
│   ├── polymer.py          # Core Polymer class
│   ├── backend/            # NumPy/PyTorch abstraction layer
│   │   ├── numpy_ops.py    # NumPy implementations
│   │   ├── torch_ops.py    # PyTorch implementations
│   │   ├── cuda_ops.py     # CUDA kernel wrappers
│   │   └── dispatch.py     # Backend selection logic
│   ├── types/              # Enums and type definitions
│   │   ├── scale.py        # Scale enum (ATOM, RESIDUE, CHAIN, MOLECULE)
│   │   ├── molecule.py     # Molecule enum (auto-generated)
│   │   └── dihedral.py     # DihedralType enum
│   ├── biochemistry/       # Chemical definitions (mostly auto-generated)
│   │   ├── _generated_*.py # Auto-generated from CCD
│   │   └── ...
│   ├── operations/         # Geometric operations
│   │   ├── alignment.py    # Kabsch alignment, RMSD
│   │   └── reduction.py    # Aggregation operations
│   ├── io/                 # File I/O
│   │   ├── loader.py       # CIF loading
│   │   └── writer.py       # CIF writing
│   ├── internal/           # Internal coordinates (Python)
│   ├── nn/                 # Neural network utilities
│   ├── visualize/          # Visualization tools
│   └── src/                # C source code
│       ├── module.c        # Python C extension entry point
│       ├── cif/            # CIF parsing (C)
│       ├── internal/       # Internal coordinates (C + CUDA)
│       │   ├── geometry.c  # CPU geometry calculations
│       │   ├── batch.c     # CPU batch operations
│       │   ├── batch.cu    # CUDA batch operations
│       │   └── cuda_module.cu  # CUDA Python bindings
│       └── hash/           # Hash tables (auto-generated)
│           ├── *.gperf     # gperf input files (auto-generated)
│           └── *.c         # gperf output files (auto-generated)
│
├── codegen/                # Code generation from PDB CCD
│   ├── __init__.py         # Main entry point (generate_all)
│   ├── generate.py         # CLI entry point
│   ├── config.py           # Constants, residue whitelist
│   ├── ccd.py              # CCD file parsing
│   ├── c_codegen.py        # C/gperf code generation
│   ├── python_codegen.py   # Python enum generation
│   └── residue.py          # Residue definition class
│
├── cuda/                   # CUDA extension package (ciffy-cuda)
│   ├── pyproject.toml      # Package metadata
│   ├── setup.py            # CUDA build script
│   └── ciffy_cuda/         # Placeholder package
│
├── tests/                  # Test suite
├── docs/                   # Documentation (MkDocs)
├── examples/               # Example scripts
│
├── setup.py                # C extension build
├── pyproject.toml          # Package metadata
└── setup_cuda.py           # Legacy CUDA build wrapper (deprecated)
```

## Code Generation

Most biochemistry definitions are auto-generated from the [PDB Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (CCD). This runs automatically during `pip install`.

### Generated Files

```
ciffy/biochemistry/_generated_atoms.py      # Atom indices per residue
ciffy/biochemistry/_generated_elements.py   # Element enum
ciffy/biochemistry/_generated_residues.py   # Residue enum and mappings
ciffy/biochemistry/_generated_dihedrals.py  # Dihedral angle definitions
ciffy/biochemistry/_generated_zmatrix.py    # Z-matrix reference tables
ciffy/types/molecule.py                     # Molecule type enum
ciffy/src/hash/*.gperf                      # gperf input files
ciffy/src/hash/*.c                          # gperf output (hash tables)
ciffy/src/hash/reverse.h                    # Reverse lookup tables
ciffy/src/internal/bond_patterns.h          # Bond connectivity
ciffy/src/internal/canonical_refs.h         # Z-matrix references
```

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
| `cuda/pyproject.toml` | ciffy-cuda package metadata |
| `cuda/setup.py` | CUDA extension build (PyTorch BuildExtension) |

### Environment Variables

**C extension (`setup.py`):**
- `CIFFY_NO_OPENMP=1` - Disable OpenMP (single-threaded builds)
- `CIFFY_PROFILE=1` - Enable profiling instrumentation
- `CIFFY_CCD_PATH` - Custom path to CCD file

**CUDA extension (`cuda/setup.py`):**
- `CIFFY_CUDA_ARCH` - Target GPU architectures (e.g., `"86"` or `"70,75,80,86"`)
- `CIFFY_CUDA_DEBUG=1` - Debug build with symbols

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

## Testing CUDA

```bash
# Check CUDA extension is loaded
python -c "from ciffy.backend.cuda_ops import HAS_CUDA_EXTENSION; print(f'CUDA: {HAS_CUDA_EXTENSION}')"

# Run tests on GPU (requires CUDA-capable PyTorch)
pytest tests/ -v -k "cuda or gpu"
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
