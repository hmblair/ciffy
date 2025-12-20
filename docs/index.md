# ciffy

**Fast CIF file parsing for molecular structures.**

ciffy is a Python package for loading and manipulating molecular structures from CIF (Crystallographic Information File) format files. It features a high-performance C parser and supports both NumPy and PyTorch backends.

## Features

- **Fast parsing** - C extension for high-performance CIF file loading
- **Dual backend** - Native support for NumPy and PyTorch arrays
- **Hierarchical data** - Work at atom, residue, chain, or molecule scale
- **GPU support** - Move structures to CUDA devices with `.to("cuda")`
- **Round-trip I/O** - Load and write CIF files

## Installation

```bash
pip install ciffy
```

## Quick Example

```python
import ciffy

# Load a structure
polymer = ciffy.load("structure.cif", backend="torch")
print(polymer)

# Work with coordinates
coords = polymer.coordinates  # (N, 3) tensor
centroids = polymer.reduce(coords, ciffy.CHAIN)  # Per-chain centroids

# Filter by molecule type
rna_only = polymer.by_type(ciffy.RNA)

# Compute RMSD between structures
rmsd = ciffy.rmsd(polymer1, polymer2)
```

## Links

- [Getting Started](getting-started.md) - Installation and basic usage
- [API Reference](api.md) - Complete API documentation
