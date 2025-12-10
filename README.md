## Overview

`ciffy` is a fast CIF file parser for molecular structures, with a C backend and Python interface. It supports both NumPy and PyTorch backends for array operations.

## Installation

### From PyPI

```bash
pip install ciffy
```

### From Source

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy
pip install -r requirements.txt
pip install -e .
```

## Backends

`ciffy` supports two array backends:

- **NumPy**: Lightweight, no additional dependencies required
- **PyTorch**: For GPU support and integration with deep learning workflows

Specify the backend when loading structures:

```python
import ciffy

# Load with NumPy backend (recommended for general use)
polymer = ciffy.load("structure.cif", backend="numpy")

# Load with PyTorch backend (for deep learning workflows)
polymer = ciffy.load("structure.cif", backend="torch")
```

Polymers can be converted between backends:

```python
# Convert to PyTorch tensors
torch_polymer = polymer.torch()

# Convert to NumPy arrays
numpy_polymer = polymer.numpy()
```

**Note:** The default backend will change from `"torch"` to `"numpy"` in v0.6.0. Specify the backend explicitly to avoid deprecation warnings.

## Usage

```python
import ciffy

# Load a structure from a CIF file
polymer = ciffy.load("structure.cif", backend="numpy")

# Basic information
print(polymer)  # Summary of chains, residues, atoms

# Access coordinates and properties
coords = polymer.coordinates      # (N, 3) array/tensor
atoms = polymer.atoms             # (N,) array/tensor of atom types
sequence = polymer.str()          # Sequence string

# Geometric operations
centered, means = polymer.center(ciffy.MOLECULE)
aligned, Q = polymer.align(ciffy.CHAIN)
distances = polymer.pd(ciffy.RESIDUE)

# Selection
rna_chains = polymer.subset(ciffy.RNA)
backbone = polymer.backbone()

# Iterate over chains
for chain in polymer.chains(ciffy.RNA):
    print(chain.id(), chain.str())

# Compute RMSD between structures
rmsd = ciffy.rmsd(polymer1, polymer2, ciffy.MOLECULE)
```

## Saving Structures

```python
# Save to CIF format (supports all molecule types)
polymer.write("output.cif")

# Save only polymer atoms (excludes water, ions, ligands)
polymer.poly().write("polymer_only.cif")
```

## Module Structure

```
ciffy/
├── backend/        # NumPy/PyTorch abstraction layer
├── types/          # Scale, Molecule enums
├── biochemistry/   # Element, Residue, nucleotide definitions
├── operations/     # Reduction, alignment operations
├── io/             # File loading and writing
└── utils/          # Helper functions and base classes
```

## Testing

```bash
pip install pytest
pytest tests/
```
