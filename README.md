## Overview

`ciffy` is a fast CIF file parser for molecular structures, with a C backend and Python interface. It supports both NumPy and PyTorch backends for array operations.

### Performance

ciffy is **50-90x faster** than BioPython and Biotite for parsing CIF files:

| Structure | Atoms | ciffy | BioPython | Biotite |
|-----------|------:|------:|----------:|--------:|
| 3SKW | 2,874 | 0.47 ms | 31 ms (66x) | 28 ms (59x) |
| 9GCM | 4,466 | 0.71 ms | 40 ms (56x) | 36 ms (51x) |
| 9MDS | 102,216 | 14 ms | 1266 ms (93x) | 911 ms (67x) |

<sub>Benchmarked on Apple M1 Max. Run `python tests/profile.py` to reproduce.</sub>

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
- **PyTorch**: For GPU support (CUDA/MPS) and integration with deep learning workflows

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

For PyTorch, move tensors to GPU:

```python
# Move to CUDA
polymer_gpu = polymer.torch().to("cuda")

# Move to Apple Silicon (MPS)
polymer_mps = polymer.torch().to("mps")
```

**Note:** The default backend is `"numpy"` as of v0.6.0. Specify the backend explicitly for clarity.

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

# Molecule type per chain (parsed from CIF _entity_poly block)
mol_types = polymer.molecule_type  # Array of Molecule enum values

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

## Command Line Interface

```bash
# View structure summary
ciffy structure.cif

# Show sequences per chain
ciffy structure.cif --sequence

# Multiple files
ciffy file1.cif file2.cif
```

Example output:
```
PDB 9GCM (numpy)
──────────────────────
   Type     Res  Atoms
A  RNA      135   1413
B  PROTEIN  132   1032
C  PROTEIN  246   1261
D  PROTEIN  485    760
──────────────────────
            998   4466
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
