# Getting Started

## Installation

Install ciffy using pip:

```bash
pip install ciffy
```

For PyTorch support:

```bash
pip install torch
```

## Quick Reference

| Task | Code |
|------|------|
| Load structure | `polymer = ciffy.load("file.cif")` |
| Get RNA only | `rna = polymer.by_type(ciffy.RNA)` |
| Get backbone | `backbone = polymer.backbone()` |
| Compute RMSD | `rmsd = ciffy.rmsd(p1, p2)` |
| Move to GPU | `polymer = polymer.to("cuda")` |
| Per-residue mean | `polymer.reduce(features, ciffy.RESIDUE)` |

## Loading Structures

```python
import ciffy

# Load with NumPy backend (default)
polymer = ciffy.load("structure.cif")

# Load with PyTorch backend
polymer = ciffy.load("structure.cif", backend="torch")

# Load with entity descriptions
polymer = ciffy.load("structure.cif", load_descriptions=True)
print(polymer.descriptions)  # ['RNA (66-MER)', 'CESIUM ION', ...]
```

## The Polymer Object

The `Polymer` class represents a molecular structure:

```python
polymer = ciffy.load("structure.cif")

# Access coordinates and atom data
coords = polymer.coordinates  # (N, 3) array of positions
atoms = polymer.atoms         # (N,) atom type indices
elements = polymer.elements   # (N,) element indices

# Structure info
print(polymer.size())              # Total atoms
print(polymer.size(ciffy.CHAIN))   # Number of chains
print(polymer.size(ciffy.RESIDUE)) # Number of residues
print(polymer.names)               # Chain names: ['A', 'B', ...]
```

## Filtering Structures

```python
# By molecule type
rna = polymer.by_type(ciffy.RNA)
protein = polymer.by_type(ciffy.PROTEIN)

# Polymer vs non-polymer
polymer_only = polymer.poly()      # Excludes water, ions, ligands
hetero = polymer.hetero()          # Only water, ions, ligands

# By chain
chain_a = polymer.by_index(0)      # First chain
chains = polymer.by_index([0, 2])  # Multiple chains

# Backbone atoms
backbone = polymer.backbone()
```

See the [Selection Guide](guides/selection.md) for advanced filtering.

## Hierarchical Operations

ciffy supports operations at different scales:

```python
# Reduce: aggregate atoms to coarser scales
chain_centroids = polymer.reduce(polymer.coordinates, ciffy.CHAIN)
residue_means = polymer.reduce(features, ciffy.RESIDUE)

# Expand: broadcast from coarse to fine scales
atom_features = polymer.expand(chain_data, ciffy.CHAIN)
```

See the [Analysis Guide](guides/analysis.md) for more operations.

## Computing RMSD

```python
# RMSD with Kabsch alignment
rmsd = ciffy.rmsd(polymer1, polymer2)

# Per-chain RMSD
per_chain = ciffy.rmsd(polymer1, polymer2, scale=ciffy.CHAIN)
```

## GPU Support (PyTorch)

```python
polymer = ciffy.load("structure.cif", backend="torch")

# Move to GPU
polymer_gpu = polymer.to("cuda")

# Mixed precision
polymer_fp16 = polymer.to(dtype=torch.float16)
```

See the [Deep Learning Guide](guides/deep-learning.md) for training workflows.

## Writing CIF Files

```python
polymer.write("output.cif")
```

## CLI Usage

```bash
# View structure summary
ciffy structure.cif

# Show entity descriptions
ciffy structure.cif --desc
```

## Next Steps

- [Selection Guide](guides/selection.md) - Molecule types, atom filtering, chain selection
- [Analysis Guide](guides/analysis.md) - RMSD, alignment, distances, reductions
- [Deep Learning Guide](guides/deep-learning.md) - PyTorch, GPU, embeddings
- [API Reference](api.md) - Complete API documentation
