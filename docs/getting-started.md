# Getting Started

This guide introduces ciffy's core concepts and walks through common workflows for loading, exploring, and saving molecular structures.

## Installation

Install ciffy from PyPI:

```bash
pip install ciffy
```

For deep learning workflows, install with PyTorch:

```bash
pip install ciffy torch
```

For visualization features:

```bash
pip install ciffy matplotlib
```

## Quick Reference

| Task | Code |
|------|------|
| Load structure | `polymer = ciffy.load("file.cif")` |
| Get RNA only | `rna = polymer.by_type(ciffy.RNA)` |
| Get backbone | `backbone = polymer.backbone()` |
| Compute RMSD | `rmsd = ciffy.rmsd(p1, p2)` |
| Superimpose structures | `ref, aligned = ciffy.align(p1, p2)` |
| Move to GPU | `polymer = polymer.to("cuda")` |
| Per-residue mean | `polymer.reduce(features, ciffy.RESIDUE)` |

## Loading Your First Structure

```python
import ciffy

# Load from local file
polymer = ciffy.load("structure.cif")

# Print summary
print(polymer)
```

Output:
```
Polymer 1ABC (numpy)
─────────────────────
   Type    Res  Atoms
─────────────────────
A  RNA      76   1629
B  PROTEIN  45    348
C  ION       -      2
─────────────────────
Σ          121   1979
─────────────────────
```

### Backend Options

```python
# Load with NumPy backend (default)
polymer = ciffy.load("structure.cif")

# Load with PyTorch backend
polymer = ciffy.load("structure.cif", backend="torch")

# Load with entity descriptions
polymer = ciffy.load("structure.cif", load_descriptions=True)
print(polymer.descriptions)  # ['RNA (66-MER)', 'CESIUM ION', ...]
```

## The Polymer Object

The `Polymer` is ciffy's central data structure. It holds:

- **Coordinates**: 3D positions of all atoms
- **Atom types**: What kind of atom (C1', N1, CA, etc.)
- **Elements**: Chemical element (C, N, O, P, etc.)
- **Sequence**: Residue types (A, G, C, U for RNA; ALA, GLY, etc. for protein)
- **Chain information**: Names, lengths, molecule types

```python
polymer = ciffy.load("structure.cif")

# Basic properties
print(f"PDB ID: {polymer.pdb_id}")
print(f"Total atoms: {polymer.size()}")
print(f"Chains: {polymer.names}")

# Access arrays
coords = polymer.coordinates      # (N, 3) float32
atoms = polymer.atoms             # (N,) int64 - atom type indices
elements = polymer.elements       # (N,) int64 - element indices
sequence = polymer.sequence       # (R,) int64 - residue type indices
```

## Hierarchical Structure

ciffy organizes structures hierarchically:

```
MOLECULE → CHAIN → RESIDUE → ATOM
```

Use `Scale` to specify which level you're working at:

```python
polymer = ciffy.load("structure.cif")

# Count at different scales
print(f"Atoms: {polymer.size(ciffy.ATOM)}")        # Same as polymer.size()
print(f"Residues: {polymer.size(ciffy.RESIDUE)}")
print(f"Chains: {polymer.size(ciffy.CHAIN)}")
```

### Atoms Per Unit

Get the number of atoms in each residue or chain:

```python
# Atoms per residue
atoms_per_res = polymer.sizes(ciffy.RESIDUE)
print(f"First residue has {atoms_per_res[0]} atoms")

# Atoms per chain
atoms_per_chain = polymer.sizes(ciffy.CHAIN)
for name, count in zip(polymer.names, atoms_per_chain):
    print(f"Chain {name}: {count} atoms")
```

## Filtering Structures

### By Molecule Type

```python
# Get only RNA chains
rna = polymer.by_type(ciffy.RNA)

# Get only protein chains
protein = polymer.by_type(ciffy.PROTEIN)

# Available types: RNA, DNA, PROTEIN, LIGAND, ION, WATER
```

### By Chain

```python
# First chain
chain_a = polymer.by_index(0)

# Multiple chains
chains_ab = polymer.by_index([0, 1])

# Iterate over chains
for chain in polymer.chains():
    print(f"{chain.names[0]}: {chain.size()} atoms")
```

### Polymer vs Heteroatoms

Separate polymer atoms from waters, ions, and ligands:

```python
# Only polymer atoms (RNA, DNA, protein)
polymer_only = polymer.poly()

# Only heteroatoms (water, ions, ligands)
hetero = polymer.hetero()

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

## Working with Coordinates

### Accessing Coordinates

```python
coords = polymer.coordinates  # Shape: (N, 3)

# Center of mass
com = coords.mean(axis=0)

# Bounding box
min_coords = coords.min(axis=0)
max_coords = coords.max(axis=0)
print(f"Size: {max_coords - min_coords}")
```

### Modifying Coordinates

```python
import numpy as np

# Translate the structure
translated_coords = polymer.coordinates + np.array([10.0, 0.0, 0.0])
translated = polymer.with_coordinates(translated_coords)

# Center at origin
centered, centroid = polymer.center(ciffy.MOLECULE)
```

## Computing RMSD

```python
# RMSD with Kabsch alignment
rmsd = ciffy.rmsd(polymer1, polymer2)

# Per-chain RMSD
per_chain = ciffy.rmsd(polymer1, polymer2, scale=ciffy.CHAIN)
```

## Sequence Information

```python
# One-letter sequence string
for chain in polymer.chains(ciffy.RNA):
    seq = chain.sequence_str()
    print(f"Chain {chain.names[0]}: {seq}")
# Output: Chain A: GCUAGCUAGCUA...
```

## Creating Structures from Sequences

Generate template polymers from sequence strings:

```python
# RNA (lowercase with u)
rna = ciffy.from_sequence("acgu")

# DNA (lowercase with t)
dna = ciffy.from_sequence("acgt")

# Protein (uppercase)
protein = ciffy.from_sequence("MGKLF")

# Multi-chain structures
multi = ciffy.from_sequence(["acgu", "MGKLF"])  # RNA + protein
```

Template polymers have correct atom types, elements, and residue sequences but zero coordinates. This is useful for generative modeling where coordinates are predicted separately.

## Saving Structures

```python
# Save to file
polymer.write("output.cif")

# Save a selection
rna_only = polymer.by_type(ciffy.RNA)
rna_only.write("rna_chains.cif")
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

## CLI Usage

```bash
# View structure summary
ciffy structure.cif

# Show entity descriptions
ciffy structure.cif --desc

# Show sequences
ciffy structure.cif --sequence
```

## Common Workflows

### Extracting a Clean Structure

```python
# Start with full structure
polymer = ciffy.load("structure.cif")

# Keep only polymer chains
clean = polymer.poly()

# Remove residues with missing atoms
clean = clean.strip(ciffy.RESIDUE)

# Save
clean.write("clean.cif")
```

### Comparing Two Structures

```python
p1 = ciffy.load("structure1.cif")
p2 = ciffy.load("structure2.cif")

# Compute RMSD
rmsd = ciffy.rmsd(p1, p2)
print(f"RMSD: {rmsd:.2f} Angstroms")

# Align structures
ref, aligned = ciffy.align(p1, p2)
aligned.write("aligned.cif")
```

### Per-Chain Analysis

```python
for chain in polymer.chains():
    name = chain.names[0]
    n_residues = chain.size(ciffy.RESIDUE)

    # Compute radius of gyration
    centered, _ = chain.center(ciffy.MOLECULE)
    coords = centered.coordinates
    rg = (coords ** 2).sum(axis=1).mean() ** 0.5

    print(f"Chain {name}: {n_residues} residues, Rg = {rg:.1f} A")
```

## Next Steps

- [Selection Guide](guides/selection.md) - Molecule types, atom filtering, chain selection
- [I/O Guide](guides/io.md) - Loading from URLs, metadata, writing files
- [Analysis Guide](guides/analysis.md) - RMSD, alignment, distances, reductions
- [Deep Learning Guide](guides/deep-learning.md) - PyTorch, GPU, embeddings
- [Visualization Guide](guides/visualization.md) - Plots and ChimeraX export
- [API Reference](api.md) - Complete API documentation
