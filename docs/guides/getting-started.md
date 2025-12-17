# Getting Started

This guide introduces ciffy's core concepts and walks through common workflows for loading, exploring, and saving molecular structures.

## Installation

Install ciffy from PyPI:

```bash
pip install ciffy
```

For deep learning workflows, install with PyTorch support:

```bash
pip install ciffy torch
```

For visualization features:

```bash
pip install ciffy matplotlib
```

## Loading Your First Structure

Load a structure from a CIF file:

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

The summary shows each chain with its molecule type, residue count, and atom count.

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
print(f"PDB ID: {polymer.id()}")
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
import ciffy

polymer = ciffy.load("structure.cif")

# Count at different scales
print(f"Atoms: {polymer.size(ciffy.ATOM)}")        # Same as polymer.size()
print(f"Residues: {polymer.size(ciffy.RESIDUE)}")
print(f"Chains: {polymer.size(ciffy.CHAIN)}")
print(f"Molecules: {polymer.size(ciffy.MOLECULE)}")  # Always 1
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

## Selecting Parts of a Structure

### By Molecule Type

Filter chains by their molecular type:

```python
# Get only RNA chains
rna = polymer.by_type(ciffy.RNA)

# Get only protein chains
protein = polymer.by_type(ciffy.PROTEIN)

# Available types: RNA, DNA, PROTEIN, LIGAND, ION, WATER
```

### By Chain

Select specific chains:

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

print(f"Polymer atoms: {polymer_only.size()}")
print(f"Heteroatoms: {hetero.size()}")
```

## Working with Coordinates

### Accessing Coordinates

```python
coords = polymer.coordinates  # Shape: (N, 3)

# Center of mass
com = coords.mean(axis=0)
print(f"Center of mass: {com}")

# Bounding box
min_coords = coords.min(axis=0)
max_coords = coords.max(axis=0)
print(f"Size: {max_coords - min_coords}")
```

### Modifying Coordinates

Create a new polymer with different coordinates:

```python
import numpy as np

# Translate the structure
translated_coords = polymer.coordinates + np.array([10.0, 0.0, 0.0])
translated = polymer.with_coordinates(translated_coords)

# Center at origin
centered, centroid = polymer.center(ciffy.MOLECULE)
print(f"Original center: {centroid}")
```

## Saving Structures

Write structures back to CIF format:

```python
# Save to file
polymer.write("output.cif")

# Save a selection
rna_only = polymer.by_type(ciffy.RNA)
rna_only.write("rna_chains.cif")
```

## Sequence Information

### Getting Sequences

```python
# One-letter sequence string
for chain in polymer.chains(ciffy.RNA):
    seq = chain.sequence_str()
    print(f"Chain {chain.names[0]}: {seq}")
# Output: Chain A: GCUAGCUAGCUA...
```

### Residue Types

```python
from ciffy.biochemistry import Residue

# Access residue information
sequence = polymer.sequence  # Integer indices

# Map to residue names
for i in range(min(5, len(sequence))):
    res_type = Residue(sequence[i])
    print(f"Residue {i}: {res_type.name}")
```

## Convenience Aliases

ciffy provides shortcuts for common operations:

```python
import ciffy

# Scale aliases
ciffy.ATOM      # Same as ciffy.Scale.ATOM
ciffy.RESIDUE   # Same as ciffy.Scale.RESIDUE
ciffy.CHAIN     # Same as ciffy.Scale.CHAIN
ciffy.MOLECULE  # Same as ciffy.Scale.MOLECULE

# Molecule type aliases
ciffy.RNA       # Same as ciffy.Molecule.RNA
ciffy.DNA       # Same as ciffy.Molecule.DNA
ciffy.PROTEIN   # Same as ciffy.Molecule.PROTEIN
ciffy.LIGAND    # Same as ciffy.Molecule.LIGAND
ciffy.ION       # Same as ciffy.Molecule.ION
ciffy.WATER     # Same as ciffy.Molecule.WATER
```

## Common Workflows

### Extracting a Clean Structure

Remove waters, ions, and unresolved residues:

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
rmsd = ciffy.rmsd(p1, p2).sqrt()
print(f"RMSD: {rmsd:.2f} Angstroms")

# Align structures
ref, aligned = ciffy.align(p1, p2)
aligned.write("aligned.cif")
```

### Per-Chain Analysis

```python
polymer = ciffy.load("structure.cif")

for chain in polymer.chains():
    # Get chain properties
    name = chain.names[0]
    n_atoms = chain.size()
    n_residues = chain.size(ciffy.RESIDUE)

    # Compute radius of gyration
    centered, _ = chain.center(ciffy.MOLECULE)
    coords = centered.coordinates
    rg = (coords ** 2).sum(axis=1).mean() ** 0.5

    print(f"Chain {name}: {n_residues} residues, Rg = {rg:.1f} A")
```

## Next Steps

- [Selection and Filtering](selection.md) - Advanced selection techniques
- [Structural Analysis](analysis.md) - RMSD, alignment, distances
- [Deep Learning](deep-learning.md) - PyTorch integration
- [Visualization](visualization.md) - Plots and ChimeraX export
