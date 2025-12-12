# Getting Started

## Installation

Install ciffy using pip:

```bash
pip install ciffy
```

For PyTorch support, ensure you have PyTorch installed:

```bash
pip install torch
```

## Loading Structures

Load a CIF file with the `load` function:

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

## Understanding the Polymer Object

The `Polymer` class represents a molecular structure with multiple scales:

```python
polymer = ciffy.load("structure.cif")

# Access coordinates and atom data
coords = polymer.coordinates  # (N, 3) array of positions
atoms = polymer.atoms         # (N,) atom type indices
elements = polymer.elements   # (N,) element indices

# Get structure info
print(polymer.size())                    # Total atoms
print(polymer.size(ciffy.CHAIN))         # Number of chains
print(polymer.size(ciffy.RESIDUE))       # Number of residues
```

## Hierarchical Operations

ciffy supports operations at different scales:

```python
# Reduce: aggregate atoms to coarser scales
centroids = polymer.reduce(polymer.coordinates, ciffy.CHAIN)  # Per-chain centroids
residue_means = polymer.reduce(features, ciffy.RESIDUE)       # Per-residue means

# Expand: broadcast from coarse to fine scales
chain_features = polymer.expand(per_chain_data, ciffy.CHAIN)  # Repeat per atom
```

## Filtering Structures

Select subsets of the structure:

```python
# By molecule type
rna = polymer.subset(ciffy.RNA)
protein = polymer.subset(ciffy.PROTEIN)

# Polymer vs non-polymer
polymer_only = polymer.poly()      # Excludes water, ions, ligands
hetero = polymer.hetero()          # Only water, ions, ligands

# By chain
chain_a = polymer.select(0)        # First chain
chains = polymer.select([0, 2])    # Multiple chains
```

## Computing RMSD

Compute root-mean-square deviation between aligned structures:

```python
# RMSD with Kabsch alignment (default: molecule scale)
rmsd = ciffy.rmsd(polymer1, polymer2)

# Per-chain RMSD
rmsd_per_chain = ciffy.rmsd(polymer1, polymer2, scale=ciffy.CHAIN)
```

## GPU Support

Move structures to GPU (PyTorch backend only):

```python
polymer = ciffy.load("structure.cif", backend="torch")

# Move to GPU
polymer_gpu = polymer.to("cuda")

# Change precision
polymer_fp16 = polymer.to(dtype=torch.float16)
```

## Writing CIF Files

Save structures back to CIF format:

```python
polymer.write("output.cif")

# Or use the function directly
ciffy.write_cif(polymer, "output.cif")
```

## CLI Usage

ciffy includes a command-line interface:

```bash
# View structure summary
ciffy structure.cif

# Show entity descriptions
ciffy structure.cif --desc
```
