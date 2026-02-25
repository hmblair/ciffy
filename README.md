## Overview

`ciffy` is a fast CIF file parser for molecular structures, with a C backend and Python interface. It supports both NumPy and PyTorch backends for array operations.

### Performance

ciffy is **55-115x faster** than BioPython and Biotite for parsing CIF files:

| Structure | Atoms | ciffy | BioPython | Biotite |
|-----------|------:|------:|----------:|--------:|
| 3SKW | 2,826 | 0.52 ms | 39 ms (75x) | 31 ms (59x) |
| 9GCM | 4,466 | 0.75 ms | 53 ms (71x) | 41 ms (55x) |
| 9MDS | 102,216 | 12 ms | 1326 ms (107x) | 1016 ms (82x) |

<sub>Run `python tests/profiling/profile_io.py` to reproduce.</sub>

## Installation

### From PyPI

```bash
pip install ciffy
```

### From Source

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy
pip install -e .
```

## Backends

`ciffy` supports two array backends:

- **NumPy**: Lightweight, no additional dependencies required
- **PyTorch**: For GPU support (CUDA/MPS) and integration with deep learning workflows

Specify the backend when loading structures:

```python
import ciffy

# Load with NumPy backend (default)
polymer = ciffy.load("structure.cif", backend="numpy")

# Load with PyTorch backend
polymer = ciffy.load("structure.cif", backend="torch")
```

Convert between backends:

```python
torch_polymer = polymer.torch()
numpy_polymer = polymer.numpy()
```

Move tensors to GPU (PyTorch only):

```python
polymer_gpu = polymer.torch().to("cuda")
polymer_mps = polymer.torch().to("mps")
```

## Loading Structures

```python
import ciffy

# Load from CIF file
polymer = ciffy.load("structure.cif")

# Load specific chains
polymer = ciffy.load("structure.cif", chains=["A", "B"])

# Load specific molecule types
polymer = ciffy.load("structure.cif", molecule_types=ciffy.RNA)

# Print summary
print(polymer)
```

Example output:
```
Polymer 9GCM [2024-08-02]
─────────────────────────
   Type     Res  Atoms
─────────────────────────
A  RNA      135   1413
B  PROTEIN  132   1032
C  PROTEIN  246   1261
D  PROTEIN  485    760
─────────────────────────
Σ  4        998   4466
─────────────────────────
```

## Working with Polymers

### Properties

```python
polymer.coordinates       # (N, 3) atom positions
polymer.atoms             # (N,) atom type indices
polymer.elements          # (N,) element indices
polymer.sequence          # (R,) residue type indices
polymer.bonds             # (B, 2) covalent bond pairs
polymer.molecule_types    # (C,) molecule type per chain
polymer.names             # Chain names ["A", "B", ...]
polymer.lengths           # (C,) residues per chain

polymer.size()                      # Total atoms
polymer.size(ciffy.RESIDUE)         # Total residues
polymer.size(ciffy.CHAIN)           # Total chains
polymer.sequence_str()              # "acgu..." sequence string
```

### Selection

```python
# Select by chain
chain_a = polymer.chain(0)
chains_ab = polymer.chain([0, 1])

# Select by residue
first_residue = polymer.residue(0)
some_residues = polymer.residue([0, 5, 10])

# Select by molecule type
rna_only = polymer.molecule_type(ciffy.RNA)
protein_only = polymer.molecule_type(ciffy.PROTEIN)

# Select by residue type
adenines = polymer.residue_type(ciffy.Residue.A)

# Structural selections
backbone = polymer.backbone()          # Backbone atoms only
bases = polymer.nucleobase()           # Nucleobase atoms (RNA/DNA)
sidechains = polymer.sidechain()       # Sidechain atoms
heavy = polymer.heavy()                # Heavy atoms (no hydrogens)

# Remove unresolved residues
resolved = polymer.strip()
```

### Iteration

```python
# Iterate over all chains
for chain in polymer.chains():
    print(chain.sequence_str())

# Iterate over RNA chains only
for chain in polymer.molecule_type(ciffy.RNA).chains():
    print(chain.pdb_id, chain.sequence_str())
```

### Hierarchy Operations

```python
# Counts at different scales
atoms_per_residue = polymer.counts(ciffy.RESIDUE)    # (R,)
residues_per_chain = polymer.counts(ciffy.CHAIN)     # (C,)

# Membership indices
chain_per_atom = polymer.membership(ciffy.CHAIN)     # (N,) chain index per atom
residue_per_atom = polymer.membership(ciffy.RESIDUE) # (N,) residue index per atom

# Reduce atom features to residue level (mean pooling)
residue_coords = polymer.reduce(polymer.coordinates, ciffy.RESIDUE)  # (R, 3)

# Expand residue features to atom level
atom_features = polymer.expand(residue_features, ciffy.RESIDUE)  # (N, ...)
```

### Geometry

```python
# Center coordinates
centered, centroids = polymer.center(ciffy.MOLECULE)
centered, centroids = polymer.center(ciffy.CHAIN)

# PCA alignment
aligned, rotations = polymer.pca(ciffy.CHAIN)

# Pairwise distances
distances = polymer.pairwise_distances()                    # Atom-atom
distances = polymer.pairwise_distances(ciffy.RESIDUE)       # Residue centroids

# K-nearest neighbors
neighbors = polymer.knn(k=16)
```

### Saving

```python
polymer.write("output.cif")
```

## Building Polymers

### From Sequence

Create template polymers (no coordinates) from sequence strings:

```python
import ciffy

# RNA (lowercase with u)
rna = ciffy.template("acguacgu")

# DNA (lowercase with t)
dna = ciffy.template("acgtacgt")

# Protein (uppercase)
protein = ciffy.template("MGKLF")

# Multi-chain
multi = ciffy.template(["acgu", "MGKLF"])
```

### Building Chains Residue-by-Residue

Build polymers incrementally using `append()`:

```python
from ciffy import Polymer, Residue

# Build a template (no coordinates)
p = Polymer()
for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
    p = p.append(res)

# Build with coordinates
p = Polymer()
p = p.append(Residue.A, coords)  # First residue with absolute coords
```

For autoregressive generation with relative positioning:

```python
from ciffy import Polymer, Residue
from ciffy.geometry import LocalCoordinates

p = Polymer()
p = p.append(Residue.A, first_coords)  # Absolute coordinates

# Subsequent residues use LocalCoordinates(coords, transform)
# where transform is an SE(3) transform [axis-angle (3), translation (3)]
p = p.append(Residue.C, LocalCoordinates(coords, transform))
p = p.append(Residue.G, LocalCoordinates(coords, transform))
```

## Structural Metrics

```python
import ciffy

# RMSD (Kabsch-aligned)
rmsd = ciffy.rmsd(polymer1, polymer2)
rmsd = ciffy.rmsd(polymer1, polymer2, scale=ciffy.CHAIN)  # Per-chain

# Also works on raw coordinates
rmsd = ciffy.rmsd(coords1, coords2)

# TM-score
tm = ciffy.tm_score(pred, ref)

# lDDT
lddt = ciffy.lddt(pred, ref)

# Radius of gyration
rg = ciffy.rg(polymer)

# Clash detection
clashes = ciffy.clashes(polymer)
```

## Command Line Interface

```bash
# View structure summary
ciffy info structure.cif

# Show sequences
ciffy info structure.cif --sequence

# Show entity descriptions
ciffy info structure.cif --desc

# Multiple files
ciffy info *.cif
```

## Testing

```bash
pytest tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
