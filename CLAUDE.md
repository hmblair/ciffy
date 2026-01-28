# Claude Code Instructions

## Project Overview

Ciffy is a library for researchers to **load, inspect, manipulate, and save macromolecular structures** (proteins, RNA, DNA).

**Design priorities:** Ease of use, performance, backend-agnostic (NumPy/PyTorch).

**Key concepts:**
- **Arrays everywhere** - All data in contiguous arrays for vectorized ops
- **Enums for readability** - `Residue.A`, `Molecule.RNA`, `Scale.ATOM`
- **Hierarchical scales** - Atoms → Residues → Chains → Molecules

## Development

### Environment

```bash
# Python
/Users/hmblair/mambaforge/bin/python

# Build
/Users/hmblair/mambaforge/bin/pip install -e .

# Test
/Users/hmblair/mambaforge/bin/python -m pytest tests/ -n auto
```

### Data Locations

```
Local RNA DB:                 /Users/hmblair/academic/data/structures/rna
Remote RNA DB (rex imp):      /home/hmblair/data/rna
Remote RNA DB (rex sherlock): /scratch/users/hmblair/structures/rna
Output dirs:                  outputs/, figures/
```

## Core API

### Loading & Creating

```python
import ciffy
from ciffy import Scale, Molecule, Residue

polymer = ciffy.load('structure.cif')      # Load from file
polymer = ciffy.template('acgu')           # Create template from sequence
polymer = ciffy.template('MGKLF')          # Protein sequence
```

### Properties

```python
polymer.coordinates          # (N, 3) atom positions
polymer.atoms                # (N,) atom type indices
polymer.sequence             # (R,) residue type indices
polymer.bonds                # (B, 2) covalent bond pairs
polymer.size()               # Total atoms
polymer.size(Scale.RESIDUE)  # Residue count
polymer.size(Scale.CHAIN)   # Chain count
polymer.sequence_str()       # "acgu"
```

### Selection

```python
polymer.chain(0)                    # First chain
polymer.residue([0, 5])             # Residues by index
polymer.residue_type(Residue.A)     # By residue type
polymer.molecule_type(Molecule.RNA) # By molecule type
polymer.backbone()                  # Backbone atoms
polymer.strip()                     # Remove unresolved residues
```

### Hierarchy Operations

```python
polymer.counts(Scale.RESIDUE)              # Atoms per residue
polymer.membership(Scale.CHAIN)            # Chain index per atom
polymer.reduce(features, Scale.RESIDUE)    # Atom → residue features
polymer.expand(features, Scale.RESIDUE)    # Residue → atom features
```

### Geometry & I/O

```python
polymer, _ = polymer.center()               # Center coordinates (second output is centroid)
polymer.numpy() / polymer.torch()
polymer.to('cuda')
polymer.write('output.cif')
```

### Operations Module

Analysis functions live in `ciffy.operations`. Import what you need:

```python
from ciffy import operations

# Geometry analysis
dists = operations.pairwise_distances(polymer)        # Distance matrix
neighbors = operations.knn(polymer, k=16)             # K-nearest neighbors
adj = operations.adjacency(polymer)                   # Adjacency matrix from bonds
bond_dists = operations.bonded_distances(polymer, Residue.A.O3p, Residue.A.P)

# Frame operations
transforms = operations.decompose(polymer)             # Extract inter-residue SE(3) transforms
rebuilt = operations.compose(polymer, transforms)      # Rebuild polymer from transforms
# transforms.data: (n_residues, 7) - [quaternion(4), translation(3)]
# transforms.source, transforms.target: frame definitions used

# Structure comparison
rmsd_val = ciffy.rmsd(polymer1, polymer2)             # Kabsch-aligned RMSD
tm = ciffy.tm_score(polymer1, polymer2)               # TM-score
lddt_val = ciffy.lddt(polymer1, polymer2)             # lDDT

# Packing for batched per-residue operations
packed, mask = operations.pack(polymer.coordinates, polymer.counts(Scale.RESIDUE))
# packed: (n_residues, max_atoms, 3), mask: (n_residues, max_atoms) boolean
unpacked = operations.unpack(packed, mask)  # Back to (N_atoms, 3)
```

Note: Many operations are also available as Polymer methods for convenience, but the operations module is the canonical location.

### Structured Atom Access

Access atoms by molecule type using `RNA`, `DNA`, `Protein` namespaces:

```python
from ciffy.biochemistry import RNA, DNA, Protein

# Backbone atoms - unified values shared across all residue types
RNA.Backbone.P       # Atom(P, 2)
RNA.Backbone.C1p     # Atom(C1', 13)
RNA.Backbone.O3p     # Atom(O3', 10)
Protein.Backbone.CA  # Atom(CA, 15)

# Base atoms - aggregated across residue types
RNA.Base.glycosidic_n       # AtomGroup with {A: N9, G: N9, C: N1, U: N1}
RNA.Base.glycosidic_n.A     # Atom(N9, 18) for adenine
RNA.Base.glycosidic_n.index()  # Array of all glycosidic N values

# Usage with bonded_distances
polymer.bonded_distances(RNA.Backbone.C1p, RNA.Base.glycosidic_n)
```

Note: `ciffy.RNA` returns `Molecule.RNA` (int) for filtering. Use `ciffy.biochemistry.RNA` for structured atom access.

### Saving Predicted Coordinates

To save predicted coordinates to a `.cif` file, create a template from the sequence and use `copy()` to assign coordinates:

```python
# Create template, assign predicted coords, save to file
template = ciffy.template('acgu')
predicted = template.copy(coordinates=pred_coords)
predicted.write('output.cif')
```

Note: Polymers are immutable - `copy()` returns a new polymer rather than modifying in place.

### Building Chains

```python
from ciffy import Polymer, Residue
from ciffy.geometry import LocalCoordinates

# Template (no coordinates)
p = Polymer()
for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
    p = p.append(res)

# With coordinates
p = Polymer()
p = p.append(Residue.A, coords1)                              # First at origin
p = p.append(Residue.C, LocalCoordinates(coords2, transform)) # Relative positioning
```

`LocalCoordinates`: Bundles (n_atoms, 3) coordinates with (7,) SE(3) transform [quaternion (4), translation (3)].

## Code Conventions

### Backend-Agnostic Code

Use operations from `ciffy.backend.ops` instead of manual type checking:

```python
# GOOD - use backend ops
from ciffy.backend import ops

result = ops.cat([a, b])
indices = ops.nonzero(mask)

# BAD - manual isinstance branches
import torch
import numpy as np

if isinstance(a, torch.Tensor):
    result = torch.cat([a, b])
else:
    result = np.concatenate([a, b])
```

Key ops: `cat`, `stack`, `cdist`, `scatter_sum/mean/max/min`, `repeat_interleave`, `nonzero`, `argwhere`, `svd`, `eigh`, `pinv`, `norm`, `where`, `topk`, `arange`, `zeros/ones/empty` (with `like=` param), `to_backend`, `convert_backend`.

### Enum Values

Always use enum `.value` attributes instead of hardcoded integers. Enum values are implementation details, auto-generated from CCD order during codegen, and may change between versions.

```python
# GOOD - use enum values
from ciffy import Residue

mask = polymer.sequence == Residue.A.value
adenine_count = (polymer.sequence == Residue.A.value).sum()

# GOOD - use structured access for atoms
from ciffy.biochemistry import RNA
p_atoms = polymer.atoms == int(RNA.Backbone.P)

# BAD - hardcoded integers
mask = polymer.sequence == 5  # What residue is this? Will it change?
mask = polymer.atoms == 2     # Atom values are auto-generated, not stable!
```
