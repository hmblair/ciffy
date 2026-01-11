# Claude Code Instructions

## Project Overview

Ciffy is a library for researchers to **load, inspect, manipulate, and predict macromolecular structures** (proteins, RNA, DNA).

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
Local RNA DB:                     /Users/hmblair/academic/data/structures/rna
Remote RNA DB (rex gpu):          /home/hmblair/data/rna
Remote RNA DB (rex sherlock-gpu): /scratch/users/hmblair/structures/rna
Output dirs:                      outputs/, figures/
```

### Git Safety

**NEVER discard unstaged changes** unrelated to the current task. **NEVER use `git add -A`** — always stage specific files. **NEVER checkout a file** without first inspecting the full diff and confirming there are no unrelated changes that would be lost. Use worktrees for complex changes:
```bash
git worktree add ../ciffy-<feature> -b <feature>
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
polymer.center()               # Center coordinates
polymer.pairwise_distances()   # Distance matrix
polymer.knn(k=16)              # K-nearest neighbors
polymer.bonded_distances(Residue.A.O3p, Residue.A.P)  # Distances between bonded atom types
polymer.numpy() / polymer.torch()
polymer.to('cuda')
polymer.write('output.cif')
```

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

`LocalCoordinates`: Bundles (n_atoms, 3) coordinates with (6,) SE(3) transform [axis-angle, translation].

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

Always use enum `.value` attributes instead of hardcoded integers. Enum values are implementation details and are not consecutive across residue types.

```python
# GOOD - use enum values
from ciffy import Residue

mask = polymer.sequence == Residue.A.value
adenine_count = (polymer.sequence == Residue.A.value).sum()

# BAD - hardcoded integers
mask = polymer.sequence == 5  # What residue is this? Will it change?
```

### Training Practices

1. **Always do a dry run locally first** - Before submitting to GPU cluster via `rex`, run a quick local test (1 epoch, small batch) to catch errors early.

2. **Always save sample predictions** - Save sample predictions/generations to `outputs/` so the user can visually inspect model quality. Use `polymer.write('outputs/sample_001.cif')` to write structures.

3. **Avoid batching complexity** - Structures have different sizes. Process one structure at a time rather than implementing complex batching logic.

4. **Keep training loops simple** - Focus on getting results fast. Avoid premature optimization or over-engineering.

5. **Use ciffy's built-in features** - `PolymerDataset`, `PolymerEmbedding`, and `Polymer` methods are battle-tested and handle the many edge cases in .cif files. Don't reimplement this functionality.

## Neural Network API

### PolymerDataset

```python
from ciffy.nn import PolymerDataset

dataset = PolymerDataset(
    "./structures/",
    scale=Scale.CHAIN,           # MOLECULE or CHAIN
    molecule_types=Molecule.RNA,
    min_residues=10,             # or max
    max_atoms=5000,              # or min
    exclude_ids=["1ABC"],
    cache=True,                  # Cache structures in memory for faster epochs
)
polymer = dataset[0]
```

### Train/Test Splitting

```python
from ciffy.nn import split_items, PolymerDataset

split = split_items(paths, train=0.8, val=0.1, test=0.1, seed=42)
train_dataset = PolymerDataset(split.train, scale=Scale.CHAIN)
val_dataset = PolymerDataset(split.val, scale=Scale.CHAIN)
```

### PolymerEmbedding

```python
from ciffy.nn import PolymerEmbedding

# Atom-level embeddings (atom + residue + element)
embed = PolymerEmbedding(
    scale=Scale.ATOM,
    atom_dim=64,      # Atom type embedding
    residue_dim=32,   # Residue type (expanded to atoms)
    element_dim=16,   # Element type embedding
)
features = embed(polymer)  # (num_atoms, 112)

# Residue-level embeddings (only residue_dim valid)
embed = PolymerEmbedding(scale=Scale.RESIDUE, residue_dim=64)
features = embed(polymer)  # (num_residues, 64)

embed.output_dim  # Total embedding dimension
```

### RMSD Loss

**Use `ciffy.rmsd` as the default loss function for structure prediction models.** It computes Kabsch-aligned RMSD with gradient support.

```python
import ciffy

# Polymer RMSD (returns per-molecule RMSD by default)
loss = ciffy.rmsd(pred_polymer, target_polymer)
loss = ciffy.rmsd(pred_polymer, target_polymer, scale=Scale.CHAIN)  # Per-chain

# Coordinate RMSD (for training loops)
loss = ciffy.rmsd(pred_coords, target_coords)           # (N, 3) -> scalar
loss = ciffy.rmsd(pred_coords, target_coords, eps=1e-8) # Gradient-stable near 0
```

The `eps` parameter adds numerical stability when RMSD approaches zero during training.
