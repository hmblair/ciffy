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
Local RNA DB:                 /Users/hmblair/academic/data/structures/rna
Remote RNA DB (rex imp):      /home/hmblair/data/rna
Remote RNA DB (rex sherlock): /scratch/users/hmblair/structures/rna
Output dirs:                  outputs/, figures/
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

## Current Goal: Per-Residue Latent Embeddings

### Objective

Learn a **per-residue latent space** (16-dim) where each RNA residue's conformation can be encoded. Each residue is represented by:
1. **Local coordinates** - Atom positions in the residue's aligned frame
2. **Transform to next residue** - SE(3) transform (quaternion + translation)

If per-residue reconstruction is exact, chain reconstruction follows automatically.

### Relevant Files

```
ciffy/nn/residue/
├── encoder.py      # ResidueEncoder
├── decoder.py      # ResidueDecoder
├── vae.py          # ResidueVAE
└── training.py     # precompute_targets(), create_batches()

scratch/train_vae.py    # Training script
```

### Running

```bash
# Local test
python scratch/train_vae.py --data tests/data --epochs 10

# Remote GPU (outputs to /scratch/users/hmblair/ciffy/vae_runs/)
rex sherlock -d --gpu scratch/train_vae.py -- --name <experiment_name>
```

### Current Best Results (935K params, latent_dim=16, coord_weight=100)

- **Per-residue**: 0.038 Å RMSD, 0.99° rotation error
- **Full chain**: 1.6-7.5 Å RMSD (20-75 residues)

Key finding: Default `coord_weight=1` causes 150x gradient imbalance (transform loss dominates). Setting `coord_weight=100` balances gradients and improves chain reconstruction 2-4x.

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
)
polymer = dataset[0]  # Caching enabled by default
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
    dropout=0.1,      # Optional dropout on output
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
