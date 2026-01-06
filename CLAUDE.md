# Claude Code Instructions

## Project Overview

Ciffy is a library for researchers to **load, inspect, manipulate, and predict macromolecular structures** (proteins, RNA, DNA).

**Design priorities:** Ease of use, performance, backend-agnostic (NumPy/PyTorch).

**Key concepts:**
- **Arrays everywhere** - All data in contiguous arrays for vectorized ops
- **Enums for readability** - `Residue.A`, `Molecule.RNA`, `Scale.ATOM`
- **Hierarchical scales** - Atoms → Residues → Chains → Molecules

## Environment

```bash
# Python
/Users/hmblair/mambaforge/bin/python

# Build
/Users/hmblair/mambaforge/bin/pip install -e .

# Test
/Users/hmblair/mambaforge/bin/python -m pytest tests/ -n auto
```

## Data Locations

```
Local RNA DB:    /Users/hmblair/academic/data/structures/rna
Remote RNA DB:   /home/hmblair/data/rna
Output dir:      outputs/
```

## Git Safety

**NEVER discard unstaged changes** unrelated to the current task. Use worktrees for complex changes:
```bash
git worktree add ../ciffy-<feature> -b <feature>
```

## Polymer API

### Loading & Creating

```python
import ciffy
from ciffy import Scale, Molecule, Residue

polymer = ciffy.load('structure.cif')           # Load from file
polymer = ciffy.from_sequence('acgu')           # Create template from sequence
polymer = ciffy.from_sequence('MGKLF')          # Protein sequence
```

### Properties

```python
polymer.coordinates          # (N, 3) atom positions
polymer.atoms                # (N,) atom type indices
polymer.sequence             # (R,) residue type indices
polymer.bonds                # (B, 2) covalent bond pairs
polymer.size()               # Total atoms
polymer.size(Scale.RESIDUE)  # Residue count
polymer.size(Scale.CHAIN)    # Chain count
polymer.sequence_str()       # "acgu"
```

### Selection

```python
polymer.chain(0)               # First chain
polymer.residue([0, 5])        # Residues by index
polymer.by_residue(Residue.A)  # By residue type
polymer.by_type(Molecule.RNA)  # By molecule type
polymer.backbone()             # Backbone atoms
polymer.strip()                # Remove unresolved residues
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
polymer.numpy() / polymer.torch()
polymer.to('cuda')
polymer.write('output.cif')
```

## Building Chains

```python
from ciffy import Polymer, Residue

# Template (no coordinates)
p = Polymer()
for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
    p = p.extend_new(res)

# With coordinates
p = Polymer()
p = p.extend_new(Residue.A, coords1)             # First at origin
p = p.extend_new(Residue.C, coords2, transform)  # Relative positioning
```

`transform`: (6,) SE(3) as [axis-angle, translation]. If `None`, uses absolute coords.

## PolymerDataset

```python
from ciffy.nn import PolymerDataset

dataset = PolymerDataset(
    "./structures/",
    scale=Scale.CHAIN,           # MOLECULE or CHAIN
    molecule_types=Molecule.RNA,
    min_residues=10,             # or max
    max_atoms=5000,              # or min
    exclude_ids=["1ABC"],
    num_workers=8,
)
polymer = dataset[0]
```

## PolymerEmbedding

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

## RMSD Loss

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

## Training Practices

1. **Always do a dry run locally first** - Before submitting to GPU cluster via `rex`, run a quick local test (1 epoch, small batch) to catch errors early.

2. **Always save sample predictions** - Save sample predictions/generations to `outputs/` so the user can visually inspect model quality. Use `polymer.write('outputs/sample_001.cif')` to write structures.

3. **Avoid batching complexity** - Structures have different sizes. Process one structure at a time rather than implementing complex batching logic.

4. **Keep training loops simple** - Focus on getting results fast. Avoid premature optimization or over-engineering.

5. **Use ciffy's built-in features** - `PolymerDataset`, `PolymerEmbedding`, and `Polymer` methods are battle-tested and handle the many edge cases in .cif files. Don't reimplement this functionality.
