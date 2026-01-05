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
Local test file:  tests/data/9MDS.cif
Local RNA DB:     /Users/hmblair/academic/data/structures/rna
Remote RNA DB:    /home/hmblair/data/rna
Output dir:       outputs/
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
polymer.poly()                 # Polymer atoms only
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
    min_residues=10,
    max_atoms=5000,
    exclude_ids=["1ABC"],
    num_workers=8,
)
polymer = dataset[0]
```

## Generative Models

### ResidueFlowModel (quick training)

```python
from ciffy import flow

model = flow.train(
    cif_paths=["data/*.cif"],
    residues="ACGU",
    output_dir="models/rna_flow",
    n_epochs=200,
    latent_dim=12,
    accelerator="gpu",
)
```

### PolymerModel (sampling)

```python
from ciffy.nn import PolymerModel

# Load trained model
model = PolymerModel.load("models/polymer", device="cuda")

# Sample from sequence
polymer = model.sample_from_sequence("acgu")
polymers = model.sample_from_sequence("acgu", n_samples=10, temperature=1.0)

# Encode/decode
latents = model.encode_polymer(polymer)  # (n_residues, latent_dim)
reconstructed = model.decode_to_polymer(latents, template)

# Save
model.save("models/polymer")
```

### ConsolidatedResidueVAE

```python
from ciffy.nn.vae.residue import ConsolidatedResidueVAE, ConsolidatedVAEConfig

model = ConsolidatedResidueVAE(
    residue_atoms=residue_atoms,  # Dict[Residue, AtomGroup]
    config=ConsolidatedVAEConfig(latent_dim=12, d_model=64),
)

coords, transforms = model.sample(n_samples=10)
latents = model.encode(coords)
```
