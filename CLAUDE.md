# Claude Code Instructions

## Project Overview

Ciffy is a library for researchers to **load, inspect, manipulate, and predict macromolecular structures** (proteins, RNA, DNA).

### Design Priorities

1. **Ease of use and predictable API** - The API should be intuitive and consistent. Researchers should be able to accomplish common tasks with minimal code. Method names should be self-explanatory.

2. **Performance** - Structures can contain hundreds of thousands of atoms. Operations must be efficient.

### Implementation Philosophy

- **Arrays everywhere**: All data is stored in contiguous arrays (NumPy or PyTorch) for vectorized operations and GPU compatibility. Avoid Python loops over atoms/residues.

- **Enums for readability**: Integer arrays store indices, but enums (like `Residue.A`, `Molecule.RNA`, `Scale.ATOM`) provide human-readable access. This gives both performance and clarity. Note: residue indices are non-contiguous (e.g., A=0, C=1, G=4, U=15) - always use `Residue.A.value` rather than assuming sequential indices.

- **Hierarchical scales**: Structures have multiple scales - atoms, residues, chains, molecules. The `Scale` enum and methods like `size(Scale.RESIDUE)` make this hierarchy explicit.

- **Backend agnostic**: Core operations work with both NumPy and PyTorch, enabling seamless CPU/GPU workflows.

## Python Environment

Always use the mambaforge Python binary for running Python commands:

```bash
/Users/hmblair/mambaforge/bin/python
```

Do NOT use `python3` or `python` directly, as this may invoke the wrong interpreter.

## Git Safety

**NEVER discard unstaged changes.** If you see modified files unrelated to the current task, leave them alone. Do not run `git checkout --`, `git restore`, or `git stash` on files you didn't modify in this session. These may be work-in-progress from other agents or tasks.

### Worktree Workflow

For complex changes, use a separate worktree:

```bash
git worktree add ../ciffy-<feature> -b <feature>  # create worktree
cd ../ciffy-<feature> && # ... make changes ...
git checkout master && git merge <feature>         # merge when approved
git worktree remove ../ciffy-<feature>             # cleanup
```

## Build Commands

```bash
/Users/hmblair/mambaforge/bin/pip install -e .
```

## Test Commands

```bash
/Users/hmblair/mambaforge/bin/python -m pytest tests/ -n auto
```

## Default Test File

Use `tests/data/9MDS.cif` for manual testing:

```python
import ciffy
polymer = ciffy.load('tests/data/9MDS.cif')
```

## Output Directory

Save all generated outputs (trained models, sampled structures, figures, etc.) to `outputs/`:

```
outputs/
├── models/          # Trained model checkpoints
├── chains/          # Sampled CIF structures
├── figures/         # Generated plots
└── ...
```

## Large RNA Structure Database

For training models or large-scale analysis, use the RNA structure database:

```
Local:  /Users/hmblair/academic/data/structures/rna
Remote: /home/hmblair/data/rna
```

This directory contains a large collection of RNA CIF files from the PDB.

## Generative Models

### Model Hierarchy

Ciffy has a two-level model hierarchy for structure generation:

1. **Residue models** - Learn distributions over single residue conformations
   - `ResidueFlowModel` - PCA + normalizing flow (exact density, interpretable latents)
   - `ResidueVAE` - Variational autoencoder (better reconstruction, learned compression)
   - `ConsolidatedResidueVAE` - Shared encoder across residue types (4x more training data)

2. **PolymerModel** - Orchestrates residue models to encode/decode full polymers
   - Works with any model implementing the `ResidueGenerativeCore` protocol
   - Chains residues using SE(3) transforms for backbone connectivity

### Training Residue Models

**ResidueFlowModel (PCA + Flow):**
```python
from ciffy import flow

# Train all RNA residue types
model = flow.train(
    cif_paths=["data/*.cif"],
    residues="ACGU",
    output_dir="models/rna_flow",
    n_epochs=200,
    latent_dim=12,
    n_layers=8,
    hidden_dim=64,
    accelerator="gpu",
)
```

**ResidueVAE:**
```python
from ciffy.nn.lightning import ResidueVAEModule, ResidueDataModule
from ciffy.nn.lightning.modules.residue_vae import ResidueVAEFullConfig, ResidueVAEModelConfig
from ciffy.biochemistry import Residue
import lightning as L

config = ResidueVAEFullConfig(
    model=ResidueVAEModelConfig(
        latent_dim=12,
        hidden_dims=[256, 128],
        beta=1.0,
        free_bits=0.5,
    ),
)

dm = ResidueDataModule(cif_paths=cif_files, residue=Residue.A)
module = ResidueVAEModule(config, residue=Residue.A)
trainer = L.Trainer(max_epochs=200, accelerator="gpu")
trainer.fit(module, dm)

model = module.get_model()
model.save("models/vae_A")
```

**ConsolidatedResidueVAE (shared encoder):**
```python
from ciffy.nn.vae.residue import ConsolidatedResidueVAE, ConsolidatedVAEConfig
from ciffy.biochemistry import Residue

model = ConsolidatedResidueVAE(
    residue_atoms=residue_atoms,  # Dict[Residue, List[Atom]]
    config=ConsolidatedVAEConfig(
        latent_dim=12,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
    ),
)
```

### Sampling from Residue Models

All residue models share the same sampling interface:

```python
# Sample random conformations
coords, transforms = model.sample(n_samples=10)  # (10, n_atoms, 3), (10, 6)

# Encode existing coordinates
latents = model.encode(coords)  # (batch, latent_dim)

# Decode latents back to coordinates
coords, transforms = model.decode(latents)
```

### PolymerModel: Assembling Residues into Chains

`PolymerModel` wraps per-residue models to encode/decode full polymer structures:

```python
from ciffy.nn import PolymerModel
from ciffy.biochemistry import Residue

# Build from trained residue models
polymer_model = PolymerModel({
    Residue.A: model_a,
    Residue.C: model_c,
    Residue.G: model_g,
    Residue.U: model_u,
})

# Sample a polymer from sequence
polymer = polymer_model.sample_from_sequence("acgu")
polymers = polymer_model.sample_from_sequence("acgu", n_samples=10, temperature=1.0)

# Encode/decode existing structures
latents = polymer_model.encode_polymer(polymer)  # (n_residues, latent_dim)
reconstructed = polymer_model.decode_to_polymer(latents, template)

# Interpolate between conformations
intermediates = polymer_model.interpolate(polymer1, polymer2, n_steps=20)

# Save/load
polymer_model.save("models/polymer")
polymer_model = PolymerModel.load("models/polymer", device="cuda")
```

**How chain assembly works:**
- Each residue is decoded independently to local coordinates + an SE(3) transform
- The transform encodes the relative position/orientation of the next residue's backbone
- Residues are positioned iteratively: first at origin, subsequent using previous transforms
- Maintains realistic backbone connectivity despite independent residue sampling

### Model Comparison

| Feature | ResidueFlowModel | ResidueVAE | ConsolidatedVAE |
|---------|------------------|-----------|-----------------|
| Compression | PCA (fixed) | Learned encoder | Shared encoder |
| Density | Exact (normalizing flow) | Approximate (VAE) | Approximate |
| Reconstruction | Good | Excellent | Excellent |
| Multi-residue | Separate per type | Separate per type | Unified |
| Latent space | Non-normalized (std~6) | Near-normalized (std~0.3) | Near-normalized |
| Diffusion-ready | No (needs normalization) | Yes | Yes |

## Polymer Class Usage

### Core Properties

```python
polymer                      # Print summary table
polymer.pdb_id               # PDB identifier
polymer.coordinates          # (N, 3) atom positions
polymer.atoms                # (N,) atom type indices
polymer.sequence             # (R,) residue type indices
polymer.bonds                # (B, 2) covalent bond pairs
polymer.size()               # Total atoms (or Scale.CHAIN, Scale.RESIDUE)
polymer.sequence_str()       # Single-letter sequence (e.g., "ACGU")
```

### Selection

```python
polymer.chain(0)             # First chain
polymer.residue([0, 5])      # Multiple residues by index
polymer.by_residue(Residue.A)  # Select by residue type
polymer.by_type(Molecule.RNA)  # Select RNA chains
polymer.poly()               # Polymer atoms only
polymer.backbone()           # Backbone atoms
polymer.nucleobase()         # Nucleobase atoms (RNA/DNA)
polymer.strip()              # Remove zero-atom (unresolved) residues
```

### Hierarchy

```python
polymer.counts(Scale.RESIDUE)              # Atoms per residue
polymer.membership(Scale.CHAIN)            # Which chain each atom belongs to
polymer.reduce(features, Scale.RESIDUE)    # Reduce atom features to residue
polymer.expand(features, Scale.RESIDUE)    # Expand residue features to atoms
```

### Geometry

```python
polymer.center()                           # Center coordinates
polymer.pairwise_distances()               # Atom-atom distance matrix
polymer.knn(k=16)                          # K-nearest neighbors
polymer.align(Scale.MOLECULE)              # Align to principal axes
```

### Backend & I/O

```python
polymer.numpy() / polymer.torch()          # Convert backend
polymer.to('cuda')                         # Move to GPU
polymer.write('output.cif')                # Write to mmCIF
```

## Building Chains

Build polymers residue-by-residue using `extend_new()`:

```python
from ciffy import Polymer, Residue

# Build template (no coordinates)
p = Polymer()
for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
    p = p.extend_new(res)

# Build with coordinates (first residue absolute, subsequent relative)
p = Polymer()
p = p.extend_new(Residue.A, coords1)                    # First residue at origin
p = p.extend_new(Residue.C, coords2, transform)         # Positioned by transform
p = p.extend_new(Residue.G, abs_coords, transform=None) # Absolute coordinates
```

- `extend_new(residue, coords, transform)` - Auto-generates atoms/elements from Residue
- `transform`: (6,) SE(3) [axis-angle, translation] positions residue relative to previous
- If `transform=None`, coordinates are used as absolute positions

## PolymerDataset

PyTorch Dataset for loading CIF files with filtering:

```python
from ciffy.nn import PolymerDataset
from ciffy import Scale, Molecule

dataset = PolymerDataset(
    "./structures/",
    scale=Scale.CHAIN,           # MOLECULE or CHAIN
    molecule_types=Molecule.RNA, # Filter by type
    min_residues=10,             # Size filters
    max_atoms=5000,
    exclude_ids=["1ABC"],        # Exclude PDB IDs
    num_workers=8,               # Parallel scanning
)
polymer = dataset[0]  # Returns Polymer object
```
