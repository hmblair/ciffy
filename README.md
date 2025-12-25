## Overview

`ciffy` is a fast CIF file parser for molecular structures, with a C backend and Python interface. It supports both NumPy and PyTorch backends for array operations.

### Performance

ciffy is **70-125x faster** than BioPython and Biotite for parsing CIF files:

| Structure | Atoms | ciffy | BioPython | Biotite |
|-----------|------:|------:|----------:|--------:|
| 3SKW | 2,874 | 0.36 ms | 39 ms (106x) | 28 ms (78x) |
| 9GCM | 4,466 | 0.54 ms | 48 ms (88x) | 38 ms (70x) |
| 9MDS | 102,216 | 11 ms | 1340 ms (126x) | 946 ms (89x) |

<sub>Benchmarked on Apple M1 Max. Run `python tests/profile.py` to reproduce.</sub>

## Installation

### From PyPI

```bash
pip install ciffy
```

### From Source

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy
pip install -r requirements.txt
pip install -e .
```

## Backends

`ciffy` supports two array backends:

- **NumPy**: Lightweight, no additional dependencies required
- **PyTorch**: For GPU support (CUDA/MPS) and integration with deep learning workflows

Specify the backend when loading structures:

```python
import ciffy

# Load with NumPy backend (recommended for general use)
polymer = ciffy.load("structure.cif", backend="numpy")

# Load with PyTorch backend (for deep learning workflows)
polymer = ciffy.load("structure.cif", backend="torch")
```

Polymers can be converted between backends:

```python
# Convert to PyTorch tensors
torch_polymer = polymer.torch()

# Convert to NumPy arrays
numpy_polymer = polymer.numpy()
```

For PyTorch, move tensors to GPU:

```python
# Move to CUDA
polymer_gpu = polymer.torch().to("cuda")

# Move to Apple Silicon (MPS)
polymer_mps = polymer.torch().to("mps")
```

**Note:** The default backend is `"numpy"` as of v0.6.0. Specify the backend explicitly for clarity.

## Usage

```python
import ciffy

# Load a structure from a CIF file
polymer = ciffy.load("structure.cif", backend="numpy")

# Basic information
print(polymer)  # Summary of chains, residues, atoms

# Access coordinates and properties
coords = polymer.coordinates      # (N, 3) array/tensor
atoms = polymer.atoms             # (N,) array/tensor of atom types
sequence = polymer.sequence_str()  # Sequence string

# Geometric operations
centered, means = polymer.center(ciffy.MOLECULE)
aligned, Q = polymer.align(ciffy.CHAIN)
distances = polymer.pairwise_distances(ciffy.RESIDUE)

# Selection
rna_chains = polymer.by_type(ciffy.RNA)
backbone = polymer.backbone()

# Molecule type per chain (parsed from CIF _entity_poly block)
mol_types = polymer.molecule_type  # Array of Molecule enum values

# Load with entity descriptions (off by default for performance)
polymer = ciffy.load("structure.cif", load_descriptions=True)
descriptions = polymer.descriptions  # List of description strings per chain

# Iterate over chains
for chain in polymer.chains(ciffy.RNA):
    print(chain.pdb_id, chain.sequence_str())

# Compute RMSD between structures (defaults to MOLECULE scale)
rmsd = ciffy.rmsd(polymer1, polymer2)
```

## Saving Structures

```python
# Save to CIF format (supports all molecule types)
polymer.write("output.cif")

# Save only polymer atoms (excludes water, ions, ligands)
polymer.poly().write("polymer_only.cif")
```

## Command Line Interface

```bash
# View structure summary
ciffy structure.cif

# Show sequences per chain
ciffy structure.cif --sequence

# Show entity descriptions per chain
ciffy structure.cif --desc

# Multiple files
ciffy file1.cif file2.cif

# Run multiple training experiments in parallel
ciffy experiment configs/*.yaml

# Run inference to generate structures from sequences
# Copy example config and customize for your setup:
# cp examples/configs/inference_example.yaml configs/inference.yaml
ciffy inference configs/inference.yaml
```

Example output:
```
PDB 9GCM (numpy)
──────────────────────
   Type     Res  Atoms
A  RNA      135   1413
B  PROTEIN  132   1032
C  PROTEIN  246   1261
D  PROTEIN  485    760
──────────────────────
            998   4466

Descriptions:
  A: U11 snRNA
  B: U11/U12 small nuclear ribonucleoprotein 25 kDa protein
  C: U11/U12 small nuclear ribonucleoprotein 35 kDa protein
  D: Programmed cell death protein 7
```

## Training Neural Networks

ciffy includes PyTorch modules for deep learning on molecular structures. See the [deep learning guide](docs/guides/deep-learning.md) for full documentation.

### Running Experiments

Train multiple models in parallel across GPUs:

```bash
# Run all configs in parallel (auto-distributes across GPUs)
ciffy experiment configs/*.yaml

# Run sequentially
ciffy experiment configs/*.yaml --sequential

# Force CPU
ciffy experiment configs/*.yaml --device cpu
```

Results are displayed in a comparison table:

```
Experiment            Status    Best Loss   Device    Time
--------------------  --------  ----------  --------  ----------
vae_small             success   0.1234      cuda:0    45.2s
vae_medium            success   0.0987      cuda:1    2m0s
vae_large             failed    N/A         cuda:0    5.3s
--------------------  --------  ----------  --------  ----------
Total: 2/3 succeeded in 2m51s
```

## Flow Models for Generative Modeling

ciffy provides a high-level API for generative modeling with normalizing flows. Generate new polymer conformations from sequences:

```python
from ciffy import flow

# Sample a polymer conformation from sequence
polymer = flow.sample("acgu")  # RNA sequence
polymer.write("output.cif")

# Generate multiple samples
samples = flow.sample("acgu", n_samples=10)
for i, p in enumerate(samples):
    p.write(f"sample_{i}.cif")
```

### Training Custom Models

```python
from ciffy import flow

# Train on your structures
model = flow.train(
    ["data/*.cif"],        # CIF files for training
    residues="ACGU",       # Residue types to model
    n_epochs=200,
    device="cuda",
)

# Sample from trained model
samples = flow.sample("acgu", n_samples=10, model=model)
```

### Latent Space Operations

```python
import ciffy
from ciffy import flow

# Encode existing structure to latent space
polymer = ciffy.load("structure.cif").poly()
latents = flow.encode(polymer)

# Modify and decode back
import torch
modified = latents + torch.randn_like(latents) * 0.1
new_polymer = flow.decode(modified, "acgu")
```

See the [flow models guide](docs/guides/flow-models.md) for comprehensive documentation.

## Testing

```bash
pytest tests/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, repository structure, and code generation details.
