# Flow Models for Generative Modeling

ciffy includes normalizing flow models for generating realistic polymer conformations. This guide covers the high-level API for sampling, training, and latent space manipulation.

## Quick Start

```python
from ciffy import flow

# Sample a polymer conformation from sequence
polymer = flow.sample("acgu")
polymer.write("output.cif")
```

## Core Functions

The `ciffy.flow` module provides five main functions:

| Function | Description |
|----------|-------------|
| `flow.sample()` | Generate conformations from a sequence |
| `flow.train()` | Train a custom flow model |
| `flow.encode()` | Encode a polymer to latent space |
| `flow.decode()` | Decode latents back to a polymer |
| `flow.load()` | Load a pre-trained model |

## Sampling Conformations

### Basic Sampling

```python
from ciffy import flow

# Sample single conformation
polymer = flow.sample("acgu")
polymer.write("sample.cif")

# Sample multiple conformations
samples = flow.sample("acgu", n_samples=10)
for i, p in enumerate(samples):
    p.write(f"sample_{i}.cif")
```

### Using GPU

```python
# Use CUDA for faster sampling
polymer = flow.sample("acgu", device="cuda")
```

### Using a Custom Model

```python
# Sample with a trained model
samples = flow.sample("acgu", n_samples=10, model=my_model)

# Or specify a model path
samples = flow.sample("acgu", model="path/to/model")
```

## Training Custom Models

### Basic Training

```python
from ciffy import flow

model = flow.train(
    ["data/*.cif"],      # Training data (supports globs)
    residues="ACGU",     # Residue types to model
    n_epochs=200,
    device="cuda",
)
```

### Training Options

```python
model = flow.train(
    cif_paths=["data/*.cif"],
    residues="ACGU",           # Can also be ["A", "C", "G", "U"]
    output_dir="models/rna",   # Save trained model

    # Training config
    latent_dim=12,             # Latent space dimension
    n_layers=8,                # Number of flow layers
    hidden_dim=64,             # Hidden layer size
    n_epochs=200,
    batch_size=256,
    learning_rate=1e-3,
    device="cuda",
)
```

### Saving and Loading Models

```python
# Save during training
model = flow.train(cif_paths, residues="ACGU", output_dir="models/my_model")

# Load later
model = flow.load("models/my_model")

# Load pre-trained model
model = flow.load("rna")  # Built-in RNA model
```

## Latent Space Operations

### Encoding Structures

```python
import ciffy
from ciffy import flow

# Load an existing structure
polymer = ciffy.load("structure.cif").poly()

# Encode to latent space
latents = flow.encode(polymer)  # Shape: (n_residues, latent_dim)
```

### Decoding Latents

```python
# Decode back to polymer
reconstructed = flow.decode(latents, "acgu")
reconstructed.write("reconstructed.cif")

# Or use original polymer as template (preserves metadata)
reconstructed = flow.decode(latents, polymer)
```

### Latent Space Manipulation

```python
import torch

# Interpolate between two structures
latents1 = flow.encode(polymer1)
latents2 = flow.encode(polymer2)

# Linear interpolation
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    interpolated = (1 - t) * latents1 + t * latents2
    polymer = flow.decode(interpolated, "acgu")
    polymer.write(f"interp_{t:.2f}.cif")

# Add noise for variation
noisy = latents + torch.randn_like(latents) * 0.1
varied = flow.decode(noisy, "acgu")
```

## Workflow Examples

### Generate Diverse Conformations

```python
from ciffy import flow

# Generate 100 conformations
samples = flow.sample("acgu" * 10, n_samples=100, device="cuda")

# Save all samples
for i, p in enumerate(samples):
    p.write(f"conformations/sample_{i:03d}.cif")
```

### Train on Custom Data

```python
from ciffy import flow
from pathlib import Path

# Collect training data
cif_files = list(Path("pdb_data").glob("*.cif"))
print(f"Training on {len(cif_files)} structures")

# Train model
model = flow.train(
    cif_files,
    residues="ACGU",
    output_dir="models/custom_rna",
    n_epochs=500,
    device="cuda",
)

# Generate samples
samples = flow.sample("acgu", n_samples=10, model=model)
```

### Conformational Analysis

```python
import ciffy
from ciffy import flow
import torch

# Load ensemble of structures
polymers = [ciffy.load(f).poly() for f in Path("ensemble").glob("*.cif")]

# Encode all to latent space
latents = torch.stack([flow.encode(p) for p in polymers])

# Analyze latent space
mean_latent = latents.mean(dim=0)
std_latent = latents.std(dim=0)

print(f"Latent variance: {std_latent.mean():.3f}")

# Generate from mean (consensus structure)
consensus = flow.decode(mean_latent, "acgu")
consensus.write("consensus.cif")
```

## Pre-trained Models

ciffy includes pre-trained models:

| Model | Residues | Description |
|-------|----------|-------------|
| `"rna"` | A, C, G, U | Standard RNA nucleotides |

```python
# Load pre-trained model
model = flow.load("rna", device="cuda")

# Check supported residues
print(model.residue_types)  # [Residue.A, Residue.C, Residue.G, Residue.U]
```

## CLI Training

For training from the command line:

```bash
# Train flow model on RNA data
ciffy train flow --data /path/to/cifs --output models/rna --epochs 200

# With custom config
ciffy train flow --data /path/to/cifs --output models/rna \
    --latent-dim 16 --n-layers 8 --hidden-dim 128

# With W&B logging
ciffy train flow --data /path/to/cifs --output models/rna --wandb
```

## Advanced Usage

For more control over the flow model architecture and training, use PyTorch Lightning:

```python
import lightning as L
from ciffy.nn.lightning import FlowDataModule, ResidueFlowModule
from ciffy.nn.lightning.modules.residue_flow import (
    ResidueFlowFullConfig,
    ResidueFlowModelConfig,
)
from ciffy.nn.flow import PolymerFlowModel
from ciffy import Residue

# Create config
config = ResidueFlowFullConfig(
    model=ResidueFlowModelConfig(latent_dim=16, n_layers=12),
)

# Train each residue type
models = {}
for residue in [Residue.A, Residue.C, Residue.G, Residue.U]:
    dm = FlowDataModule(cif_paths, residue)
    module = ResidueFlowModule(config, residue)
    trainer = L.Trainer(max_epochs=200)
    trainer.fit(module, dm)
    models[residue] = module.get_model()

# Combine into polymer model
polymer_model = PolymerFlowModel(models)
```

See the [Deep Learning Guide](deep-learning.md) for detailed documentation of the lower-level API.

## API Reference

### flow.sample

```python
flow.sample(
    sequence: str,
    n_samples: int = 1,
    model: str | PolymerFlowModel = "rna",
    device: str = "cpu",
) -> Polymer | list[Polymer]
```

### flow.train

```python
flow.train(
    cif_paths: list[str | Path],
    residues: list[str] | str = "ACGU",
    output_dir: str | Path | None = None,
    **config_kwargs,  # latent_dim, n_epochs, device, etc.
) -> PolymerFlowModel
```

### flow.encode

```python
flow.encode(
    polymer: Polymer,
    model: str | PolymerFlowModel = "rna",
    device: str = "cpu",
) -> torch.Tensor  # Shape: (n_residues, latent_dim)
```

### flow.decode

```python
flow.decode(
    latents: torch.Tensor,
    template: Polymer | str,
    model: str | PolymerFlowModel = "rna",
    device: str = "cpu",
) -> Polymer
```

### flow.load

```python
flow.load(
    name: str = "rna",
    device: str = "cpu",
) -> PolymerFlowModel
```
