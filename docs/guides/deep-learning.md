# Deep Learning Integration

This guide covers using ciffy with PyTorch for deep learning applications.

## PyTorch Backend

Load structures directly as PyTorch tensors:

```python
import ciffy

# Load with PyTorch backend
polymer = ciffy.load("structure.cif", backend="torch")

# All arrays are now torch tensors
print(type(polymer.coordinates))  # <class 'torch.Tensor'>
print(polymer.coordinates.dtype)   # torch.float32
```

### Converting Between Backends

```python
# NumPy to PyTorch
polymer_np = ciffy.load("structure.cif", backend="numpy")
polymer_torch = polymer_np.torch()

# PyTorch to NumPy
polymer_np = polymer_torch.numpy()

# Check current backend
print(polymer.backend)  # 'numpy' or 'torch'
```

## GPU Operations

Move structures to GPU:

```python
import torch

polymer = ciffy.load("structure.cif", backend="torch")

# Move to GPU
polymer_gpu = polymer.to("cuda")

# Move to specific GPU
polymer_gpu = polymer.to("cuda:0")

# Move back to CPU
polymer_cpu = polymer_gpu.to("cpu")
```

### Mixed Precision

Convert coordinate dtype for memory efficiency:

```python
# Convert to float16
polymer_fp16 = polymer.to(dtype=torch.float16)

# Combine device and dtype
polymer_gpu_fp16 = polymer.to("cuda", torch.float16)

# Convert to bfloat16 (better for training)
polymer_bf16 = polymer.to(dtype=torch.bfloat16)
```

!!! note
    Only coordinates are converted to the specified dtype. Integer tensors (atoms, elements, sequence) remain as int64.

## Embedding Layers

ciffy provides vocabulary sizes for creating embedding layers:

```python
import torch.nn as nn
import ciffy

# Vocabulary sizes
print(f"Atom types: {ciffy.NUM_ATOMS}")
print(f"Residue types: {ciffy.NUM_RESIDUES}")
print(f"Element types: {ciffy.NUM_ELEMENTS}")

# Create embeddings
atom_embedding = nn.Embedding(ciffy.NUM_ATOMS, 64)
residue_embedding = nn.Embedding(ciffy.NUM_RESIDUES, 64)
element_embedding = nn.Embedding(ciffy.NUM_ELEMENTS, 64)
```

### Using Embeddings

```python
polymer = ciffy.load("structure.cif", backend="torch")

# Embed atom types
atom_features = atom_embedding(polymer.atoms)  # (N, 64)

# Embed residue types
residue_features = residue_embedding(polymer.sequence)  # (R, 64)

# Embed elements
element_features = element_embedding(polymer.elements)  # (N, 64)

# Combine features
combined = torch.cat([atom_features, element_features], dim=-1)  # (N, 128)
```

## Differentiable Operations

Most ciffy operations are differentiable:

```python
polymer = ciffy.load("structure.cif", backend="torch")
polymer = polymer.to("cuda")

# Coordinates with gradients
coords = polymer.coordinates.requires_grad_(True)
polymer = polymer.with_coordinates(coords)

# Compute per-residue centroids (differentiable)
centroids = polymer.reduce(polymer.coordinates, ciffy.RESIDUE)

# Compute loss and backprop
target_centroids = get_target()
loss = ((centroids - target_centroids) ** 2).mean()
loss.backward()

print(coords.grad)  # Gradients flow back to coordinates
```

### Differentiable RMSD

```python
p1 = ciffy.load("pred.cif", backend="torch")
p2 = ciffy.load("target.cif", backend="torch")

# Enable gradients on predicted coordinates
coords = p1.coordinates.requires_grad_(True)
p1 = p1.with_coordinates(coords)

# RMSD is differentiable
rmsd_sq = ciffy.rmsd(p1, p2)
rmsd_sq.backward()

# Gradients for structure optimization
print(coords.grad.shape)
```

## Index Mapping

Use `index()` to get the containing unit index for each atom:

```python
import ciffy

polymer = ciffy.load("structure.cif", backend="torch")

# Get residue index for each atom (0 to num_residues-1)
residue_idx = polymer.index(ciffy.RESIDUE)  # (N,)

# Get chain index for each atom (0 to num_chains-1)
chain_idx = polymer.index(ciffy.CHAIN)  # (N,)

# Use for attention masking (same-residue attention)
same_residue_mask = residue_idx[:, None] == residue_idx[None, :]

# Use for chain-aware masking
same_chain_mask = chain_idx[:, None] == chain_idx[None, :]
```

This is useful for:
- Positional encodings in transformers
- Chain-aware or residue-aware attention masking
- Grouping atoms for aggregation operations

## Hierarchical Aggregation

Aggregate atom features to coarser scales:

```python
class RNAEncoder(nn.Module):
    def __init__(self, atom_dim=64, hidden_dim=128):
        super().__init__()
        self.atom_embed = nn.Embedding(ciffy.NUM_ATOMS, atom_dim)
        self.atom_mlp = nn.Linear(atom_dim + 3, hidden_dim)
        self.residue_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.chain_mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, polymer):
        # Atom-level features
        atom_feats = self.atom_embed(polymer.atoms)
        atom_feats = torch.cat([atom_feats, polymer.coordinates], dim=-1)
        atom_feats = self.atom_mlp(atom_feats)

        # Aggregate to residue level
        residue_feats = polymer.reduce(atom_feats, ciffy.RESIDUE)
        residue_feats = self.residue_mlp(residue_feats)

        # Aggregate to chain level
        chain_feats = polymer.rreduce(residue_feats, ciffy.CHAIN)
        chain_feats = self.chain_mlp(chain_feats)

        return chain_feats
```

## Batching Strategies

ciffy doesn't have built-in batching, but here are common patterns:

### List of Polymers

```python
# Load multiple structures
files = ["struct1.cif", "struct2.cif", "struct3.cif"]
polymers = [ciffy.load(f, backend="torch").to("cuda") for f in files]

# Process each
outputs = [model(p) for p in polymers]

# Pad and stack if needed
max_atoms = max(p.size() for p in polymers)
padded = [F.pad(p.coordinates, (0, 0, 0, max_atoms - p.size())) for p in polymers]
batch = torch.stack(padded)
```

### DataLoader Integration

```python
from torch.utils.data import Dataset, DataLoader

class CIFDataset(Dataset):
    def __init__(self, cif_files):
        self.files = cif_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        polymer = ciffy.load(self.files[idx], backend="torch")
        return {
            'coordinates': polymer.coordinates,
            'atoms': polymer.atoms,
            'elements': polymer.elements,
            'sequence': polymer.sequence,
        }

def collate_fn(batch):
    # Custom collation for variable-size structures
    return batch  # Keep as list, or implement padding

dataset = CIFDataset(cif_files)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
import ciffy

class StructurePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ciffy.NUM_ATOMS, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.coord_head = nn.Linear(64, 3)

    def forward(self, polymer):
        # One-hot encode atoms
        atoms_onehot = nn.functional.one_hot(
            polymer.atoms, ciffy.NUM_ATOMS
        ).float()

        # Encode
        features = self.encoder(atoms_onehot)

        # Predict coordinate deltas
        delta = self.coord_head(features)

        return polymer.coordinates + delta

# Training loop
model = StructurePredictor().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    polymer = ciffy.load("structure.cif", backend="torch").to("cuda")
    target = ciffy.load("target.cif", backend="torch").to("cuda")

    # Forward
    pred_coords = model(polymer)
    pred = polymer.with_coordinates(pred_coords)

    # RMSD loss
    loss = ciffy.rmsd(pred, target)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: RMSD = {loss.sqrt().item():.3f}")
```

## Generative Modeling

Use `from_sequence()` to create template structures for generative models that predict coordinates:

```python
import ciffy

# Create template from sequence (zero coordinates)
template = ciffy.from_sequence("acgu", backend="torch")
template = template.to("cuda")

# Template has correct structure but zero coordinates
print(template.coordinates.sum())  # 0.0
print(template.size())  # Total atoms
print(template.atoms)  # Atom type indices

# Predict coordinates with your model
predicted_coords = model(template)  # Shape: (N, 3)

# Attach predicted coordinates
result = template.with_coordinates(predicted_coords)

# Save the predicted structure
result.write("predicted.cif")
```

### Multi-chain Generation

Generate complex structures with multiple chains:

```python
# RNA-protein complex
sequences = ["acguacgu", "MGKLF"]
template = ciffy.from_sequence(sequences, backend="torch")

print(template.size(ciffy.CHAIN))  # 2
print(template.names)  # ['A', 'B']

# Each chain has correct molecule type
for chain in template.chains():
    print(f"{chain.names[0]}: {chain.molecule_type[0]}")
# A: Molecule.RNA
# B: Molecule.PROTEIN
```

### Structure Prediction Training

```python
class StructurePredictor(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.atom_embed = nn.Embedding(ciffy.NUM_ATOMS, hidden_dim)
        self.encoder = nn.TransformerEncoder(...)
        self.coord_head = nn.Linear(hidden_dim, 3)

    def forward(self, template):
        # Embed atom types
        h = self.atom_embed(template.atoms)

        # Encode structure
        h = self.encoder(h)

        # Predict coordinates
        return self.coord_head(h)

# Training loop
model = StructurePredictor().cuda()
optimizer = optim.Adam(model.parameters())

for sequence, target_cif in dataset:
    # Create template from sequence
    template = ciffy.from_sequence(sequence, backend="torch").to("cuda")

    # Load ground truth
    target = ciffy.load(target_cif, backend="torch").to("cuda")

    # Predict coordinates
    pred_coords = model(template)
    pred = template.with_coordinates(pred_coords)

    # RMSD loss
    loss = ciffy.rmsd(pred, target)
    loss.backward()
    optimizer.step()
```

## Internal Coordinates

Polymer supports dual representation: you can access both Cartesian (XYZ) and internal (bond length, angle, dihedral) coordinates transparently. Conversions happen automatically with lazy evaluation.

```python
import ciffy

# Load structure - internal coords computed on first access
polymer = ciffy.load("structure.cif", backend="torch")

# Access internal coordinate arrays (computed lazily)
print(polymer.distances.shape)   # (N,) bond lengths
print(polymer.angles.shape)      # (N,) bond angles
print(polymer.dihedrals.shape)   # (N,) dihedral angles

# Access named backbone dihedrals using enum
phi = polymer.dihedral(ciffy.DihedralType.PHI)    # Protein phi angles
psi = polymer.dihedral(ciffy.DihedralType.PSI)    # Protein psi angles
omega = polymer.dihedral(ciffy.DihedralType.OMEGA)  # Protein omega angles

# Nucleic acid dihedrals
alpha = polymer.dihedral(ciffy.DihedralType.ALPHA)
beta = polymer.dihedral(ciffy.DihedralType.BETA)
# ... gamma, delta, epsilon, zeta, chi_purine, chi_pyrimidine
```

### Differentiable Reconstruction

Modifications to internal coordinates trigger automatic Cartesian reconstruction (fully differentiable):

```python
polymer = ciffy.load("structure.cif", backend="torch")

# Enable gradients on dihedrals
dihedrals = polymer.dihedrals.requires_grad_(True)
polymer.dihedrals = dihedrals

# Cartesian coordinates auto-reconstructed on next access (differentiable)
coords = polymer.coordinates  # NERF reconstruction happens here

# Compute loss and backprop
target = ciffy.load("target.cif", backend="torch")
loss = ciffy.rmsd(polymer, target)
loss.backward()

# Gradients flow to dihedral angles
print(dihedrals.grad)
```

### Conformational Sampling

Modify dihedrals to generate new conformations:

```python
import torch

polymer = ciffy.load("structure.cif", backend="torch")

# Perturb backbone dihedrals
noise = torch.randn_like(polymer.dihedrals) * 0.1
new_dihedrals = polymer.dihedrals + noise
polymer.dihedrals = new_dihedrals

# Cartesian coordinates automatically recomputed
new_coords = polymer.coordinates
```

### Modifying Named Dihedrals

Set specific backbone angles while preserving others:

```python
polymer = ciffy.load("structure.cif", backend="torch")

# Get current phi angles
phi = polymer.dihedral(ciffy.DihedralType.PHI)

# Modify phi angles
new_phi = phi + torch.randn_like(phi) * 0.1
polymer.set_dihedral(ciffy.DihedralType.PHI, new_phi)

# Cartesian coordinates auto-update on next access
coords = polymer.coordinates
```

### Structure Generation from Dihedrals

Generate structures by predicting dihedral angles:

```python
class DihedralPredictor(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = nn.TransformerEncoder(...)
        self.dihedral_head = nn.Linear(hidden_dim, 1)

    def forward(self, template):
        # Predict dihedral angles for each atom
        h = self.encoder(...)
        pred_dihedrals = self.dihedral_head(h).squeeze(-1)

        # Set dihedrals (Cartesian auto-recomputed)
        template.dihedrals = pred_dihedrals
        return template.coordinates
```

## Built-in Neural Network Modules

ciffy provides ready-to-use PyTorch modules in `ciffy.nn`.

### PolymerEmbedding

Learnable embeddings for polymer features:

```python
from ciffy.nn import PolymerEmbedding
from ciffy import Scale

# Create embedding layer
embed = PolymerEmbedding(
    scale=Scale.ATOM,      # Output per-atom features
    atom_dim=64,           # Embed atom types
    residue_dim=32,        # Embed residue types (expanded to atoms)
    element_dim=16,        # Embed element types
)

polymer = ciffy.load("structure.cif", backend="torch")
features = embed(polymer)  # (num_atoms, 112)

# Or at residue scale
embed_res = PolymerEmbedding(scale=Scale.RESIDUE, residue_dim=64)
features = embed_res(polymer)  # (num_residues, 64)
```

### PolymerDataset

PyTorch Dataset for loading CIF files:

```python
from ciffy.nn import PolymerDataset
from ciffy import Scale, Molecule

# Load all structures from a directory
dataset = PolymerDataset(
    directory="./data/cif",
    scale=Scale.CHAIN,              # Iterate over chains (more samples)
    min_atoms=10,                   # Filter by size
    max_atoms=5000,
    molecule_types=(Molecule.PROTEIN, Molecule.RNA),  # Filter by type
    backend="torch",
)

print(f"Found {len(dataset)} chains")

# Access individual items
polymer = dataset[0]
```

### Modern Transformer

A reusable transformer with modern best practices:

```python
from ciffy.nn import (
    Transformer,
    TransformerBlock,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)

# Full transformer encoder
model = Transformer(
    d_model=256,
    num_layers=6,
    num_heads=8,
    use_rope=True,       # Rotary Position Embeddings
    use_swiglu=True,     # SwiGLU activation (vs GELU)
    use_rmsnorm=True,    # RMSNorm (vs LayerNorm)
)

x = torch.randn(batch, seq_len, 256)
out = model(x, mask=padding_mask)  # (batch, seq_len, 256)
```

Architecture features:

- **Pre-LN**: LayerNorm before attention/FFN for stable training
- **RoPE**: Rotary Position Embeddings for better length generalization
- **SwiGLU**: Gated activation (used in LLaMA, PaLM)
- **RMSNorm**: Simpler, faster normalization
- **Flash Attention**: Used automatically when available (PyTorch 2.0+)

Individual components can be composed:

```python
# Build custom architectures
class CustomBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, use_rope=True)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(self.norm(x))
        return x
```

## Variational Autoencoder (VAE)

ciffy includes a VAE for polymer conformations that operates on backbone dihedral angles:

```python
from ciffy.nn import PolymerVAE

# Create VAE
vae = PolymerVAE(
    latent_dim=64,      # Latent space dimension
    hidden_dim=256,     # Transformer hidden dim
    num_layers=4,       # Transformer layers
    num_heads=8,        # Attention heads
    beta=1.0,           # KL weight (beta-VAE)
)

polymer = ciffy.load("structure.cif", backend="torch")

# Encode to latent space
z_mu, z_logvar = vae.encode(polymer)

# Decode back to polymer
reconstructed = vae.decode(z_mu, polymer)

# Sample new conformations
samples = vae.sample(polymer, n_samples=10, temperature=1.0)

# Interpolate between conformations
interp = vae.interpolate(polymer1, polymer2, n_steps=10)

# Training
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
losses = vae.compute_loss(polymer)
losses["loss"].backward()  # Contains recon_loss + beta * kl_loss
optimizer.step()
```

The VAE:

- Works on backbone dihedrals (φ/ψ/ω for proteins, α/β/γ/δ/ε/ζ/χ for RNA)
- Uses sin/cos encoding to handle angle periodicity
- Outputs von Mises distributions (proper circular probability)
- Supports both proteins and nucleic acids

### Training Script

A complete training script is provided:

```bash
# Train with config file
python scripts/train_vae.py configs/vae_example.yaml

# Resume from checkpoint
python scripts/train_vae.py config.yaml --resume checkpoints/checkpoint_epoch0050.pt
```

Example config (`configs/vae_example.yaml`):

```yaml
model:
  latent_dim: 64
  hidden_dim: 256
  num_layers: 4

data:
  data_dir: ./data/cif
  scale: chain
  molecule_types: [protein, rna]

training:
  epochs: 100
  lr: 0.0001
  device: cuda

output:
  checkpoint_dir: ./checkpoints
  sample_dir: ./samples
  n_perturbations: 5  # Latent perturbations saved each epoch
```

At each epoch, the script saves perturbed samples to visualize how the latent space captures conformational variation.

### Running Multiple Experiments

Use the `ciffy experiment` command to run multiple training configurations in parallel across GPUs:

```bash
# Run all configs in a directory (parallel by default)
ciffy experiment configs/*.yaml

# Run specific configs
ciffy experiment configs/vae_small.yaml configs/vae_large.yaml

# Run sequentially (one at a time)
ciffy experiment configs/*.yaml --sequential

# Force specific device
ciffy experiment configs/*.yaml --device cpu
```

Experiments are automatically distributed across available GPUs in round-robin fashion. Each experiment runs in a separate process for memory isolation, so a failed experiment won't affect others.

Example output:

```
============================================================
Ciffy Experiment Runner
============================================================
Configs: 3
Parallel: True
Device: auto

  1. configs/vae_small.yaml
  2. configs/vae_medium.yaml
  3. configs/vae_large.yaml

Running experiments...
------------------------------------------------------------
  [+] vae_small: loss=0.1234 (45.2s on cuda:0)
  [+] vae_medium: loss=0.0987 (2m0s on cuda:1)
  [X] vae_large: FAILED - CUDA out of memory

============================================================
Results
============================================================
Experiment            Status    Best Loss   Recon       KL          Epochs      Device    Time
--------------------  --------  ----------  ----------  ----------  ----------  --------  ----------
vae_small             success   0.1234      0.0812      0.0422      100/100     cuda:0    45.2s
vae_medium            success   0.0987      0.0654      0.0333      100/100     cuda:1    2m0s
vae_large             failed    N/A         N/A         N/A         0/100       cuda:0    5.3s
--------------------  --------  ----------  ----------  ----------  ----------  --------  ----------
Total: 2/3 succeeded in 2m51s
```

You can also use the experiment runner programmatically:

```python
from ciffy.nn import run_experiments, format_results_table

results = run_experiments(
    config_paths=["small.yaml", "medium.yaml", "large.yaml"],
    parallel=True,
    device="auto",  # auto, cuda, mps, or cpu
)

# Access individual results
for r in results:
    if r.status == "success":
        print(f"{r.name}: best_loss={r.best_loss:.4f}")

# Print formatted table
print(format_results_table(results))
```

## Inference

After training a model, use ciffy's inference system to generate structures from sequences.

### Loading Trained Models

```python
from ciffy.nn import load_vae

# Load a trained VAE from checkpoint
vae = load_vae("checkpoints/vae_best.pt", device="cuda")
print(f"Latent dimension: {vae.latent_dim}")
```

### Sampling from Sequences

Generate new structures for a given sequence:

```python
import ciffy
from ciffy.nn import load_vae, generate_samples

# Load trained model
vae = load_vae("checkpoints/vae_best.pt", device="cuda")

# Generate structures from a sequence
samples = generate_samples(
    vae,
    sequence="MGKLF",      # Protein sequence
    n_samples=10,          # Generate 10 conformations
    temperature=1.0,       # Sampling temperature
    output_dir="./generated",  # Save to disk
    prefix="gen_",         # Filename prefix
)

print(f"Generated {len(samples)} structures")
```

### Multiple Sequences and Multi-chain Complexes

```python
# Generate for multiple sequences
samples = generate_samples(
    vae,
    sequence=["MGKLF", "acgu"],  # Protein and RNA
    n_samples=5,
    output_dir="./complexes",
)

# Or load from FASTA file
with open("sequences.fasta", "w") as f:
    f.write(">protein1\nMGKLF\n>rna1\nacgu\n")

samples = generate_samples(
    vae,
    sequence=[],  # Not used when loading from file
    n_samples=5,
    output_dir="./complexes",
)
```

### Reconstruction and Interpolation

```python
from ciffy.nn.vae import reconstruct_polymer, interpolate_structures

# Load existing structure
p1 = ciffy.load("structure1.cif", backend="torch").to("cuda")

# Reconstruct through VAE
recon = reconstruct_polymer(vae, p1, sample_latent=False)
recon.numpy().write("reconstructed.cif")

# Interpolate between two structures in latent space
p2 = ciffy.load("structure2.cif", backend="torch").to("cuda")
frames = interpolate_structures(
    vae, p1, p2,
    n_steps=20,
    output_dir="./interpolation"
)

print(f"Generated {len(frames)} interpolation frames")
```

### Batch Inference via CLI

Run inference on multiple sequences with a YAML configuration file:

**Config file** (`examples/configs/inference_example.yaml`):

```yaml
model:
  checkpoint_path: ./checkpoints/vae_best.pt
  model_type: vae
  device: auto

input:
  # Option 1: Inline sequences
  sequences:
    - MGKLF
    - acgu

  # Option 2: Load from FASTA file
  # sequence_file: sequences.fasta

sampling:
  n_samples: 10
  temperature: 1.0
  seed: 42

output:
  output_dir: ./inference_output
  id_prefix: gen_
```

**Run inference**:

```bash
# Copy example config to personal directory first:
cp examples/configs/inference_example.yaml configs/inference.yaml

# Single config
ciffy inference configs/inference.yaml

# Multiple configs (parallel across GPUs)
ciffy inference configs/*.yaml

# Sequential execution
ciffy inference configs/*.yaml --sequential

# Specific device
ciffy inference configs/*.yaml --device cpu
```

**Results**:

```
============================================================
Ciffy Inference Runner
============================================================
Configs: 2
Parallel: True
Device: auto

  1. configs/inference_small.yaml
  2. configs/inference_large.yaml

Running inference...
------------------------------------------------------------

============================================================
Results
============================================================
Job              Status     Structures  Sequences  Device    Time
-----------      --------   ----------  ---------  --------  ----------
inference_small  success    100         10         cuda:0    5.2s
inference_large  success    500         50         cuda:1    12.3s
-----------      --------   ----------  ---------  --------  ----------
Total: 2/2 succeeded, 600 structures in 17.5s
```

### Using FASTA Files for Sequences

Create a FASTA file with sequences:

```fasta
>protein1
MGKLF
>rna1
acgu
>protein2
ARNDCEQGHILKMFPSTWYV
```

Copy and use the FASTA example config:

```bash
# Copy FASTA example config
cp examples/configs/inference_from_fasta.yaml configs/inference_fasta.yaml

# Edit with your FASTA file path
# Then run:
ciffy inference configs/inference_fasta.yaml
```

Or create your own config with:

```yaml
input:
  sequence_file: sequences.fasta  # Auto-detects format
```

The sequence IDs (lines starting with '>') are used for output filenames.

### Protocol and Extension

The inference system is built on protocols, making it easy to add new model types:

```python
from ciffy.nn import PolymerGenerativeModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ciffy import Polymer

# Any model implementing this protocol works with the inference system:
class MyModel:
    def sample(
        self,
        template: Polymer,
        n_samples: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ) -> list[Polymer]:
        """Generate conformations from template."""
        ...

# Then use with generate_samples:
from ciffy.nn import generate_samples

samples = generate_samples(
    MyModel(),
    sequence="MGKLF",
    n_samples=10,
    output_dir="./outputs",
)
```

### Programmatic Inference

```python
from ciffy.nn import load_model_from_checkpoint, generate_samples

# Load any registered model
model, checkpoint_info = load_model_from_checkpoint(
    "checkpoints/vae_best.pt",
    device="cuda"
)

print(f"Model: {model.__class__.__name__}")
print(f"Trained for {checkpoint_info['epoch']} epochs")
print(f"Best loss: {checkpoint_info['best_loss']:.4f}")

# Generate samples
samples = generate_samples(
    model,
    sequence="MGKLF",
    n_samples=10,
    output_dir="./outputs",
)
```

## Performance Tips

1. **Load once, reuse**: Parse CIF files once and keep polymers in memory
2. **Use GPU**: Move to CUDA for large structures
3. **Mixed precision**: Use `torch.float16` or `torch.bfloat16` for large batches
4. **Avoid repeated conversions**: Stay in one backend throughout training
5. **Parallel inference**: Use `ciffy inference` with multiple GPUs for batch generation

```python
# Good: Load once
polymers = [ciffy.load(f, backend="torch").to("cuda") for f in files]

# Bad: Load repeatedly
for epoch in range(100):
    polymer = ciffy.load(file, backend="torch")  # Slow!

# Parallel inference via CLI (auto-distributes across GPUs)
# ciffy inference configs/*.yaml
```
