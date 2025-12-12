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

## Performance Tips

1. **Load once, reuse**: Parse CIF files once and keep polymers in memory
2. **Use GPU**: Move to CUDA for large structures
3. **Mixed precision**: Use `torch.float16` or `torch.bfloat16` for large batches
4. **Avoid repeated conversions**: Stay in one backend throughout training

```python
# Good: Load once
polymers = [ciffy.load(f, backend="torch").to("cuda") for f in files]

# Bad: Load repeatedly
for epoch in range(100):
    polymer = ciffy.load(file, backend="torch")  # Slow!
```
