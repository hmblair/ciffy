# RNA Structure Analysis

This guide covers RNA-specific analysis in ciffy, including backbone geometry, base analysis, and nucleotide-level operations.

## Loading RNA Structures

```python
import ciffy

# Load structure
polymer = ciffy.load("rna_structure.cif")

# Extract only RNA chains
rna = polymer.molecule_type(ciffy.RNA)

# Check molecule type
for chain in polymer.chains():
    if chain.istype(ciffy.RNA):
        print(f"Chain {chain.names[0]} is RNA")
```

## RNA Sequence

### Getting the Sequence

```python
rna = ciffy.load("structure.cif").molecule_type(ciffy.RNA)

# One-letter sequence
for chain in rna.chains():
    seq = chain.sequence_str()
    print(f"Chain {chain.names[0]}: {seq}")
# Output: Chain A: GCUAGCUAGCUA...
```

### Residue Types

```python
from ciffy import Residue

# Standard RNA residues
Residue.A    # Adenosine
Residue.C    # Cytidine
Residue.G    # Guanosine
Residue.U    # Uridine

# Access residue information
sequence = rna.sequence
print(f"First residue: {Residue(sequence[0]).name}")
```

### Selecting by Residue Type

```python
from ciffy import Residue

# Get all adenosines
adenosines = rna.residue_type(Residue.A)

# Get purines (A and G)
purines = rna.residue_type([Residue.A, Residue.G])

# Get pyrimidines (C and U)
pyrimidines = rna.residue_type([Residue.C, Residue.U])
```

## Structural Components

### Backbone Atoms

The RNA backbone consists of sugar-phosphate atoms:

```python
# Get backbone atoms
backbone = rna.backbone()
print(f"Backbone atoms: {backbone.size()}")

# Backbone includes: P, OP1, OP2, O5', C5', C4', C3', O3', C2', O2', C1', O4'
```

### Nucleobase Atoms

```python
# Get nucleobase atoms (ring atoms only)
bases = rna.nucleobase()
print(f"Base atoms: {bases.size()}")

# Nucleobase atoms: N1, C2, N3, C4, C5, C6, N7, C8, N9 (purines)
#                   N1, C2, O2, N3, C4, O4/N4, C5, C6 (pyrimidines)
```

### Phosphate Groups

```python
# Get phosphate atoms only
phosphates = rna.phosphate()
print(f"Phosphate atoms: {phosphates.size()}")

# Phosphate includes: P, OP1, OP2, OP3
```

## Base Geometry

### Nucleobase Centers

Compute the center of mass for each nucleobase:

```python
rna = ciffy.load("structure.cif").molecule_type(ciffy.RNA)

# Get nucleobase atoms and their centers
bases = rna.nucleobase()
_, base_centers = bases.center(ciffy.RESIDUE)

print(f"Base centers shape: {base_centers.shape}")  # (num_residues, 3)
```

### Base-Base Distances

```python
import numpy as np

# Compute pairwise distances between base centers
distances = np.linalg.norm(
    base_centers[:, None, :] - base_centers[None, :, :],
    axis=-1
)

# Find close bases (potential base pairs)
close_pairs = np.argwhere((distances < 12.0) & (distances > 0))
for i, j in close_pairs:
    if i < j:  # Avoid duplicates
        print(f"Residues {i+1} - {j+1}: {distances[i,j]:.1f} Å")
```

### Base Stacking Analysis

```python
import numpy as np

rna = ciffy.load("structure.cif").molecule_type(ciffy.RNA)

# Get sequential base-base distances
bases = rna.nucleobase()
_, centers = bases.center(ciffy.RESIDUE)

# Distance between consecutive bases
stack_distances = np.linalg.norm(centers[1:] - centers[:-1], axis=1)

print(f"Mean stacking distance: {stack_distances.mean():.2f} Å")
print(f"Stacking distance std: {stack_distances.std():.2f} Å")

# Typical stacking: 3.3-3.5 Å
unstacked = np.sum(stack_distances > 5.0)
print(f"Potentially unstacked positions: {unstacked}")
```

## Phosphate-Phosphate Distances

P-P distances are useful for secondary structure analysis:

```python
import numpy as np
from ciffy import Residue

rna = ciffy.load("structure.cif").molecule_type(ciffy.RNA)

# Get P atom coordinates using atom_type
p_atoms = rna.atom_type(Residue.A.P)
p_coords = p_atoms.coordinates

# Compute P-P distances
pp_distances = np.linalg.norm(
    p_coords[:, None, :] - p_coords[None, :, :],
    axis=-1
)

# Sequential P-P distance (should be ~5.8-6.0 Å for A-form)
sequential_pp = np.diag(pp_distances, k=1)
print(f"Mean sequential P-P: {sequential_pp.mean():.2f} Å")
```

## Multi-Chain RNA Analysis

### Analyzing Each Chain

```python
rna = ciffy.load("ribosome.cif").molecule_type(ciffy.RNA)

for chain in rna.chains():
    name = chain.names[0]
    n_res = chain.size(ciffy.RESIDUE)
    seq = chain.sequence_str()

    # Compute radius of gyration
    centered, _ = chain.center(ciffy.MOLECULE)
    rg = (centered.coordinates ** 2).sum(axis=1).mean() ** 0.5

    print(f"Chain {name}: {n_res} nt, Rg = {rg:.1f} Å")
    print(f"  Sequence: {seq[:20]}...")
```

### Inter-Chain Contacts

```python
import numpy as np

rna = ciffy.load("complex.cif").molecule_type(ciffy.RNA)

chains = list(rna.chains())
for i, c1 in enumerate(chains):
    for j, c2 in enumerate(chains):
        if i >= j:
            continue

        coords1 = c1.coordinates
        coords2 = c2.coordinates

        # Cross-chain distances
        cross_dist = np.linalg.norm(
            coords1[:, None, :] - coords2[None, :, :],
            axis=-1
        )
        min_dist = cross_dist.min()

        print(f"Chains {c1.names[0]}-{c2.names[0]}: min distance = {min_dist:.1f} Å")
```

## Complete RNA Analysis Example

```python
import ciffy
import numpy as np

def analyze_rna(cif_file):
    """Complete RNA structure analysis."""
    polymer = ciffy.load(cif_file)
    rna = polymer.molecule_type(ciffy.RNA)

    if rna.size() == 0:
        print("No RNA chains found")
        return

    print(f"Structure: {polymer.pdb_id}")
    print(f"RNA chains: {len(rna.names)}")
    print(f"Total nucleotides: {rna.size(ciffy.RESIDUE)}")
    print()

    # Per-chain analysis
    for chain in rna.chains():
        name = chain.names[0]
        n_res = chain.size(ciffy.RESIDUE)

        print(f"Chain {name} ({n_res} nt)")
        print(f"  Sequence: {chain.sequence_str()[:50]}...")

        # Radius of gyration
        centered, _ = chain.center(ciffy.MOLECULE)
        rg = np.sqrt((centered.coordinates ** 2).sum(axis=1).mean())
        print(f"  Radius of gyration: {rg:.1f} Å")
        print()

# Run analysis
analyze_rna("structure.cif")
```

## Working with Modified Nucleotides

RNA structures often contain modified nucleotides:

```python
from ciffy import Residue

rna = ciffy.load("trna.cif").molecule_type(ciffy.RNA)

# Check for common modifications
sequence = rna.sequence
for i, res_idx in enumerate(sequence):
    res = Residue(res_idx)
    if res not in [Residue.A, Residue.C, Residue.G, Residue.U]:
        print(f"Position {i+1}: {res.name} (modified)")
```

## Next Steps

- [Selection and Filtering](selection.md) - More selection techniques
- [Structural Analysis](analysis.md) - RMSD, alignment
- [Deep Learning](deep-learning.md) - Using RNA structures with PyTorch
- [Protein Analysis](protein.md) - Working with proteins
