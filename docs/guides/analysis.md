# Structural Analysis

This guide covers structural analysis operations in ciffy, including RMSD computation, alignment, and distance calculations.

## Computing RMSD

The `rmsd()` function computes root-mean-square deviation between two structures using Kabsch alignment:

```python
import ciffy

p1 = ciffy.load("structure1.cif")
p2 = ciffy.load("structure2.cif")

# Compute RMSD (returns squared distance)
distance_sq = ciffy.rmsd(p1, p2)
rmsd_value = distance_sq.sqrt()  # Take sqrt for actual RMSD
```

### RMSD at Different Scales

Control the alignment scale with the `scale` parameter:

```python
# Molecule-level RMSD (align entire structure, single value)
mol_rmsd = ciffy.rmsd(p1, p2, scale=ciffy.MOLECULE)

# Per-chain RMSD (align each chain independently)
chain_rmsd = ciffy.rmsd(p1, p2, scale=ciffy.CHAIN)
print(f"Per-chain RMSD: {chain_rmsd}")  # Array with one value per chain
```

!!! tip
    Use `scale=ciffy.CHAIN` when comparing multi-chain structures where chains may have moved relative to each other.

## Centering Structures

Center coordinates around the centroid:

```python
# Center entire molecule
centered, centroid = polymer.center(ciffy.MOLECULE)
print(f"Original centroid: {centroid}")
# centered.coordinates now has mean [0, 0, 0]

# Center each chain independently
centered, centroids = polymer.center(ciffy.CHAIN)
# Each chain now centered at origin
```

## Superimposing Two Structures

Use `ciffy.align()` to superimpose one structure onto another using the Kabsch algorithm:

```python
import ciffy

# Load reference and mobile structures
reference = ciffy.load("reference.cif")
mobile = ciffy.load("mobile.cif")

# Align mobile onto reference
ref, aligned = ciffy.align(reference, mobile)

# ref is unchanged, aligned is mobile rotated/translated to minimize RMSD
print(f"RMSD after alignment: {ciffy.rmsd(ref, aligned).sqrt():.3f} Å")
```

The `align()` function:

- Returns a tuple `(reference, aligned_mobile)`
- Reference structure is unchanged
- Mobile structure is optimally rotated and translated
- Uses Kabsch algorithm (SVD-based) for optimal superposition

### Visualizing Alignment

```python
# Save aligned structures for visualization
reference.write("reference.cif")
aligned.write("aligned_mobile.cif")
# Open both in PyMOL/ChimeraX to verify alignment
```

### Alignment with Metrics

```python
# Compute TM-score on aligned structures
ref, aligned = ciffy.align(pred, target)
tm = ciffy.tm_score(aligned, ref)
print(f"TM-score: {tm:.3f}")
```

## Aligning to Principal Axes

Use `polymer.align()` to align a single structure to its principal axes:

```python
# Align to principal components
aligned, rotation_matrices = polymer.align(ciffy.MOLECULE)

# The aligned structure has:
# - Centered coordinates
# - Covariance matrix is diagonal
# - Consistent sign orientation
```

The alignment:
1. Centers the structure
2. Rotates to diagonalize the covariance matrix
3. Fixes signs based on third moments for reproducibility

## Pairwise Distances

Compute distance matrices:

```python
# Atom-atom distances
atom_distances = polymer.pairwise_distances()
print(f"Shape: {atom_distances.shape}")  # (N, N)

# Distances between chain centroids
chain_distances = polymer.pairwise_distances(scale=ciffy.CHAIN)
print(f"Shape: {chain_distances.shape}")  # (C, C)

# Distances between residue centroids
residue_distances = polymer.pairwise_distances(scale=ciffy.RESIDUE)
```

## Reduction Operations

Aggregate features from fine to coarse scales.

### Basic Reductions

```python
# Per-residue centroid (mean of atom coordinates)
residue_centers = polymer.reduce(polymer.coordinates, ciffy.RESIDUE)

# Per-chain centroid
chain_centers = polymer.reduce(polymer.coordinates, ciffy.CHAIN)

# Whole molecule centroid
mol_center = polymer.reduce(polymer.coordinates, ciffy.MOLECULE)
```

### Reduction Types

```python
from ciffy import Reduction

# Mean (default)
means = polymer.reduce(features, ciffy.RESIDUE, Reduction.MEAN)

# Sum
sums = polymer.reduce(features, ciffy.RESIDUE, Reduction.SUM)

# Min (returns values and indices)
min_vals, min_indices = polymer.reduce(features, ciffy.RESIDUE, Reduction.MIN)

# Max (returns values and indices)
max_vals, max_indices = polymer.reduce(features, ciffy.RESIDUE, Reduction.MAX)
```

| Reduction | Returns | Use Case |
|-----------|---------|----------|
| `MEAN` | Averaged values | Centroids, mean features |
| `SUM` | Summed values | Total charge, counts |
| `MIN` | (values, indices) | Find closest atom |
| `MAX` | (values, indices) | Find furthest atom |

### Per-Residue Reductions

For features already at residue level, use `rreduce()`:

```python
# Aggregate per-residue features to per-chain
per_residue = polymer.sequence  # One value per residue
per_chain = polymer.rreduce(per_residue, ciffy.CHAIN, Reduction.SUM)
```

## Expanding Features

Broadcast features from coarse to fine scales:

```python
# Broadcast chain features to atoms
chain_features = get_chain_features(polymer)  # Shape: (C, D)
atom_features = polymer.expand(chain_features, ciffy.CHAIN)  # Shape: (N, D)

# Broadcast chain features to residues
residue_features = polymer.expand(chain_features, ciffy.CHAIN, ciffy.RESIDUE)
```

## Counting

Count atoms or True values per unit:

```python
# Atoms per residue
atoms_per_res = polymer.sizes(ciffy.RESIDUE)

# Atoms per chain
atoms_per_chain = polymer.sizes(ciffy.CHAIN)

# Count specific atoms per residue
nitrogen_mask = polymer.elements == 7
nitrogens_per_res = polymer.count(nitrogen_mask, ciffy.RESIDUE)
```

## Statistical Moments

Compute moments of coordinate distributions:

```python
# First moment (mean) - same as reduce with MEAN
mean = polymer.moment(1, ciffy.CHAIN)

# Second moment (related to variance)
second = polymer.moment(2, ciffy.CHAIN)

# Third moment (related to skewness)
third = polymer.moment(3, ciffy.CHAIN)
```

## Complete Analysis Example

```python
import ciffy

# Load structure
polymer = ciffy.load("ribosome.cif", backend="torch")

# Get RNA chains only
rna = polymer.by_type(ciffy.RNA)

# Compute per-chain analysis
for i, chain in enumerate(rna.chains()):
    # Center the chain
    centered, centroid = chain.center(ciffy.MOLECULE)

    # Compute radius of gyration
    coords = centered.coordinates
    rg = (coords ** 2).sum(dim=1).mean().sqrt()

    # Get residue count
    n_residues = chain.size(ciffy.RESIDUE)

    print(f"Chain {chain.id()}: {n_residues} residues, Rg = {rg:.2f} Å")

# Compare two conformations
p1 = ciffy.load("conf1.cif", backend="torch")
p2 = ciffy.load("conf2.cif", backend="torch")

# Per-chain RMSD
rmsd_per_chain = ciffy.rmsd(p1, p2, scale=ciffy.CHAIN).sqrt()
for name, rmsd in zip(p1.names, rmsd_per_chain):
    print(f"Chain {name}: RMSD = {rmsd:.2f} Å")
```
