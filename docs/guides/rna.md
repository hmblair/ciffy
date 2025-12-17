# RNA Structure Analysis

This guide covers RNA-specific analysis in ciffy, including backbone geometry, base analysis, and nucleotide-level operations.

## Loading RNA Structures

```python
import ciffy

# Load structure
polymer = ciffy.load("rna_structure.cif")

# Extract only RNA chains
rna = polymer.by_type(ciffy.RNA)

# Check molecule type
for chain in polymer.chains():
    if chain.istype(ciffy.RNA):
        print(f"Chain {chain.names[0]} is RNA")
```

## RNA Sequence

### Getting the Sequence

```python
rna = ciffy.load("structure.cif").by_type(ciffy.RNA)

# One-letter sequence
for chain in rna.chains():
    seq = chain.sequence_str()
    print(f"Chain {chain.names[0]}: {seq}")
# Output: Chain A: GCUAGCUAGCUA...
```

### Residue Types

```python
from ciffy.biochemistry import Residue

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
from ciffy.biochemistry import Residue

# Get all adenosines
adenosines = rna.by_residue(Residue.A)

# Get purines (A and G)
purines = rna.by_residue([Residue.A, Residue.G])

# Get pyrimidines (C and U)
pyrimidines = rna.by_residue([Residue.C, Residue.U])
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

## Backbone Dihedrals

RNA backbone conformation is defined by six torsion angles per residue:

| Angle | Atoms | Description |
|-------|-------|-------------|
| α (alpha) | O3'(i-1) - P - O5' - C5' | Phosphate rotation |
| β (beta) | P - O5' - C5' - C4' | O5'-C5' bond |
| γ (gamma) | O5' - C5' - C4' - C3' | C5'-C4' bond |
| δ (delta) | C5' - C4' - C3' - O3' | Sugar pucker related |
| ε (epsilon) | C4' - C3' - O3' - P(i+1) | C3'-O3' bond |
| ζ (zeta) | C3' - O3' - P(i+1) - O5'(i+1) | O3'-P bond |

### Accessing Backbone Dihedrals

```python
import ciffy
import numpy as np

rna = ciffy.load("structure.cif").by_type(ciffy.RNA)

# Get individual dihedral types
alpha = rna.dihedral(ciffy.DihedralType.ALPHA)
beta = rna.dihedral(ciffy.DihedralType.BETA)
gamma = rna.dihedral(ciffy.DihedralType.GAMMA)
delta = rna.dihedral(ciffy.DihedralType.DELTA)
epsilon = rna.dihedral(ciffy.DihedralType.EPSILON)
zeta = rna.dihedral(ciffy.DihedralType.ZETA)

# Convert to degrees
alpha_deg = np.degrees(alpha)
print(f"Alpha angles: mean={alpha_deg.mean():.1f}, std={alpha_deg.std():.1f}")
```

### Glycosidic Angle (χ)

The chi angle describes base orientation relative to sugar:

```python
# Chi angle for purines (A, G): O4' - C1' - N9 - C4
chi_pur = rna.dihedral(ciffy.DihedralType.CHI_PURINE)

# Chi angle for pyrimidines (C, U): O4' - C1' - N1 - C2
chi_pyr = rna.dihedral(ciffy.DihedralType.CHI_PYRIMIDINE)

# Syn vs anti conformation
# Anti: chi ~ -160° to -60° (most common)
# Syn: chi ~ 30° to 90° (rare, found in Z-DNA/RNA)
```

### Analyzing Backbone Conformations

```python
import numpy as np

rna = ciffy.load("structure.cif").by_type(ciffy.RNA)

# Get all backbone dihedrals for analysis
dihedrals = {
    'alpha': np.degrees(rna.dihedral(ciffy.DihedralType.ALPHA)),
    'beta': np.degrees(rna.dihedral(ciffy.DihedralType.BETA)),
    'gamma': np.degrees(rna.dihedral(ciffy.DihedralType.GAMMA)),
    'delta': np.degrees(rna.dihedral(ciffy.DihedralType.DELTA)),
    'epsilon': np.degrees(rna.dihedral(ciffy.DihedralType.EPSILON)),
    'zeta': np.degrees(rna.dihedral(ciffy.DihedralType.ZETA)),
}

# Print statistics
for name, angles in dihedrals.items():
    valid = angles[~np.isnan(angles)]
    print(f"{name:8s}: mean={valid.mean():7.1f}°, std={valid.std():5.1f}°")
```

## Sugar Pucker Analysis

The delta angle correlates with sugar pucker:

```python
import numpy as np

rna = ciffy.load("structure.cif").by_type(ciffy.RNA)
delta = np.degrees(rna.dihedral(ciffy.DihedralType.DELTA))

# C3'-endo (A-form): delta ~ 80°
# C2'-endo (B-form): delta ~ 140°
c3_endo = np.sum((delta > 60) & (delta < 100))
c2_endo = np.sum((delta > 120) & (delta < 160))

print(f"C3'-endo (A-form): {c3_endo} residues")
print(f"C2'-endo (B-form): {c2_endo} residues")
```

## Base Geometry

### Nucleobase Centers

Compute the center of mass for each nucleobase:

```python
rna = ciffy.load("structure.cif").by_type(ciffy.RNA)

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

rna = ciffy.load("structure.cif").by_type(ciffy.RNA)

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
rna = ciffy.load("structure.cif").by_type(ciffy.RNA)

# Get phosphate atoms
phosphates = rna.phosphate()

# Get P atom coordinates (one per residue)
# Filter to just P atoms (not OP1, OP2)
from ciffy.biochemistry import Backbone
p_atoms = rna.by_atom(Backbone.A_P)  # Adenosine P atom index

# Compute P-P distances
p_coords = p_atoms.coordinates
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
rna = ciffy.load("ribosome.cif").by_type(ciffy.RNA)

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
rna = ciffy.load("complex.cif").by_type(ciffy.RNA)

chains = list(rna.chains())
for i, c1 in enumerate(chains):
    for j, c2 in enumerate(chains):
        if i >= j:
            continue

        # Compute minimum distance between chains
        d1 = c1.pairwise_distances()  # Within c1
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
    rna = polymer.by_type(ciffy.RNA)

    if rna.size() == 0:
        print("No RNA chains found")
        return

    print(f"Structure: {polymer.id()}")
    print(f"RNA chains: {len(rna.names)}")
    print(f"Total nucleotides: {rna.size(ciffy.RESIDUE)}")
    print()

    # Per-chain analysis
    for chain in rna.chains():
        name = chain.names[0]
        n_res = chain.size(ciffy.RESIDUE)

        print(f"Chain {name} ({n_res} nt)")
        print(f"  Sequence: {chain.sequence_str()[:50]}...")

        # Backbone geometry
        delta = np.degrees(chain.dihedral(ciffy.DihedralType.DELTA))
        valid_delta = delta[~np.isnan(delta)]
        c3_endo = np.sum((valid_delta > 60) & (valid_delta < 100))

        print(f"  Sugar pucker: {100*c3_endo/len(valid_delta):.0f}% C3'-endo")

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
from ciffy.biochemistry import Residue

rna = ciffy.load("trna.cif").by_type(ciffy.RNA)

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
