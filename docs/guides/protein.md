# Protein Structure Analysis

This guide covers protein-specific analysis in ciffy, including backbone geometry and residue-level operations.

## Loading Protein Structures

```python
import ciffy

# Load structure
polymer = ciffy.load("protein.cif")

# Extract only protein chains
protein = polymer.molecule_type(ciffy.PROTEIN)

# Check molecule type
for chain in polymer.chains():
    if chain.istype(ciffy.PROTEIN):
        print(f"Chain {chain.names[0]} is protein")
```

## Protein Sequence

### Getting the Sequence

```python
protein = ciffy.load("structure.cif").molecule_type(ciffy.PROTEIN)

# One-letter sequence
for chain in protein.chains():
    seq = chain.sequence_str()
    print(f"Chain {chain.names[0]}: {seq}")
# Output: Chain A: MGKLVFFAED...
```

### Residue Types

```python
from ciffy import Residue

# Standard amino acids
Residue.ALA  # Alanine
Residue.GLY  # Glycine
Residue.VAL  # Valine
# ... all 20 standard amino acids

# Access residue information
sequence = protein.sequence
print(f"First residue: {Residue(sequence[0]).name}")
```

### Selecting by Residue Type

```python
from ciffy import Residue

# Get all glycines
glycines = protein.residue_type(Residue.GLY)

# Get hydrophobic residues
hydrophobic = protein.residue_type([
    Residue.ALA, Residue.VAL, Residue.LEU,
    Residue.ILE, Residue.MET, Residue.PHE,
    Residue.TRP, Residue.PRO
])

# Get charged residues
charged = protein.residue_type([
    Residue.ASP, Residue.GLU,  # Negative
    Residue.LYS, Residue.ARG, Residue.HIS  # Positive
])
```

## Structural Components

### Backbone Atoms

The protein backbone consists of N-CA-C-O atoms:

```python
# Get backbone atoms
backbone = protein.backbone()
print(f"Backbone atoms: {backbone.size()}")

# Backbone includes: N, CA, C, O (4 atoms per residue)
```

### Sidechain Atoms

```python
# Get sidechain atoms
sidechains = protein.sidechain()
print(f"Sidechain atoms: {sidechains.size()}")

# Sidechains: all atoms except N, CA, C, O, and hydrogens
```

### CA Trace

Extract alpha-carbon coordinates:

```python
from ciffy import Residue

# Get CA atoms for a specific residue type
ala_ca = protein.atom_type(Residue.ALA.CA)

# For all CA atoms, use reduce to get one coordinate per residue
ca_coords = protein.reduce(protein.coordinates, ciffy.RESIDUE)

print(f"CA trace: {ca_coords.shape}")  # (num_residues, 3)
```

## Contact Analysis

### Residue-Residue Contacts

```python
import numpy as np
from ciffy import operations

protein = ciffy.load("structure.cif").molecule_type(ciffy.PROTEIN)

# Compute CA-CA distance matrix
distances = operations.pairwise_distances(protein, scale=ciffy.RESIDUE)

# Contact map (8 Å cutoff)
contacts = distances < 8.0

# Count contacts per residue
contact_count = contacts.sum(axis=1)
print(f"Mean contacts per residue: {contact_count.mean():.1f}")
```

### Long-Range Contacts

```python
import numpy as np
from ciffy import operations

protein = ciffy.load("structure.cif").molecule_type(ciffy.PROTEIN)
distances = operations.pairwise_distances(protein, scale=ciffy.RESIDUE)

# Find contacts between residues far apart in sequence
n_res = len(distances)
sequence_distance = np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :])

# Long-range: > 12 residues apart in sequence, < 8 Å in space
long_range = (sequence_distance > 12) & (distances < 8.0)

print(f"Long-range contacts: {long_range.sum() // 2}")  # Divide by 2 for unique pairs

# Find the pairs
pairs = np.argwhere(long_range & (np.triu(np.ones_like(long_range), k=1) > 0))
for i, j in pairs[:10]:  # First 10
    print(f"  Residues {i+1} - {j+1}: {distances[i,j]:.1f} Å")
```

## Sidechain Analysis

### Sidechain Centroids

```python
import numpy as np
from ciffy import Residue

protein = ciffy.load("structure.cif").molecule_type(ciffy.PROTEIN)

# Analyze sidechain centroids
sidechains = protein.sidechain()

if sidechains.size() > 0:
    _, sc_centers = sidechains.center(ciffy.RESIDUE)
    print(f"Sidechain centers: {sc_centers.shape}")
```

### Sidechain-Backbone Distances

```python
import numpy as np
from ciffy import Residue

protein = ciffy.load("structure.cif").molecule_type(ciffy.PROTEIN)

backbone = protein.backbone()
sidechains = protein.sidechain()

# Get per-residue centers
_, bb_centers = backbone.center(ciffy.RESIDUE)
_, sc_centers = sidechains.center(ciffy.RESIDUE)

# Distance from backbone to sidechain center
bb_sc_dist = np.linalg.norm(bb_centers - sc_centers, axis=1)

# Glycine has no sidechain
sequence = protein.sequence
gly_mask = np.array([Residue(s) == Residue.GLY for s in sequence])

print(f"Mean backbone-sidechain distance: {bb_sc_dist[~gly_mask].mean():.2f} Å")
```

## Multi-Chain Protein Analysis

### Interface Analysis

```python
import numpy as np

protein = ciffy.load("complex.cif").molecule_type(ciffy.PROTEIN)

chains = list(protein.chains())
if len(chains) >= 2:
    c1, c2 = chains[0], chains[1]

    # Compute cross-chain distances
    coords1 = c1.coordinates
    coords2 = c2.coordinates

    cross_dist = np.linalg.norm(
        coords1[:, None, :] - coords2[None, :, :],
        axis=-1
    )

    # Interface atoms: within 4 Å of other chain
    interface1 = (cross_dist.min(axis=1) < 4.0)
    interface2 = (cross_dist.min(axis=0) < 4.0)

    print(f"Chain {c1.names[0]} interface atoms: {interface1.sum()}")
    print(f"Chain {c2.names[0]} interface atoms: {interface2.sum()}")
```

### Per-Chain Analysis

```python
import numpy as np

protein = ciffy.load("complex.cif").molecule_type(ciffy.PROTEIN)

for chain in protein.chains():
    name = chain.names[0]
    n_res = chain.size(ciffy.RESIDUE)
    seq = chain.sequence_str()

    # Radius of gyration
    centered, _ = chain.center(ciffy.MOLECULE)
    rg = np.sqrt((centered.coordinates ** 2).sum(axis=1).mean())

    print(f"Chain {name}: {n_res} residues")
    print(f"  Sequence: {seq[:30]}...")
    print(f"  Radius of gyration: {rg:.1f} Å")
```

## Complete Protein Analysis Example

```python
import ciffy
import numpy as np

def analyze_protein(cif_file):
    """Complete protein structure analysis."""
    polymer = ciffy.load(cif_file)
    protein = polymer.molecule_type(ciffy.PROTEIN)

    if protein.size() == 0:
        print("No protein chains found")
        return

    print(f"Structure: {polymer.pdb_id}")
    print(f"Protein chains: {len(protein.names)}")
    print(f"Total residues: {protein.size(ciffy.RESIDUE)}")
    print()

    for chain in protein.chains():
        name = chain.names[0]
        n_res = chain.size(ciffy.RESIDUE)

        print(f"Chain {name} ({n_res} residues)")
        print(f"  Sequence: {chain.sequence_str()[:50]}...")

        # Radius of gyration
        centered, _ = chain.center(ciffy.MOLECULE)
        rg = np.sqrt((centered.coordinates ** 2).sum(axis=1).mean())
        print(f"  Radius of gyration: {rg:.1f} Å")
        print()

# Run analysis
analyze_protein("structure.cif")
```

## Working with Ligands

Proteins often have bound ligands:

```python
from ciffy import Molecule
import numpy as np

polymer = ciffy.load("protein_ligand.cif")

# Get protein and ligand separately
protein = polymer.molecule_type(ciffy.PROTEIN)
ligands = polymer.molecule_type(Molecule.LIGAND)

print(f"Protein atoms: {protein.size()}")
print(f"Ligand atoms: {ligands.size()}")

# Find protein atoms near ligand
if ligands.size() > 0:
    protein_coords = protein.coordinates
    ligand_coords = ligands.coordinates

    distances = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :],
        axis=-1
    )

    binding_site = distances.min(axis=1) < 4.0
    print(f"Binding site atoms: {binding_site.sum()}")
```

## Next Steps

- [Selection and Filtering](selection.md) - More selection techniques
- [Structural Analysis](analysis.md) - RMSD, alignment
- [Deep Learning](deep-learning.md) - Using protein structures with PyTorch
- [RNA Analysis](rna.md) - Working with nucleic acids
