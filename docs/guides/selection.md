# Selection and Filtering

This guide covers how to select and filter molecular structures in ciffy.

## Molecule Type Selection

ciffy supports various molecule types. Use `by_type()` to filter by type:

```python
import ciffy

polymer = ciffy.load("structure.cif")

# Select by molecule type
rna_chains = polymer.by_type(ciffy.RNA)
protein_chains = polymer.by_type(ciffy.PROTEIN)
dna_chains = polymer.by_type(ciffy.DNA)
```

### Available Molecule Types

| Type | Description |
|------|-------------|
| `ciffy.PROTEIN` | Standard proteins (polypeptide L) |
| `ciffy.RNA` | RNA (polyribonucleotide) |
| `ciffy.DNA` | DNA (polydeoxyribonucleotide) |
| `Molecule.HYBRID` | DNA/RNA hybrids |
| `Molecule.LIGAND` | Small molecules, cofactors |
| `Molecule.ION` | Metal ions (Mg, K, etc.) |
| `Molecule.WATER` | Water molecules |

```python
from ciffy.types import Molecule

# Access all molecule types
ligands = polymer.by_type(Molecule.LIGAND)
ions = polymer.by_type(Molecule.ION)
```

### Iterating Over Chains

Use `chains()` to iterate, optionally filtering by type:

```python
# Iterate over all chains
for chain in polymer.chains():
    print(f"{chain.id()}: {chain.size()} atoms")

# Iterate over only RNA chains
for chain in polymer.chains(ciffy.RNA):
    print(f"RNA chain {chain.id()}: {chain.size(ciffy.RESIDUE)} residues")

# Check molecule type
if polymer.istype(ciffy.RNA):
    print("This is a single RNA chain")
```

## Chain Selection

Select specific chains by index:

```python
# Select first chain
chain_a = polymer.by_index(0)

# Select multiple chains
chains_ac = polymer.by_index([0, 2])

# Chain names are preserved
print(polymer.names)  # ['A', 'B', 'C', ...]
```

## Polymer vs Non-Polymer

Separate polymer atoms from heteroatoms (water, ions, ligands):

```python
# Get only polymer atoms (RNA, DNA, protein)
polymer_only = polymer.poly()

# Get only heteroatoms (water, ions, ligands)
heteroatoms = polymer.hetero()

# Check counts
print(f"Polymer atoms: {polymer.polymer_count}")
print(f"Non-polymer atoms: {polymer.nonpoly}")
```

!!! note
    The `poly()` result has valid residue information and supports residue-scale operations. The `hetero()` result does not have residue structure.

## Atom Selection

### By Atom Type Index

Use `by_atom()` to select atoms by their type index:

```python
from ciffy.biochemistry import Adenosine, Guanosine

# Get all N1 atoms from adenosines
n1_atoms = polymer.by_atom(Adenosine.N1)

# Get multiple atom types
c1_prime = polymer.by_atom([
    Adenosine.C1_PRIME,
    Guanosine.C1_PRIME,
])
```

### Backbone Atoms

Select backbone atoms (sugar-phosphate for RNA, N-CA-C-O for proteins):

```python
backbone = polymer.backbone()
print(f"Backbone atoms: {backbone.size()}")
```

### Specific Nucleotide Atoms

ciffy provides enums for all standard atoms:

```python
from ciffy.biochemistry import (
    Adenosine,   # A nucleotide atoms
    Cytosine,    # C nucleotide atoms
    Guanosine,   # G nucleotide atoms
    Uridine,     # U nucleotide atoms
)

# Examples of available atoms
Adenosine.N1      # N1 atom
Adenosine.N3      # N3 atom
Adenosine.C1_PRIME  # C1' sugar atom
Adenosine.P       # Phosphate
```

### Reference Frame Atoms

For structural analysis, ciffy provides predefined atom groups:

```python
from ciffy.biochemistry import COARSE, FRAMES, Backbone

# N1/N3 atoms for base pairing analysis
n1_n3_atoms = polymer.by_atom(COARSE.index())

# Reference frame atoms (C2, C4, C6 of each nucleotide)
frame_atoms = polymer.by_atom(FRAMES.index())

# Backbone atoms
backbone_atoms = polymer.by_atom(Backbone.index())
```

| Group | Atoms | Use Case |
|-------|-------|----------|
| `COARSE` | N1, N3 | Base pairing, coarse-grained models |
| `FRAMES` | C2, C4, C6 | Reference frame construction |
| `Backbone` | P, O5', C5', C4', C3', O3' | Backbone analysis |

## Boolean Masking

Use boolean masks for flexible selection:

```python
import numpy as np

# Create a mask
mask = polymer.elements == 7  # Nitrogen atoms only
nitrogen_atoms = polymer[mask]

# Combine conditions
mask = (polymer.elements == 7) & (polymer.atoms < 100)
filtered = polymer[mask]
```

### Creating Masks at Different Scales

Use `mask()` to create masks from indices:

```python
# Create atom mask from chain indices
atom_mask = polymer.mask([0, 2], source=ciffy.CHAIN, dest=ciffy.ATOM)

# Create residue mask from chain indices
res_mask = polymer.mask([0], source=ciffy.CHAIN, dest=ciffy.RESIDUE)
```

## Slicing

Select contiguous ranges of atoms:

```python
# First 100 atoms
first_100 = polymer[:100]

# Atoms 50-150
middle = polymer[50:150]

# Last 50 atoms
last_50 = polymer[-50:]
```

## Resolved Residues

Find and filter unresolved (missing) residues:

```python
# Get mask of resolved residues
resolved_mask = polymer.resolved(ciffy.RESIDUE)

# Remove unresolved residues
clean = polymer.strip(ciffy.RESIDUE)
```

## Combining Selections

Chain multiple selections together:

```python
# Get backbone atoms of RNA chains only
rna_backbone = polymer.by_type(ciffy.RNA).backbone()

# Get N1/N3 of first chain
from ciffy.biochemistry import COARSE
chain_a_n1n3 = polymer.by_index(0).by_atom(COARSE.index())

# Polymer-only, then by chain
clean = polymer.poly().by_index([0, 1])
```
