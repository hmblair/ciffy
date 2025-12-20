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
    print(f"{chain.names[0]}: {chain.size()} atoms")

# Iterate over only RNA chains
for chain in polymer.chains(ciffy.RNA):
    print(f"RNA chain {chain.names[0]}: {chain.size(ciffy.RESIDUE)} residues")

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

## Residue Selection

Select specific residue types using `by_residue()`:

```python
from ciffy.biochemistry import Residue

# Get all adenosine residues
adenosines = polymer.by_residue(Residue.ADE)

# Get all purines (A and G)
purines = polymer.by_residue([Residue.ADE, Residue.GUA])

# Get all pyrimidines (C and U)
pyrimidines = polymer.by_residue([Residue.CYT, Residue.URA])

# Amino acids use 3-letter codes too
alanines = polymer.by_residue(Residue.ALA)
```

## Atom Selection

### By Atom Type Index

Use `by_atom()` to select atoms by their type index:

```python
from ciffy.biochemistry import Residue

# Get all N1 atoms from adenosines
n1_atoms = polymer.by_atom(Residue.A.N1)

# Get multiple atom types
c1_prime = polymer.by_atom([
    Residue.A.C1p,
    Residue.G.C1p,
])
```

### Backbone Atoms

Select backbone atoms (works for RNA, DNA, and proteins):

```python
backbone = polymer.backbone()
print(f"Backbone atoms: {backbone.size()}")
```

### Specific Nucleotide Atoms

ciffy provides enums for all standard atoms:

```python
from ciffy.biochemistry import Residue

# Access nucleotide atoms via Residue enum
# Residue.A = Adenosine, Residue.G = Guanosine, etc.

# Examples of available atoms
Residue.A.N1   # N1 atom in adenosine
Residue.A.N3   # N3 atom in adenosine
Residue.A.C1p  # C1' sugar atom
Residue.A.P    # Phosphate

# Same pattern for other nucleotides
Residue.G.N1   # N1 in guanosine
Residue.C.N1   # N1 in cytosine
Residue.U.N1   # N1 in uridine
```

### Specialized Selection Methods

ciffy provides convenience methods for common structural selections:

```python
# Works on all molecule types
backbone = polymer.backbone()     # Sugar-phosphate (RNA/DNA) or N-CA-C-O (protein)

# Nucleic acid specific
bases = polymer.nucleobase()      # RNA nucleobases only
phosphates = polymer.phosphate()  # RNA/DNA phosphate groups

# Protein specific
sidechains = polymer.sidechain()  # Amino acid sidechains
```

| Method | Molecule | Atoms | Use Case |
|--------|----------|-------|----------|
| `backbone()` | RNA, DNA, Protein | Sugar-phosphate or N-CA-C-O | Backbone analysis |
| `nucleobase()` | RNA | Ring atoms (N1-N9, C2-C8) | Base pairing, stacking |
| `phosphate()` | RNA, DNA | P, OP1, OP2, OP3 | Phosphate contacts |
| `sidechain()` | Protein | CB onwards | Sidechain packing |

### Computing Nucleobase Centers

Get the center of mass for each nucleobase in an RNA:

```python
import ciffy

rna = ciffy.load("structure.cif").by_type(ciffy.RNA)

# Get nucleobase atoms and compute per-residue centers
_, nucleobase_centers = rna.nucleobase().center(ciffy.RESIDUE)
print(nucleobase_centers.shape)  # (num_residues, 3)
```

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

# Get nucleobases of first chain
chain_a_bases = polymer.by_index(0).nucleobase()

# Polymer-only, then by chain
clean = polymer.poly().by_index([0, 1])
```
