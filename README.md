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
sequence = polymer.str()          # Sequence string

# Geometric operations
centered, means = polymer.center(ciffy.MOLECULE)
aligned, Q = polymer.align(ciffy.CHAIN)
distances = polymer.pairwise_distances(ciffy.RESIDUE)

# Selection
rna_chains = polymer.subset(ciffy.RNA)
backbone = polymer.backbone()

# Molecule type per chain (parsed from CIF _entity_poly block)
mol_types = polymer.molecule_type  # Array of Molecule enum values

# Load with entity descriptions (off by default for performance)
polymer = ciffy.load("structure.cif", load_descriptions=True)
descriptions = polymer.descriptions  # List of description strings per chain

# Iterate over chains
for chain in polymer.chains(ciffy.RNA):
    print(chain.id(), chain.str())

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

## Module Structure

```
ciffy/
├── backend/        # NumPy/PyTorch abstraction layer
├── types/          # Scale, Molecule enums (Molecule is auto-generated)
├── biochemistry/   # Element, Residue, atom definitions (mostly auto-generated)
├── operations/     # Reduction, alignment operations
├── io/             # File loading and writing
├── src/codegen/    # Code generation from PDB Chemical Component Dictionary
└── utils/          # Helper functions and base classes
```

## Development

### Code Generation

Most biochemistry definitions are auto-generated from the [PDB Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (CCD). The generator runs automatically during `pip install`:

```bash
# Generated files (in .gitignore):
ciffy/biochemistry/_generated_atoms.py      # Atom indices per residue
ciffy/biochemistry/_generated_elements.py   # Element enum with atomic numbers
ciffy/biochemistry/_generated_residues.py   # Residue enum and mappings
ciffy/types/molecule.py                     # Molecule type enum
ciffy/src/hash/*.c                          # C hash tables for fast lookup
```

To regenerate manually (requires gperf):

```bash
# CCD is auto-downloaded to ~/.cache/ciffy/ on first run
python ciffy/src/codegen/generate.py ~/.cache/ciffy/components.cif
```

To add new residues, edit `RESIDUE_WHITELIST` in `ciffy/src/codegen/generate.py`.

## Testing

```bash
pip install pytest
pytest tests/
```
