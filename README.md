## Overview

`ciffy` is a fast CIF file parser for molecular structures, with a C backend and Python interface.

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
pip install torch-scatter --no-build-isolation
pip install -e .
```

### Note on torch-scatter

`ciffy` requires `torch-scatter`, which may need to be installed separately depending on your platform:

**Linux/Windows with CUDA:**
```bash
# Replace CUDA with your version (e.g., cu118, cu121) or cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+${CUDA}.html
```

**macOS or if pre-built wheels are unavailable:**
```bash
pip install torch-scatter --no-build-isolation
```

## Usage

```python
import ciffy

# Load a structure from a CIF file
polymer = ciffy.load("structure.cif")

# Basic information
print(polymer)  # Summary of chains, residues, atoms

# Access coordinates and properties
coords = polymer.coordinates      # (N, 3) tensor
atoms = polymer.atoms             # (N,) tensor of atom types
sequence = polymer.str()          # Sequence string

# Geometric operations
centered, means = polymer.center(ciffy.MOLECULE)
aligned, Q = polymer.align(ciffy.CHAIN)
distances = polymer.pd(ciffy.RESIDUE)

# Selection
rna_chains = polymer.subset(ciffy.RNA)
backbone = polymer.backbone()

# Iterate over chains
for chain in polymer.chains(ciffy.RNA):
    print(chain.id(), chain.str())

# Compute RMSD between structures
rmsd = ciffy.rmsd(polymer1, polymer2, ciffy.MOLECULE)
```

## Module Structure

```
ciffy/
├── types/          # Scale, Molecule enums
├── biochemistry/   # Element, Residue, nucleotide definitions
├── operations/     # Reduction, alignment operations
├── io/             # File loading and writing
└── utils/          # Helper functions and base classes
```

## Testing

```bash
pip install pytest
pytest tests/
```
