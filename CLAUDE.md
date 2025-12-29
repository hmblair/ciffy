# Claude Code Instructions

## Project Overview

Ciffy is a library for researchers to **load, inspect, manipulate, and predict macromolecular structures** (proteins, RNA, DNA).

### Design Priorities

1. **Ease of use and predictable API** - The API should be intuitive and consistent. Researchers should be able to accomplish common tasks with minimal code. Method names should be self-explanatory.

2. **Performance** - Structures can contain hundreds of thousands of atoms. Operations must be efficient.

### Implementation Philosophy

- **Arrays everywhere**: All data is stored in contiguous arrays (NumPy or PyTorch) for vectorized operations and GPU compatibility. Avoid Python loops over atoms/residues.

- **Enums for readability**: Integer arrays store indices, but enums (like `Residue.A`, `Molecule.RNA`, `Scale.ATOM`) provide human-readable access. This gives both performance and clarity.

- **Hierarchical scales**: Structures have multiple scales - atoms, residues, chains, molecules. The `Scale` enum and methods like `size(Scale.RESIDUE)` make this hierarchy explicit.

- **Backend agnostic**: Core operations work with both NumPy and PyTorch, enabling seamless CPU/GPU workflows.

## Python Environment

Always use the mambaforge Python binary for running Python commands:

```bash
/Users/hmblair/mambaforge/bin/python
```

Do NOT use `python3` or `python` directly, as this may invoke the wrong interpreter.

## Remote GPU Execution with rpy

When you need to run code that requires a GPU (training models, large batch inference, CUDA operations), use `rpy` to execute on the remote GPU cluster. The local machine has no GPU.

### When to Use rpy

- **Training neural networks** - Any PyTorch/TensorFlow training loop
- **GPU-accelerated inference** - Batch predictions on large datasets
- **CUDA operations** - Anything requiring `torch.cuda` or GPU tensors
- **Memory-intensive operations** - When local RAM is insufficient

### Data Location
RNA .cif files are located at `/home/hmblair/data/rna`.

### Project Sync

Before running code that imports ciffy, sync the project to the remote:

```bash
# Sync ciffy to remote and reinstall (run from ciffy directory)
rpy gpu --sync

# Sync without reinstalling (faster, use when only scripts changed)
rpy gpu --sync --no-install
```

This rsyncs the project to `/home/hmblair/academic/software/ciffy` and runs `pip install -e .` on the remote. **Sync again whenever you modify ciffy source files.**

### Basic Workflow

```bash
# Check available GPUs first
rpy gpu --gpus

# Run a script and stream output back
rpy gpu train.py --epochs 10 # or pipe python code

# For long-running jobs, detach to survive disconnection
rpy gpu -d train.py --epochs 100
# Returns: job ID like 20251229-161516

# Monitor detached jobs
rpy gpu --status 20251229-161516  # exit 0=running, 1=done
rpy gpu --log 20251229-161516     # view output
rpy gpu --jobs                     # list all jobs
rpy gpu --kill 20251229-161516    # stop if needed
```

### Agent Workflow for Long Jobs

1. `rpy gpu --gpus` - verify GPUs are available
2. `rpy gpu --sync` - ensure latest ciffy code is on remote
3. `rpy gpu -d script.py` - launch detached, note the job ID
4. Poll `rpy gpu --status <job>` periodically (exit code 0 = still running)
5. When done (exit code 1), fetch results with `rpy gpu --log <job>`

### Notes

- The `gpu` alias is configured in `~/.config/rpy` with the correct host and Python venv
- Scripts are copied to the remote, executed, and output streams back
- For piped code: `echo 'print(torch.cuda.device_count())' | rpy gpu`

## Git Safety

**NEVER discard unstaged changes.** If you see modified files unrelated to the current task, leave them alone. Do not run `git checkout --`, `git restore`, or `git stash` on files you didn't modify in this session. These may be work-in-progress from other agents or tasks.

### Worktree Workflow for Complex Changes

When working on a complex change (multiple files, significant refactoring, new features), use a separate worktree to avoid conflicts with other agents:

```bash
# 1. Create a new branch and worktree
git worktree add ../ciffy-<feature-name> -b <feature-name>
cd ../ciffy-<feature-name>

# 2. Make changes, commit as needed
# ... work in this directory ...
git add <files>
git commit -m "Description of changes"

# 3. When ready, ask user for permission to merge
# "The feature is complete. May I merge <feature-name> into master?"

# 4. After user approval, merge into master
cd /Users/hmblair/academic/software/ciffy
git checkout master
git merge <feature-name>

# 5. Clean up the worktree
git worktree remove ../ciffy-<feature-name>
git branch -d <feature-name>
```

This keeps each agent's work isolated until explicitly merged.

## Build Commands

```bash
/Users/hmblair/mambaforge/bin/pip install -e .
```

## Test Commands

```bash
CIFFY_LOG_LEVEL=WARNING /Users/hmblair/mambaforge/bin/python -m pytest tests/ -n auto
```

## Default Test File

Use `tests/data/9MDS.cif` for manual testing:

```python
import ciffy
polymer = ciffy.load('tests/data/9MDS.cif')
```

## Large RNA Structure Database

For training models or large-scale analysis, use the RNA structure database:

```
/Users/hmblair/academic/data/structures/rna
```

This directory contains a large collection of RNA CIF files from the PDB.

## Polymer Class Usage

### Basic Info

```python
polymer                      # Print summary table (chains, types, residues, atoms)
polymer.pdb_id               # PDB identifier string
polymer.resolution           # Structure resolution in Angstroms (if available)
polymer.size()               # Total atom count
polymer.size(Scale.CHAIN)    # Number of chains
polymer.size(Scale.RESIDUE)  # Number of residues
polymer.empty()              # Check if polymer has no atoms
polymer.sequence_str()       # Single-letter sequence (e.g., "ACGU")
polymer.atom_names()         # List of atom name strings
```

### Fields and Arrays

```python
polymer.coordinates          # (N, 3) array of atom positions
polymer.atoms                # (N,) array of atom type indices
polymer.elements             # (N,) array of element indices
polymer.bfactors             # (N,) array of B-factors
polymer.sequence             # (R,) array of residue type indices
polymer.lengths              # (C,) array of residues per chain
polymer.names                # List of chain name strings
polymer.molecule_types       # (C,) array of molecule type per chain
polymer.bonds                # (B, 2) array of covalent bond atom pairs
```

### Counting and Hierarchy

```python
polymer.counts(Scale.RESIDUE)                    # Atoms per residue
polymer.counts(Scale.CHAIN)                      # Atoms per chain
polymer.counts(Scale.RESIDUE, per=Scale.CHAIN)   # Residues per chain
polymer.membership(Scale.RESIDUE)                # Which residue each atom belongs to
polymer.membership(Scale.CHAIN)                  # Which chain each atom belongs to
polymer.polymer_count                            # Number of polymer atoms
polymer.nonpoly()                                # Number of non-polymer atoms
```

### Selection Methods

```python
from ciffy import Scale, Molecule
from ciffy.biochemistry import Residue

polymer.by_index(0)              # Select chain by index
polymer.by_residue(Residue.A)    # Select residues by type
polymer.by_type(Molecule.RNA)    # Select chains by molecule type
polymer.select(0, Scale.RESIDUE) # Select residue by index
polymer.select([0, 2], Scale.CHAIN)  # Select multiple chains
polymer.poly()                   # Polymer atoms only (no HETATM)
polymer.hetero()                 # Non-polymer atoms only (water, ions)
polymer.canonical()              # Only canonical residue types (A/C/G/U, amino acids)
polymer.backbone()               # Backbone atoms
polymer.nucleobase()             # Nucleobase atoms (RNA/DNA)
polymer.phosphate()              # Phosphate atoms (RNA/DNA)
polymer.sidechain()              # Sidechain atoms (protein)
polymer.strip()                  # Remove unresolved (0-atom) residues
```

### Chain Iteration

```python
for chain in polymer.chains():           # Iterate all chains
    print(chain.pdb_id, chain.names[0])
```

### Geometry Operations

```python
polymer.center()                         # Center coordinates, return (centered, centroids)
polymer.pairwise_distances()             # Atom-atom distance matrix
polymer.pairwise_distances(Scale.RESIDUE)  # Residue centroid distances
polymer.knn(k=16)                        # K-nearest neighbors (k, N) indices
polymer.align(Scale.MOLECULE)            # Align to principal axes
polymer.bonded_distances(atom1, atom2)   # Distances between bonded atom types
```

### Reduction Operations

```python
polymer.reduce(features, Scale.RESIDUE)  # Reduce atom features to residue (mean)
polymer.reduce(features, Scale.CHAIN, rtype=Reduction.SUM)  # Sum to chain
polymer.expand(features, Scale.RESIDUE)  # Expand residue features to atoms
```

### Backend Conversion

```python
polymer.backend              # 'numpy' or 'torch'
polymer.device               # Device string ('cpu', 'cuda:0') or None for numpy
polymer.numpy()              # Convert to NumPy arrays
polymer.torch()              # Convert to PyTorch tensors
polymer.to('cuda')           # Move to GPU (torch only)
polymer.cpu()                # Move to CPU (torch only)
polymer.cuda()               # Shorthand for to('cuda')
polymer.copy()               # Deep copy of polymer
polymer.with_coordinates(coords)  # Copy with new coordinates
```

### I/O

```python
polymer.write('output.cif')  # Write to mmCIF file
```
