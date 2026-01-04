# TODO

## HIGH Priority

### Handle Missing Atoms in PolymerFlowModel (Imputation)

**Goal**: Allow flow model to encode/decode structures with missing atoms.

**Context**: Currently, structures with missing atoms are filtered out during training because the flow model expects a fixed number of atoms per residue type. This can discard a significant portion of real PDB structures.

**Proposed approach (imputation)**:
1. During encoding, identify missing atoms by comparing actual vs expected count
2. Fill missing positions with the residue's mean coordinates (stored in PCA)
3. Encode the "completed" residue normally
4. On decode, complete structures are generated naturally

**Implementation**:
```python
def encode(self, coords, sequence):
    # For each residue, check if atoms are missing
    for i, res_type in enumerate(sequence):
        expected = self._atom_counts[res_type]
        actual = len(residue_coords)
        if actual < expected:
            # Impute missing atoms with mean positions
            residue_coords = self._impute_missing(residue_coords, res_type)
    # Continue with normal encoding...
```

**Files affected**:
- `ciffy/nn/flow/polymer.py` - Add `_impute_missing()` method to `PolymerFlowModel.encode()`
- `ciffy/nn/flow/residue/model.py` - Store atom masks or expected positions

**Effort**: ~50-100 lines, 2-4 hours
**Impact**: Fewer samples filtered during training, better data utilization

**Alternative approach (masked training)**:

Train the flow model to handle missing atoms natively by using masked inputs during training:

1. During training, randomly mask some atoms (set to zero or learnable mask token)
2. Model learns to produce valid latents even with partial input
3. At inference, naturally handles incomplete structures

```python
class MaskedPCAFlow(PCAFlow):
    def __init__(self, V, mean, mask_token=None):
        super().__init__(V, mean)
        # Learnable embedding for masked positions
        self.mask_embedding = nn.Parameter(torch.zeros(3))

    def encode(self, coords, mask=None):
        if mask is not None:
            # Replace masked positions with learned embedding
            coords = coords.clone()
            coords[mask] = self.mask_embedding
        return super().encode(coords)
```

**Pros**:
- Model learns robust representations
- No information loss from imputation
- Works even with many missing atoms

**Cons**:
- Requires retraining flow models
- More complex training procedure
- May need larger models for robustness

**Effort**: 1-2 days (includes retraining)
**Impact**: Most robust solution for incomplete structures

---

## MEDIUM Priority

### Refactor CLI Module to Reduce Duplication

**Goal**: Extract repeated patterns from `ciffy/cli/__main__.py` to reduce maintenance burden.

**Context**: The CLI module is 1,691 lines with significant code duplication identified during codebase audit.

**Duplicated patterns**:

| Pattern | Count | Lines |
|---------|-------|-------|
| Accelerator detection | 5 | 318-325, 447-454, 578-585, 726-733, 797-804 |
| Error handling | 4+ | 51-56, 72-77, 113-118, 173-178 |
| Output writing | 3 | 699-712, 773-784, 834-846 |
| Training header | 3 | 327-337, 486-495, 606-615 |
| W&B logger setup | 3 | 340-346, 498-504, 619-624 |

**Proposed helpers** (create `ciffy/cli/helpers.py`):

```python
def resolve_accelerator(accelerator: str) -> str:
    """Resolve 'auto' to actual accelerator (gpu/mps/cpu)."""
    if accelerator == "auto":
        import torch
        if torch.cuda.is_available():
            return "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return accelerator

def load_structure(filepath: str, **kwargs) -> Polymer | None:
    """Load structure with standardized error handling."""
    try:
        return load(filepath, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return None

def save_polymers(polymers: list[Polymer], output: Path, quiet: bool = False) -> None:
    """Save polymers to file(s) with standardized output."""
    if len(polymers) == 1:
        out_path = output if output.suffix == ".cif" else output.with_suffix(".cif")
        polymers[0].write(str(out_path))
        if not quiet:
            print(f"Saved to {out_path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for i, polymer in enumerate(polymers):
            out_path = output / f"sample_{i:03d}.cif"
            polymer.write(str(out_path))
            if not quiet:
                print(f"Saved {out_path}")

def setup_wandb_logger(enable: bool, project: str, name: str | None = None):
    """Set up W&B logger if enabled."""
    if not enable:
        return None
    from lightning.pytorch.loggers import WandbLogger
    return WandbLogger(project=project, name=name)

def print_training_header(model_type: str, **info) -> None:
    """Print standardized training header."""
    print()
    print("=" * 60)
    print(f"Ciffy {model_type} Training")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
```

**Optional: Split into submodules**:
```
cli/
├── __init__.py
├── __main__.py      # Entry point, argparse setup
├── helpers.py       # Common utilities
├── commands/
│   ├── info.py      # info, split, map, template
│   ├── train.py     # train flow/latent-diffusion/coord-diffusion
│   ├── predict.py   # predict flow/latent-diffusion/coord-diffusion
│   └── download.py  # download command
```

**Files affected**:
- `ciffy/cli/__main__.py` - Extract helpers, update commands to use them
- `ciffy/cli/helpers.py` (new) - Shared CLI utilities

**Effort**: 2-4 hours for helper extraction, 1 day for full submodule split
**Impact**: ~200 fewer lines, easier maintenance, consistent behavior

---

### Consolidate Residue Extraction Code

**Goal**: Reduce duplication between `ciffy/operations/extract.py` and `ciffy/nn/flow/residue/data.py`.

**Context**: Both modules extract residue coordinates into dense `(n, n_atoms, 3)` arrays with similar logic but different implementations.

**Detailed comparison**:

| Concern | `operations/extract.py` | `nn/flow/residue/data.py` |
|---------|-------------------------|---------------------------|
| Find common atoms | Set intersection (all instances) | Counter + min_coverage threshold |
| Build dense array | Mask-based on sorted atoms | Dict lookup remapping |
| Filter residue type | `poly.by_residue(residue.value)` | Manual `seq[i] != residue_type.value` |
| Input | Single `Polymer` object | List of CIF file paths |
| Output | `(n, n_atoms, 3)` coords only | coords + SE(3) link transforms |

**Duplicated patterns**:

1. **Common atom discovery** (`extract.py:134-142` vs `data.py:187-193`):
```python
# extract.py - strict intersection
atom_sets = [set(to_numpy(a).tolist()) for a in per_res_atoms]
common_atoms = set.intersection(*atom_sets)

# data.py - coverage threshold
atom_counts = Counter()
for coords_i, atoms_i, *_ in all_instances:
    atom_counts.update(atoms_i)
common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])
```

2. **Dense array construction** (`extract.py:163-173` vs `data.py:212-217`):
```python
# extract.py - mask-based
result = np.zeros((n_residues, n_atoms, 3), dtype=np.float32)
for i in range(n_residues):
    mask = np.isin(res_atoms, common_atoms_arr)
    result[i] = res_coords[mask]

# data.py - dict remapping via _remap_to_common()
coords_out = np.zeros((n, n_atoms, 3), dtype=np.float32)
coords_out[idx] = _remap_to_common(c_i, a_i, common_atoms)
```

**Proposed refactor**:

Option A: Extend `extract()` to support both modes:
```python
def extract(
    poly: Polymer,
    residue: AtomGroup,
    atoms: list | None = None,
    min_coverage: float = 1.0,  # 1.0 = strict intersection (current default)
    ...
) -> tuple[Array, list[int]]:
```

Option B: Create unified `Ensemble` class:
```python
class Ensemble:
    coords: Array              # (n, n_atoms, 3)
    atoms: list[int]
    residue: AtomGroup
    transforms: Array | None   # (n, 6) optional SE(3) link transforms

    @classmethod
    def from_polymer(cls, poly, residue, ...): ...

    @classmethod
    def from_cif_files(cls, paths, residue, with_transforms=False, ...): ...
```

`ResidueDataset` becomes `Ensemble.from_cif_files(..., with_transforms=True)`.

**Files affected**:
- `ciffy/operations/extract.py` - Add `min_coverage` parameter or refactor to shared utility
- `ciffy/nn/flow/residue/data.py` - Use shared extraction logic, keep link transform computation
- `ciffy/ensemble.py` (optional) - Unified interface

**Effort**: 2-4 hours

---

## LOW Priority

## Test Coverage Gaps

The following modules lack test coverage. Adding tests would improve reliability.

### High Priority

| Module | Issue |
|--------|-------|
| `ciffy/ensemble.py` | Core feature for conformational analysis, no tests |
| `ciffy/operations/extract.py` | Used by template system, no tests |

### Medium Priority

| Module | Issue |
|--------|-------|
| `ciffy/nn/dataset.py` | ML pipeline data loading, no direct tests |
| `ciffy/nn/inference.py` | Inference utilities, no tests |
| `ciffy/nn/runners/` | Training infrastructure (experiment_runner, inference_runner), no tests |

### Low Priority

| Module | Issue |
|--------|-------|
| `ciffy/utils/formatting.py` | Utility functions |
| `ciffy/utils/helpers.py` | Utility functions |
| `ciffy/cli/` | CLI commands |
