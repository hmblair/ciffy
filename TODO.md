# TODO

## HIGH Priority

### PolymerFlowModel Training Pipeline

**Goal**: Create end-to-end training pipeline for PolymerFlowModel.

**Components needed**:
- Training script with configuration
- Data loading for polymer structures
- Per-residue model training
- Model checkpointing and evaluation
- Example training configs for RNA and protein

**Files likely affected**:
- `scripts/train_polymer_flow.py` (new)
- `ciffy/nn/flow/training.py` (new)

---

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

---

## MEDIUM Priority

### Chain-Level Positioning Primitives (Partial)

**Completed**:
- ✅ `ciffy.join(*polymers)` - Combine multiple poly-only Polymers
- ✅ `Polymer.extend(residue)` - Append residue to single-chain polymer

**Remaining**:
```python
def insert_residue(chain: Polymer, position: int, residue: Residue) -> Polymer:
    """Insert residue at position, reposition all downstream residues."""

def replace_residue(chain: Polymer, position: int, residue: Residue) -> Polymer:
    """Replace residue at position, keeping backbone frame alignment."""
```

---

## LOW Priority

### Extract Frame Computation to Geometry Helper

**Goal**: Decouple frame computation from flow models.

**Context**: Currently `ResidueFlowModel` stores pre-resolved frame column indices (`prev_frame_cols`, `next_frame_cols`) and `PolymerFlowModel.decode()` uses these to position residues. This couples geometry computation with the flow model.

**Proposed refactor**:
- Create a `ResidueFrameResolver` or similar helper in `ciffy/geometry.py`
- Move frame index pre-resolution there
- Flow models would use the helper rather than storing frame indices
- This allows frame computation to be reused outside of flow models

**Benefits**:
- Separation of concerns (flow = encoding/decoding, geometry = positioning)
- Frame computation reusable for other use cases
- Cleaner flow model API

**Note**: Frame cols could also be stored as `np.ndarray` shape `(3,)` with `-1` sentinel for `None`, enabling vectorized operations across multiple residues.

---

### Clean Up Deprecated C Code

**Goal**: Remove unused internal coordinate C functions from the extension module.

The following C functions in `ciffy/src/internal/` are no longer used by the Python API:
- `py_cartesian_to_internal*`
- `py_nerf_reconstruct*`
- `py_build_zmatrix*`
- `py_build_canonical_zmatrix`

**Keep** (still used):
- `py_build_bond_graph`
- `py_edges_to_csr`
- `py_find_connected_components`

**Files affected**:
- `ciffy/src/internal/internal_module.c`
- `ciffy/src/internal/internal_module.h`
- `ciffy/src/module.c`

---

## CIF Parsing Performance Optimizations

Current: ~11ms for 100K atoms (9M atoms/sec). Bottleneck is memory-bound, not compute-bound.

### Binary Cache Format (3-5x faster on repeated loads)

Cache parsed structures in binary format to skip ASCII parsing on subsequent loads.

```python
polymer = ciffy.load('file.cif')  # First: 11ms (parses + writes cache)
polymer = ciffy.load('file.cif')  # Second: ~2ms (reads binary cache)
```

**Effort**: 1-2 days
**Impact**: 3-5x faster for ML training loops

### Columnar Pre-scan (1.5-2x faster)

Parse one column at a time across all rows instead of row-by-row with scattered column access. Improves CPU prefetcher efficiency.

**Effort**: 4-6 hours
**Impact**: 1.5-2x faster parsing

### Integer Hash Keys (1.3-1.5x faster)

Replace string-based gperf hash lookups with integer-encoded keys for direct array lookup.

**Effort**: 1 day
**Impact**: 1.3-1.5x faster element/atom type lookups

---

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
