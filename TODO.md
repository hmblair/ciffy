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
