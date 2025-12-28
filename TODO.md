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

### Refactor Dataset Validation into Reusable Helpers

**Goal**: Extract error checking and logging from `LatentEncodingDataset` into reusable modules.

**Context**: `latent_trainer.py` now has extensive validation logic (residue count filtering, unknown residue detection, atom count validation, detailed logging). This should be reusable across different trainers and datasets.

**Proposed refactor**:
```python
# ciffy/nn/dataset_validation.py
@dataclass
class ValidationStats:
    total: int
    valid: int
    too_small: int
    too_large: int
    unknown_residues: int
    incomplete: int
    errors: int

def validate_polymer_for_flow(
    polymer: Polymer,
    flow_model: PolymerFlowModel,
    min_residues: int,
    max_residues: int,
) -> tuple[bool, str]:
    """Check if polymer is valid for flow model training.

    Returns (is_valid, reason) tuple.
    """

def filter_dataset(
    dataset: PolymerDataset,
    validator: Callable[[Polymer], tuple[bool, str]],
) -> tuple[list[int], ValidationStats]:
    """Filter dataset and return valid indices with stats."""
```

**Files affected**:
- `ciffy/nn/dataset_validation.py` (new)
- `ciffy/nn/diffusion/latent_trainer.py` - Use new helpers
- `ciffy/nn/base_trainer.py` - Optional integration

**Effort**: 2-3 hours
**Impact**: Cleaner code, reusable validation, consistent error reporting

---

### Enhanced Training Diagnostics and Metric Tracking

**Goal**: Add comprehensive metric tracking to diagnose NaN/Inf, convergence issues, and training instabilities.

**Context**: Currently we detect NaN/Inf after they occur. Better to track warning signs (gradient norms, loss variance, parameter statistics) to catch issues early.

**Proposed features**:

1. **Gradient health tracking**:
   - Per-layer gradient norms
   - Gradient explosion detection (norm > threshold)
   - Gradient vanishing detection (norm < threshold)

2. **Loss stability metrics**:
   - Rolling loss variance
   - Loss spike detection
   - Early stopping on divergence

3. **Parameter statistics**:
   - Weight norm tracking
   - Parameter update magnitudes
   - Dead neuron detection

4. **Convergence diagnostics**:
   - Learning rate vs loss correlation
   - Plateau detection
   - Suggested interventions (reduce LR, increase batch size)

**Implementation**:
```python
# ciffy/nn/diagnostics.py
class TrainingDiagnostics:
    def __init__(self, model, check_every: int = 100):
        self.gradient_history = []
        self.loss_history = []

    def on_backward(self, loss):
        """Called after loss.backward()."""
        self._check_gradients()
        self._update_loss_stats(loss)

    def get_warnings(self) -> list[str]:
        """Return list of current warning messages."""

    def should_stop_early(self) -> bool:
        """Return True if training should stop due to instability."""
```

**Integration with wandb/tensorboard**:
- Auto-log diagnostic metrics
- Alert on anomalies
- Visualization of health metrics

**Files affected**:
- `ciffy/nn/diagnostics.py` (new or extend existing)
- `ciffy/nn/training.py` - Integrate diagnostics
- `ciffy/nn/base_trainer.py` - Add diagnostics config

**Effort**: 4-6 hours
**Impact**: Earlier detection of training issues, better debugging, more stable training

---

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

### GNM (Gaussian Network Model) Extensions

**Goal**: Extend `ciffy/operations/gnm.py` with commonly-used GNM operations for structural biology research.

**Current state**: Basic functions exist (`graph_laplacian`, `gnm_correlations`, `gnm_variances`).

**Missing high-priority functions**:

| Function | Description |
|----------|-------------|
| `contact_map(polymer, cutoff=7.0)` | Build adjacency matrix from Polymer coordinates (Cα or centroid distances). Essential preprocessing step. |
| `gnm_modes(adj, k=None)` | Extract eigenvectors (normal modes) of Kirchhoff matrix. Returns (eigenvalues, eigenvectors). |
| `gnm_eigenvalues(adj)` | Get eigenvalues (squared frequencies) - cheaper than full mode decomposition. |
| `cross_correlations(adj)` | Normalized correlation matrix (range [-1, 1]) for identifying coupled motions. |

**Implementation notes**:

```python
def contact_map(polymer: Polymer, cutoff: float = 7.0, scale: Scale = Scale.RESIDUE) -> Array:
    """Build adjacency matrix from inter-residue distances."""
    dists = polymer.pairwise_distances(scale=scale)
    return (dists < cutoff).astype(float)

def gnm_modes(adj: Array, k: int | None = None) -> tuple[Array, Array]:
    """Compute GNM normal modes (eigenvectors of Kirchhoff matrix)."""
    L = graph_laplacian(adj)
    eigenvalues, eigenvectors = eigh(L)
    # Skip trivial zero mode, return slowest k modes
    return eigenvalues[1:k+1], eigenvectors[:, 1:k+1]

def cross_correlations(adj: Array) -> Array:
    """Normalized cross-correlation matrix."""
    corr = gnm_correlations(adj)
    std = sqrt(diagonal(corr))
    return corr / outer(std, std)
```

**Design consideration**: Consider wrapping in a `GNM` class to compute the pseudo-inverse once and reuse it across multiple queries (correlations, variances, cross-correlations all need the same pinv):

```python
class GNM:
    def __init__(self, adj: Array, rtol: float = 1e-2):
        self.adj = adj
        self.laplacian = graph_laplacian(adj)
        self._pinv = pinv(self.laplacian, rtol=rtol)  # Computed once

    @property
    def correlations(self) -> Array:
        return self._pinv

    @property
    def variances(self) -> Array:
        return diagonal(self._pinv)

    @property
    def cross_correlations(self) -> Array:
        std = sqrt(self.variances)
        return self._pinv / outer(std, std)

    def modes(self, k: int | None = None) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = eigh(self.laplacian)
        return eigenvalues[1:k+1], eigenvectors[:, 1:k+1]
```

**Files affected**:
- `ciffy/operations/gnm.py`
- `tests/test_gnm.py`

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

### ~~Clean Up Deprecated C Code~~ ✅ DONE

**Completed**: Removed dead Z-matrix code from C extension (commit 80db388).

**Removed**:
- `py_build_zmatrix_parallel`
- `py_build_canonical_zmatrix`
- `py_build_atom_indexed_zmatrix_parallel`
- All Z-matrix C implementations (~2000 lines)

**Kept** (still used):
- `py_cartesian_to_internal*` - Internal coordinate conversion
- `py_nerf_reconstruct*` - NERF-based coordinate reconstruction
- `py_build_bond_graph` - Bond graph construction
- `py_edges_to_csr` - CSR format conversion
- `py_find_connected_components` - Graph component detection

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
