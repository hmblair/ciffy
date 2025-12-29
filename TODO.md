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

### Load `_struct_conn` Bond Data from CIF Files

**Goal**: Parse the `_struct_conn` mmCIF category to extract explicit bond/connection data.

**Context**: The CIF loader currently does not parse `_struct_conn`. The C code defines `BLOCK_CONN` but no fields extract data from it. Bond connectivity is currently derived from:
1. **Intra-residue bonds** - Template bonds from `Residue.A.bonds` (chemical component dictionary)
2. **Inter-residue backbone** - Static linking rules in `ciffy/biochemistry/linking.py`

This works for standard polymers but misses connections involving modified residues or non-standard linkages.

**What `_struct_conn` contains**:

| Type | Description | Use Case |
|------|-------------|----------|
| `covale` | Covalent bonds to/from **non-standard residues** only | Modified nucleotides (2MG, H2U, OMC, etc.) |
| `hydrog` | Hydrogen bonds (base pairs, etc.) | Base-pair annotations, secondary structure |
| `metalc` | Metal coordination | Mg²⁺, Zn²⁺ binding sites |
| `disulf` | Disulfide bridges | Protein cross-links |

**What `_struct_conn` does NOT contain**:
- Standard backbone bonds (O3'-P between A/U/G/C, peptide bonds)
- Intra-residue bonds (C-C, C-N within nucleotides)
- These are implicit and derived from sequence + chemical knowledge

**Typical sizes**:
- RNA structures: 20-150 connections (mostly H-bonds)
- 9MDS (102K atoms): 4,404 hydrogen bonds, 0 covalent
- 1EHZ tRNA (76 residues): 25 covale (modified residues only), 70 hydrog, 47 metalc

**Performance impact**: Minimal. Parsing ~100-200 rows adds <0.1ms. The main cost is building a reverse lookup table `(chain_id, residue_number, atom_name) → global_atom_index`, which would add 1-3ms (~10-30% overhead on large structures). Could be made optional via `ciffy.load(..., connections=True)`.

**Implementation steps**:

1. **C layer** (`ciffy/src/cif/`):
   - Add `FIELD_CONNECTIONS` to registry with attributes: `ptnr1_label_asym_id`, `ptnr1_label_seq_id`, `ptnr1_label_atom_id`, `ptnr2_*`, `conn_type_id`
   - Build reverse lookup hash during atom parsing
   - Map connection records to global atom indices
   - Store as `(n_connections, 2)` int array + connection type array

2. **Python layer**:
   - Add `connections` field to `mmCIF` struct and Polymer class
   - Expose connection types (covale, hydrog, metalc, disulf)
   - Optional parameter: `ciffy.load(..., connections=True)`

3. **Integration**:
   - Extend `build_bond_graph()` to include `_struct_conn` covalent bonds
   - Add `Polymer.base_pairs` property using hydrog connections (optional)

**Files affected**:
- `ciffy/src/cif/registry.c/h` - Add connection field definitions
- `ciffy/src/cif/parser.c/h` - Add connection parsing, reverse lookup
- `ciffy/io/loader.py` - Pass connections to Polymer
- `ciffy/polymer/polymer.py` - Add connections field
- `ciffy/backend/graph.py` - Integrate covale bonds into bond graph

**Effort**: 1-2 days
**Impact**: Correct bond graphs for modified residues, explicit base-pair annotations, metal binding site identification

---

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
