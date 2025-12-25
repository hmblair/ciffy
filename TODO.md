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

### Polymer Join Function

**Goal**: Implement `ciffy.join(*polymers)` to combine multiple Polymer objects into a single Polymer containing all chains.

#### Use Cases
- Combining separately loaded structures for multi-chain analysis
- Reassembling chains after parallel processing
- Building complexes from individual components

#### API Design

```python
# Basic usage
combined = ciffy.join(polymer_a, polymer_b)
combined = ciffy.join(polymer_a, polymer_b, polymer_c)

# From list
polymers = [ciffy.load(f) for f in files]
combined = ciffy.join(*polymers)
```

#### Data to Concatenate

**Simple concatenation (axis 0):**
- `coordinates` - (N, 3) atom positions
- `atoms` - (N,) atom type indices
- `elements` - (N,) element indices
- `sequence` - (R,) residue type indices
- `sizes[Scale.RESIDUE]` - atoms per residue
- `sizes[Scale.CHAIN]` - atoms per chain
- `lengths` - residues per chain
- `names` - chain name list
- `strands` - strand identifier list
- `molecule_types` - chain type array (if present)
- `descriptions` - chain description list (if present)

**Recomputed values:**
- `sizes[Scale.MOLECULE]` - sum of all atoms → `[total_atoms]`
- `polymer_count` - sum of all input polymer_counts

#### Implementation Steps

1. **Initial version: polymer-only join** (no HETATM support)
   - Add `join()` function in `ciffy/__init__.py` (or new `ciffy/operations/join.py`)
   - Validate inputs (non-empty, compatible backends)
   - Handle edge cases (0, 1 input)
   - **Raise `ValueError` if any input has `nonpoly > 0`**
   - Simple concatenation of all arrays (no reordering needed)

2. **Extend to support HETATM atoms** (future enhancement)
   - Remove the `nonpoly > 0` check
   - Reorder atoms to maintain polymer/HETATM separation

#### Files Affected

- `ciffy/__init__.py` - Export `join` function
- `ciffy/operations/join.py` (new) - Implementation
- `tests/test_polymer.py` - Test cases

---

### Chain-Level Positioning Primitives

**Goal**: Build chain manipulation functions on top of the residue positioning primitives in `ciffy/geometry.py`.

**Context**: The `position_residue()` function in `geometry.py` handles positioning a single residue relative to a previous residue using frame-based SE(3) alignment. These higher-level functions would compose that primitive for common chain operations.

**Functions to add**:

```python
def extend_chain(chain: Polymer, residue_coords: Array, residue: Residue,
                 transform: Array | None = None) -> Polymer:
    """Append a residue to the end of a chain."""

def insert_residue(chain: Polymer, position: int, residue_coords: Array,
                   residue: Residue) -> Polymer:
    """Insert residue at position, reposition all downstream residues."""

def replace_residue(chain: Polymer, position: int, residue_coords: Array,
                    residue: Residue) -> Polymer:
    """Replace residue at position, keeping backbone frame alignment."""
```

**Files likely affected**:
- `ciffy/geometry.py` - Add chain-level functions
- `ciffy/__init__.py` - Export new functions
- `tests/test_geometry.py` - Test cases

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
