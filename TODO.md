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

### Remove Deprecated Polymer Methods (API Cleanup)

**Goal**: Remove deprecated methods from Polymer class that now live in the operations module.

**Context**: As part of the API refactoring (commit forthcoming), analysis methods were moved from Polymer to `ciffy.operations`. The old methods remain on Polymer with deprecation warnings for backward compatibility.

**Methods to remove from Polymer**:
- `pairwise_distances()` → `operations.pairwise_distances(polymer)`
- `knn()` → `operations.knn(polymer, k)`
- `adjacency()` → `operations.adjacency(polymer)`
- `bonded_distances()` → `operations.bonded_distances(polymer, ...)`
- `pca()` → `operations.pca(polymer, scale)`
- `moment()` → `operations.moment(polymer, n, scale)`
- `frames()` → `operations.frames(polymer)`
- `align()` → `operations.align_to_frame(polymer)` (renamed to `align_to_frame()`)
- `unalign()` → `operations.unalign(polymer, Rs, origins)`
- `local_transforms()` → `operations.local_transforms(polymer, ...)`
- `apply_local_transforms()` → `operations.apply_local_transforms(polymer, ...)`
- `gather()` → `operations.gather(polymer, groups)`
- `sort_atoms()` → `operations.sort_atoms(polymer)`

**Timeline**: Remove after 1-2 minor versions to give users time to migrate.

**Files affected**:
- `ciffy/polymer/polymer.py` - Remove deprecated method wrappers (~100 lines)

**Effort**: 1 hour
**Impact**: Cleaner API, smaller Polymer class

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

### Properly Parse CIF Semicolon Multi-line Text Blocks

**Goal**: Fully parse CIF files containing semicolon-delimited multi-line text values instead of skipping them.

**Context**: CIF format uses lines starting with `;` to delimit multi-line text values. Example file: `3H5X.cif`.
```
_pdbx_entity_nonpoly.entity_id
_pdbx_entity_nonpoly.name
_pdbx_entity_nonpoly.comp_id
4 'MANGANESE (II) ION'    MN
5
;2'-amino-2'-deoxycytidine 5'-(tetrahydrogen triphosphate)
;
CSG
```

Entity 5's name spans multiple lines. Currently, `_scan_lines()` in `io.c` skips these blocks entirely, meaning rows with multi-line values have missing fields.

**Current workaround**: Rows with semicolon blocks are indexed but have incomplete data. `_get_field_ptr()` returns NULL for missing fields. Most structures don't use this format, so impact is minimal.

**Proper fix**:
1. Modify `_scan_lines()` to correctly index rows that span multiple lines
2. Modify `_get_field_ptr()` to detect semicolon values and return the full multi-line content
3. Store block boundaries (start pointer + length) rather than assuming single-line fields

**Affected files**:
- `ciffy/src/cif/io.c` - `_scan_lines()`, `_get_field_ptr()`, field parsing functions
- `ciffy/src/cif/io.h` - May need new field type for multi-line values

**Effort**: 4-8 hours
**Impact**: Complete CIF format support, affects ~1% of PDB structures

---

## LOW Priority

### ~~Structured AtomGroup Access by Molecule Type~~ (DONE)

Implemented in `ciffy/biochemistry/groups.py`. Access via:
```python
from ciffy.biochemistry import RNA, DNA, Protein

RNA.Backbone.P           # Atom(P, 2) - unified value
RNA.Backbone.C1p         # Atom(C1', 13)
RNA.Base.glycosidic_n    # AtomGroup with {A: N9, G: N9, C: N1, U: N1}
RNA.Base.glycosidic_n.A  # Atom(N9, 18)
Protein.Backbone.CA      # Atom(CA, 15)
```

---

### ~~Auto-generate Backbone Atom Values in Codegen~~ (DONE)

Backbone atom values are now auto-generated from CCD order during codegen, exactly like non-backbone atoms:
- `codegen/__init__.py` `_build_indices()` assigns backbone values as atoms are encountered
- Values exported to `ciffy/biochemistry/_generated_atoms.py` as `UNIFIED_BACKBONE_VALUES`
- Config only defines *which* atoms are backbone (name sets), not their values

New CCD-based ordering:
```python
# OP3=1, P=2, OP1=3, OP2=4, O5'=5, C5'=6, C4'=7, O4'=8, C3'=9, O3'=10, C2'=11, O2'=12, C1'=13
# N=14, CA=15, C=16, O=17
```

Note: This is a breaking change from the old hardcoded ordering (P=1, OP1=2, ...).

---

### Derive Terminal Atoms from CCD

**Goal**: Replace hardcoded `TERMINAL_ATOMS` in `codegen/config.py` with values derived from the Chemical Component Dictionary (CCD).

**Context**: Terminal atoms (OP3, HOP3, HO3' for RNA; H2, OXT, HXT for protein) are currently hardcoded. The CCD contains relevant flags that could derive these automatically.

**CCD fields available**:
- `pdbx_leaving_atom_flag` - atoms removed during polymerization
- `pdbx_n_terminal_atom_flag` - N-terminal atoms (proteins)
- `pdbx_c_terminal_atom_flag` - C-terminal atoms (proteins)

**Caveat for nucleic acids**: CCD marks OP3 and HO3' as `leaving=Y`, but HOP3 (hydrogen bonded to OP3) is **not** marked. Need transitive closure: atoms bonded to leaving atoms should also be considered terminal.

**Implementation**:
1. Parse `leaving_atom_flag`, `n_terminal_atom_flag`, `c_terminal_atom_flag` in `codegen/ccd.py`
2. For nucleic acids: compute transitive closure (atoms bonded to leaving atoms)
3. Generate per-residue terminal atoms (instead of per-molecule-type)
4. Update `AtomGroup.terminal()` to use per-residue data

**Files affected**:
- `codegen/ccd.py` - Parse additional CCD fields
- `codegen/python_codegen.py` - Generate per-residue terminal atoms
- `ciffy/biochemistry/atom.py` - Update `terminal()` to use per-residue data

**Effort**: 4-6 hours
**Impact**: More robust handling of non-standard residues, single source of truth

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
