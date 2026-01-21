# TODO

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
| `ciffy/operations/extract.py` | Used by template system, no tests |

### Low Priority

| Module | Issue |
|--------|-------|
| `ciffy/utils/formatting.py` | Utility functions |
| `ciffy/utils/helpers.py` | Utility functions |
| `ciffy/cli/` | CLI commands |
