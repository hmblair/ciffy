# Known Issues - Dihedral System Refactoring

## Summary

This document tracks known issues discovered during the dihedral system refactoring for RNA backbone sampling.

---

## Background: Z-Matrix Representations

### What is a Z-Matrix?

A Z-matrix (also called internal coordinates) represents molecular geometry using bond lengths, bond angles, and dihedral angles instead of Cartesian coordinates. Each atom (except the first three) is defined by:

1. **Distance reference**: The atom it's bonded to (bond length)
2. **Angle reference**: A second atom defining the bond angle
3. **Dihedral reference**: A third atom defining the dihedral angle

```
Atom i: [i, j, k, l]  →  i is placed at distance d(i,j) from j,
                          angle θ(i,j,k) from the j-k bond,
                          dihedral φ(i,j,k,l) around the j-k axis
```

### NERF Reconstruction

The Natural Extension Reference Frame (NERF) algorithm reconstructs Cartesian coordinates from internal coordinates. It processes atoms level-by-level, where an atom's level must be greater than all its reference atoms' levels (dependency ordering).

### Two Z-Matrix Construction Approaches

This codebase has two algorithms for building Z-matrices:

#### 1. BFS-Based Z-Matrix (Original, Working)

**Location**: `build_zmatrix_from_csr()` in `ciffy/src/internal/graph.c`

**Algorithm**:
1. Start BFS traversal from a root atom (typically first atom of each connected component)
2. Visit atoms in breadth-first order
3. For each atom, select references from already-visited neighbors
4. Attempt to match dihedral patterns (PHI, PSI, CHI, etc.) when selecting the dihedral reference

**Characteristics**:
- Atoms are reordered by BFS traversal order
- Reference selection depends on traversal order
- Dihedral assignment is opportunistic: if the correct reference atoms happen to be visited first, the dihedral is captured; otherwise it may be assigned to a different atom
- **Works correctly for coordinate roundtrips**

**Problem**: The BFS order is arbitrary. For dihedral-owning atoms (e.g., the C atom that "owns" PHI), the correct reference atoms (C-N-CA-C for PHI) may not all be visited yet, causing the dihedral to be assigned to the wrong atom or missed entirely.

#### 2. Canonical Z-Matrix (New, Experimental)

**Location**: `build_canonical_zmatrix_c()` in `ciffy/src/internal/graph.c`

**Algorithm**:
1. Process atoms in natural order (0, 1, 2, ..., N-1)
2. For each atom, look up precomputed canonical references from `ATOM_CANONICAL_REFS` table
3. These references are generated at codegen time from residue topology
4. Fall back to bond-graph neighbor selection if no canonical refs exist

**Characteristics**:
- Atoms remain in natural order (no reordering)
- `zmatrix[i]` always corresponds to `atoms[i]`
- Dihedral-owning atoms are guaranteed correct references
- Uses precomputed tables from `ciffy/src/internal/canonical_refs.h`

**The canonical refs table structure**:
```c
// ATOM_CANONICAL_REFS[atom_type][6] = {dist_ref, ang_ref, dih_ref, dist_off, ang_off, dih_off}
// If offset == 0: ref value is an atom TYPE (resolve within same residue)
// If offset != 0: ref value is a BACKBONE_NAME_ID (resolve in adjacent residue)
```

**Problem**: Currently causes roundtrip failures (1.9-3.9 Å RMSD). The root cause is unknown but likely involves:
- Incorrect reference resolution for inter-residue references
- Missing or incorrect canonical refs for some atom types
- Possible issues with the offset/backbone name resolution logic

---

## Completed Fixes

### 1. Residue.T and Residue.DU Missing (Fixed)
- **Location**: `ciffy/sampling/gmm_registry.py`
- **Issue**: `Residue.T` and `Residue.DU` enum values don't exist, causing `AttributeError` when sampling 't' or 'u' sequences
- **Fix**: Removed non-existent enum references from GMM registry

### 2. Chi Angle Assignment to Wrong Residue Types (Fixed)
- **Location**: `codegen/residue.py`, `codegen/config.py`
- **Issue**: Both `chi_purine` AND `chi_pyrimidine` patterns were assigned to ALL nucleotides. Purines (A, G) incorrectly got CHI_PYRIMIDINE because they have N1 and C2 atoms in their fused ring.
- **Fix**: Added `PURINE_RESIDUES` and `PYRIMIDINE_RESIDUES` sets in `codegen/config.py`, made chi pattern assignment conditional on residue type

### 3. Chi Angles Break Base Ring Structure (Fixed)
- **Location**: `ciffy/sampling/backbone.py`
- **Issue**: Setting chi dihedral angles breaks the nucleobase ring structure. The Z-matrix places ring atoms through different reference chains, so rotating chi moves only the owner atom (C4 for purines, C2 for pyrimidines), breaking ring closure.
- **Fix**: Skip chi angle sampling entirely for nucleotides - only backbone dihedrals (alpha, beta, gamma, delta, epsilon, zeta) are sampled

---

## Outstanding Issues

### 1. Canonical Z-Matrix Roundtrip Failures (OPEN)

- **Severity**: HIGH
- **Location**: `ciffy/backend/graph.py` - `_build_canonical_zmatrix_from_topology()`
- **Issue**: The new canonical Z-matrix implementation causes roundtrip test failures with RMSD of 1.9-3.9 Å (thresholds are 1e-5 to 1e-4 Å)

**Failing Tests**:
- `test_roundtrip[a-roundtrip_single_residue]` - RMSD 1.896725
- `test_roundtrip[acgu-roundtrip_small]` - RMSD 3.869783
- `test_protein_roundtrip`
- `test_small_perturbation_roundtrip`
- `test_zero_perturbation_preserves_structure`
- `test_sugar_ring_preserved_on_chi_rotation`
- `test_roundtrip_on_gpu[mps]`
- `test_pdb_roundtrip_on_gpu[mps]`
- `test_multichain_relative_orientation`
- `test_rna_structure_per_chain`
- `test_torch_roundtrip`

**Likely Root Causes to Investigate**:
1. Inter-residue reference resolution (backbone name → atom index mapping)
2. Missing canonical refs for certain atom types (falling back incorrectly)
3. NERF level computation for canonical order vs BFS order
4. First three atoms of each residue/chain needing special handling

**Workaround**: Set `use_canonical=False` in `_build_zmatrix_indices_from_topology()` to use BFS-based Z-matrix

### 2. Geometric GNM Test Failure (OPEN)
- **Severity**: LOW (unrelated to dihedral changes)
- **Location**: `tests/ml/test_geometric_gnm.py`
- **Issue**: `test_positive_semidefinite` failing - likely pre-existing or unrelated

---

## Test Status

- **Sampling tests**: 26/26 passing
- **Full test suite**: 1106 passed, 12 failed, 4 skipped

---

## Next Steps

1. **Investigate canonical Z-matrix roundtrip failures**
   - Add debug logging to trace reference resolution
   - Compare canonical vs BFS Z-matrices for a simple molecule
   - Check NERF level assignments

2. **Consider reverting to BFS-based Z-matrix** if canonical cannot be fixed
   - Change `use_canonical=True` default back to `False` in `graph.py`

3. **Chi angle limitation is fundamental**
   - Ring structures cannot have internal dihedrals freely modified without rigid-body treatment
   - Would require treating nucleobases as rigid bodies with only glycosidic bond rotation
