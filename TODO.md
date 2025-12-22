HIGH Priority

RNA CHI Angle Optimization Without Ring Deformation

Goal: Enable modification of RNA glycosidic (CHI) dihedral angles without deforming nucleobase rings.

Currently, optimizing CHI_PURINE or CHI_PYRIMIDINE dihedrals causes ring atoms to move inconsistently because they reference a mix of sugar atoms (C1') and other ring atoms in the Z-matrix. When CHI changes, atoms referencing the sugar move differently than atoms referencing other ring atoms, breaking planarity.

Potential approaches:
- Compute Jacobian of ring dihedrals w.r.t. Z-matrix dihedrals and apply compensating updates
- Ensure all ring atoms reference only other ring atoms (except the glycosidic bond attachment)
- Use a hybrid representation: Z-matrix for backbone, rigid body for bases

Files likely affected:
- ciffy/src/codegen/residue.py - Canonical reference definitions for base atoms
- ciffy/src/internal/graph.c - Z-matrix construction
- ciffy/internal/coordinates.py - Coordinate manager dihedral methods

See test: tests/test_internal.py::TestRingPreservation::test_ring_torsion_during_backbone_optimization

MEDIUM Priority
Improved Polymer Template Construction

Goal: Enhance from_sequence method with ideal dihedral angles.

Files likely affected:

    ciffy/template.py - Template construction logic
    ciffy/biochemistry/_generated_residues.py - Ideal coordinates data
    ciffy/biochemistry/constants.py - Dihedral angle constants
    tests/test_template.py - Template validation tests

CUDA Polymer Conversions

Goal: GPU-native conversion algorithms to avoid CPU-GPU memory transfers.

Files likely affected:

    ciffy/backend/torch_ops.py - CUDA operation implementations
    ciffy/src/internal/ - New CUDA source files
    ciffy/internal/nerf.py - GPU-aware NERF algorithm
    tests/test_device.py - GPU testing

---

## Polymer Join Function

**Goal:** Implement `ciffy.join(*polymers)` to combine multiple Polymer objects into a single Polymer containing all chains.

### Use Cases
- Combining separately loaded structures for multi-chain analysis
- Reassembling chains after parallel processing
- Building complexes from individual components

### API Design

```python
# Basic usage
combined = ciffy.join(polymer_a, polymer_b)
combined = ciffy.join(polymer_a, polymer_b, polymer_c)

# From list
polymers = [ciffy.load(f) for f in files]
combined = ciffy.join(*polymers)
```

### Data to Concatenate

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

### Key Implementation Challenges

#### 1. Atom Ordering Constraint
Polymer atoms must precede non-polymer atoms: `[0, polymer_count)` polymer, `[polymer_count, total)` HETATM.

**Solution:** Reorder during join:
```python
# Collect all polymer atoms first, then all non-polymer
polymer_coords = [p.coordinates[:p.polymer_count] for p in polymers]
hetero_coords = [p.coordinates[p.polymer_count:] for p in polymers]
coordinates = ops.cat(polymer_coords + hetero_coords, axis=0)
new_polymer_count = sum(p.polymer_count for p in polymers)
```

Must apply same reordering to `atoms` and `elements` arrays.

#### 2. Size Array Recalculation for CHAIN Scale
After reordering, chain sizes change because HETATM atoms move to the end.

**Solution:**
- For polymer chains (with residues): chain size = sum of residue sizes
- For HETATM-only chains: track separately and add at end

```python
# Polymer chain sizes from residue sizes
poly_chain_sizes = [ops.segment_sum(p.sizes[Scale.RESIDUE], p.lengths) for p in polymers]
# HETATM chain sizes (chains with length=0)
hetero_chain_sizes = [p.sizes[Scale.CHAIN][p.lengths == 0] for p in polymers]
```

#### 3. PDB ID Handling
Multiple inputs have different pdb_ids.

**Options (recommend Option A):**
- A) Use "JOINED" or "MULTI" as synthetic ID
- B) Comma-separated list: "1ABC,2XYZ"
- C) Use first input's ID

#### 4. Backend/Device Compatibility
All inputs must be compatible (same backend, same device for torch).

**Solution:** Use `check_compatible()` on all inputs before processing:
```python
for p in polymers[1:]:
    check_compatible(polymers[0].coordinates, p.coordinates, "coordinates")
```

#### 5. Empty Polymer Handling
Skip empty polymers gracefully:
```python
polymers = [p for p in polymers if not p.empty()]
if not polymers:
    return Polymer.create_empty()
if len(polymers) == 1:
    return polymers[0]
```

#### 6. Topology Invalidation
Joined polymer has different Z-matrix structure. CoordinateManager handles this automatically when constructed with new topology.

### Implementation Steps

1. **Initial version: polymer-only join** (no HETATM support)
   - Add `join()` function in `ciffy/__init__.py` (or new `ciffy/operations/join.py`)
   - Validate inputs (non-empty, compatible backends)
   - Handle edge cases (0, 1 input)
   - **Raise `ValueError` if any input has `nonpoly > 0`**
   - Simple concatenation of all arrays (no reordering needed)
   - This avoids atom reordering complexity for the initial implementation

2. **Extend to support HETATM atoms** (future enhancement)
   - Remove the `nonpoly > 0` check
   - Reorder atoms to maintain polymer/HETATM separation
   - Concatenate all polymer atoms first
   - Concatenate all HETATM atoms second
   - Apply same reordering to atoms, elements arrays

3. **Concatenate simple arrays**
   - sequence, lengths, names, strands
   - sizes[Scale.RESIDUE]
   - molecule_types, descriptions (if present in all)

4. **Recompute derived values**
   - sizes[Scale.CHAIN] after reordering
   - sizes[Scale.MOLECULE] = [total_atoms]
   - polymer_count = sum of input polymer_counts

5. **Construct result Polymer**
   - Call Polymer.__init__ which validates consistency
   - TopologyInfo rebuilt automatically

6. **Add tests in `tests/test_polymer.py`**
   - Two single-chain polymers
   - Multi-chain polymers
   - Polymers with HETATM atoms
   - Empty polymer inputs
   - Mixed molecule types (protein + RNA)
   - Backend compatibility (numpy, torch)

### Files Affected

- `ciffy/__init__.py` - Export `join` function
- `ciffy/operations/join.py` (new) - Implementation
- `tests/test_polymer.py` - Test cases

### Estimated Complexity

**Initial version (polymer-only):**

| Aspect | Difficulty |
|--------|------------|
| Basic array concatenation | Low |
| Edge case handling | Low |
| Test coverage | Low |

**Initial version: Low-Medium (4/10)** - Straightforward concatenation with no reordering.

**Full version (with HETATM support):**

| Aspect | Difficulty |
|--------|------------|
| Atom reordering for polymer/HETATM | Medium |
| Chain size recalculation | Medium |
| Test coverage | Medium |

**Full version: Medium (6/10)** - Requires careful handling of atom ordering and size array consistency.

### Code Skeleton (Initial Version - Polymer Only)

```python
def join(*polymers: Polymer) -> Polymer:
    """
    Combine multiple Polymer objects into one.

    Args:
        *polymers: Polymer objects to join. Must have compatible backends
            and contain only polymer atoms (no HETATM).

    Returns:
        New Polymer containing all chains from all inputs.

    Raises:
        ValueError: If polymers have incompatible backends/devices,
            or if any polymer contains non-polymer (HETATM) atoms.

    Example:
        >>> chain_a = ciffy.load("chain_a.cif").poly()
        >>> chain_b = ciffy.load("chain_b.cif").poly()
        >>> combined = ciffy.join(chain_a, chain_b)
    """
    # Filter empty, validate compatibility
    polymers = [p for p in polymers if not p.empty()]
    if not polymers:
        return Polymer.create_empty()
    if len(polymers) == 1:
        return polymers[0]

    for p in polymers[1:]:
        check_compatible(polymers[0].coordinates, p.coordinates, "input polymers")

    # Initial version: reject HETATM atoms
    for i, p in enumerate(polymers):
        if p.nonpoly > 0:
            raise ValueError(
                f"Polymer at index {i} contains {p.nonpoly} non-polymer atoms. "
                "Use polymer.poly() to remove HETATM atoms before joining."
            )

    # Simple concatenation (no reordering needed for polymer-only)
    coordinates = ops.cat([p.coordinates for p in polymers], axis=0)
    atoms = ops.cat([p.atoms for p in polymers], axis=0)
    elements = ops.cat([p.elements for p in polymers], axis=0)
    sequence = ops.cat([p.sequence for p in polymers], axis=0)
    lengths = ops.cat([p.lengths for p in polymers], axis=0)
    names = sum([p.names for p in polymers], [])
    strands = sum([p.strands for p in polymers], [])

    # Size arrays
    res_sizes = ops.cat([p.sizes(Scale.RESIDUE) for p in polymers], axis=0)
    chn_sizes = ops.cat([p.sizes(Scale.CHAIN) for p in polymers], axis=0)
    total_atoms = sum(p.size() for p in polymers)

    sizes = {
        Scale.RESIDUE: res_sizes,
        Scale.CHAIN: chn_sizes,
        Scale.MOLECULE: ops.array([total_atoms], like=polymers[0].coordinates),
    }

    # Optional arrays (only if present in all inputs)
    mol_types = None
    if all(p._molecule_types is not None for p in polymers):
        mol_types = ops.cat([p._molecule_types for p in polymers], axis=0)

    descriptions = None
    if all(p.descriptions is not None for p in polymers):
        descriptions = sum([p.descriptions for p in polymers], [])

    return Polymer(
        coordinates=coordinates,
        atoms=atoms,
        elements=elements,
        sequence=sequence,
        sizes=sizes,
        id="JOINED",
        names=names,
        strands=strands,
        lengths=lengths,
        polymer_count=total_atoms,  # All atoms are polymer atoms
        molecule_types=mol_types,
        descriptions=descriptions,
    )
```
