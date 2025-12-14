# Input/Output

This guide covers loading, saving, and file handling in ciffy.

## Loading Structures

### From Local Files

```python
import ciffy

# Load from CIF file
polymer = ciffy.load("structure.cif")

# Load with specific backend
polymer_np = ciffy.load("structure.cif", backend="numpy")
polymer_torch = ciffy.load("structure.cif", backend="torch")
```

### From URLs

ciffy can fetch structures directly from the RCSB PDB:

```python
# Load by PDB ID (downloads from RCSB)
polymer = ciffy.load("1ehz")  # tRNA structure

# Explicit URL
polymer = ciffy.load("https://files.rcsb.org/download/1ehz.cif")
```

Downloaded files are cached in `~/.cache/ciffy/` for faster subsequent loads.

### Loading Multiple Structures

```python
import ciffy
from pathlib import Path

# Load all CIF files in a directory
cif_dir = Path("structures/")
polymers = [ciffy.load(f) for f in cif_dir.glob("*.cif")]

# Load with progress tracking
from tqdm import tqdm
polymers = [ciffy.load(f) for f in tqdm(cif_files)]
```

### Metadata-Only Loading

For large datasets, load only metadata without coordinates:

```python
# Load metadata only (fast)
metadata = ciffy.load_metadata("large_structure.cif")

print(f"PDB ID: {metadata['id']}")
print(f"Chains: {metadata['chain_count']}")
print(f"Residues: {metadata['residue_count']}")
print(f"Atoms: {metadata['atom_count']}")
print(f"Polymer atoms: {metadata['polymer_count']}")
```

Use this to filter structures before loading full coordinates:

```python
# Only load structures with > 1000 residues
for cif_file in cif_files:
    meta = ciffy.load_metadata(cif_file)
    if meta['residue_count'] > 1000:
        polymer = ciffy.load(cif_file)
        process(polymer)
```

### Loading Entity Descriptions

CIF files contain entity descriptions (what each chain represents):

```python
# Load with descriptions
polymer = ciffy.load("structure.cif", load_descriptions=True)

# Access descriptions
for name, desc in zip(polymer.names, polymer.descriptions):
    print(f"Chain {name}: {desc}")
# Output:
# Chain A: 16S ribosomal RNA
# Chain B: 30S ribosomal protein S1
```

## Saving Structures

### Basic Writing

```python
polymer = ciffy.load("input.cif")

# Modify structure
clean = polymer.poly().strip(ciffy.RESIDUE)

# Save to new file
clean.write("output.cif")
```

### Functional Style

```python
# Alternative: use write_cif function
ciffy.write_cif(polymer, "output.cif")
```

### Writing Selections

```python
polymer = ciffy.load("complex.cif")

# Save individual chains
for chain in polymer.chains():
    chain.write(f"chain_{chain.names[0]}.cif")

# Save only RNA
polymer.by_type(ciffy.RNA).write("rna_only.cif")

# Save polymer without heteroatoms
polymer.poly().write("polymer_only.cif")
```

## Splitting Structures

### Split by Chain

Use the CLI or Python to split multi-chain structures:

```bash
# CLI: split into per-chain files
ciffy split complex.cif -o chains/
```

```python
# Python equivalent
polymer = ciffy.load("complex.cif")

for chain in polymer.chains():
    name = chain.names[0]
    chain.write(f"chains/chain_{name}.cif")
```

### Split Polymer from Heteroatoms

```python
polymer = ciffy.load("structure.cif")

# Separate polymer and heteroatoms
poly = polymer.poly()
hetero = polymer.hetero()

poly.write("polymer.cif")
hetero.write("heteroatoms.cif")
```

## Building Structures

### From Sequence

Create template structures from sequences:

```python
# Single RNA chain
rna = ciffy.from_sequence("acguacgu")
print(rna.size())  # Atoms for 8-nucleotide RNA

# Single protein chain
protein = ciffy.from_sequence("MGKLF")
print(protein.size())  # Atoms for 5-residue protein

# Multi-chain complex
complex = ciffy.from_sequence(["acgu", "MGKLF"])
print(complex.names)  # ['A', 'B']
```

Templates have zero coordinates. Use them for:
- Generative model inputs
- Computing expected atom counts
- Testing pipelines

### From Existing Structure

Extract a template (sequence + structure) from an existing polymer:

```python
from ciffy import from_extract

# Original structure
polymer = ciffy.load("structure.cif")

# Extract specific chains
template = from_extract(polymer, chains=[0, 1])

# Use with predicted coordinates
predicted_coords = model(template)
result = template.with_coordinates(predicted_coords)
```

## File Format Details

### What ciffy Reads

ciffy parses these mmCIF categories:

| Category | Data Extracted |
|----------|----------------|
| `_atom_site` | Coordinates, atom types, elements, residue assignments |
| `_struct_asym` | Chain definitions and names |
| `_pdbx_poly_seq_scheme` | Polymer sequence information |
| `_entity_poly` | Molecule type (RNA, DNA, protein) |
| `_entity` | Entity descriptions (optional) |

### What ciffy Writes

Output CIF files contain:

```
data_XXXX
#
loop_
_struct_asym.id
_struct_asym.entity_id
A  1
B  2
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
...
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
...
```

## Performance Tips

### Batch Processing

For large datasets, use metadata filtering:

```python
# Filter first, then load
good_files = []
for f in cif_files:
    meta = ciffy.load_metadata(f)
    if meta['atom_count'] < 10000:  # Skip huge structures
        good_files.append(f)

# Load filtered set
polymers = [ciffy.load(f) for f in good_files]
```

### Memory Management

For very large structures on GPU:

```python
import torch

# Load to GPU directly
polymer = ciffy.load("ribosome.cif", backend="torch").to("cuda")

# Process in chunks if needed
for chain in polymer.chains():
    result = process(chain)
    save_result(result)
    # Chain goes out of scope, memory freed
```

## Error Handling

```python
import ciffy

try:
    polymer = ciffy.load("structure.cif")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Parse error: {e}")
except KeyError as e:
    print(f"Missing required data: {e}")
```

Common errors:

| Error | Cause |
|-------|-------|
| `FileNotFoundError` | File doesn't exist |
| `ValueError` | Invalid CIF format |
| `KeyError` | Missing required mmCIF category |
| `IOError` | File read/write error |

## Integration with Other Tools

### PyMOL

```python
# Save for PyMOL visualization
polymer.write("for_pymol.cif")
# Then: pymol for_pymol.cif
```

### ChimeraX

```python
# Save structure
polymer.write("structure.cif")

# Export attribute file for coloring
from ciffy.visualize import to_defattr
values = compute_values(polymer)
to_defattr(polymer, values, "values.defattr", scale=ciffy.RESIDUE)

# In ChimeraX:
# open structure.cif
# open values.defattr
# color byattribute ciffy_value
```

### MDAnalysis

```python
import MDAnalysis as mda

# Save ciffy structure, load in MDAnalysis
polymer.write("structure.cif")
universe = mda.Universe("structure.cif")
```
