# Visualization

This guide covers visualization tools in ciffy, including the command-line interface, matplotlib plots, and ChimeraX integration.

## Command-Line Interface

ciffy provides a CLI for quick structure inspection.

### Basic Usage

```bash
# View structure summary
ciffy structure.cif

# View multiple structures
ciffy *.cif

# Show sequence
ciffy structure.cif --sequence

# Show atom counts per residue
ciffy structure.cif --atoms

# Show entity descriptions
ciffy structure.cif --desc
```

Example output:
```
Polymer 1EHZ (numpy)
─────────────────────
   Type    Res  Atoms
─────────────────────
A  RNA      76   1629
─────────────────────
Σ           76   1629
─────────────────────
```

### Contact Maps

Generate contact maps from the CLI:

```bash
# Display contact map (opens matplotlib window)
ciffy map structure.cif

# Save to file
ciffy map structure.cif -o contact_map.png

# Customize appearance
ciffy map structure.cif --scale residue --power 2 --cmap viridis

# Specific chain
ciffy map structure.cif --chain A
```

### Splitting Structures

Split multi-chain structures into separate files:

```bash
# Split into per-chain files
ciffy split complex.cif -o chains/

# Include all chains (not just polymer)
ciffy split complex.cif --all -o all_chains/
```

## Contact Maps

Visualize residue-residue or atom-atom proximity:

```python
import ciffy
from ciffy.visualize import contact_map
import matplotlib.pyplot as plt

polymer = ciffy.load("structure.cif")

# Basic contact map
fig, ax = plt.subplots(figsize=(8, 8))
contact_map(polymer, ax=ax)
plt.savefig("contact_map.png", dpi=150)
```

### Customization Options

```python
# Residue-level (default)
contact_map(polymer, scale=ciffy.RESIDUE)

# Atom-level (higher resolution)
contact_map(polymer, scale=ciffy.ATOM)

# Adjust distance transformation
contact_map(polymer, power=1)    # 1/r (linear)
contact_map(polymer, power=2)    # 1/r^2 (default)
contact_map(polymer, power=6)    # 1/r^6 (LJ-like, emphasizes close contacts)

# Custom colormap
contact_map(polymer, cmap="viridis")
contact_map(polymer, cmap="Blues")

# Without colorbar
contact_map(polymer, colorbar=False)

# Custom value range
contact_map(polymer, vmin=0, vmax=0.1)
```

### Per-Chain Contact Maps

```python
polymer = ciffy.load("complex.cif")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, chain in enumerate(polymer.chains(ciffy.RNA)):
    contact_map(chain, ax=axes[i], title=f"Chain {chain.names[0]}")

plt.tight_layout()
plt.savefig("chain_contacts.png")
```

## Profile Plots

Plot per-residue values along the sequence:

```python
import ciffy
import numpy as np
from ciffy.visualize import plot_profile
import matplotlib.pyplot as plt

polymer = ciffy.load("structure.cif")

# Compute some per-residue metric
values = np.random.rand(polymer.size(ciffy.RESIDUE))

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
plot_profile(polymer, values, ax=ax, ylabel="Random Value")
plt.savefig("profile.png")
```

### Profile Options

```python
# Fill under curve (default)
plot_profile(polymer, values, fill=True)

# Line only
plot_profile(polymer, values, fill=False)

# Custom color
plot_profile(polymer, values, color="steelblue")

# At atom scale
atom_values = np.random.rand(polymer.size(ciffy.ATOM))
plot_profile(polymer, atom_values, scale=ciffy.ATOM)

# Add title
plot_profile(polymer, values, title=f"{polymer.id()} B-factors")
```

### Multi-Chain Profiles

```python
polymer = ciffy.load("complex.cif")

fig, axes = plt.subplots(len(polymer.names), 1, figsize=(12, 8))

for i, chain in enumerate(polymer.chains()):
    values = compute_values(chain)
    plot_profile(chain, values, ax=axes[i], title=f"Chain {chain.names[0]}")

plt.tight_layout()
plt.savefig("chain_profiles.png")
```

## ChimeraX Integration

Export per-residue attributes for coloring in ChimeraX:

```python
import ciffy
from ciffy.visualize import to_defattr

polymer = ciffy.load("structure.cif")

# Compute per-residue values
values = compute_conservation_scores(polymer)

# Export as defattr file
to_defattr(
    polymer,
    values,
    "conservation.defattr",
    scale=ciffy.RESIDUE,
    name="conservation"
)
```

### Using in ChimeraX

1. Open the structure:
   ```
   open structure.cif
   ```

2. Load the attribute file:
   ```
   open conservation.defattr
   ```

3. Color by attribute:
   ```
   color byattribute conservation palette blue:white:red
   ```

### Attribute File Format

The defattr file format:
```
attribute: conservation
match mode: 1-to-1
recipient: residues

	:1	0.85
	:2	0.72
	:3	0.91
	...
```

### Per-Atom Attributes

```python
# Atom-level attributes
atom_values = compute_atom_scores(polymer)

to_defattr(
    polymer,
    atom_values,
    "atom_scores.defattr",
    scale=ciffy.ATOM,
    name="atom_score"
)
```

In ChimeraX:
```
color byattribute atom_score atoms palette viridis
```

## Comparing Structures

### Side-by-Side Contact Maps

```python
import ciffy
from ciffy.visualize import contact_map
import matplotlib.pyplot as plt

p1 = ciffy.load("structure1.cif")
p2 = ciffy.load("structure2.cif")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

contact_map(p1, ax=ax1, title=p1.id())
contact_map(p2, ax=ax2, title=p2.id())

plt.tight_layout()
plt.savefig("comparison.png")
```

### Difference Contact Maps

```python
import numpy as np

p1 = ciffy.load("conf1.cif").poly()
p2 = ciffy.load("conf2.cif").poly()

# Compute distance matrices
d1 = p1.pairwise_distances(ciffy.RESIDUE)
d2 = p2.pairwise_distances(ciffy.RESIDUE)

# Difference
diff = np.abs(d1 - d2)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(diff, cmap="RdBu_r", vmin=-5, vmax=5)
plt.colorbar(im, label="Distance change (Å)")
ax.set_title("Conformational Change")
plt.savefig("diff_map.png")
```

## Ramachandran Plots

For protein structures, visualize backbone dihedrals:

```python
import ciffy
import numpy as np
import matplotlib.pyplot as plt

protein = ciffy.load("protein.cif").by_type(ciffy.PROTEIN)

# Get phi/psi angles
phi = protein.dihedral(ciffy.DihedralType.PHI)
psi = protein.dihedral(ciffy.DihedralType.PSI)

# Convert to degrees
phi_deg = np.degrees(phi)
psi_deg = np.degrees(psi)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(phi_deg, psi_deg, alpha=0.5, s=10)
ax.set_xlabel("φ (degrees)")
ax.set_ylabel("ψ (degrees)")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title("Ramachandran Plot")
plt.savefig("ramachandran.png")
```

## Radius of Gyration Profile

```python
import ciffy
import numpy as np
from ciffy.visualize import plot_profile
import matplotlib.pyplot as plt

polymer = ciffy.load("structure.cif").poly()

# Compute per-residue distance from center
centered, _ = polymer.center(ciffy.MOLECULE)
coords = centered.coordinates

# Distance from center for each atom
distances = np.linalg.norm(coords, axis=1)

# Aggregate to per-residue (mean distance)
res_distances = polymer.reduce(distances, ciffy.RESIDUE)

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
plot_profile(polymer, res_distances, ax=ax, ylabel="Distance from center (Å)")
ax.set_title(f"{polymer.id()} Radial Distribution")
plt.savefig("radial_profile.png")
```

## Publication-Ready Figures

```python
import matplotlib.pyplot as plt

# Set up publication style
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

polymer = ciffy.load("structure.cif")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Contact map
contact_map(polymer, ax=ax1, cmap="Purples")
ax1.set_title("Contact Map", fontsize=14, fontweight='bold')

# Profile
values = compute_values(polymer)
plot_profile(polymer, values, ax=ax2)
ax2.set_title("Conservation", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("figure.pdf", dpi=300, bbox_inches='tight')
```
