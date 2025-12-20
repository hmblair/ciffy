# Protein Structure Analysis

This guide covers protein-specific analysis in ciffy, including backbone geometry, Ramachandran analysis, and residue-level operations.

## Loading Protein Structures

```python
import ciffy

# Load structure
polymer = ciffy.load("protein.cif")

# Extract only protein chains
protein = polymer.by_type(ciffy.PROTEIN)

# Check molecule type
for chain in polymer.chains():
    if chain.istype(ciffy.PROTEIN):
        print(f"Chain {chain.names[0]} is protein")
```

## Protein Sequence

### Getting the Sequence

```python
protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

# One-letter sequence
for chain in protein.chains():
    seq = chain.sequence_str()
    print(f"Chain {chain.names[0]}: {seq}")
# Output: Chain A: MGKLVFFAED...
```

### Residue Types

```python
from ciffy.biochemistry import Residue

# Standard amino acids
Residue.ALA  # Alanine
Residue.GLY  # Glycine
Residue.VAL  # Valine
# ... all 20 standard amino acids

# Access residue information
sequence = protein.sequence
print(f"First residue: {Residue(sequence[0]).name}")
```

### Selecting by Residue Type

```python
from ciffy.biochemistry import Residue

# Get all glycines
glycines = protein.by_residue(Residue.GLY)

# Get hydrophobic residues
hydrophobic = protein.by_residue([
    Residue.ALA, Residue.VAL, Residue.LEU,
    Residue.ILE, Residue.MET, Residue.PHE,
    Residue.TRP, Residue.PRO
])

# Get charged residues
charged = protein.by_residue([
    Residue.ASP, Residue.GLU,  # Negative
    Residue.LYS, Residue.ARG, Residue.HIS  # Positive
])
```

## Structural Components

### Backbone Atoms

The protein backbone consists of N-CA-C-O atoms:

```python
# Get backbone atoms
backbone = protein.backbone()
print(f"Backbone atoms: {backbone.size()}")

# Backbone includes: N, CA, C, O (4 atoms per residue)
```

### Sidechain Atoms

```python
# Get sidechain atoms
sidechains = protein.sidechain()
print(f"Sidechain atoms: {sidechains.size()}")

# Sidechains: all atoms except N, CA, C, O, and hydrogens
```

### CA Trace

Extract alpha-carbon coordinates:

```python
from ciffy.biochemistry import Residue

# Get CA atoms for a specific residue type using .value
ala_ca = protein.by_atom(Residue.ALA.CA.value)

# For all CA atoms, use reduce to get one coordinate per residue
# The backbone already includes CA for proteins
backbone = protein.backbone()
ca_coords = protein.reduce(protein.coordinates, ciffy.RESIDUE)

print(f"CA trace: {ca_coords.shape}")  # (num_residues, 3)
```

## Backbone Dihedrals

Protein backbone conformation is defined by three torsion angles:

| Angle | Atoms | Description |
|-------|-------|-------------|
| φ (phi) | C(i-1) - N - CA - C | N-CA bond rotation |
| ψ (psi) | N - CA - C - N(i+1) | CA-C bond rotation |
| ω (omega) | CA - C - N(i+1) - CA(i+1) | Peptide bond (usually ~180°) |

### Accessing Backbone Dihedrals

```python
import ciffy
import numpy as np

protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

# Get phi, psi, omega angles
phi = protein.dihedral(ciffy.DihedralType.PHI)
psi = protein.dihedral(ciffy.DihedralType.PSI)
omega = protein.dihedral(ciffy.DihedralType.OMEGA)

# Convert to degrees
phi_deg = np.degrees(phi)
psi_deg = np.degrees(psi)
omega_deg = np.degrees(omega)

print(f"Phi: mean={np.nanmean(phi_deg):.1f}°, std={np.nanstd(phi_deg):.1f}°")
print(f"Psi: mean={np.nanmean(psi_deg):.1f}°, std={np.nanstd(psi_deg):.1f}°")
print(f"Omega: mean={np.nanmean(omega_deg):.1f}° (should be ~180°)")
```

## Ramachandran Analysis

The Ramachandran plot shows the distribution of phi/psi angles:

```python
import numpy as np
import matplotlib.pyplot as plt

protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

# Get phi and psi
phi = np.degrees(protein.dihedral(ciffy.DihedralType.PHI))
psi = np.degrees(protein.dihedral(ciffy.DihedralType.PSI))

# Remove NaN values (terminal residues)
valid = ~(np.isnan(phi) | np.isnan(psi))
phi = phi[valid]
psi = psi[valid]

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(phi, psi, alpha=0.5, s=20)
ax.set_xlabel("φ (degrees)", fontsize=12)
ax.set_ylabel("ψ (degrees)", fontsize=12)
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title("Ramachandran Plot")
ax.set_aspect('equal')
plt.savefig("ramachandran.png", dpi=150)
```

### Secondary Structure Regions

```python
import numpy as np

protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

phi = np.degrees(protein.dihedral(ciffy.DihedralType.PHI))
psi = np.degrees(protein.dihedral(ciffy.DihedralType.PSI))

# Alpha helix region: phi ~ -60°, psi ~ -45°
alpha_helix = (
    (phi > -80) & (phi < -40) &
    (psi > -60) & (psi < -20)
)

# Beta sheet region: phi ~ -120°, psi ~ 120°
beta_sheet = (
    (phi > -150) & (phi < -90) &
    (psi > 90) & (psi < 150)
)

n_residues = len(phi)
print(f"Alpha helix: {np.sum(alpha_helix)} residues ({100*np.sum(alpha_helix)/n_residues:.1f}%)")
print(f"Beta sheet: {np.sum(beta_sheet)} residues ({100*np.sum(beta_sheet)/n_residues:.1f}%)")
```

### Glycine and Proline Special Cases

```python
from ciffy.biochemistry import Residue

protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

# Get residue types
sequence = protein.sequence

# Find glycines (can access left-hand region)
gly_mask = np.array([Residue(s) == Residue.GLY for s in sequence])

# Find prolines (restricted phi)
pro_mask = np.array([Residue(s) == Residue.PRO for s in sequence])

phi = np.degrees(protein.dihedral(ciffy.DihedralType.PHI))

print(f"Glycine phi range: {phi[gly_mask].min():.0f}° to {phi[gly_mask].max():.0f}°")
print(f"Proline phi range: {phi[pro_mask].min():.0f}° to {phi[pro_mask].max():.0f}°")
```

## Cis/Trans Peptide Bonds

Most peptide bonds are trans (ω ~ 180°), but cis bonds (ω ~ 0°) occur:

```python
import numpy as np

protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

omega = np.degrees(protein.dihedral(ciffy.DihedralType.OMEGA))
valid_omega = omega[~np.isnan(omega)]

# Cis peptide bonds: omega close to 0°
cis_bonds = np.abs(valid_omega) < 30
trans_bonds = np.abs(valid_omega) > 150

print(f"Trans peptide bonds: {np.sum(trans_bonds)}")
print(f"Cis peptide bonds: {np.sum(cis_bonds)}")

# Cis bonds are often before prolines
if np.any(cis_bonds):
    cis_positions = np.where(cis_bonds)[0]
    print(f"Cis bond positions: {cis_positions + 1}")  # 1-indexed
```

## Contact Analysis

### Residue-Residue Contacts

```python
import numpy as np

protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

# Compute CA-CA distance matrix
backbone = protein.backbone()
_, ca_centers = backbone.center(ciffy.RESIDUE)

distances = np.linalg.norm(
    ca_centers[:, None, :] - ca_centers[None, :, :],
    axis=-1
)

# Contact map (8 Å cutoff)
contacts = distances < 8.0

# Count contacts per residue
contact_count = contacts.sum(axis=1)
print(f"Mean contacts per residue: {contact_count.mean():.1f}")
```

### Long-Range Contacts

```python
import numpy as np

# Find contacts between residues far apart in sequence
n_res = len(distances)
sequence_distance = np.abs(np.arange(n_res)[:, None] - np.arange(n_res)[None, :])

# Long-range: > 12 residues apart in sequence, < 8 Å in space
long_range = (sequence_distance > 12) & (distances < 8.0)

print(f"Long-range contacts: {long_range.sum() // 2}")  # Divide by 2 for unique pairs

# Find the pairs
pairs = np.argwhere(long_range & (np.triu(np.ones_like(long_range), k=1) > 0))
for i, j in pairs[:10]:  # First 10
    print(f"  Residues {i+1} - {j+1}: {distances[i,j]:.1f} Å")
```

## Sidechain Analysis

### Rotamer Analysis

```python
import numpy as np
from ciffy.biochemistry import Residue

protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

# Sidechain chi angles would require computing additional dihedrals
# For now, analyze sidechain centroids
sidechains = protein.sidechain()

if sidechains.size() > 0:
    _, sc_centers = sidechains.center(ciffy.RESIDUE)
    print(f"Sidechain centers: {sc_centers.shape}")
```

### Sidechain-Backbone Distances

```python
protein = ciffy.load("structure.cif").by_type(ciffy.PROTEIN)

backbone = protein.backbone()
sidechains = protein.sidechain()

# Get per-residue centers
_, bb_centers = backbone.center(ciffy.RESIDUE)
_, sc_centers = sidechains.center(ciffy.RESIDUE)

# Distance from backbone to sidechain center
bb_sc_dist = np.linalg.norm(bb_centers - sc_centers, axis=1)

# Glycine has no sidechain
sequence = protein.sequence
gly_mask = np.array([Residue(s) == Residue.GLY for s in sequence])

print(f"Mean backbone-sidechain distance: {bb_sc_dist[~gly_mask].mean():.2f} Å")
```

## Multi-Chain Protein Analysis

### Interface Analysis

```python
import numpy as np

protein = ciffy.load("complex.cif").by_type(ciffy.PROTEIN)

chains = list(protein.chains())
if len(chains) >= 2:
    c1, c2 = chains[0], chains[1]

    # Compute cross-chain distances
    coords1 = c1.coordinates
    coords2 = c2.coordinates

    cross_dist = np.linalg.norm(
        coords1[:, None, :] - coords2[None, :, :],
        axis=-1
    )

    # Interface atoms: within 4 Å of other chain
    interface1 = (cross_dist.min(axis=1) < 4.0)
    interface2 = (cross_dist.min(axis=0) < 4.0)

    print(f"Chain {c1.names[0]} interface atoms: {interface1.sum()}")
    print(f"Chain {c2.names[0]} interface atoms: {interface2.sum()}")
```

### Per-Chain Analysis

```python
protein = ciffy.load("complex.cif").by_type(ciffy.PROTEIN)

for chain in protein.chains():
    name = chain.names[0]
    n_res = chain.size(ciffy.RESIDUE)
    seq = chain.sequence_str()

    # Secondary structure estimate
    phi = np.degrees(chain.dihedral(ciffy.DihedralType.PHI))
    psi = np.degrees(chain.dihedral(ciffy.DihedralType.PSI))

    alpha = np.sum((phi > -80) & (phi < -40) & (psi > -60) & (psi < -20))
    beta = np.sum((phi > -150) & (phi < -90) & (psi > 90) & (psi < 150))

    print(f"Chain {name}: {n_res} residues")
    print(f"  Sequence: {seq[:30]}...")
    print(f"  ~{100*alpha/n_res:.0f}% helix, ~{100*beta/n_res:.0f}% sheet")
```

## Complete Protein Analysis Example

```python
import ciffy
import numpy as np

def analyze_protein(cif_file):
    """Complete protein structure analysis."""
    polymer = ciffy.load(cif_file)
    protein = polymer.by_type(ciffy.PROTEIN)

    if protein.size() == 0:
        print("No protein chains found")
        return

    print(f"Structure: {polymer.id()}")
    print(f"Protein chains: {len(protein.names)}")
    print(f"Total residues: {protein.size(ciffy.RESIDUE)}")
    print()

    for chain in protein.chains():
        name = chain.names[0]
        n_res = chain.size(ciffy.RESIDUE)

        print(f"Chain {name} ({n_res} residues)")
        print(f"  Sequence: {chain.sequence_str()[:50]}...")

        # Backbone geometry
        phi = np.degrees(chain.dihedral(ciffy.DihedralType.PHI))
        psi = np.degrees(chain.dihedral(ciffy.DihedralType.PSI))
        omega = np.degrees(chain.dihedral(ciffy.DihedralType.OMEGA))

        # Secondary structure
        valid = ~(np.isnan(phi) | np.isnan(psi))
        alpha = np.sum((phi[valid] > -80) & (phi[valid] < -40) &
                       (psi[valid] > -60) & (psi[valid] < -20))
        beta = np.sum((phi[valid] > -150) & (phi[valid] < -90) &
                      (psi[valid] > 90) & (psi[valid] < 150))

        print(f"  Secondary structure: {100*alpha/n_res:.0f}% helix, {100*beta/n_res:.0f}% sheet")

        # Cis peptide bonds
        cis = np.sum(np.abs(omega[~np.isnan(omega)]) < 30)
        if cis > 0:
            print(f"  Cis peptide bonds: {cis}")

        # Radius of gyration
        centered, _ = chain.center(ciffy.MOLECULE)
        rg = np.sqrt((centered.coordinates ** 2).sum(axis=1).mean())
        print(f"  Radius of gyration: {rg:.1f} Å")
        print()

# Run analysis
analyze_protein("structure.cif")
```

## Working with Ligands

Proteins often have bound ligands:

```python
from ciffy import Molecule

polymer = ciffy.load("protein_ligand.cif")

# Get protein and ligand separately
protein = polymer.by_type(ciffy.PROTEIN)
ligands = polymer.by_type(Molecule.LIGAND)

print(f"Protein atoms: {protein.size()}")
print(f"Ligand atoms: {ligands.size()}")

# Find protein atoms near ligand
if ligands.size() > 0:
    protein_coords = protein.coordinates
    ligand_coords = ligands.coordinates

    distances = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :],
        axis=-1
    )

    binding_site = distances.min(axis=1) < 4.0
    print(f"Binding site atoms: {binding_site.sum()}")
```

## Next Steps

- [Selection and Filtering](selection.md) - More selection techniques
- [Structural Analysis](analysis.md) - RMSD, alignment
- [Deep Learning](deep-learning.md) - Using protein structures with PyTorch
- [RNA Analysis](rna.md) - Working with nucleic acids
