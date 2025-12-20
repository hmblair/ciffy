# API Reference

## Core

### load

::: ciffy.load

### Polymer

::: ciffy.Polymer
    options:
      members:
        - __init__
        - id
        - size
        - sizes
        - per
        - molecule_type
        - istype
        - reduce
        - rreduce
        - expand
        - count
        - index
        - center
        - pairwise_distances
        - align
        - moment
        - mask
        - __getitem__
        - by_index
        - by_atom
        - by_residue
        - by_type
        - poly
        - hetero
        - chains
        - resolved
        - strip
        - backbone
        - str
        - atom_names
        - backend
        - numpy
        - torch
        - to
        - write
        - with_coordinates
        - distances
        - angles
        - dihedrals
        - dihedral
        - set_dihedral

---

## Operations

### rmsd

::: ciffy.rmsd

### align

::: ciffy.align

### Reduction

::: ciffy.Reduction

---

## Types

### Scale

::: ciffy.Scale

### Molecule

::: ciffy.Molecule

### DihedralType

::: ciffy.DihedralType

### kabsch_rotation

::: ciffy.kabsch_rotation

### kabsch_align

::: ciffy.kabsch_align

### tm_score

::: ciffy.tm_score

### lddt

::: ciffy.lddt

---

## I/O

### write_cif

::: ciffy.write_cif

### load_metadata

::: ciffy.load_metadata

### from_sequence

::: ciffy.from_sequence

### from_extract

::: ciffy.from_extract

---

## Sampling

### randomize_backbone

::: ciffy.randomize_backbone

---

## Ensemble

::: ciffy.Ensemble

---

## Neural Networks (ciffy.nn)

!!! note "Optional Dependency"
    The `ciffy.nn` module requires PyTorch. Install with `pip install torch`.

### PolymerEmbedding

::: ciffy.nn.PolymerEmbedding

### PolymerDataset

::: ciffy.nn.PolymerDataset

### Transformer

::: ciffy.nn.Transformer

### TransformerBlock

::: ciffy.nn.TransformerBlock

### MultiHeadAttention

::: ciffy.nn.MultiHeadAttention

### RMSNorm

::: ciffy.nn.RMSNorm

### RotaryPositionEmbedding

::: ciffy.nn.RotaryPositionEmbedding

### SwiGLU

::: ciffy.nn.SwiGLU

### PolymerVAE

::: ciffy.nn.PolymerVAE

### DihedralEncoder

::: ciffy.nn.DihedralEncoder

### DihedralDecoder

::: ciffy.nn.DihedralDecoder

---

## Constants

### Vocabulary Sizes

::: ciffy.NUM_ELEMENTS

::: ciffy.NUM_RESIDUES

::: ciffy.NUM_ATOMS

### Dihedral Constants

::: ciffy.PROTEIN_BACKBONE

::: ciffy.RNA_BACKBONE

::: ciffy.RNA_GLYCOSIDIC

::: ciffy.DIHEDRAL_ATOMS

::: ciffy.DIHEDRAL_NAME_TO_TYPE

---

## Visualization (ciffy.visualize)

### to_defattr

::: ciffy.to_defattr

### plot_profile

::: ciffy.plot_profile

### contact_map

::: ciffy.contact_map
