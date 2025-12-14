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

### InternalPolymer (Deprecated)

!!! warning "Deprecated"
    `InternalPolymer` is deprecated. Use `Polymer` directly - it now supports both Cartesian and internal coordinates transparently.

::: ciffy.InternalPolymer

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

---

## I/O

### write_cif

::: ciffy.write_cif

### from_sequence

::: ciffy.from_sequence

---

## Constants

### Vocabulary Sizes

::: ciffy.NUM_ELEMENTS

::: ciffy.NUM_RESIDUES

::: ciffy.NUM_ATOMS
