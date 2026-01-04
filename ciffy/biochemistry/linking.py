"""
Inter-residue linking definitions for polymer chains.

Defines the atoms and coordinate frames involved in linking consecutive
residues, enabling both template positioning and flow model decoding.

Supported polymer types:
- RNA, DNA, HYBRID: Phosphodiester linkage (O3' -> P)
- PROTEIN, PROTEIN_D, CYCLIC_PEPTIDE: Peptide bond (C -> N)

Unsupported polymer types (no linking definition):
- POLYSACCHARIDE: Glycosidic bonds vary by sugar type
- PNA: Synthetic backbone with different linkage
- LIGAND, ION, WATER, OTHER, UNKNOWN: Non-polymeric
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._generated_molecule import Molecule
from ._generated_linking import (
    NUCLEIC_ACID_LINK_GEOMETRY,
    PEPTIDE_LINK_GEOMETRY,
)
from .constants import (
    Sugar, PhosphateGroup, ProteinBackbone,
    PurineBase, PyrimidineBase,
)
from .atom import AtomGroup

if TYPE_CHECKING:
    pass


# Unified backbone atom values (from codegen/config.py)
# These are shared across all residue types, enabling robust frame resolution
# for modified residues with standard backbones.
BACKBONE_ATOM_VALUES: dict[str, int] = {
    # Nucleic acid backbone (Python names with p for apostrophe)
    "P": 1, "OP1": 2, "OP2": 3, "OP3": 4,
    "O5p": 5, "C5p": 6, "C4p": 7, "O4p": 8,
    "C3p": 9, "O3p": 10, "C2p": 11, "O2p": 12, "C1p": 13,
    # Protein backbone
    "N": 14, "CA": 15, "C": 16, "O": 17,
}


@dataclass
class FrameDefinition:
    """
    Definition for computing a local coordinate frame from 3 atoms.

    Frame convention:
    - Z-axis points FROM origin TOWARD axis_ref
    - X-axis is perpendicular to Z, toward plane_ref (via Gram-Schmidt)
    - Y-axis completes right-handed system (Y = Z × X)

    Atoms are specified as AtomGroups (e.g., Sugar.O3p, PurineBase.N9).
    For residue-specific atom values, use AtomGroup.for_residue(residue).

    Example:
        O3' frame for RNA:
        - Origin at O3', Z points toward C3' (into residue)
        - X perpendicular in C4'-O3'-C3' plane

        >>> positions = O3P_FRAME.extract(coords, atoms, Residue.A)
        >>> origin, R = frame_from_positions(positions)
    """

    origin: AtomGroup      # Frame location, Z-axis starts here
    axis_ref: AtomGroup    # Z-axis points toward here
    plane_ref: AtomGroup   # X-axis derived toward here (perpendicular to Z)


@dataclass
class LinkingDefinition:
    """
    Definition for inter-residue bonding and coordinate frames.

    Attributes:
        prev_atom: AtomGroup for the atom on residue N that forms the bond.
        next_atom: AtomGroup for the atom on residue N+1 that forms the bond.
        bond_length: Standard bond length in Angstroms.
        prev_frame: Frame definition for outgoing link (at prev_atom).
        next_frame: Frame definition for incoming link (at next_atom).

    Example:
        For RNA, residue N's O3' binds to residue N+1's P with ~1.6A bond.
        The O3' frame defines the outgoing direction, P frame the incoming.
    """

    prev_atom: AtomGroup
    next_atom: AtomGroup
    bond_length: float
    prev_frame: FrameDefinition
    next_frame: FrameDefinition


# =============================================================================
# Nucleic Acid Linking Frames
# =============================================================================
# All frames use Z-primary convention: Z points origin → axis_ref

# O3' frame: Z points O3'→C3' (into residue)
O3P_FRAME = FrameDefinition(
    origin=Sugar.O3p,
    axis_ref=Sugar.C3p,
    plane_ref=Sugar.C4p,
)

# P frame: Z points P→O5' (toward 5' end)
P_FRAME = FrameDefinition(
    origin=PhosphateGroup.P,
    axis_ref=Sugar.O5p,
    plane_ref=PhosphateGroup.OP1,
)

# Phosphodiester bond: O3' of residue N to P of residue N+1
NUCLEIC_ACID_LINK = LinkingDefinition(
    prev_atom=Sugar.O3p,
    next_atom=PhosphateGroup.P,
    bond_length=NUCLEIC_ACID_LINK_GEOMETRY.bond_length,
    prev_frame=O3P_FRAME,
    next_frame=P_FRAME,
)


# =============================================================================
# Protein Linking Frames
# =============================================================================

# C frame: Z points C→CA (into residue)
C_FRAME = FrameDefinition(
    origin=ProteinBackbone.C,
    axis_ref=ProteinBackbone.CA,
    plane_ref=ProteinBackbone.O,
)

# N frame: Z points N→CA (along backbone)
# Note: plane_ref uses O from same residue for now (arbitrary perpendicular)
N_FRAME = FrameDefinition(
    origin=ProteinBackbone.N,
    axis_ref=ProteinBackbone.CA,
    plane_ref=ProteinBackbone.C,  # Using C as plane_ref (within same residue)
)

# Peptide bond: C of residue N to N of residue N+1
PEPTIDE_LINK = LinkingDefinition(
    prev_atom=ProteinBackbone.C,
    next_atom=ProteinBackbone.N,
    bond_length=PEPTIDE_LINK_GEOMETRY.bond_length,
    prev_frame=C_FRAME,
    next_frame=N_FRAME,
)


# =============================================================================
# Molecule Type → Linking Definition Map
# =============================================================================

LINKING_BY_TYPE: dict[int, LinkingDefinition] = {
    # Nucleic acids (phosphodiester linkage)
    Molecule.RNA.value: NUCLEIC_ACID_LINK,
    Molecule.DNA.value: NUCLEIC_ACID_LINK,
    Molecule.HYBRID.value: NUCLEIC_ACID_LINK,
    # Peptides (peptide bond)
    Molecule.PROTEIN.value: PEPTIDE_LINK,
    Molecule.PROTEIN_D.value: PEPTIDE_LINK,
    Molecule.CYCLIC_PEPTIDE.value: PEPTIDE_LINK,
}


# =============================================================================
# Alignment Frames (for residue encoding)
# =============================================================================
# Frame definitions for aligning residues to a canonical local frame.
# Used by Polymer.align() to put residues in consistent orientations.

# Purine glycosidic: Z points C1'→N9 (toward base)
PURINE_GLYCOSIDIC_FRAME = FrameDefinition(
    origin=Sugar.C1p,
    axis_ref=PurineBase.N9,
    plane_ref=PurineBase.C4,
)

# Pyrimidine glycosidic: Z points C1'→N1 (toward base)
PYRIMIDINE_GLYCOSIDIC_FRAME = FrameDefinition(
    origin=Sugar.C1p,
    axis_ref=PyrimidineBase.N1,
    plane_ref=PyrimidineBase.C2,
)

# Protein backbone: Z points CA→N
PROTEIN_BACKBONE_FRAME = FrameDefinition(
    origin=ProteinBackbone.CA,
    axis_ref=ProteinBackbone.N,
    plane_ref=ProteinBackbone.C,
)

# Unified glycosidic frame for all nucleotides (purines + pyrimidines)
# Uses combined AtomGroups that match N9/C4 for purines OR N1/C2 for pyrimidines
from .constants import NucleotideAxisRef, NucleotidePlaneRef
GLYCOSIDIC_FRAME = FrameDefinition(
    origin=Sugar.C1p,
    axis_ref=NucleotideAxisRef,
    plane_ref=NucleotidePlaneRef,
)
