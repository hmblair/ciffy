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

if TYPE_CHECKING:
    from . import Residue


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
    Definition for computing a local coordinate frame at a linking atom.

    Frame construction:
    - Origin at `origin` atom
    - Z-axis from `z_ref` toward `origin` (if z_toward_origin=True) or vice versa
    - X-axis perpendicular, toward `perp_ref` (or arbitrary if None)
    - Y-axis completes right-handed system

    Atom names use Python naming convention (apostrophe -> p, e.g., "O3p" for O3').

    Example:
        O3' frame for RNA outgoing link:
        - Origin at O3'
        - Z-axis along C3'->O3' bond (outward)
        - X-axis perpendicular, in C4'-C3'-O3' plane
    """

    origin: str              # Atom at frame origin (Python name, e.g., "O3p")
    z_ref: str               # Atom for Z-axis reference
    perp_ref: str | None     # Atom for perpendicular reference (None = arbitrary)
    z_toward_origin: bool    # True if Z points from z_ref toward origin

    def resolve(
        self,
        residue: "Residue",
        atom_to_col: dict[int, int],
    ) -> tuple[int, int, int | None]:
        """
        Convert atom names to column indices for a specific residue.

        This should be called once at model initialization and cached.
        The resulting indices can be used with compute_frame_from_indices()
        for fast, vectorizable frame computation.

        Args:
            residue: Residue enum (e.g., Residue.A) providing atom lookups.
            atom_to_col: Dict mapping atom type values to column indices.

        Returns:
            Tuple of (origin_col, z_ref_col, perp_ref_col) where perp_ref_col
            may be None if perp_ref is None.

        Raises:
            KeyError: If required atoms are not in atom_to_col.
            AttributeError: If residue doesn't have the specified atom.
        """
        origin_col = atom_to_col[getattr(residue, self.origin).value]
        z_ref_col = atom_to_col[getattr(residue, self.z_ref).value]
        perp_ref_col = (
            atom_to_col[getattr(residue, self.perp_ref).value]
            if self.perp_ref else None
        )
        return origin_col, z_ref_col, perp_ref_col

    def resolve_by_value(
        self,
        atom_to_col: dict[int, int],
    ) -> tuple[int, int, int | None]:
        """
        Resolve frame using unified backbone atom values.

        This method uses the fixed backbone atom values (BACKBONE_ATOM_VALUES)
        instead of looking up atoms via a Residue enum. This enables robust
        frame resolution for modified residues with standard backbones.

        Args:
            atom_to_col: Dict mapping atom type values to column indices.

        Returns:
            Tuple of (origin_col, z_ref_col, perp_ref_col) where perp_ref_col
            may be None if perp_ref is None.

        Raises:
            KeyError: If required backbone atoms are not in atom_to_col,
                or if atom names are not in BACKBONE_ATOM_VALUES.
        """
        origin_val = BACKBONE_ATOM_VALUES[self.origin]
        z_ref_val = BACKBONE_ATOM_VALUES[self.z_ref]

        origin_col = atom_to_col[origin_val]
        z_ref_col = atom_to_col[z_ref_val]
        perp_ref_col = None
        if self.perp_ref:
            perp_ref_val = BACKBONE_ATOM_VALUES[self.perp_ref]
            perp_ref_col = atom_to_col[perp_ref_val]

        return origin_col, z_ref_col, perp_ref_col

    def atom_values(self) -> tuple[int, int, int]:
        """
        Get atom type values for frame atoms as an ordered tuple.

        Returns:
            (origin_val, z_ref_val, perp_ref_val) where perp_ref_val is -1
            if perp_ref is None.
        """
        origin_val = BACKBONE_ATOM_VALUES[self.origin]
        z_ref_val = BACKBONE_ATOM_VALUES[self.z_ref]
        perp_ref_val = BACKBONE_ATOM_VALUES[self.perp_ref] if self.perp_ref else -1
        return origin_val, z_ref_val, perp_ref_val

    def required_backbone_values(self) -> set[int]:
        """
        Get the set of backbone atom values required for this frame.

        Returns:
            Set of unified backbone atom values needed to compute this frame.
        """
        required = {BACKBONE_ATOM_VALUES[self.origin]}
        required.add(BACKBONE_ATOM_VALUES[self.z_ref])
        if self.perp_ref:
            required.add(BACKBONE_ATOM_VALUES[self.perp_ref])
        return required

    def required_atoms(self, residue: "Residue") -> set[int]:
        """
        Get the set of atom values required for this frame.

        Args:
            residue: Residue type (e.g., Residue.A).

        Returns:
            Set of atom type values needed to compute this frame.
        """
        required = {getattr(residue, self.origin).value}
        required.add(getattr(residue, self.z_ref).value)
        if self.perp_ref:
            required.add(getattr(residue, self.perp_ref).value)
        return required


@dataclass
class LinkingDefinition:
    """
    Definition for inter-residue bonding and coordinate frames.

    Attributes:
        prev_atom: Atom name on residue N that forms the bond (e.g., "O3p", "C").
                   Uses Python naming convention (apostrophe -> p).
        next_atom: Atom name on residue N+1 that forms the bond (e.g., "P", "N").
        bond_length: Standard bond length in Angstroms.
        prev_frame: Frame definition for outgoing link (at prev_atom).
        next_frame: Frame definition for incoming link (at next_atom).

    Example:
        For RNA, residue N's O3' binds to residue N+1's P with ~1.6A bond.
        The O3' frame defines the outgoing direction, P frame the incoming.
    """

    prev_atom: str
    next_atom: str
    bond_length: float
    prev_frame: FrameDefinition
    next_frame: FrameDefinition

    def required_atoms(self, residue: "Residue") -> set[int]:
        """
        Get all atoms required for link computation (both frames).

        Args:
            residue: Residue type.

        Returns:
            Set of atom type values needed for linking.
        """
        return self.prev_frame.required_atoms(residue) | self.next_frame.required_atoms(residue)

    def validate_atoms(
        self,
        residue: "Residue",
        available_atoms: set[int],
        which: str = "both",
    ) -> list[str]:
        """
        Validate that required atoms are available for frame computation.

        Args:
            residue: Residue type.
            available_atoms: Set of available atom type values.
            which: "prev", "next", or "both" to check specific frames.

        Returns:
            List of missing atom names (empty if all present).
        """
        if which == "prev":
            required = self.prev_frame.required_atoms(residue)
        elif which == "next":
            required = self.next_frame.required_atoms(residue)
        else:
            required = self.required_atoms(residue)

        missing_values = required - available_atoms
        if not missing_values:
            return []

        # Convert values back to names for error messages
        value_to_name = {int(a): a.name for a in residue}
        return [value_to_name.get(v, f"atom_{v}") for v in missing_values]


# Phosphodiester bond: O3' of residue N to P of residue N+1
# O3' frame: origin at O3', Z along C3'->O3', X toward C4'
# P frame: origin at P, Z along O5'->P, X toward OP1
# Bond length from MonomerLibrary (links_and_mods.cif)
NUCLEIC_ACID_LINK = LinkingDefinition(
    prev_atom="O3p",
    next_atom="P",
    bond_length=NUCLEIC_ACID_LINK_GEOMETRY.bond_length,
    prev_frame=FrameDefinition(
        origin="O3p",
        z_ref="C3p",
        perp_ref="C4p",
        z_toward_origin=True,
    ),
    next_frame=FrameDefinition(
        origin="P",
        z_ref="O5p",
        perp_ref="OP1",
        z_toward_origin=True,
    ),
)

# Peptide bond: C of residue N to N of residue N+1
# C frame: origin at C, Z along CA->C, X toward O
# N frame: origin at N, Z along N->CA (inverted), no perpendicular ref
# Bond length from MonomerLibrary (links_and_mods.cif)
PEPTIDE_LINK = LinkingDefinition(
    prev_atom="C",
    next_atom="N",
    bond_length=PEPTIDE_LINK_GEOMETRY.bond_length,
    prev_frame=FrameDefinition(
        origin="C",
        z_ref="CA",
        perp_ref="O",
        z_toward_origin=True,
    ),
    next_frame=FrameDefinition(
        origin="N",
        z_ref="CA",
        perp_ref=None,
        z_toward_origin=False,
    ),
)

# Map molecule type to linking definition
# Only polymer types with well-defined inter-residue linkages are included
# Keys are molecule type int values for easy lookup from residue.molecule_type
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
# Alignment Frames
# =============================================================================
# Frame definitions for aligning residues to a canonical local frame.
# Used by Polymer.align() to put residues in consistent orientations.

# Glycosidic frame for purines (A, G): C1' origin, X toward N9
PURINE_GLYCOSIDIC_FRAME = FrameDefinition(
    origin="C1p",
    z_ref="N9",
    perp_ref="C4",
    z_toward_origin=False,  # Z from C1' toward N9
)

# Glycosidic frame for pyrimidines (C, U, T): C1' origin, X toward N1
PYRIMIDINE_GLYCOSIDIC_FRAME = FrameDefinition(
    origin="C1p",
    z_ref="N1",
    perp_ref="C2",
    z_toward_origin=False,  # Z from C1' toward N1
)

# Backbone frame for proteins: CA origin, X toward N
PROTEIN_BACKBONE_FRAME = FrameDefinition(
    origin="CA",
    z_ref="N",
    perp_ref="C",
    z_toward_origin=False,  # Z from CA toward N
)
