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

if TYPE_CHECKING:
    from . import Residue


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


# Phosphodiester bond: O3' of residue N to P of residue N+1
# O3' frame: origin at O3', Z along C3'->O3', X toward C4'
# P frame: origin at P, Z along O5'->P, X toward OP1
NUCLEIC_ACID_LINK = LinkingDefinition(
    prev_atom="O3p",
    next_atom="P",
    bond_length=1.60,
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
PEPTIDE_LINK = LinkingDefinition(
    prev_atom="C",
    next_atom="N",
    bond_length=1.33,
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
