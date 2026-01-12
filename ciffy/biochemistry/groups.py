"""
Molecule-type specific atom groups for intuitive hierarchical access.

Provides RNA and Protein namespaces with structured atom access:

    from ciffy import RNA, Protein

    # Backbone atoms - single unified values (shared across all residue types)
    RNA.Backbone.P       # Atom(P, 1)
    RNA.Backbone.O3p     # Atom(O3', 10)
    RNA.Backbone.C1p     # Atom(C1', 13)

    # Base atoms - aggregated across residue types
    RNA.Base.glycosidic_n       # AtomGroup with {A: N9, G: N9, C: N1, U: N1}
    RNA.Base.glycosidic_n.A     # Atom(N9, 20) for adenine
    RNA.Base.glycosidic_n.index()  # array of all glycosidic N values

    # Protein backbone
    Protein.Backbone.CA  # Atom(CA, 15)

Key insight: Backbone atoms have unified values shared across residue types,
so RNA.Backbone.P returns a single Atom. Base atoms vary per residue type,
so they return AtomGroup instances.
"""

from __future__ import annotations

from .atom import Atom, AtomGroup, build_atom_group
from ._generated_residues import Residue


# =============================================================================
# BACKBONE VALUES (auto-generated from CCD)
# =============================================================================

# These are the unified backbone atom values, shared across all residue types.
# Auto-generated during codegen based on CCD atom order.
from ._generated_atoms import UNIFIED_BACKBONE_VALUES as _BACKBONE_VALUES


# =============================================================================
# RNA NAMESPACE
# =============================================================================

class _RNABackbone:
    """
    RNA backbone atoms (phosphate + sugar).

    All backbone atoms have unified values shared across all RNA residues.
    Access via RNA.Backbone.P, RNA.Backbone.C1p, etc.
    """

    # Phosphate group
    P = Atom("P", _BACKBONE_VALUES["P"])
    OP1 = Atom("OP1", _BACKBONE_VALUES["OP1"])
    OP2 = Atom("OP2", _BACKBONE_VALUES["OP2"])
    OP3 = Atom("OP3", _BACKBONE_VALUES["OP3"])

    # Sugar (ribose)
    O5p = Atom("O5'", _BACKBONE_VALUES["O5'"])
    C5p = Atom("C5'", _BACKBONE_VALUES["C5'"])
    C4p = Atom("C4'", _BACKBONE_VALUES["C4'"])
    O4p = Atom("O4'", _BACKBONE_VALUES["O4'"])
    C3p = Atom("C3'", _BACKBONE_VALUES["C3'"])
    O3p = Atom("O3'", _BACKBONE_VALUES["O3'"])
    C2p = Atom("C2'", _BACKBONE_VALUES["C2'"])
    O2p = Atom("O2'", _BACKBONE_VALUES["O2'"])  # RNA-specific (not in DNA)
    C1p = Atom("C1'", _BACKBONE_VALUES["C1'"])


# Residue groupings for building base atom groups
_RNA_RESIDUES = [("A", Residue.A), ("C", Residue.C), ("G", Residue.G), ("U", Residue.U)]
_RNA_PURINES = [("A", Residue.A), ("G", Residue.G)]
_RNA_PYRIMIDINES = [("C", Residue.C), ("U", Residue.U)]

# Build glycosidic nitrogen group: N9 for purines, N1 for pyrimidines
_purine_glycosidic = build_atom_group("_purine_n9", _RNA_PURINES, {"N9"})
_pyrimidine_glycosidic = build_atom_group("_pyrimidine_n1", _RNA_PYRIMIDINES, {"N1"})

# Merge into a single AtomGroup with all residue types
_glycosidic_members: dict[str, Atom] = {}
_purine_glycosidic._ensure_loaded()
_pyrimidine_glycosidic._ensure_loaded()
if _purine_glycosidic._members and "N9" in _purine_glycosidic._members:
    n9_group = _purine_glycosidic._members["N9"]
    if isinstance(n9_group, AtomGroup):
        n9_group._ensure_loaded()
        if n9_group._members:
            for k, v in n9_group._members.items():
                if isinstance(v, Atom):
                    _glycosidic_members[k] = v
if _pyrimidine_glycosidic._members and "N1" in _pyrimidine_glycosidic._members:
    n1_group = _pyrimidine_glycosidic._members["N1"]
    if isinstance(n1_group, AtomGroup):
        n1_group._ensure_loaded()
        if n1_group._members:
            for k, v in n1_group._members.items():
                if isinstance(v, Atom):
                    _glycosidic_members[k] = v

_glycosidic_n = AtomGroup("glycosidic_n", _glycosidic_members)


class _RNABase:
    """
    RNA base atoms aggregated across residue types.

    Base atoms vary between purines (A, G) and pyrimidines (C, U),
    so they are represented as AtomGroups.

    Example:
        RNA.Base.glycosidic_n.A     # Atom for adenine N9
        RNA.Base.glycosidic_n.index()  # All glycosidic N values
    """

    glycosidic_n = _glycosidic_n


class RNA:
    """
    RNA-specific atom groups for intuitive hierarchical access.

    Attributes:
        Backbone: Phosphate and sugar atoms (unified values)
        Base: Nucleobase atoms (aggregated across residue types)

    Example:
        >>> RNA.Backbone.P
        Atom(P, 1)
        >>> RNA.Backbone.C1p
        Atom(C1', 13)
        >>> RNA.Base.glycosidic_n.A
        Atom(N9, ...)
    """

    Backbone = _RNABackbone
    Base = _RNABase


# =============================================================================
# DNA NAMESPACE
# =============================================================================

class _DNABackbone:
    """
    DNA backbone atoms (phosphate + sugar).

    All backbone atoms have unified values shared across all DNA residues.
    Note: DNA lacks O2' (2'-hydroxyl) present in RNA.
    """

    # Phosphate group
    P = Atom("P", _BACKBONE_VALUES["P"])
    OP1 = Atom("OP1", _BACKBONE_VALUES["OP1"])
    OP2 = Atom("OP2", _BACKBONE_VALUES["OP2"])
    OP3 = Atom("OP3", _BACKBONE_VALUES["OP3"])

    # Sugar (deoxyribose)
    O5p = Atom("O5'", _BACKBONE_VALUES["O5'"])
    C5p = Atom("C5'", _BACKBONE_VALUES["C5'"])
    C4p = Atom("C4'", _BACKBONE_VALUES["C4'"])
    O4p = Atom("O4'", _BACKBONE_VALUES["O4'"])
    C3p = Atom("C3'", _BACKBONE_VALUES["C3'"])
    O3p = Atom("O3'", _BACKBONE_VALUES["O3'"])
    C2p = Atom("C2'", _BACKBONE_VALUES["C2'"])
    C1p = Atom("C1'", _BACKBONE_VALUES["C1'"])


# DNA residue groupings
_DNA_RESIDUES = [("DA", Residue.DA), ("DC", Residue.DC), ("DG", Residue.DG), ("DT", Residue.DT)]
_DNA_PURINES = [("DA", Residue.DA), ("DG", Residue.DG)]
_DNA_PYRIMIDINES = [("DC", Residue.DC), ("DT", Residue.DT)]

# Build glycosidic nitrogen group for DNA
_dna_purine_glycosidic = build_atom_group("_dna_purine_n9", _DNA_PURINES, {"N9"})
_dna_pyrimidine_glycosidic = build_atom_group("_dna_pyrimidine_n1", _DNA_PYRIMIDINES, {"N1"})

_dna_glycosidic_members: dict[str, Atom] = {}
_dna_purine_glycosidic._ensure_loaded()
_dna_pyrimidine_glycosidic._ensure_loaded()
if _dna_purine_glycosidic._members and "N9" in _dna_purine_glycosidic._members:
    dna_n9_group = _dna_purine_glycosidic._members["N9"]
    if isinstance(dna_n9_group, AtomGroup):
        dna_n9_group._ensure_loaded()
        if dna_n9_group._members:
            for k, v in dna_n9_group._members.items():
                if isinstance(v, Atom):
                    _dna_glycosidic_members[k] = v
if _dna_pyrimidine_glycosidic._members and "N1" in _dna_pyrimidine_glycosidic._members:
    dna_n1_group = _dna_pyrimidine_glycosidic._members["N1"]
    if isinstance(dna_n1_group, AtomGroup):
        dna_n1_group._ensure_loaded()
        if dna_n1_group._members:
            for k, v in dna_n1_group._members.items():
                if isinstance(v, Atom):
                    _dna_glycosidic_members[k] = v

_dna_glycosidic_n = AtomGroup("glycosidic_n", _dna_glycosidic_members)


class _DNABase:
    """DNA base atoms aggregated across residue types."""

    glycosidic_n = _dna_glycosidic_n


class DNA:
    """
    DNA-specific atom groups for intuitive hierarchical access.

    Attributes:
        Backbone: Phosphate and sugar atoms (unified values)
        Base: Nucleobase atoms (aggregated across residue types)
    """

    Backbone = _DNABackbone
    Base = _DNABase


# =============================================================================
# PROTEIN NAMESPACE
# =============================================================================

class _ProteinBackbone:
    """
    Protein backbone atoms (N, CA, C, O).

    All backbone atoms have unified values shared across all amino acids.
    """

    N = Atom("N", _BACKBONE_VALUES["N"])
    CA = Atom("CA", _BACKBONE_VALUES["CA"])
    C = Atom("C", _BACKBONE_VALUES["C"])
    O = Atom("O", _BACKBONE_VALUES["O"])


class Protein:
    """
    Protein-specific atom groups for intuitive hierarchical access.

    Attributes:
        Backbone: Peptide backbone atoms (N, CA, C, O)

    Example:
        >>> Protein.Backbone.CA
        Atom(CA, 15)
        >>> Protein.Backbone.N
        Atom(N, 14)
    """

    Backbone = _ProteinBackbone


__all__ = ["RNA", "DNA", "Protein"]
