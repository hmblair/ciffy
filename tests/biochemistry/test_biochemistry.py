"""Tests for biochemistry constants, atom groups, and molecule enums."""

import numpy as np


class TestBiochemistryConstants:
    """Test biochemistry constants are correctly defined."""

    def test_nucleotide_consistency(self):
        from ciffy.biochemistry import Residue, Backbone

        # All nucleotides should have P atom (accessed via Residue)
        assert hasattr(Residue.A, 'P')
        assert hasattr(Residue.C, 'P')
        assert hasattr(Residue.G, 'P')
        assert hasattr(Residue.U, 'P')

        # Backbone atoms share values across residue types (unified backbone)
        assert Residue.A.P.value == Residue.C.P.value == Residue.G.P.value == Residue.U.P.value

        # Sidechain/base atoms should be unique within each residue
        backbone_values = set(b.value for b in Backbone)
        for nuc in [Residue.A, Residue.C, Residue.G, Residue.U]:
            sidechain_values = set()
            for member in nuc.atoms:
                if member.value not in backbone_values:
                    assert member.value not in sidechain_values, f"Duplicate sidechain value {member.value}"
                    sidechain_values.add(member.value)

    def test_backbone_contains_phosphate(self):
        from ciffy.biochemistry import Backbone, Phosphate

        # All phosphate atoms should be in backbone
        phosphate_values = set(p.value for p in Phosphate)
        backbone_values = set(b.value for b in Backbone)
        assert phosphate_values.issubset(backbone_values)

    def test_backbone_contains_rna_dna_protein(self):
        """Test that Backbone includes atoms from all molecule types."""
        from ciffy.biochemistry import Backbone, Residue
        backbone_values = set(b.value for b in Backbone)

        # RNA backbone (sugar-phosphate)
        assert Residue.A.P.value in backbone_values
        assert Residue.A.C4p.value in backbone_values

        # DNA backbone (sugar-phosphate)
        assert Residue.DA.P.value in backbone_values
        assert Residue.DA.C4p.value in backbone_values

        # Protein backbone (N-CA-C-O)
        assert Residue.ALA.N.value in backbone_values
        assert Residue.ALA.CA.value in backbone_values
        assert Residue.ALA.C.value in backbone_values
        assert Residue.ALA.O.value in backbone_values

        # Sidechains should NOT be in backbone
        assert Residue.ALA.CB.value not in backbone_values

    def test_nucleobase_excludes_backbone(self):
        """Test that Nucleobase only contains base atoms, not backbone."""
        from ciffy.biochemistry import Nucleobase, Residue
        nucleobase_values = set(n.value for n in Nucleobase)

        # Base atoms should be included
        assert Residue.A.N1.value in nucleobase_values
        assert Residue.A.C2.value in nucleobase_values

        # Backbone atoms should NOT be included
        assert Residue.A.P.value not in nucleobase_values
        assert Residue.A.C4p.value not in nucleobase_values

    def test_sidechain_excludes_backbone(self):
        """Test that Sidechain excludes backbone atoms."""
        from ciffy.biochemistry import Sidechain, Residue
        sidechain_values = set(s.value for s in Sidechain)

        # Sidechain atoms should be included
        assert Residue.ALA.CB.value in sidechain_values
        assert Residue.LYS.CE.value in sidechain_values

        # Backbone atoms should NOT be included
        assert Residue.ALA.N.value not in sidechain_values
        assert Residue.ALA.CA.value not in sidechain_values
        assert Residue.ALA.C.value not in sidechain_values
        assert Residue.ALA.O.value not in sidechain_values


class TestAtomGroup:
    """Test AtomGroup and atom group functionality."""

    def test_build_atom_group(self):
        """Test build_atom_group creates correct structure."""
        from ciffy.biochemistry import build_atom_group
        from ciffy.biochemistry import Residue

        # Build a simple atom group from purines
        sources = [("A", Residue.A), ("G", Residue.G)]
        TestGroup = build_atom_group("TestGroup", sources, {"N1", "N9"})

        # Should have N1 and N9 as attributes
        assert hasattr(TestGroup, "N1")
        assert hasattr(TestGroup, "N9")

        # Each should be an AtomGroup with A and G members
        assert hasattr(TestGroup.N1, "A")
        assert hasattr(TestGroup.N1, "G")
        assert int(TestGroup.N1.A) == int(Residue.A.N1)
        assert int(TestGroup.N1.G) == int(Residue.G.N1)

    def test_single_source_of_truth(self):
        """Test that hierarchical enums reference same values as Residue."""
        from ciffy.biochemistry import (
            Residue, PurineBase, PyrimidineBase, Sugar, PhosphateGroup
        )

        # Purine atoms
        assert PurineBase.N1.A.value == Residue.A.N1.value
        assert PurineBase.N1.G.value == Residue.G.N1.value
        assert PurineBase.N1.DA.value == Residue.DA.N1.value
        assert PurineBase.N1.DG.value == Residue.DG.N1.value

        assert PurineBase.N9.A.value == Residue.A.N9.value
        assert PurineBase.C8.G.value == Residue.G.C8.value

        # Pyrimidine atoms
        assert PyrimidineBase.N1.C.value == Residue.C.N1.value
        assert PyrimidineBase.N1.U.value == Residue.U.N1.value

        # Sugar atoms
        assert Sugar.C5p.A.value == Residue.A.C5p.value
        assert Sugar.C5p.G.value == Residue.G.C5p.value
        assert Sugar.C5p.C.value == Residue.C.C5p.value
        assert Sugar.C5p.U.value == Residue.U.C5p.value

        # Phosphate atoms
        assert PhosphateGroup.P.A.value == Residue.A.P.value
        assert PhosphateGroup.OP1.G.value == Residue.G.OP1.value

    def test_purine_hierarchy(self):
        """Test PurineBase = PurineImidazole | PurinePyrimidine."""
        from ciffy.biochemistry import PurineBase, PurineImidazole, PurinePyrimidine

        imidazole_values = set(PurineImidazole.index().tolist())
        pyrimidine_values = set(PurinePyrimidine.index().tolist())
        base_values = set(PurineBase.index().tolist())

        # Union should equal PurineBase
        assert imidazole_values | pyrimidine_values == base_values

        # Imidazole and pyrimidine share C4 and C5
        shared = imidazole_values & pyrimidine_values
        assert len(shared) > 0  # C4 and C5 are shared

    def test_hierarchical_enum_methods(self):
        """Test all AtomGroup methods on hierarchical groups."""
        from ciffy.biochemistry import PurineBase

        # index() returns numpy array of all atom values
        idx = PurineBase.index()
        assert isinstance(idx, np.ndarray)
        assert idx.dtype == np.int64

        # list() returns list of member names (atom position names)
        names = PurineBase.list()
        assert isinstance(names, list)
        assert "N1" in names
        assert "N9" in names

        # dict() returns leaf atoms only (empty for hierarchical groups)
        d = PurineBase.dict()
        assert isinstance(d, dict)
        assert len(d) == 0

        # Access nested AtomGroup for each atom position
        assert hasattr(PurineBase, "N1")
        assert set(PurineBase.N1.list()) == {"A", "G", "DA", "DG"}
        assert PurineBase.N1.dict() == {
            "A": int(PurineBase.N1.A),
            "G": int(PurineBase.N1.G),
            "DA": int(PurineBase.N1.DA),
            "DG": int(PurineBase.N1.DG),
        }

    def test_atom_groups_with_polymer(self):
        """Test using atom groups with Polymer.atom_type()."""
        from ciffy import template
        from ciffy.biochemistry import Sugar, PurineBase, PyrimidineBase

        polymer = template("acgu")
        total_atoms = polymer.size()

        # Select sugar atoms - should be present in all 4 residues
        sugar = polymer.atom_type(Sugar.index())
        assert sugar.size() > 0
        assert sugar.size() < total_atoms

        # Select purine base atoms - only A and G
        purine = polymer.atom_type(PurineBase.index())
        assert purine.size() > 0

        # Select pyrimidine base atoms - only C and U
        pyrimidine = polymer.atom_type(PyrimidineBase.index())
        assert pyrimidine.size() > 0

        # Purine + pyrimidine should not overlap (different chemical identity)
        purine_values = set(PurineBase.index().tolist())
        pyrimidine_values = set(PyrimidineBase.index().tolist())
        assert purine_values.isdisjoint(pyrimidine_values)

    def test_specific_atom_selection(self):
        """Test selecting specific atoms like all C5' or all N1."""
        from ciffy import template
        from ciffy.biochemistry import Sugar, PurineBase

        polymer = template("acgu")

        # Select all C5' atoms (one per residue)
        c5p = polymer.atom_type(Sugar.C5p.index())
        assert c5p.size() == 4  # One per residue

        # Select all purine N1 atoms (only A and G have purine N1)
        n1_purine = polymer.atom_type(PurineBase.N1.index())
        assert n1_purine.size() == 2  # A and G only

    def test_iteration_and_containment(self):
        """Test __iter__ and __contains__ on AtomGroup (hierarchical)."""
        from ciffy.biochemistry import PurineBase

        # In v2, __iter__ only yields leaf Atoms. PurineBase has nested AtomGroups
        # so iteration is empty. Use list() for member names instead.
        members = list(PurineBase)
        assert len(members) == 0  # No leaf atoms, only nested AtomGroups

        # Use list() to get member names (both Atoms and AtomGroups)
        names = PurineBase.list()
        assert len(names) > 0
        assert "N1" in names

        # String containment checks member names
        assert "N1" in PurineBase
        assert "INVALID" not in PurineBase

        # Nested AtomGroup access via index
        assert PurineBase.N1.index() is not None


class TestMoleculeEnum:
    """Test Molecule enum functionality."""

    def test_molecule_type_function(self):
        from ciffy.biochemistry import molecule_type, Molecule
        assert molecule_type(0) == Molecule.PROTEIN
        assert molecule_type(1) == Molecule.RNA
        assert molecule_type(2) == Molecule.DNA
