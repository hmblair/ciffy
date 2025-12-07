"""
Tests for module imports and backward compatibility.
"""

import pytest


class TestPublicAPI:
    """Test main public API imports."""

    def test_core_imports(self):
        from ciffy import Polymer, Scale, Molecule, Reduction, load, rmsd
        assert Polymer is not None
        assert Scale is not None
        assert Molecule is not None
        assert Reduction is not None
        assert load is not None
        assert rmsd is not None

    def test_convenience_aliases(self):
        from ciffy import RESIDUE, CHAIN, MOLECULE, PROTEIN, RNA, DNA
        from ciffy import Scale, Molecule
        assert RESIDUE == Scale.RESIDUE
        assert CHAIN == Scale.CHAIN
        assert MOLECULE == Scale.MOLECULE
        assert PROTEIN == Molecule.PROTEIN
        assert RNA == Molecule.RNA
        assert DNA == Molecule.DNA

    def test_version(self):
        import ciffy
        assert ciffy.__version__ == "0.4.0"


class TestBackwardCompatibility:
    """Test backward-compatible imports from old module locations."""

    def test_enum_imports(self):
        from ciffy.enum import (
            IndexEnum, PairEnum,
            Residue, Element,
            Adenosine, Cytosine, Guanosine, Uridine,
            RibonucleicAcid, RibonucleicAcidNoPrefix,
            RES_ABBREV,
            FRAMES, FRAME1, FRAME2, FRAME3, COARSE,
            Backbone, Nucleobase, Phosphate,
        )
        assert IndexEnum is not None
        assert Residue.A.value == 0
        assert Element.C.value == 6

    def test_reduction_imports(self):
        from ciffy.reduction import Reduction, REDUCTIONS, _Reduction
        assert Reduction.MEAN is not None
        assert REDUCTIONS is not None

    def test_rmsd_imports(self):
        from ciffy.rmsd import _kabsch_distance, _coordinate_covariance
        assert _kabsch_distance is not None
        assert _coordinate_covariance is not None


class TestNewModuleStructure:
    """Test imports from new module organization."""

    def test_utils_imports(self):
        from ciffy.utils import IndexEnum, PairEnum, all_equal, filter_by_mask
        assert IndexEnum is not None
        assert PairEnum is not None
        assert all_equal(1, 1, 1) is True
        assert all_equal(1, 2) is False

    def test_types_imports(self):
        from ciffy.types import Scale, Molecule
        assert Scale.ATOM.value == 0
        assert Scale.RESIDUE.value == 1
        assert Scale.CHAIN.value == 2
        assert Scale.MOLECULE.value == 3
        assert Molecule.RNA.value == 1

    def test_biochemistry_imports(self):
        from ciffy.biochemistry import (
            Element, Residue, RES_ABBREV,
            Adenosine, Cytosine, Guanosine, Uridine,
            RibonucleicAcid,
            FRAMES, Backbone, Nucleobase, Phosphate, COARSE,
        )
        assert Element.C.value == 6
        assert Residue.A.value == 0
        assert RES_ABBREV['ALA'] == 'A'
        assert Adenosine.P.value == 2

    def test_operations_imports(self):
        from ciffy.operations import Reduction, REDUCTIONS, kabsch_distance
        assert Reduction.MEAN is not None
        assert kabsch_distance is not None

    def test_io_imports(self):
        from ciffy.io import load, write_pdb
        assert load is not None
        assert write_pdb is not None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_all_equal(self):
        from ciffy.utils import all_equal
        assert all_equal(1, 1, 1) is True
        assert all_equal(1, 2, 1) is False
        assert all_equal(1) is True
        assert all_equal() is True

    def test_filter_by_mask(self):
        import torch
        from ciffy.utils import filter_by_mask

        items = ['a', 'b', 'c', 'd']
        mask = torch.tensor([True, False, True, False])
        result = filter_by_mask(items, mask)
        assert result == ['a', 'c']

    def test_index_enum(self):
        import torch
        from ciffy.utils import IndexEnum

        TestEnum = IndexEnum("TestEnum", {"A": 1, "B": 2, "C": 3})
        assert TestEnum.A.value == 1

        indices = TestEnum.index()
        assert torch.equal(indices, torch.tensor([1, 2, 3]))

        d = TestEnum.dict()
        assert d == {"A": 1, "B": 2, "C": 3}

        rd = TestEnum.revdict()
        assert rd == {1: "A", 2: "B", 3: "C"}


class TestBiochemistryConstants:
    """Test biochemistry constants are correctly defined."""

    def test_element_values(self):
        from ciffy.biochemistry import Element
        assert Element.H.value == 1
        assert Element.C.value == 6
        assert Element.N.value == 7
        assert Element.O.value == 8
        assert Element.P.value == 15
        assert Element.S.value == 16

    def test_nucleotide_consistency(self):
        from ciffy.biochemistry import Adenosine, Cytosine, Guanosine, Uridine

        # All nucleotides should have P atom
        assert hasattr(Adenosine, 'P')
        assert hasattr(Cytosine, 'P')
        assert hasattr(Guanosine, 'P')
        assert hasattr(Uridine, 'P')

        # Values should be unique across nucleotides
        all_values = set()
        for nuc in [Adenosine, Cytosine, Guanosine, Uridine]:
            for member in nuc:
                assert member.value not in all_values, f"Duplicate value {member.value}"
                all_values.add(member.value)

    def test_backbone_contains_phosphate(self):
        from ciffy.biochemistry import Backbone, Phosphate

        # All phosphate atoms should be in backbone
        phosphate_values = set(p.value for p in Phosphate)
        backbone_values = set(b.value for b in Backbone)
        assert phosphate_values.issubset(backbone_values)


class TestScaleEnum:
    """Test Scale enum functionality."""

    def test_scale_ordering(self):
        from ciffy.types import Scale
        assert Scale.ATOM.value < Scale.RESIDUE.value
        assert Scale.RESIDUE.value < Scale.CHAIN.value
        assert Scale.CHAIN.value < Scale.MOLECULE.value


class TestMoleculeEnum:
    """Test Molecule enum functionality."""

    def test_molecule_types(self):
        from ciffy.types import Molecule
        assert Molecule.PROTEIN.value == 0
        assert Molecule.RNA.value == 1
        assert Molecule.DNA.value == 2
        assert hasattr(Molecule, 'OTHER')
        assert hasattr(Molecule, 'MISSING')

    def test_molecule_type_function(self):
        from ciffy.types.molecule import molecule_type, Molecule
        assert molecule_type(0) == Molecule.PROTEIN
        assert molecule_type(1) == Molecule.RNA
        assert molecule_type(2) == Molecule.DNA


class TestReduction:
    """Test reduction operations."""

    def test_reduction_enum(self):
        from ciffy.operations import Reduction
        assert Reduction.NONE.value == 0
        assert Reduction.COLLATE.value == 1
        assert Reduction.MEAN.value == 2
        assert Reduction.SUM.value == 3
        assert Reduction.MIN.value == 4
        assert Reduction.MAX.value == 5

    def test_reductions_dict(self):
        from ciffy.operations import Reduction, REDUCTIONS
        assert Reduction.NONE in REDUCTIONS
        assert Reduction.MEAN in REDUCTIONS
        assert Reduction.SUM in REDUCTIONS

    def test_create_reduction_index(self):
        import torch
        from ciffy.operations.reduction import create_reduction_index

        result = create_reduction_index(3, torch.tensor([2, 1, 3]))
        expected = torch.tensor([0, 0, 1, 2, 2, 2])
        assert torch.equal(result, expected)
