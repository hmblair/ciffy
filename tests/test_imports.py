"""
Tests for module imports and backward compatibility.

Includes tests for both numpy and torch backends.
"""

import pytest
import numpy as np


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
        assert hasattr(ciffy, "__version__")
        assert isinstance(ciffy.__version__, str)


class TestModuleStructure:
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
        from ciffy.io import load, write_cif
        assert load is not None
        assert write_cif is not None


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
        import numpy as np
        from ciffy.utils import IndexEnum

        TestEnum = IndexEnum("TestEnum", {"A": 1, "B": 2, "C": 3})
        assert TestEnum.A.value == 1

        indices = TestEnum.index()
        assert np.array_equal(indices, np.array([1, 2, 3]))

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
        assert hasattr(Molecule, 'HYBRID')
        assert hasattr(Molecule, 'PROTEIN_D')
        assert hasattr(Molecule, 'POLYSACCHARIDE')
        assert hasattr(Molecule, 'PNA')
        assert hasattr(Molecule, 'CYCLIC_PEPTIDE')
        assert hasattr(Molecule, 'LIGAND')
        assert hasattr(Molecule, 'ION')
        assert hasattr(Molecule, 'WATER')
        assert hasattr(Molecule, 'OTHER')
        assert hasattr(Molecule, 'UNKNOWN')

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


class TestBackendOperations:
    """Test backend operations work with both numpy and torch."""

    def test_backend_detection_numpy(self):
        from ciffy.backend import get_backend, Backend, is_numpy, is_torch

        arr = np.array([1, 2, 3])
        assert get_backend(arr) == Backend.NUMPY
        assert is_numpy(arr)
        assert not is_torch(arr)

    def test_backend_detection_torch(self):
        import torch
        from ciffy.backend import get_backend, Backend, is_numpy, is_torch

        arr = torch.tensor([1, 2, 3])
        assert get_backend(arr) == Backend.TORCH
        assert is_torch(arr)
        assert not is_numpy(arr)

    def test_backend_conversion_numpy_to_torch(self):
        import torch
        from ciffy.backend import to_torch

        np_arr = np.array([1.0, 2.0, 3.0])
        torch_arr = to_torch(np_arr)
        assert isinstance(torch_arr, torch.Tensor)
        assert np.allclose(np_arr, torch_arr.numpy())

    def test_backend_conversion_torch_to_numpy(self):
        import torch
        from ciffy.backend import to_numpy

        torch_arr = torch.tensor([1.0, 2.0, 3.0])
        np_arr = to_numpy(torch_arr)
        assert isinstance(np_arr, np.ndarray)
        assert np.allclose(np_arr, torch_arr.numpy())

    def test_scatter_sum_numpy(self):
        from ciffy.backend.ops import scatter_sum

        src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = np.array([0, 1, 0])
        result = scatter_sum(src, index, dim_size=2)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        # index 0: [1,2] + [5,6] = [6,8]; index 1: [3,4]
        assert np.allclose(result, [[6.0, 8.0], [3.0, 4.0]])

    def test_scatter_sum_torch(self):
        import torch
        from ciffy.backend.ops import scatter_sum

        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([0, 1, 0])
        result = scatter_sum(src, index, dim_size=2)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)
        expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_scatter_mean_numpy(self):
        from ciffy.backend.ops import scatter_mean

        src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = np.array([0, 1, 0])
        result = scatter_mean(src, index, dim_size=2)

        assert isinstance(result, np.ndarray)
        # index 0: mean([1,2], [5,6]) = [3,4]; index 1: [3,4]
        assert np.allclose(result, [[3.0, 4.0], [3.0, 4.0]])

    def test_scatter_mean_torch(self):
        import torch
        from ciffy.backend.ops import scatter_mean

        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([0, 1, 0])
        result = scatter_mean(src, index, dim_size=2)

        assert isinstance(result, torch.Tensor)
        expected = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_cdist_numpy(self):
        from ciffy.backend.ops import cdist

        x1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        x2 = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = cdist(x1, x2)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        # Distances: [0,0]->[0,0]=0, [0,0]->[0,1]=1, [1,0]->[0,0]=1, [1,0]->[0,1]=sqrt(2)
        expected = np.array([[0.0, 1.0], [1.0, np.sqrt(2)]])
        assert np.allclose(result, expected)

    def test_cdist_torch(self):
        import torch
        from ciffy.backend.ops import cdist

        x1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        x2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        result = cdist(x1, x2)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)
        expected = torch.tensor([[0.0, 1.0], [1.0, 2**0.5]])
        assert torch.allclose(result, expected)

    def test_cat_numpy(self):
        from ciffy.backend.ops import cat

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = cat([a, b])

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6]))

    def test_cat_torch(self):
        import torch
        from ciffy.backend.ops import cat

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = cat([a, b])

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1, 2, 3, 4, 5, 6]))

    def test_repeat_interleave_numpy(self):
        from ciffy.backend.ops import repeat_interleave

        arr = np.array([[1, 2], [3, 4], [5, 6]])
        repeats = np.array([2, 1, 3])
        result = repeat_interleave(arr, repeats)

        assert isinstance(result, np.ndarray)
        expected = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [5, 6], [5, 6]])
        assert np.array_equal(result, expected)

    def test_repeat_interleave_torch(self):
        import torch
        from ciffy.backend.ops import repeat_interleave

        arr = torch.tensor([[1, 2], [3, 4], [5, 6]])
        repeats = torch.tensor([2, 1, 3])
        result = repeat_interleave(arr, repeats)

        assert isinstance(result, torch.Tensor)
        expected = torch.tensor([[1, 2], [1, 2], [3, 4], [5, 6], [5, 6], [5, 6]])
        assert torch.equal(result, expected)

    def test_multiply_numpy(self):
        from ciffy.backend.ops import multiply

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = multiply(a, b)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [2.0, 6.0, 12.0])

    def test_multiply_torch(self):
        import torch
        from ciffy.backend.ops import multiply

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 3.0, 4.0])
        result = multiply(a, b)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor([2.0, 6.0, 12.0]))


class TestKabschDistance:
    """Test Kabsch distance (aligned RMSD) computation."""

    def test_rotation_zero_rmsd(self):
        """Rotating a polymer should give zero RMSD after alignment."""
        import copy
        from ciffy import load, Scale
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif")

        # Create a rotation matrix (90 degrees around z-axis)
        theta = np.pi / 2
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Create rotated copy
        rotated = copy.deepcopy(polymer)
        rotated.coordinates = polymer.coordinates @ rotation.T

        # Kabsch distance should be ~0 (rotation is aligned out)
        dist = kabsch_distance(polymer, rotated, Scale.MOLECULE)
        assert np.allclose(dist, 0, atol=1e-5)

    def test_translation_zero_rmsd(self):
        """Translating a polymer should give zero RMSD after alignment."""
        import copy
        from ciffy import load, Scale
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif")

        # Create translated copy
        translated = copy.deepcopy(polymer)
        translated.coordinates = polymer.coordinates + np.array([100.0, -50.0, 25.0])

        # Kabsch distance should be ~0 (translation is centered out)
        dist = kabsch_distance(polymer, translated, Scale.MOLECULE)
        assert np.allclose(dist, 0, atol=1e-5)

    def test_flip_nonzero_rmsd(self):
        """Flipping/reflecting coordinates should give nonzero RMSD."""
        import copy
        from ciffy import load, Scale
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif")

        # Create reflected copy (mirror across xy-plane)
        flipped = copy.deepcopy(polymer)
        flipped.coordinates = polymer.coordinates * np.array([1, 1, -1])

        # Kabsch distance should be nonzero (reflection cannot be aligned)
        dist = kabsch_distance(polymer, flipped, Scale.MOLECULE)
        assert dist > 0.1  # Should be significantly nonzero

    def test_numpy_backend(self):
        """Test RMSD with NumPy backend explicitly."""
        from ciffy import load
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif", backend="numpy")
        assert polymer.backend == "numpy"

        # RMSD of structure with itself should be ~0
        dist = kabsch_distance(polymer, polymer)
        assert np.allclose(dist, 0, atol=1e-10)

    def test_default_scale(self):
        """Test that rmsd defaults to MOLECULE scale."""
        from ciffy import load, Scale
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif")

        # These should be equivalent
        dist_default = kabsch_distance(polymer, polymer)
        dist_explicit = kabsch_distance(polymer, polymer, Scale.MOLECULE)

        assert np.allclose(dist_default, dist_explicit)

    def test_identical_structures(self):
        """Test RMSD of identical structures is zero."""
        from ciffy import load
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif")
        dist = kabsch_distance(polymer, polymer)

        assert np.allclose(dist, 0, atol=1e-10)

    def test_single_chain(self):
        """Test RMSD works on single-chain polymers."""
        import copy
        from ciffy import load
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif")
        # Select first chain only
        chain = polymer.select(0)

        # Add small perturbation
        perturbed = copy.deepcopy(chain)
        perturbed.coordinates = chain.coordinates + np.random.randn(*chain.coordinates.shape) * 0.1

        dist = kabsch_distance(chain, perturbed)
        assert dist.shape == (1,)  # Single molecule
        assert dist[0] > 0  # Should be nonzero due to perturbation

    def test_chain_scale(self):
        """Test RMSD at CHAIN scale."""
        import copy
        from ciffy import load, Scale
        from ciffy.operations.alignment import kabsch_distance

        polymer = load("tests/data/3SKW.cif")

        # Perturb one chain more than others
        perturbed = copy.deepcopy(polymer)
        coords = perturbed.coordinates.copy()
        # Add larger noise to first chain's atoms
        n_first_chain = polymer._sizes[Scale.CHAIN][0]
        coords[:n_first_chain] += np.random.randn(n_first_chain, 3) * 1.0
        perturbed.coordinates = coords

        dist = kabsch_distance(polymer, perturbed, Scale.CHAIN)
        assert dist.shape[0] == polymer.size(Scale.CHAIN)
        # First chain should have larger RMSD
        assert dist[0] > dist[1:].mean()
