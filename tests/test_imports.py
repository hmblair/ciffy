"""
Tests for utility functions, biochemistry constants, and backend operations.

Includes tests for both numpy and torch backends.
"""

import pytest
import numpy as np

from tests.utils import skip_if_no_torch


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

    def test_nucleotide_consistency(self):
        from ciffy.biochemistry import Residue

        # All nucleotides should have P atom (accessed via Residue)
        assert hasattr(Residue.A, 'P')
        assert hasattr(Residue.C, 'P')
        assert hasattr(Residue.G, 'P')
        assert hasattr(Residue.U, 'P')

        # Values should be unique across nucleotides
        all_values = set()
        for nuc in [Residue.A, Residue.C, Residue.G, Residue.U]:
            for member in nuc.atoms:
                assert member.value not in all_values, f"Duplicate value {member.value}"
                all_values.add(member.value)

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


class TestHierarchicalEnum:
    """Test HierarchicalEnum and atom group functionality."""

    def test_build_hierarchical_enum(self):
        """Test basic HierarchicalEnum creation."""
        import numpy as np
        from ciffy.utils import build_hierarchical_enum, IndexEnum

        # Create nested structure
        Inner = IndexEnum("Inner", {"X": 10, "Y": 20})
        Outer = build_hierarchical_enum("Outer", {"inner": Inner, "leaf": 30})

        # Test attribute access
        assert Outer.inner is Inner
        assert Outer.leaf == 30

        # Test index aggregates all values
        idx = Outer.index()
        assert set(idx.tolist()) == {10, 20, 30}

        # Test list returns member names
        assert set(Outer.list()) == {"inner", "leaf"}

    def test_build_atom_group(self):
        """Test build_atom_group creates correct structure."""
        import numpy as np
        from ciffy.utils import build_atom_group
        from ciffy.biochemistry import Residue

        # Build a simple atom group from purines
        sources = [("A", Residue.A), ("G", Residue.G)]
        TestGroup = build_atom_group("TestGroup", sources, {"N1", "N9"})

        # Should have N1 and N9 as attributes
        assert hasattr(TestGroup, "N1")
        assert hasattr(TestGroup, "N9")

        # Each should be an IndexEnum with A and G members
        assert hasattr(TestGroup.N1, "A")
        assert hasattr(TestGroup.N1, "G")
        assert TestGroup.N1.A.value == Residue.A.N1.value
        assert TestGroup.N1.G.value == Residue.G.N1.value

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
        """Test all IndexEnum-like methods on HierarchicalEnum."""
        import numpy as np
        from ciffy.biochemistry import PurineBase

        # index() returns numpy array
        idx = PurineBase.index()
        assert isinstance(idx, np.ndarray)
        assert idx.dtype == np.int64

        # list() returns list of atom names
        names = PurineBase.list()
        assert isinstance(names, list)
        assert "N1" in names
        assert "N9" in names

        # dict() returns name -> subenum mapping
        d = PurineBase.dict()
        assert isinstance(d, dict)
        assert "N1" in d

        # Nested IndexEnum has full functionality
        assert PurineBase.N1.list() == ["A", "G", "DA", "DG"]
        assert PurineBase.N1.dict() == {
            "A": PurineBase.N1.A.value,
            "G": PurineBase.N1.G.value,
            "DA": PurineBase.N1.DA.value,
            "DG": PurineBase.N1.DG.value,
        }

    def test_atom_groups_with_polymer(self):
        """Test using atom groups with Polymer.by_atom()."""
        from ciffy import from_sequence
        from ciffy.biochemistry import Sugar, PurineBase, PyrimidineBase

        polymer = from_sequence("acgu")
        total_atoms = polymer.coordinates.shape[0]

        # Select sugar atoms - should be present in all 4 residues
        sugar = polymer.by_atom(Sugar.index())
        assert sugar.coordinates.shape[0] > 0
        assert sugar.coordinates.shape[0] < total_atoms

        # Select purine base atoms - only A and G
        purine = polymer.by_atom(PurineBase.index())
        assert purine.coordinates.shape[0] > 0

        # Select pyrimidine base atoms - only C and U
        pyrimidine = polymer.by_atom(PyrimidineBase.index())
        assert pyrimidine.coordinates.shape[0] > 0

        # Purine + pyrimidine should not overlap (different chemical identity)
        purine_values = set(PurineBase.index().tolist())
        pyrimidine_values = set(PyrimidineBase.index().tolist())
        assert purine_values.isdisjoint(pyrimidine_values)

    def test_specific_atom_selection(self):
        """Test selecting specific atoms like all C5' or all N1."""
        from ciffy import from_sequence
        from ciffy.biochemistry import Sugar, PurineBase

        polymer = from_sequence("acgu")

        # Select all C5' atoms (one per residue)
        c5p = polymer.by_atom(Sugar.C5p.index())
        assert c5p.coordinates.shape[0] == 4  # One per residue

        # Select all purine N1 atoms (only A and G have purine N1)
        n1_purine = polymer.by_atom(PurineBase.N1.index())
        assert n1_purine.coordinates.shape[0] == 2  # A and G only

    def test_iteration_and_containment(self):
        """Test __iter__ and __contains__ on HierarchicalEnum."""
        from ciffy.biochemistry import PurineBase

        # Iteration yields subenums
        members = list(PurineBase)
        assert len(members) > 0

        # String containment
        assert "N1" in PurineBase
        assert "INVALID" not in PurineBase

        # Subenum containment
        assert PurineBase.N1 in PurineBase


class TestMoleculeEnum:
    """Test Molecule enum functionality."""

    def test_molecule_type_function(self):
        from ciffy.biochemistry import molecule_type, Molecule
        assert molecule_type(0) == Molecule.PROTEIN
        assert molecule_type(1) == Molecule.RNA
        assert molecule_type(2) == Molecule.DNA


class TestReduction:
    """Test reduction operations."""

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
        from ciffy.backend import scatter_sum

        src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = np.array([0, 1, 0])
        result = scatter_sum(src, index, dim_size=2)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        # index 0: [1,2] + [5,6] = [6,8]; index 1: [3,4]
        assert np.allclose(result, [[6.0, 8.0], [3.0, 4.0]])

    def test_scatter_sum_torch(self):
        import torch
        from ciffy.backend import scatter_sum

        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([0, 1, 0])
        result = scatter_sum(src, index, dim_size=2)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)
        expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_scatter_mean_numpy(self):
        from ciffy.backend import scatter_mean

        src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = np.array([0, 1, 0])
        result = scatter_mean(src, index, dim_size=2)

        assert isinstance(result, np.ndarray)
        # index 0: mean([1,2], [5,6]) = [3,4]; index 1: [3,4]
        assert np.allclose(result, [[3.0, 4.0], [3.0, 4.0]])

    def test_scatter_mean_torch(self):
        import torch
        from ciffy.backend import scatter_mean

        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([0, 1, 0])
        result = scatter_mean(src, index, dim_size=2)

        assert isinstance(result, torch.Tensor)
        expected = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
        assert torch.allclose(result, expected)

    def test_cdist_numpy(self):
        from ciffy.backend import cdist

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
        from ciffy.backend import cdist

        x1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        x2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        result = cdist(x1, x2)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)
        expected = torch.tensor([[0.0, 1.0], [1.0, 2**0.5]])
        assert torch.allclose(result, expected)

    def test_cat_numpy(self):
        from ciffy.backend import cat

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = cat([a, b])

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6]))

    def test_cat_torch(self):
        import torch
        from ciffy.backend import cat

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = cat([a, b])

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1, 2, 3, 4, 5, 6]))

    def test_repeat_interleave_numpy(self):
        from ciffy.backend import repeat_interleave

        arr = np.array([[1, 2], [3, 4], [5, 6]])
        repeats = np.array([2, 1, 3])
        result = repeat_interleave(arr, repeats)

        assert isinstance(result, np.ndarray)
        expected = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [5, 6], [5, 6]])
        assert np.array_equal(result, expected)

    def test_repeat_interleave_torch(self):
        import torch
        from ciffy.backend import repeat_interleave

        arr = torch.tensor([[1, 2], [3, 4], [5, 6]])
        repeats = torch.tensor([2, 1, 3])
        result = repeat_interleave(arr, repeats)

        assert isinstance(result, torch.Tensor)
        expected = torch.tensor([[1, 2], [1, 2], [3, 4], [5, 6], [5, 6], [5, 6]])
        assert torch.equal(result, expected)

    def test_multiply_numpy(self):
        from ciffy.backend import multiply

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = multiply(a, b)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [2.0, 6.0, 12.0])

    def test_multiply_torch(self):
        import torch
        from ciffy.backend import multiply

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 3.0, 4.0])
        result = multiply(a, b)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor([2.0, 6.0, 12.0]))


class TestRMSD:
    """Test RMSD (aligned Kabsch distance) computation.

    Uses parametrized any_cif fixture to run on all test PDBs.
    """

    def test_rotation_zero_rmsd(self, any_cif):
        """Rotating a polymer should give zero RMSD after alignment."""
        import copy
        from ciffy import load, Scale, rmsd

        polymer = load(any_cif)

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

        # RMSD should be ~0 (rotation is aligned out)
        dist = rmsd(polymer, rotated, Scale.MOLECULE)
        assert np.allclose(dist, 0, atol=1e-3)  # RMSD in Angstroms

    def test_translation_zero_rmsd(self, any_cif):
        """Translating a polymer should give zero RMSD after alignment."""
        import copy
        from ciffy import load, Scale, rmsd

        polymer = load(any_cif)

        # Create translated copy
        translated = copy.deepcopy(polymer)
        translated.coordinates = polymer.coordinates + np.array([100.0, -50.0, 25.0])

        # RMSD should be ~0 (translation is centered out)
        dist = rmsd(polymer, translated, Scale.MOLECULE)
        assert np.allclose(dist, 0, atol=1e-3)  # RMSD in Angstroms

    def test_flip_nonzero_rmsd(self, any_cif):
        """Flipping/reflecting coordinates should give nonzero RMSD."""
        import copy
        from ciffy import load, Scale, rmsd

        polymer = load(any_cif)

        # Create reflected copy (mirror across xy-plane)
        flipped = copy.deepcopy(polymer)
        flipped.coordinates = polymer.coordinates * np.array([1, 1, -1])

        # RMSD should be nonzero (reflection cannot be aligned)
        dist = rmsd(polymer, flipped, Scale.MOLECULE)
        assert dist > 0.1  # Should be significantly nonzero

    def test_numpy_backend(self, any_cif):
        """Test RMSD with NumPy backend explicitly."""
        from ciffy import load, rmsd

        polymer = load(any_cif, backend="numpy")
        assert polymer.backend == "numpy"

        # RMSD of structure with itself should be ~0
        # Tolerance scales with structure size due to float32 accumulation errors
        dist = rmsd(polymer, polymer)
        n_atoms = polymer.coordinates.shape[0]
        tolerance = max(1e-6, (n_atoms ** 0.5) * 1e-7 * 100)
        assert np.allclose(dist, 0, atol=tolerance)

    def test_default_scale(self, any_cif):
        """Test that rmsd defaults to MOLECULE scale."""
        from ciffy import load, Scale, rmsd

        polymer = load(any_cif)

        # These should be equivalent
        dist_default = rmsd(polymer, polymer)
        dist_explicit = rmsd(polymer, polymer, Scale.MOLECULE)

        assert np.allclose(dist_default, dist_explicit)

    def test_identical_structures(self, any_cif):
        """Test RMSD of identical structures is zero."""
        from ciffy import load, rmsd

        polymer = load(any_cif)
        dist = rmsd(polymer, polymer)

        # Tolerance scales with structure size due to float32 accumulation errors
        n_atoms = polymer.coordinates.shape[0]
        tolerance = max(1e-6, (n_atoms ** 0.5) * 1e-7 * 100)
        assert np.allclose(dist, 0, atol=tolerance)

    def test_single_chain(self, any_cif):
        """Test RMSD works on single-chain polymers."""
        import copy
        from ciffy import load, rmsd

        polymer = load(any_cif)
        # Select first chain only
        chain = polymer.by_index(0)

        # Add small perturbation
        perturbed = copy.deepcopy(chain)
        perturbed.coordinates = chain.coordinates + np.random.randn(*chain.coordinates.shape) * 0.1

        dist = rmsd(chain, perturbed)
        assert dist.shape == (1,)  # Single molecule
        assert dist[0] > 0  # Should be nonzero due to perturbation

    def test_chain_scale(self, any_cif):
        """Test RMSD at CHAIN scale."""
        import copy
        from ciffy import load, Scale, rmsd

        polymer = load(any_cif)

        # Perturb one chain more than others
        perturbed = copy.deepcopy(polymer)
        coords = perturbed.coordinates.copy()
        # Add larger noise to first chain's atoms
        n_first_chain = polymer._sizes[Scale.CHAIN][0]
        coords[:n_first_chain] += np.random.randn(n_first_chain, 3) * 1.0
        perturbed.coordinates = coords

        dist = rmsd(polymer, perturbed, Scale.CHAIN)
        assert dist.shape[0] == polymer.size(Scale.CHAIN)
        # First chain should have larger RMSD
        assert dist[0] > dist[1:].mean()


class TestRMSDEdgeCases:
    """Edge case tests for rmsd with small atom counts.

    The Kabsch algorithm uses SVD on a 3x3 covariance matrix. With fewer than
    3 atoms, the covariance matrix is rank-deficient (degenerate), which can
    cause numerical instability.
    """

    @staticmethod
    def _create_small_polymer(n_atoms: int, backend: str = "numpy", seed: int = 42):
        """Create a minimal polymer with n_atoms for testing.

        Args:
            n_atoms: Number of atoms (1, 2, or 3).
            backend: "numpy" or "torch".
            seed: Random seed for reproducibility.

        Returns:
            Polymer with random coordinates.
        """
        from ciffy import Polymer, Scale

        # Random coordinates with fixed seed for reproducibility
        rng = np.random.RandomState(seed)
        coords = rng.randn(n_atoms, 3).astype(np.float32) * 10.0

        # Minimal metadata
        atoms = np.zeros(n_atoms, dtype=np.int64)
        elements = np.ones(n_atoms, dtype=np.int64) * 8  # Oxygen
        sequence = np.zeros(1, dtype=np.int64)  # Single residue

        sizes = {
            Scale.RESIDUE: np.array([n_atoms], dtype=np.int64),
            Scale.CHAIN: np.array([n_atoms], dtype=np.int64),
            Scale.MOLECULE: np.array([n_atoms], dtype=np.int64),
        }

        polymer = Polymer(
            coordinates=coords,
            atoms=atoms,
            elements=elements,
            sequence=sequence,
            sizes=sizes,
            id="test",
            names=["A"],
            strands=["A"],
            lengths=np.array([1], dtype=np.int64),
        )

        if backend == "torch":
            return polymer.torch()
        return polymer

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_single_atom_rmsd(self, backend):
        """rmsd with single atom should return exactly 0.

        With 1 atom, the centered coordinates are all zero, making the
        covariance matrix all zeros. RMSD is trivially 0.
        """
        skip_if_no_torch(backend)

        from ciffy import rmsd

        p = self._create_small_polymer(1, backend)

        # Single atom: after centering, coordinates are at origin
        # Covariance is 0, variance is 0, so RMSD should be exactly 0
        dist = rmsd(p, p)

        assert dist.shape == (1,)
        if backend == "torch":
            assert dist.item() == 0.0
        else:
            assert dist[0] == 0.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_two_atom_rmsd(self, backend):
        """rmsd with two atoms should return ~0 for self-comparison.

        With 2 atoms, the covariance matrix has rank 1, but self-comparison
        should still give 0 since any 2 points can be perfectly aligned.
        """
        skip_if_no_torch(backend)

        from ciffy import rmsd

        p = self._create_small_polymer(2, backend)
        dist = rmsd(p, p)

        assert dist.shape == (1,)
        # Torch's SVD has lower precision on rank-deficient matrices,
        # causing errors up to ~0.01 for degenerate cases
        if backend == "torch":
            assert dist.item() < 0.02
        else:
            assert dist[0] < 1e-5

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_three_atom_rmsd(self, backend):
        """rmsd with three atoms should return ~0 for self-comparison.

        With 3 atoms, the covariance matrix can be rank-deficient depending
        on the point configuration.
        """
        skip_if_no_torch(backend)

        from ciffy import rmsd

        p = self._create_small_polymer(3, backend)
        dist = rmsd(p, p)

        assert dist.shape == (1,)
        # Torch's SVD has lower precision on rank-deficient matrices,
        # causing errors up to ~0.01 for degenerate cases
        if backend == "torch":
            assert dist.item() < 0.02
        else:
            assert dist[0] < 1e-5
