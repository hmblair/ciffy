"""Tests for geometric deep learning module.

Tests cover:
- Irrep, ProductIrrep, Repr, ProductRepr (representation theory)
- DenseNetwork (MLP building block)
- Basic layer tests (where possible without sphericart)
"""
from __future__ import annotations

import pytest
import torch

from ciffy.nn.dense_network import DenseNetwork
from ciffy.nn.geometric.representations import Irrep, ProductIrrep, Repr, ProductRepr


# ============================================================================
# TEST: Irrep
# ============================================================================


class TestIrrep:
    """Tests for the Irrep class."""

    def test_construction(self):
        """Test basic Irrep construction."""
        irrep = Irrep(l=2, mult=3)
        assert irrep.l == 2
        assert irrep.mult == 3

    def test_default_multiplicity(self):
        """Test default multiplicity is 1."""
        irrep = Irrep(l=1)
        assert irrep.mult == 1

    def test_dimension(self):
        """Test dimension calculation (2l+1)."""
        assert Irrep(l=0).dim() == 1  # Scalar
        assert Irrep(l=1).dim() == 3  # Vector
        assert Irrep(l=2).dim() == 5  # Rank-2 tensor
        assert Irrep(l=3).dim() == 7
        assert Irrep(l=4).dim() == 9

    def test_mvals(self):
        """Test magnetic quantum numbers."""
        assert Irrep(l=0).mvals() == [0]
        assert Irrep(l=1).mvals() == [-1, 0, 1]
        assert Irrep(l=2).mvals() == [-2, -1, 0, 1, 2]

    def test_offset(self):
        """Test offset calculation."""
        assert Irrep(l=0).offset() == 0
        assert Irrep(l=1).offset() == 1
        assert Irrep(l=2).offset() == 4
        assert Irrep(l=3).offset() == 9

    def test_equality(self):
        """Test Irrep equality."""
        assert Irrep(l=2, mult=3) == Irrep(l=2, mult=3)
        assert Irrep(l=2, mult=3) != Irrep(l=2, mult=4)
        assert Irrep(l=2, mult=3) != Irrep(l=3, mult=3)
        assert Irrep(l=1) != "not an irrep"

    def test_hash(self):
        """Test Irrep hashing (for use in sets/dicts)."""
        irreps = {Irrep(l=1, mult=2), Irrep(l=1, mult=2), Irrep(l=2, mult=2)}
        assert len(irreps) == 2

    def test_invalid_degree_negative(self):
        """Test that negative degree raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            Irrep(l=-1)

    def test_invalid_degree_float(self):
        """Test that float degree raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            Irrep(l=1.5)

    def test_invalid_multiplicity_zero(self):
        """Test that zero multiplicity raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            Irrep(l=1, mult=0)

    def test_raising_operator_shape(self):
        """Test raising operator has correct shape."""
        for l in range(5):
            irrep = Irrep(l=l)
            J_plus = irrep.raising()
            assert J_plus.shape == (irrep.dim(), irrep.dim())

    def test_lowering_operator_shape(self):
        """Test lowering operator has correct shape."""
        for l in range(5):
            irrep = Irrep(l=l)
            J_minus = irrep.lowering()
            assert J_minus.shape == (irrep.dim(), irrep.dim())

    def test_raising_lowering_relation(self):
        """Test relationship between J_+ and J_-."""
        for l in range(1, 4):
            irrep = Irrep(l=l)
            J_plus = irrep.raising()
            J_minus = irrep.lowering()
            assert torch.allclose(J_minus, -J_plus.T, atol=1e-6)

    def test_toreal_unitary(self):
        """Test conversion matrix is unitary."""
        for l in range(4):
            irrep = Irrep(l=l)
            Q = irrep.toreal()
            identity = torch.eye(irrep.dim(), dtype=torch.complex128)
            assert torch.allclose(Q @ Q.conj().T, identity, atol=1e-10)

    def test_generators_shape(self):
        """Test generators have correct shape."""
        for l in range(4):
            irrep = Irrep(l=l)
            gens = irrep._generators()
            assert gens.shape == (3, irrep.dim(), irrep.dim())

    def test_generators_antisymmetric(self):
        """Test generators are antisymmetric (property of so(3))."""
        for l in range(1, 4):
            irrep = Irrep(l=l)
            gens = irrep._generators()
            for i in range(3):
                assert torch.allclose(gens[i], -gens[i].T, atol=1e-6)


# ============================================================================
# TEST: ProductIrrep
# ============================================================================


class TestProductIrrep:
    """Tests for the ProductIrrep class."""

    def test_construction(self):
        """Test ProductIrrep construction."""
        rep1 = Irrep(l=1)
        rep2 = Irrep(l=2)
        prod = ProductIrrep(rep1, rep2)
        assert prod.rep1 == rep1
        assert prod.rep2 == rep2

    def test_lmin_lmax(self):
        """Test min/max degree calculation."""
        prod = ProductIrrep(Irrep(l=1), Irrep(l=2))
        assert prod.lmin == 1
        assert prod.lmax == 3

        prod = ProductIrrep(Irrep(l=2), Irrep(l=2))
        assert prod.lmin == 0
        assert prod.lmax == 4

    def test_decomposition_irreps(self):
        """Test irreps in decomposition."""
        prod = ProductIrrep(Irrep(l=1), Irrep(l=1))
        degrees = [rep.l for rep in prod.reps]
        assert degrees == [0, 1, 2]

    def test_dimension(self):
        """Test total dimension equals product of input dims."""
        for l1 in range(4):
            for l2 in range(4):
                prod = ProductIrrep(Irrep(l=l1), Irrep(l=l2))
                expected_dim = (2*l1+1) * (2*l2+1)
                assert prod.dim() == expected_dim

    def test_nreps(self):
        """Test number of irreps in decomposition."""
        prod = ProductIrrep(Irrep(l=2), Irrep(l=3))
        assert prod.nreps() == 5


# ============================================================================
# TEST: Repr
# ============================================================================


class TestRepr:
    """Tests for the Repr class."""

    def test_construction(self):
        """Test basic Repr construction."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        assert repr.lvals == [0, 1, 2]
        assert repr.mult == 4

    def test_default_lvals(self):
        """Test default lvals is [1]."""
        repr = Repr()
        assert repr.lvals == [1]
        assert repr.mult == 1

    def test_dimension(self):
        """Test total dimension calculation."""
        assert Repr(lvals=[0, 1, 2]).dim() == 9
        assert Repr(lvals=[1]).dim() == 3
        assert Repr(lvals=[0]).dim() == 1

    def test_nreps(self):
        """Test number of irreps."""
        assert Repr(lvals=[0, 1, 2]).nreps() == 3
        assert Repr(lvals=[1]).nreps() == 1
        assert Repr(lvals=[0, 1, 1, 2]).nreps() == 4

    def test_lmax(self):
        """Test maximum degree."""
        assert Repr(lvals=[0, 1, 2]).lmax() == 2
        assert Repr(lvals=[1, 3]).lmax() == 3

    def test_cumdims(self):
        """Test cumulative dimensions."""
        repr = Repr(lvals=[0, 1, 2])
        assert repr.cumdims() == [0, 1, 4, 9]

    def test_indices(self):
        """Test indices mapping dims to irreps."""
        repr = Repr(lvals=[0, 1])
        assert repr.indices() == [0, 1, 1, 1]

    def test_iteration(self):
        """Test iterating over irreps."""
        repr = Repr(lvals=[0, 1, 2], mult=2)
        irreps = list(repr)
        assert len(irreps) == 3
        assert irreps[0].l == 0
        assert irreps[1].l == 1
        assert irreps[2].l == 2

    def test_equality(self):
        """Test Repr equality."""
        assert Repr(lvals=[0, 1], mult=2) == Repr(lvals=[0, 1], mult=2)
        assert Repr(lvals=[0, 1], mult=2) != Repr(lvals=[0, 1], mult=3)

    def test_hash(self):
        """Test Repr hashing."""
        reprs = {Repr(lvals=[0, 1]), Repr(lvals=[0, 1]), Repr(lvals=[0, 2])}
        assert len(reprs) == 2

    def test_verify_correct_shape(self):
        """Test verify with correct shape."""
        repr = Repr(lvals=[0, 1], mult=4)
        tensor = torch.randn(10, 4, 4)
        assert repr.verify(tensor) is True

    def test_verify_wrong_mult(self):
        """Test verify with wrong multiplicity."""
        repr = Repr(lvals=[0, 1], mult=4)
        tensor = torch.randn(10, 3, 4)
        assert repr.verify(tensor) is False

    def test_dot_product(self):
        """Test dot product between spherical tensors."""
        repr = Repr(lvals=[0, 1], mult=2)
        st1 = torch.randn(5, 2, 4)
        st2 = torch.randn(5, 2, 4)
        result = repr.dot(st1, st2)
        assert result.shape == (5, 2, 2)

    def test_find_scalar(self):
        """Test finding scalar representations."""
        repr = Repr(lvals=[0, 1, 0, 2])
        count, locs = repr.find_scalar()
        assert count == 2
        assert locs == [0, 4]

    def test_invalid_empty_lvals(self):
        """Test that empty lvals raises error."""
        with pytest.raises(ValueError, match="at least one degree"):
            Repr(lvals=[])

    def test_generators_shape(self):
        """Test that generators have correct shape."""
        repr = Repr(lvals=[0, 1, 2])
        gens = repr._generators()
        dim = repr.dim()
        assert gens.shape == (3, dim, dim)


# ============================================================================
# TEST: ProductRepr
# ============================================================================


class TestProductRepr:
    """Tests for the ProductRepr class."""

    def test_construction(self):
        """Test ProductRepr construction."""
        rep1 = Repr(lvals=[0, 1], mult=2)
        rep2 = Repr(lvals=[1, 2], mult=3)
        prod = ProductRepr(rep1, rep2)
        assert prod.rep1 == rep1
        assert prod.rep2 == rep2

    def test_nreps(self):
        """Test number of product irreps."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prod = ProductRepr(rep1, rep2)
        assert prod.nreps() == 4

    def test_dim(self):
        """Test total dimension."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prod = ProductRepr(rep1, rep2)
        assert prod.dim() == 12

    def test_equality(self):
        """Test ProductRepr equality."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prod1 = ProductRepr(rep1, rep2)
        prod2 = ProductRepr(rep1, rep2)
        assert prod1 == prod2

    def test_hash(self):
        """Test ProductRepr hashing."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prods = {ProductRepr(rep1, rep2), ProductRepr(rep1, rep2)}
        assert len(prods) == 1

    def test_lmax(self):
        """Test maximum degree in decomposition."""
        rep1 = Repr(lvals=[1], mult=1)
        rep2 = Repr(lvals=[2], mult=1)
        prod = ProductRepr(rep1, rep2)
        assert prod.lmax() == 3


# ============================================================================
# TEST: DenseNetwork
# ============================================================================


class TestDenseNetwork:
    """Tests for DenseNetwork class."""

    def test_forward_shape_no_hidden(self):
        """Test output shape with no hidden layers."""
        mlp = DenseNetwork(64, 10)
        x = torch.randn(32, 64)
        output = mlp(x)
        assert output.shape == (32, 10)

    def test_forward_shape_with_hidden(self):
        """Test output shape with hidden layers."""
        mlp = DenseNetwork(64, 10, hidden_sizes=[128, 64])
        x = torch.randn(32, 64)
        output = mlp(x)
        assert output.shape == (32, 10)

    def test_forward_no_nan(self):
        """Test output contains no NaN values."""
        mlp = DenseNetwork(64, 10, hidden_sizes=[128])
        x = torch.randn(32, 64)
        output = mlp(x)
        assert not torch.isnan(output).any()

    def test_batched_input(self):
        """Test with extra batch dimensions."""
        mlp = DenseNetwork(64, 10, hidden_sizes=[128])
        x = torch.randn(8, 16, 64)
        output = mlp(x)
        assert output.shape == (8, 16, 10)

    def test_bias_false(self):
        """Test network without bias."""
        mlp = DenseNetwork(64, 10, hidden_sizes=[128], bias=False)
        for layer in mlp.layers:
            assert layer.bias is None

    def test_dropout_eval(self):
        """Test dropout is disabled during eval."""
        mlp = DenseNetwork(64, 10, hidden_sizes=[128], dropout=0.5)
        mlp.eval()
        x = torch.randn(32, 64)
        output1 = mlp(x)
        output2 = mlp(x)
        assert torch.allclose(output1, output2)

    def test_backward_gradients(self):
        """Test gradients flow correctly."""
        mlp = DenseNetwork(64, 10, hidden_sizes=[128, 64])
        x = torch.randn(32, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_layer_count(self):
        """Test correct number of layers."""
        mlp = DenseNetwork(64, 10, hidden_sizes=[128, 64, 32])
        assert len(mlp.layers) == 4

    def test_invalid_in_size(self):
        """Test invalid in_size raises error."""
        with pytest.raises(ValueError, match="in_size must be a positive integer"):
            DenseNetwork(0, 10)

    def test_invalid_out_size(self):
        """Test invalid out_size raises error."""
        with pytest.raises(ValueError, match="out_size must be a positive integer"):
            DenseNetwork(64, 0)

    def test_invalid_dropout(self):
        """Test invalid dropout raises error."""
        with pytest.raises(ValueError, match="dropout must be in"):
            DenseNetwork(64, 10, dropout=1.5)


# ============================================================================
# TEST: Integration
# ============================================================================


class TestGeometricIntegration:
    """Integration tests for geometric module."""

    def test_repr_with_single_irrep(self):
        """Test Repr behaves correctly with single irrep."""
        repr = Repr(lvals=[2], mult=4)
        assert repr.dim() == 5
        assert repr.nreps() == 1

    def test_high_degree_irrep(self):
        """Test high degree irreps work correctly."""
        irrep = Irrep(l=10)
        assert irrep.dim() == 21
        gens = irrep._generators()
        assert gens.shape == (3, 21, 21)
        assert not torch.isnan(gens).any()

    def test_large_multiplicity(self):
        """Test large multiplicity works."""
        repr = Repr(lvals=[0, 1], mult=100)
        tensor = torch.randn(10, 100, 4)
        assert repr.verify(tensor)
