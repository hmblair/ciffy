"""Tests for geometric deep learning module.

Tests cover:
- Irrep, ProductIrrep, Repr, ProductRepr (representation theory)
- Wigner D-matrices and rotation equivariance
- SphericalHarmonic, RadialBasisFunctions (primitives)
- RepNorm, EquivariantLinear, EquivariantLayerNorm (layers)
- build_knn_graph (utilities)
- EquivariantTransformer (integration)
"""
from __future__ import annotations

import pytest
import torch
import math

from ciffy.nn.geometric.representations import Irrep, ProductIrrep, Repr, ProductRepr

# Check if sphericart is available for tests that require it
try:
    import sphericart.torch
    SPHERICART_AVAILABLE = True
except ImportError:
    SPHERICART_AVAILABLE = False

requires_sphericart = pytest.mark.skipif(
    not SPHERICART_AVAILABLE,
    reason="sphericart not installed"
)


def random_rotation_matrix(device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random rotation matrix and its axis-angle representation.

    Returns:
        Tuple of (rotation_matrix, axis, angle)
    """
    # Random axis (unit vector)
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()

    # Random angle
    angle = torch.rand(1, device=device).squeeze() * 2 * math.pi

    # Rodrigues formula for rotation matrix
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], device=device, dtype=axis.dtype)

    R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    return R, axis, angle


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

    def test_generators_commutation_relations(self):
        """Test generators satisfy so(3) Lie algebra: [Li, Lj] = ε_ijk Lk."""
        for l in range(1, 5):
            irrep = Irrep(l=l)
            gens = irrep._generators()
            Lx, Ly, Lz = gens[0], gens[1], gens[2]

            # [Lx, Ly] = Lz
            comm_xy = Lx @ Ly - Ly @ Lx
            assert torch.allclose(comm_xy, Lz, atol=1e-5), f"[Lx, Ly] != Lz for l={l}"

            # [Ly, Lz] = Lx
            comm_yz = Ly @ Lz - Lz @ Ly
            assert torch.allclose(comm_yz, Lx, atol=1e-5), f"[Ly, Lz] != Lx for l={l}"

            # [Lz, Lx] = Ly
            comm_zx = Lz @ Lx - Lx @ Lz
            assert torch.allclose(comm_zx, Ly, atol=1e-5), f"[Lz, Lx] != Ly for l={l}"


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
# TEST: Wigner D-matrices (Repr.rot)
# ============================================================================


class TestWignerDMatrix:
    """Tests for Wigner D-matrix computation via Repr.rot()."""

    def test_rot_output_shape(self):
        """Test rotation matrix has correct shape."""
        repr = Repr(lvals=[0, 1, 2])
        axis = torch.tensor([0., 0., 1.])
        angle = torch.tensor(0.5)
        D = repr.rot(axis, angle)
        assert D.shape == (repr.dim(), repr.dim())

    def test_rot_identity_for_zero_angle(self):
        """Test that zero rotation gives identity."""
        repr = Repr(lvals=[0, 1, 2])
        axis = torch.tensor([0., 0., 1.])
        angle = torch.tensor(0.0)
        D = repr.rot(axis, angle)
        assert torch.allclose(D, torch.eye(repr.dim()), atol=1e-6)

    def test_rot_orthogonality(self):
        """Test rotation matrix is orthogonal: D @ D^T = I."""
        repr = Repr(lvals=[0, 1, 2])
        for _ in range(5):
            _, axis, angle = random_rotation_matrix()
            D = repr.rot(axis, angle)
            identity = torch.eye(repr.dim())
            assert torch.allclose(D @ D.T, identity, atol=1e-5)

    def test_rot_determinant_one(self):
        """Test rotation matrix has determinant 1."""
        repr = Repr(lvals=[0, 1, 2])
        for _ in range(5):
            _, axis, angle = random_rotation_matrix()
            D = repr.rot(axis, angle)
            det = torch.linalg.det(D)
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)

    def test_rot_scalars_invariant(self):
        """Test scalar components are invariant under rotation."""
        repr = Repr(lvals=[0, 1, 0, 2])  # Two scalar irreps
        _, axis, angle = random_rotation_matrix()
        D = repr.rot(axis, angle)

        # Scalar positions should have D[i,i] = 1, D[i,j] = 0 for j != i
        scalar_count, scalar_locs = repr.find_scalar()
        for loc in scalar_locs:
            assert torch.allclose(D[loc, loc], torch.tensor(1.0), atol=1e-6)
            # Check off-diagonal in that row/col are zero
            for j in range(repr.dim()):
                if j != loc:
                    assert torch.allclose(D[loc, j], torch.tensor(0.0), atol=1e-6)
                    assert torch.allclose(D[j, loc], torch.tensor(0.0), atol=1e-6)

    def test_rot_vectors_match_3d_rotation(self):
        """Test l=1 rotation matches standard 3D rotation matrix."""
        repr = Repr(lvals=[1])  # Just vectors

        for _ in range(5):
            R, axis, angle = random_rotation_matrix()
            D = repr.rot(axis, angle)

            # Apply both to a random vector
            v = torch.randn(3)
            v_rotated_R = R @ v
            v_rotated_D = D @ v

            assert torch.allclose(v_rotated_R, v_rotated_D, atol=1e-5)

    def test_rot_composition(self):
        """Test D(R1) @ D(R2) = D(R1 @ R2) (approximately)."""
        repr = Repr(lvals=[0, 1, 2])

        # Two random rotations
        R1, axis1, angle1 = random_rotation_matrix()
        R2, axis2, angle2 = random_rotation_matrix()

        D1 = repr.rot(axis1, angle1)
        D2 = repr.rot(axis2, angle2)

        # Compose in D space
        D_composed = D1 @ D2

        # Compose in R space and get D
        R_composed = R1 @ R2

        # Extract axis-angle from composed rotation
        # Using Rodrigues formula inverse
        trace = R_composed.trace()
        angle_composed = torch.acos((trace - 1) / 2).clamp(-math.pi, math.pi)

        if angle_composed.abs() > 1e-6:
            axis_composed = torch.stack([
                R_composed[2, 1] - R_composed[1, 2],
                R_composed[0, 2] - R_composed[2, 0],
                R_composed[1, 0] - R_composed[0, 1],
            ]) / (2 * torch.sin(angle_composed))
            axis_composed = axis_composed / axis_composed.norm()

            D_from_composed = repr.rot(axis_composed, angle_composed)

            # Should be equal (up to sign ambiguity for 180 degree rotations)
            assert torch.allclose(D_composed, D_from_composed, atol=1e-4) or \
                   torch.allclose(D_composed, -D_from_composed, atol=1e-4)

    def test_rot_batched(self):
        """Test rotation with batched inputs."""
        repr = Repr(lvals=[0, 1])
        batch_size = 10

        axes = torch.randn(batch_size, 3)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        angles = torch.rand(batch_size) * 2 * math.pi

        D = repr.rot(axes, angles)
        assert D.shape == (batch_size, repr.dim(), repr.dim())

        # Each should be orthogonal
        for i in range(batch_size):
            assert torch.allclose(D[i] @ D[i].T, torch.eye(repr.dim()), atol=1e-5)


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

    @pytest.mark.parametrize("rank", [0, 1, 2, None])
    def test_num_basis(self, rank):
        """Test num_basis computation for different ranks."""
        rep = Repr(lvals=[0, 1, 2], mult=1)
        prod = ProductRepr(rep, rep)
        num_basis = prod.num_basis(rank)

        # num_basis should increase with rank
        assert num_basis > 0
        if rank is not None and rank > 0:
            assert num_basis > prod.num_basis(0)

    @pytest.mark.parametrize("rank", [0, 1, None])
    def test_cg_tensor(self, rank):
        """Test cg_tensor returns correct shape."""
        rep = Repr(lvals=[0, 1], mult=1)
        prod = ProductRepr(rep, rep)

        tensor, num_basis = prod.cg_tensor(rank)

        # Check num_basis matches
        assert num_basis == prod.num_basis(rank)

        # Check tensor shape: (sh_dim, num_basis * dim1 * dim2)
        sh_dim = (prod.lmax() + 1) ** 2
        expected_cols = num_basis * rep.dim() * rep.dim()
        assert tensor.shape == (sh_dim, expected_cols)

    def test_cg_tensor_rank_ordering(self):
        """Test that higher rank gives more basis elements."""
        rep = Repr(lvals=[0, 1, 2], mult=1)
        prod = ProductRepr(rep, rep)

        nb_0 = prod.num_basis(0)
        nb_1 = prod.num_basis(1)
        nb_2 = prod.num_basis(2)
        nb_full = prod.num_basis(None)

        assert nb_0 < nb_1 < nb_2 <= nb_full


# ============================================================================
# TEST: EquivariantBasis
# ============================================================================


@requires_sphericart
class TestEquivariantBasis:
    """Tests for EquivariantBasis layer."""

    @pytest.mark.parametrize("rank", [0, 1, 2, None])
    def test_output_shape(self, rank):
        """Test basis output shape for different ranks."""
        from ciffy.nn.geometric import EquivariantBasis

        rep = Repr(lvals=[0, 1], mult=1)
        prepr = ProductRepr(rep, rep)
        basis = EquivariantBasis(prepr, rank=rank)

        N = 50
        x = torch.randn(N, 3)
        output = basis(x)

        expected_num_basis = prepr.num_basis(rank)
        assert output.shape == (N, expected_num_basis, rep.dim(), rep.dim())

    @pytest.mark.parametrize("rank", [0, 1, None])
    def test_no_nan_output(self, rank):
        """Test basis produces no NaN values."""
        from ciffy.nn.geometric import EquivariantBasis

        rep = Repr(lvals=[0, 1, 2], mult=1)
        prepr = ProductRepr(rep, rep)
        basis = EquivariantBasis(prepr, rank=rank)

        x = torch.randn(100, 3)
        output = basis(x)

        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("rank", [0, 1, None])
    def test_equivariance(self, rank):
        """Test basis satisfies equivariance: B(Rx) = D_out @ B(x) @ D_in.T"""
        from ciffy.nn.geometric import EquivariantBasis

        rep = Repr(lvals=[0, 1], mult=1)
        prepr = ProductRepr(rep, rep)
        basis = EquivariantBasis(prepr, rank=rank)

        x = torch.randn(30, 3)
        R, axis, angle = random_rotation_matrix()
        D = rep.rot(axis, angle)

        B_x = basis(x)
        B_Rx = basis(x @ R.T)

        # B(Rx) should equal D @ B(x) @ D.T
        B_x_rotated = torch.einsum('ij,...jk,lk->...il', D, B_x, D)

        assert torch.allclose(B_Rx, B_x_rotated, atol=1e-5)

    def test_rank_increases_basis_count(self):
        """Test higher rank gives more basis elements."""
        from ciffy.nn.geometric import EquivariantBasis

        rep = Repr(lvals=[0, 1, 2], mult=1)
        prepr = ProductRepr(rep, rep)

        basis_0 = EquivariantBasis(prepr, rank=0)
        basis_1 = EquivariantBasis(prepr, rank=1)
        basis_full = EquivariantBasis(prepr, rank=None)

        assert basis_0.num_basis < basis_1.num_basis < basis_full.num_basis


# ============================================================================
# TEST: RepNorm
# ============================================================================


class TestRepNorm:
    """Tests for RepNorm layer."""

    @pytest.fixture
    def repr_and_norm(self):
        """Create a representation and RepNorm for testing."""
        from ciffy.nn.geometric import RepNorm
        repr = Repr(lvals=[0, 1, 2])
        norm = RepNorm(repr)
        return repr, norm

    def test_output_shape(self, repr_and_norm):
        """Test output has correct shape."""
        repr, norm = repr_and_norm
        st = torch.randn(32, repr.dim())
        result = norm(st)
        assert result.shape == (32, repr.nreps())

    def test_output_shape_batched(self, repr_and_norm):
        """Test with extra batch dimensions."""
        repr, norm = repr_and_norm
        st = torch.randn(8, 16, repr.dim())
        result = norm(st)
        assert result.shape == (8, 16, repr.nreps())

    def test_norms_positive(self, repr_and_norm):
        """Test all norms are non-negative."""
        repr, norm = repr_and_norm
        st = torch.randn(32, repr.dim())
        result = norm(st)
        assert (result >= 0).all()

    def test_zero_input_gives_zero_norms(self, repr_and_norm):
        """Test zero input produces zero norms."""
        repr, norm = repr_and_norm
        st = torch.zeros(32, repr.dim())
        result = norm(st)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_scaling_behavior(self, repr_and_norm):
        """Test norm scales linearly with input magnitude."""
        repr, norm = repr_and_norm
        st = torch.randn(32, repr.dim())
        scale = 3.5

        norm1 = norm(st)
        norm2 = norm(st * scale)

        assert torch.allclose(norm2, norm1 * scale, atol=1e-5)

    def test_rotation_invariance(self, repr_and_norm):
        """Test norms are invariant under rotation."""
        repr, norm = repr_and_norm
        st = torch.randn(10, repr.dim())

        # Get rotation matrix
        _, axis, angle = random_rotation_matrix()
        D = repr.rot(axis, angle)

        # Rotate the spherical tensor
        st_rotated = st @ D.T

        # Norms should be the same
        norms_original = norm(st)
        norms_rotated = norm(st_rotated)

        assert torch.allclose(norms_original, norms_rotated, atol=1e-5)


# ============================================================================
# TEST: RadialBasisFunctions
# ============================================================================


class TestRadialBasisFunctions:
    """Tests for RadialBasisFunctions."""

    def test_gaussian_output_shape(self):
        """Test Gaussian RBF output shape."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")
        distances = torch.rand(100) * 10
        features = rbf(distances)
        assert features.shape == (100, 16)

    def test_bessel_output_shape(self):
        """Test Bessel RBF output shape."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")
        distances = torch.rand(100) * 10
        features = rbf(distances)
        assert features.shape == (100, 16)

    def test_polynomial_output_shape(self):
        """Test polynomial RBF output shape."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")
        distances = torch.rand(100) * 10
        features = rbf(distances)
        assert features.shape == (100, 16)

    def test_gaussian_no_nan(self):
        """Test Gaussian RBF produces no NaN values."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")
        distances = torch.rand(100) * 10
        features = rbf(distances)
        assert not torch.isnan(features).any()

    def test_bessel_no_nan(self):
        """Test Bessel RBF produces no NaN values."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")
        distances = torch.rand(100) * 10
        features = rbf(distances)
        assert not torch.isnan(features).any()

    def test_polynomial_no_nan(self):
        """Test polynomial RBF produces no NaN values."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")
        distances = torch.rand(100) * 10
        features = rbf(distances)
        assert not torch.isnan(features).any()

    def test_gaussian_at_zero(self):
        """Test Gaussian RBF at zero distance."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")
        distances = torch.zeros(10)
        features = rbf(distances)
        assert not torch.isnan(features).any()

    def test_bessel_smooth_cutoff(self):
        """Test Bessel RBF goes to zero at r_max."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")
        distances = torch.tensor([10.0, 11.0, 15.0])
        features = rbf(distances)
        # At and beyond r_max, envelope should be zero
        assert torch.allclose(features, torch.zeros_like(features), atol=1e-6)

    def test_polynomial_smooth_cutoff(self):
        """Test polynomial RBF goes to zero at r_max."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")
        distances = torch.tensor([10.0, 11.0, 15.0])
        features = rbf(distances)
        # At r_max, (1 - r/r_max) = 0
        assert torch.allclose(features, torch.zeros_like(features), atol=1e-6)

    def test_batched_input(self):
        """Test RBF with batched input."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")
        distances = torch.rand(8, 32) * 10
        features = rbf(distances)
        assert features.shape == (8, 32, 16)

    def test_invalid_rbf_type(self):
        """Test invalid RBF type raises error."""
        from ciffy.nn.geometric import RadialBasisFunctions
        with pytest.raises(ValueError, match="rbf_type must be"):
            RadialBasisFunctions(16, rbf_type="invalid")

    def test_learnable_gaussian_parameters(self):
        """Test Gaussian RBF has learnable parameters."""
        from ciffy.nn.geometric import RadialBasisFunctions
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")
        assert hasattr(rbf, 'mu')
        assert hasattr(rbf, 'sigma')
        assert rbf.mu.requires_grad
        assert rbf.sigma.requires_grad


# ============================================================================
# TEST: SphericalHarmonic
# ============================================================================


@requires_sphericart
class TestSphericalHarmonic:
    """Tests for SphericalHarmonic layer."""

    def test_output_shape(self):
        """Test output has correct shape."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=3)
        coords = torch.randn(100, 3)
        features = sh(coords)
        expected_dim = (3 + 1) ** 2  # (lmax+1)^2 = 16
        assert features.shape == (100, expected_dim)

    def test_output_shape_batched(self):
        """Test with extra batch dimensions."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=2)
        coords = torch.randn(8, 100, 3)
        features = sh(coords)
        expected_dim = (2 + 1) ** 2  # 9
        assert features.shape == (8, 100, expected_dim)

    def test_no_nan_for_nonzero_input(self):
        """Test no NaN values for non-zero coordinates."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=3)
        coords = torch.randn(100, 3) + 0.1  # Avoid exact zero
        features = sh(coords)
        assert not torch.isnan(features).any()

    def test_zero_vector_handling(self):
        """Test zero vectors produce no NaN and only l=0 is non-zero."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=2)
        coords = torch.zeros(10, 3)
        features = sh(coords)
        assert not torch.isnan(features).any()
        # l=0 (Y_0^0 = 1/sqrt(4*pi)) is constant, l>0 should be zero
        assert torch.allclose(features[:, 1:], torch.zeros(10, 8), atol=1e-6)

    def test_l0_is_constant(self):
        """Test l=0 spherical harmonic is constant."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=2)
        coords = torch.randn(100, 3)
        coords = coords / coords.norm(dim=-1, keepdim=True)  # Unit vectors
        features = sh(coords)

        # l=0 component (first element) should be constant for all directions
        l0_values = features[:, 0]
        assert torch.allclose(l0_values, l0_values.mean().expand_as(l0_values), atol=1e-5)

    def test_pairwise_shape(self):
        """Test pairwise spherical harmonics shape."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=2)
        coords = torch.randn(50, 3)
        features = sh.pairwise(coords)
        expected_dim = (2 + 1) ** 2
        assert features.shape == (50, 50, expected_dim)

    def test_pairwise_diagonal_l_gt_0_zero(self):
        """Test pairwise diagonal l>0 components are zero (self-interactions)."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=2)
        coords = torch.randn(20, 3)
        features = sh.pairwise(coords)

        # Diagonal l>0 should be zeros (zero displacement direction undefined)
        # l=0 is constant regardless of direction
        for i in range(20):
            assert torch.allclose(features[i, i, 1:], torch.zeros(8), atol=1e-6)

    def test_rotation_equivariance_l1(self):
        """Test l=1 spherical harmonics rotate like vectors."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=1)
        repr = Repr(lvals=[0, 1])

        # Random points
        coords = torch.randn(50, 3)

        # Get rotation
        R, axis, angle = random_rotation_matrix()
        D = repr.rot(axis, angle)

        # Compute SH before and after rotation
        sh_original = sh(coords)
        sh_rotated = sh(coords @ R.T)

        # SH should transform as: Y(Rx) = D @ Y(x)
        sh_transformed = sh_original @ D.T

        assert torch.allclose(sh_rotated, sh_transformed, atol=1e-4)


# ============================================================================
# TEST: build_knn_graph
# ============================================================================


class TestBuildKnnGraph:
    """Tests for build_knn_graph utility."""

    def test_output_shape(self):
        """Test output has correct shape."""
        from ciffy.nn.geometric import build_knn_graph
        coords = torch.randn(100, 3)
        neighbor_idx = build_knn_graph(coords, k=16)
        assert neighbor_idx.shape == (100, 16)

    def test_output_dtype(self):
        """Test output has integer dtype."""
        from ciffy.nn.geometric import build_knn_graph
        coords = torch.randn(100, 3)
        neighbor_idx = build_knn_graph(coords, k=16)
        assert neighbor_idx.dtype == torch.long

    def test_excludes_self(self):
        """Test self is not included in neighbors."""
        from ciffy.nn.geometric import build_knn_graph
        coords = torch.randn(50, 3)
        neighbor_idx = build_knn_graph(coords, k=10)

        for i in range(50):
            assert i not in neighbor_idx[i]

    def test_neighbors_are_valid_indices(self):
        """Test all neighbor indices are valid."""
        from ciffy.nn.geometric import build_knn_graph
        N = 100
        coords = torch.randn(N, 3)
        neighbor_idx = build_knn_graph(coords, k=16)

        assert (neighbor_idx >= 0).all()
        assert (neighbor_idx < N).all()

    def test_neighbors_are_actually_nearest(self):
        """Test returned neighbors are actually k-nearest."""
        from ciffy.nn.geometric import build_knn_graph
        coords = torch.randn(50, 3)
        k = 10
        neighbor_idx = build_knn_graph(coords, k=k)

        # For each node, verify neighbors are among k-nearest
        dists = torch.cdist(coords, coords)

        for i in range(50):
            # Get distances to neighbors
            neighbor_dists = dists[i, neighbor_idx[i]]

            # Get all distances (excluding self)
            all_dists = dists[i].clone()
            all_dists[i] = float('inf')  # Exclude self

            # k-th smallest distance
            kth_dist = all_dists.topk(k, largest=False).values[-1]

            # All neighbor distances should be <= k-th smallest
            assert (neighbor_dists <= kth_dist + 1e-6).all()

    def test_small_n_edge_case(self):
        """Test when N <= k."""
        from ciffy.nn.geometric import build_knn_graph
        coords = torch.randn(5, 3)
        neighbor_idx = build_knn_graph(coords, k=10)
        assert neighbor_idx.shape == (5, 10)

    def test_chunked_computation(self):
        """Test large N uses chunked computation correctly."""
        from ciffy.nn.geometric import build_knn_graph
        # Use N larger than default chunk_size
        coords = torch.randn(3000, 3)
        neighbor_idx = build_knn_graph(coords, k=16, chunk_size=1024)
        assert neighbor_idx.shape == (3000, 16)

        # Verify neighbors are valid
        assert (neighbor_idx >= 0).all()
        assert (neighbor_idx < 3000).all()


# ============================================================================
# TEST: EquivariantLinear
# ============================================================================


class TestEquivariantLinear:
    """Tests for EquivariantLinear layer."""

    def test_output_shape(self):
        """Test output has correct shape."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr_in = Repr(lvals=[0, 1, 2], mult=8)
        repr_out = Repr(lvals=[0, 1, 2], mult=16)
        layer = EquivariantLinear(repr_in, repr_out)

        x = torch.randn(32, 8, 9)
        y = layer(x)
        assert y.shape == (32, 16, 9)

    def test_mismatched_lvals_raises(self):
        """Test mismatched lvals raises error."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr_in = Repr(lvals=[0, 1], mult=8)
        repr_out = Repr(lvals=[0, 1, 2], mult=8)

        with pytest.raises(ValueError, match="cannot modify the degrees"):
            EquivariantLinear(repr_in, repr_out)

    def test_no_nan_output(self):
        """Test output contains no NaN values."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr = Repr(lvals=[0, 1, 2], mult=8)
        layer = EquivariantLinear(repr, repr)

        x = torch.randn(32, 8, 9)
        y = layer(x)
        assert not torch.isnan(y).any()

    def test_gradients_flow(self):
        """Test gradients flow correctly."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr = Repr(lvals=[0, 1], mult=4)
        layer = EquivariantLinear(repr, repr)

        x = torch.randn(16, 4, 4, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_bias_only_on_scalars(self):
        """Test bias is only applied to scalar components."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr = Repr(lvals=[0, 1, 2], mult=4)  # Has scalars
        layer = EquivariantLinear(repr, repr, bias=True)

        # Zero input
        x = torch.zeros(16, 4, 9)
        y = layer(x)

        # Scalar component (index 0) may have non-zero output (bias)
        # Vector and higher components should be zero (no bias)
        assert torch.allclose(y[..., 1:4], torch.zeros(16, 4, 3), atol=1e-6)  # l=1
        assert torch.allclose(y[..., 4:9], torch.zeros(16, 4, 5), atol=1e-6)  # l=2

    def test_no_bias_option(self):
        """Test layer can be created without bias."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr = Repr(lvals=[0, 1], mult=4)
        layer = EquivariantLinear(repr, repr, bias=False)

        x = torch.zeros(16, 4, 4)
        y = layer(x)

        # Without bias, zero input should give zero output
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


# ============================================================================
# TEST: EquivariantLayerNorm
# ============================================================================


class TestEquivariantLayerNorm:
    """Tests for EquivariantLayerNorm layer."""

    def test_output_shape(self):
        """Test output has same shape as input."""
        from ciffy.nn.geometric.layers import EquivariantLayerNorm
        repr = Repr(lvals=[0, 1, 2], mult=8)
        ln = EquivariantLayerNorm(repr)

        x = torch.randn(32, 8, 9)
        y = ln(x)
        assert y.shape == x.shape

    def test_no_nan_output(self):
        """Test output contains no NaN values."""
        from ciffy.nn.geometric.layers import EquivariantLayerNorm
        repr = Repr(lvals=[0, 1, 2], mult=8)
        ln = EquivariantLayerNorm(repr)

        x = torch.randn(32, 8, 9)
        y = ln(x)
        assert not torch.isnan(y).any()

    def test_zero_input_handling(self):
        """Test zero input doesn't produce NaN."""
        from ciffy.nn.geometric.layers import EquivariantLayerNorm
        repr = Repr(lvals=[0, 1], mult=4)
        ln = EquivariantLayerNorm(repr)

        x = torch.zeros(16, 4, 4)
        y = ln(x)
        assert not torch.isnan(y).any()

    def test_mult_one_edge_case(self):
        """Test with multiplicity 1 (edge case for LayerNorm)."""
        from ciffy.nn.geometric.layers import EquivariantLayerNorm
        repr = Repr(lvals=[0, 1], mult=1)
        ln = EquivariantLayerNorm(repr)

        x = torch.randn(16, 1, 4)
        y = ln(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_gradients_flow(self):
        """Test gradients flow correctly."""
        from ciffy.nn.geometric.layers import EquivariantLayerNorm
        repr = Repr(lvals=[0, 1], mult=4)
        ln = EquivariantLayerNorm(repr)

        x = torch.randn(16, 4, 4, requires_grad=True)
        y = ln(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================================================
# TEST: EquivariantTransformer
# ============================================================================


@requires_sphericart
class TestEquivariantTransformer:
    """Tests for the main EquivariantTransformer model."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple transformer for testing (rank=0, fastest)."""
        from ciffy.nn.geometric import EquivariantTransformer
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        model = EquivariantTransformer(
            in_repr, out_repr, hidden_repr,
            hidden_layers=2,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            nheads=2,
            rank=0,
        )
        return model, in_repr, out_repr

    @pytest.fixture(params=[0, 1, None])
    def model_with_rank(self, request):
        """Create transformer with different rank values."""
        from ciffy.nn.geometric import EquivariantTransformer
        rank = request.param
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        model = EquivariantTransformer(
            in_repr, out_repr, hidden_repr,
            hidden_layers=2,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            nheads=2,
            rank=rank,
        )
        return model, in_repr, out_repr, rank

    def test_forward_shape(self, simple_model):
        """Test forward pass produces correct output shape."""
        model, in_repr, out_repr = simple_model

        N = 50
        coords = torch.randn(N, 3)
        features = torch.randn(N, in_repr.mult, in_repr.dim())

        output = model(coords, features)
        assert output.shape == (N, out_repr.mult, out_repr.dim())

    def test_forward_no_nan(self, simple_model):
        """Test forward pass produces no NaN values."""
        model, in_repr, out_repr = simple_model

        coords = torch.randn(50, 3)
        features = torch.randn(50, in_repr.mult, in_repr.dim())

        output = model(coords, features)
        assert not torch.isnan(output).any()

    def test_backward_gradients(self, simple_model):
        """Test gradients flow correctly."""
        model, in_repr, out_repr = simple_model

        coords = torch.randn(30, 3)
        features = torch.randn(30, in_repr.mult, in_repr.dim(), requires_grad=True)

        output = model(coords, features)
        loss = output.sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_invalid_input_shape_raises(self, simple_model):
        """Test wrong input shape raises error."""
        model, in_repr, out_repr = simple_model

        coords = torch.randn(50, 3)
        # Wrong multiplicity
        features = torch.randn(50, 3, in_repr.dim())

        with pytest.raises(ValueError, match="does not match"):
            model(coords, features)

    def test_with_mask(self, simple_model):
        """Test forward pass with attention mask."""
        model, in_repr, out_repr = simple_model

        N = 50
        k = 8
        coords = torch.randn(N, 3)
        features = torch.randn(N, in_repr.mult, in_repr.dim())
        mask = torch.zeros(N, k, dtype=torch.bool)
        mask[:, -2:] = True  # Mask last 2 neighbors

        output = model(coords, features, mask=mask)
        assert output.shape == (N, out_repr.mult, out_repr.dim())
        assert not torch.isnan(output).any()

    def test_with_seq_pos(self):
        """Test forward pass with sequence position encoding."""
        from ciffy.nn.geometric import EquivariantTransformer
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        model = EquivariantTransformer(
            in_repr, out_repr, hidden_repr,
            hidden_layers=2,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            nheads=2,
            seq_pos_dim=8,  # Enable sequence position encoding
        )

        N = 50
        coords = torch.randn(N, 3)
        features = torch.randn(N, in_repr.mult, in_repr.dim())
        seq_pos = torch.arange(N)

        output = model(coords, features, seq_pos=seq_pos)
        assert output.shape == (N, out_repr.mult, out_repr.dim())

    def test_seq_pos_required_when_configured(self):
        """Test seq_pos must be provided when seq_pos_dim > 0."""
        from ciffy.nn.geometric import EquivariantTransformer
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        model = EquivariantTransformer(
            in_repr, out_repr, hidden_repr,
            hidden_layers=1,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            seq_pos_dim=8,
        )

        coords = torch.randn(30, 3)
        features = torch.randn(30, in_repr.mult, in_repr.dim())

        with pytest.raises(ValueError, match="seq_pos must be provided"):
            model(coords, features)

    def test_translation_invariance(self, simple_model):
        """Test output is invariant to global translation."""
        model, in_repr, out_repr = simple_model
        model.eval()

        # Use seed for reproducibility
        torch.manual_seed(42)
        coords = torch.randn(30, 3)
        features = torch.randn(30, in_repr.mult, in_repr.dim())

        # Translation (moderate magnitude to avoid floating point issues)
        translation = torch.randn(3) * 10
        coords_translated = coords + translation

        with torch.no_grad():
            output1 = model(coords, features)
            output2 = model(coords_translated, features)

        # Use relative tolerance due to floating point accumulation in deep networks
        assert torch.allclose(output1, output2, atol=1e-3, rtol=1e-3)

    def test_rotation_equivariance(self, model_with_rank):
        """Test output rotates correctly under input rotation for all ranks."""
        model, in_repr, out_repr, rank = model_with_rank
        model.eval()

        coords = torch.randn(30, 3)
        features = torch.randn(30, in_repr.mult, in_repr.dim())

        # Random rotation
        R, axis, angle = random_rotation_matrix()
        D_in = in_repr.rot(axis, angle)
        D_out = out_repr.rot(axis, angle)

        # Rotate coordinates and features
        coords_rotated = coords @ R.T
        features_rotated = features @ D_in.T

        with torch.no_grad():
            output1 = model(coords, features)
            output2 = model(coords_rotated, features_rotated)

        # Output should be rotated by D_out
        output1_rotated = output1 @ D_out.T

        assert torch.allclose(output1_rotated, output2, atol=1e-4)

    def test_different_attention_types(self):
        """Test both attention types work."""
        from ciffy.nn.geometric import EquivariantTransformer
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        for attn_type in ["node_wise", "edge_wise"]:
            model = EquivariantTransformer(
                in_repr, out_repr, hidden_repr,
                hidden_layers=1,
                edge_dim=16,
                edge_hidden_dim=32,
                k_neighbors=8,
                attention_type=attn_type,
            )

            coords = torch.randn(30, 3)
            features = torch.randn(30, in_repr.mult, in_repr.dim())

            output = model(coords, features)
            assert output.shape == (30, out_repr.mult, out_repr.dim())
            assert not torch.isnan(output).any()

    def test_different_rbf_types(self):
        """Test all RBF types work in the model."""
        from ciffy.nn.geometric import EquivariantTransformer
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        for rbf_type in ["gaussian", "bessel", "polynomial"]:
            model = EquivariantTransformer(
                in_repr, out_repr, hidden_repr,
                hidden_layers=1,
                edge_dim=16,
                edge_hidden_dim=32,
                k_neighbors=8,
                rbf_type=rbf_type,
            )

            coords = torch.randn(30, 3)
            features = torch.randn(30, in_repr.mult, in_repr.dim())

            output = model(coords, features)
            assert not torch.isnan(output).any()


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_scalar_only_representation(self):
        """Test representation with only l=0 (scalars)."""
        repr = Repr(lvals=[0], mult=4)
        assert repr.dim() == 1
        assert repr.nreps() == 1

        # Rotation should be identity for scalars
        _, axis, angle = random_rotation_matrix()
        D = repr.rot(axis, angle)
        assert torch.allclose(D, torch.ones(1, 1), atol=1e-6)

    def test_duplicate_lvals(self):
        """Test representation with duplicate l values."""
        repr = Repr(lvals=[1, 1, 1], mult=2)
        assert repr.dim() == 9  # 3 + 3 + 3
        assert repr.nreps() == 3

        # Should still be valid
        st = torch.randn(10, 2, 9)
        assert repr.verify(st)

        # Rotation should work
        _, axis, angle = random_rotation_matrix()
        D = repr.rot(axis, angle)
        assert D.shape == (9, 9)
        assert torch.allclose(D @ D.T, torch.eye(9), atol=1e-5)

    def test_rotation_180_degrees(self):
        """Test rotation by exactly π (180 degrees) - numerical edge case."""
        repr = Repr(lvals=[0, 1, 2])
        axis = torch.tensor([0., 0., 1.])
        angle = torch.tensor(math.pi)

        D = repr.rot(axis, angle)

        # Should still be orthogonal
        assert torch.allclose(D @ D.T, torch.eye(repr.dim()), atol=1e-5)
        # Determinant should be 1
        assert torch.allclose(torch.linalg.det(D), torch.tensor(1.0), atol=1e-5)

    def test_rotation_negative_angle(self):
        """Test rotation with negative angle."""
        repr = Repr(lvals=[0, 1])
        axis = torch.tensor([1., 0., 0.])
        angle_pos = torch.tensor(0.5)
        angle_neg = torch.tensor(-0.5)

        D_pos = repr.rot(axis, angle_pos)
        D_neg = repr.rot(axis, angle_neg)

        # D(-θ) = D(θ)^T for rotations
        assert torch.allclose(D_neg, D_pos.T, atol=1e-5)

    def test_single_point_knn(self):
        """Test k-NN graph with single point."""
        from ciffy.nn.geometric import build_knn_graph
        coords = torch.randn(1, 3)
        neighbor_idx = build_knn_graph(coords, k=5)

        # Should still return valid shape
        assert neighbor_idx.shape == (1, 5)

    def test_coincident_points_knn(self):
        """Test k-NN graph with coincident (duplicate) points."""
        from ciffy.nn.geometric import build_knn_graph
        # All points at same location
        coords = torch.zeros(10, 3)
        neighbor_idx = build_knn_graph(coords, k=5)

        # Should not crash, indices should be valid
        assert neighbor_idx.shape == (10, 5)
        assert (neighbor_idx >= 0).all()
        assert (neighbor_idx < 10).all()

    @requires_sphericart
    def test_collinear_points_spherical_harmonic(self):
        """Test spherical harmonics with collinear points."""
        from ciffy.nn.geometric import SphericalHarmonic
        sh = SphericalHarmonic(lmax=2)

        # All points on z-axis
        coords = torch.tensor([[0., 0., 1.], [0., 0., 2.], [0., 0., -1.]])
        features = sh(coords)

        assert not torch.isnan(features).any()
        assert features.shape == (3, 9)

    def test_float64_repnorm(self):
        """Test RepNorm works with float64 inputs."""
        from ciffy.nn.geometric import RepNorm
        repr = Repr(lvals=[0, 1, 2])
        norm = RepNorm(repr)

        st = torch.randn(10, repr.dim(), dtype=torch.float64)
        result = norm(st)

        assert result.dtype == torch.float64
        assert not torch.isnan(result).any()

    def test_float64_equivariant_linear(self):
        """Test EquivariantLinear works with float64 inputs."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr = Repr(lvals=[0, 1], mult=4)
        layer = EquivariantLinear(repr, repr).double()

        x = torch.randn(10, 4, 4, dtype=torch.float64)
        y = layer(x)

        assert y.dtype == torch.float64
        assert not torch.isnan(y).any()

    def test_float64_repr_rot(self):
        """Test Repr.rot() works with float64 inputs."""
        repr = Repr(lvals=[0, 1, 2])

        # Float64 axis and angle
        axis = torch.tensor([1., 0., 0.], dtype=torch.float64)
        angle = torch.tensor(0.5, dtype=torch.float64)

        D = repr.rot(axis, angle)

        # Result should be float64
        assert D.dtype == torch.float64
        # Should be orthogonal
        assert torch.allclose(D @ D.T, torch.eye(repr.dim(), dtype=torch.float64), atol=1e-10)
        # No NaN values
        assert not torch.isnan(D).any()

    def test_repnorm_scalar_only(self):
        """Test RepNorm with scalar-only representation."""
        from ciffy.nn.geometric import RepNorm
        repr = Repr(lvals=[0, 0, 0], mult=1)  # Three scalars
        norm = RepNorm(repr)

        st = torch.randn(10, 3)
        result = norm(st)

        # Each scalar's "norm" is just its absolute value
        assert result.shape == (10, 3)
        assert torch.allclose(result, st.abs(), atol=1e-6)

    @requires_sphericart
    def test_transformer_single_point(self):
        """Test transformer with single point (N=1)."""
        from ciffy.nn.geometric import EquivariantTransformer
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        model = EquivariantTransformer(
            in_repr, out_repr, hidden_repr,
            hidden_layers=1,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
        )

        coords = torch.randn(1, 3)
        features = torch.randn(1, in_repr.mult, in_repr.dim())

        output = model(coords, features)
        assert output.shape == (1, out_repr.mult, out_repr.dim())
        assert not torch.isnan(output).any()

    @requires_sphericart
    def test_transformer_all_masked(self):
        """Test transformer behavior when all neighbors are masked."""
        from ciffy.nn.geometric import EquivariantTransformer
        in_repr = Repr(lvals=[0, 1], mult=4)
        out_repr = Repr(lvals=[0, 1], mult=2)
        hidden_repr = Repr(lvals=[0, 1], mult=8)

        model = EquivariantTransformer(
            in_repr, out_repr, hidden_repr,
            hidden_layers=1,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
        )

        N, k = 20, 8
        coords = torch.randn(N, 3)
        features = torch.randn(N, in_repr.mult, in_repr.dim())
        mask = torch.ones(N, k, dtype=torch.bool)  # All masked

        # Should not crash (softmax over all -inf gives nan, but skip connections help)
        output = model(coords, features, mask=mask)
        assert output.shape == (N, out_repr.mult, out_repr.dim())

    def test_equivariant_linear_scalar_only(self):
        """Test EquivariantLinear with scalar-only representation."""
        from ciffy.nn.geometric.layers import EquivariantLinear
        repr = Repr(lvals=[0, 0], mult=4)  # Two scalar channels
        layer = EquivariantLinear(repr, repr, bias=True)

        x = torch.randn(10, 4, 2)
        y = layer(x)

        assert y.shape == (10, 4, 2)
        assert not torch.isnan(y).any()


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
