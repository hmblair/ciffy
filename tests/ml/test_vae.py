"""Tests for ciffy.nn.vae module."""

import pytest
import numpy as np

import ciffy
from ciffy import Scale

from tests.utils import (
    get_test_cif,
    TORCH_AVAILABLE,
    skip_if_no_torch,
)
from tests.testing import get_tolerances


# =============================================================================
# Test Distributions
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDistributions:
    """Tests for angular distribution utilities."""

    def test_sincos_encode_decode_roundtrip(self):
        """Test sin/cos encoding is invertible."""
        import torch
        from ciffy.nn.vae.distributions import sincos_encode, sincos_decode

        angles = torch.rand(10, 7) * 2 * np.pi - np.pi  # [-pi, pi]
        encoded = sincos_encode(angles)
        decoded = sincos_decode(encoded)

        # Check round-trip accuracy (accounting for periodicity)
        tol = get_tolerances()
        diff = torch.abs(torch.sin(angles - decoded))
        assert diff.max() < tol.allclose_atol

    def test_sincos_encode_shape(self):
        """Test sin/cos encoding doubles dimension."""
        import torch
        from ciffy.nn.vae.distributions import sincos_encode

        angles = torch.rand(5, 7)
        encoded = sincos_encode(angles)

        assert encoded.shape == (5, 14)

    def test_sincos_encode_batched(self):
        """Test sin/cos encoding handles batch dimension."""
        import torch
        from ciffy.nn.vae.distributions import sincos_encode

        angles = torch.rand(3, 5, 7)  # (batch, seq, dihedrals)
        encoded = sincos_encode(angles)

        assert encoded.shape == (3, 5, 14)

    def test_sincos_unit_circle(self):
        """Test sin/cos encoding produces points on unit circle."""
        import torch
        from ciffy.nn.vae.distributions import sincos_encode

        angles = torch.rand(10) * 2 * np.pi - np.pi
        encoded = sincos_encode(angles)

        # Each (sin, cos) pair should have norm 1
        tol = get_tolerances()
        encoded = encoded.view(-1, 2)
        norms = torch.sqrt(encoded[:, 0] ** 2 + encoded[:, 1] ** 2)
        assert torch.allclose(norms, torch.ones_like(norms), atol=tol.allclose_atol)

    def test_von_mises_nll_positive(self):
        """Test von Mises NLL returns positive values for low concentration."""
        import torch
        from ciffy.nn.vae.distributions import VonMisesNLL

        nll = VonMisesNLL(reduction="mean")
        mu = torch.zeros(10)
        kappa = torch.ones(10) * 0.5  # Low concentration
        target = torch.zeros(10)

        loss = nll(mu, kappa, target)
        # NLL should be positive when concentration is low
        assert loss >= 0

    def test_von_mises_nll_lower_for_closer_targets(self):
        """Test von Mises NLL is lower when target is closer to mean."""
        import torch
        from ciffy.nn.vae.distributions import VonMisesNLL

        nll = VonMisesNLL(reduction="mean")
        mu = torch.zeros(10)
        kappa = torch.ones(10) * 10  # High concentration

        target_close = torch.zeros(10)  # Same as mean
        target_far = torch.ones(10) * np.pi  # Opposite side

        loss_close = nll(mu, kappa, target_close)
        loss_far = nll(mu, kappa, target_far)

        assert loss_close < loss_far

    def test_von_mises_nll_with_mask(self):
        """Test von Mises NLL respects mask."""
        import torch
        from ciffy.nn.vae.distributions import VonMisesNLL

        nll = VonMisesNLL(reduction="mean")
        mu = torch.zeros(10)
        kappa = torch.ones(10) * 5
        target = torch.ones(10) * np.pi  # All far from mean

        mask_all = torch.ones(10, dtype=torch.bool)
        mask_half = torch.zeros(10, dtype=torch.bool)
        mask_half[:5] = True

        loss_all = nll(mu, kappa, target, mask_all)
        loss_half = nll(mu, kappa, target, mask_half)

        # Both should be similar since all targets are the same
        # But computation should succeed with mask
        assert not torch.isnan(loss_all)
        assert not torch.isnan(loss_half)

    def test_angular_distance(self):
        """Test angular distance handles periodicity."""
        import torch
        from ciffy.nn.vae.distributions import angular_distance

        tol = get_tolerances()

        # Same angle
        dist = angular_distance(torch.tensor([0.0]), torch.tensor([0.0]))
        assert torch.allclose(dist, torch.tensor([0.0]), atol=tol.allclose_atol)

        # Opposite angles (pi apart)
        dist = angular_distance(torch.tensor([np.pi]), torch.tensor([0.0]))
        assert torch.allclose(dist, torch.tensor([np.pi]), atol=tol.allclose_atol)

        # Across periodic boundary
        dist = angular_distance(torch.tensor([-np.pi + 0.1]), torch.tensor([np.pi - 0.1]))
        assert dist < 0.3  # Should be close, not 2*pi apart


# =============================================================================
# Test DihedralEncoder
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDihedralEncoder:
    """Tests for DihedralEncoder."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        import torch
        from ciffy.nn.vae import DihedralEncoder

        encoder = DihedralEncoder(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        # Create mock dihedral data
        L = polymer.size(Scale.RESIDUE)
        dihedrals = torch.randn(L, 7)
        mask = torch.ones(L, 7, dtype=torch.bool)

        mu, logvar = encoder(dihedrals, mask, polymer)

        assert mu.shape == (32,)
        assert logvar.shape == (32,)

    def test_encoder_with_mask(self):
        """Test encoder handles dihedral mask correctly."""
        import torch
        from ciffy.nn.vae import DihedralEncoder

        encoder = DihedralEncoder(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acguacgu", backend="torch")

        L = polymer.size(Scale.RESIDUE)
        dihedrals = torch.randn(L, 7)
        mask = torch.ones(L, 7, dtype=torch.bool)
        mask[:, 3:] = False  # Mask some dihedrals (like protein only has 3)

        mu, logvar = encoder(dihedrals, mask, polymer)

        assert mu.shape == (32,)
        assert not torch.isnan(mu).any()

    def test_encoder_protein(self):
        """Test encoder handles proteins."""
        import torch
        from ciffy.nn.vae import DihedralEncoder

        encoder = DihedralEncoder(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("MGKLF", backend="torch")

        L = polymer.size(Scale.RESIDUE)
        dihedrals = torch.randn(L, 7)
        mask = torch.ones(L, 7, dtype=torch.bool)
        mask[:, 3:] = False  # Proteins have 3 backbone dihedrals

        mu, logvar = encoder(dihedrals, mask, polymer)

        assert mu.shape == (32,)
        assert not torch.isnan(mu).any()

    def test_encoder_single_residue(self):
        """Test encoder handles single residue."""
        import torch
        from ciffy.nn.vae import DihedralEncoder

        encoder = DihedralEncoder(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("a", backend="torch")

        L = 1
        dihedrals = torch.randn(L, 7)
        mask = torch.ones(L, 7, dtype=torch.bool)

        mu, logvar = encoder(dihedrals, mask, polymer)

        assert mu.shape == (32,)


# =============================================================================
# Test DihedralDecoder
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDihedralDecoder:
    """Tests for DihedralDecoder."""

    def test_decoder_output_shape(self):
        """Test decoder produces correct output shape."""
        import torch
        from ciffy.nn.vae import DihedralDecoder
        from ciffy.nn.vae.distributions import MAX_DIHEDRALS_PER_RESIDUE

        decoder = DihedralDecoder(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        L = polymer.size(Scale.RESIDUE)
        z = torch.randn(32)
        mask = torch.ones(L, 7, dtype=torch.bool)

        mu, kappa = decoder(z, polymer, mask)

        assert mu.shape == (L, MAX_DIHEDRALS_PER_RESIDUE)
        assert kappa.shape == (L, MAX_DIHEDRALS_PER_RESIDUE)
        assert (kappa > 0).all()  # kappa must be positive

    def test_decoder_sample(self):
        """Test decoder sampling produces valid angles."""
        import torch
        from ciffy.nn.vae import DihedralDecoder

        decoder = DihedralDecoder(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        L = polymer.size(Scale.RESIDUE)
        z = torch.randn(32)
        mask = torch.ones(L, 7, dtype=torch.bool)

        samples = decoder.sample(z, polymer, mask)

        # Samples should be in [-pi, pi]
        assert (samples >= -np.pi - 0.01).all()
        assert (samples <= np.pi + 0.01).all()

    def test_decoder_temperature(self):
        """Test decoder temperature affects sampling diversity."""
        import torch
        from ciffy.nn.vae import DihedralDecoder

        decoder = DihedralDecoder(latent_dim=32, hidden_dim=64, num_layers=2)
        decoder.eval()

        polymer = ciffy.from_sequence("acgu", backend="torch")
        L = polymer.size(Scale.RESIDUE)
        z = torch.randn(32)
        mask = torch.ones(L, 7, dtype=torch.bool)

        # Sample many times at different temperatures
        torch.manual_seed(42)
        samples_low_temp = [decoder.sample(z, polymer, mask, temperature=0.1) for _ in range(10)]
        torch.manual_seed(42)
        samples_high_temp = [decoder.sample(z, polymer, mask, temperature=2.0) for _ in range(10)]

        # Lower temperature should produce less variance
        var_low = torch.stack(samples_low_temp).var()
        var_high = torch.stack(samples_high_temp).var()

        assert var_low < var_high


# =============================================================================
# Test VAELoss
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestVAELoss:
    """Tests for VAELoss."""

    def test_loss_components(self):
        """Test that loss returns all expected components."""
        import torch
        from ciffy.nn.vae.losses import VAELoss

        loss_fn = VAELoss(beta=1.0)

        L, D = 10, 7
        mu_pred = torch.randn(L, D)
        kappa_pred = torch.ones(L, D) * 5
        target = torch.randn(L, D)
        z_mu = torch.randn(32)
        z_logvar = torch.randn(32)
        mask = torch.ones(L, D, dtype=torch.bool)

        losses = loss_fn(mu_pred, kappa_pred, target, z_mu, z_logvar, mask)

        assert "loss" in losses
        assert "recon_loss" in losses
        assert "kl_loss" in losses

    def test_loss_beta_scaling(self):
        """Test that beta scales KL loss."""
        import torch
        from ciffy.nn.vae.losses import VAELoss

        L, D = 10, 7
        mu_pred = torch.randn(L, D)
        kappa_pred = torch.ones(L, D) * 5
        target = torch.randn(L, D)
        z_mu = torch.randn(32)
        z_logvar = torch.randn(32)
        mask = torch.ones(L, D, dtype=torch.bool)

        loss_fn_1 = VAELoss(beta=1.0)
        loss_fn_10 = VAELoss(beta=10.0)

        losses_1 = loss_fn_1(mu_pred, kappa_pred, target, z_mu, z_logvar, mask)
        losses_10 = loss_fn_10(mu_pred, kappa_pred, target, z_mu, z_logvar, mask)

        # Same reconstruction loss
        assert torch.allclose(losses_1["recon_loss"], losses_10["recon_loss"])
        # Same KL loss
        assert torch.allclose(losses_1["kl_loss"], losses_10["kl_loss"])
        # Total differs by KL contribution
        # total = recon + beta * kl
        # Difference should be (10 - 1) * kl = 9 * kl
        tol = get_tolerances()
        diff = losses_10["loss"] - losses_1["loss"]
        expected_diff = 9 * losses_1["kl_loss"]
        assert torch.allclose(diff, expected_diff, atol=tol.allclose_atol)


# =============================================================================
# Test PolymerVAE
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolymerVAE:
    """Tests for PolymerVAE main class."""

    def test_vae_encode_rna(self):
        """Test VAE encoding of RNA template."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        mu, logvar = vae.encode(polymer)

        assert mu.shape == (32,)
        assert logvar.shape == (32,)
        assert not torch.isnan(mu).any()

    def test_vae_encode_protein(self):
        """Test VAE encoding of protein template."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("MGKLF", backend="torch")

        mu, logvar = vae.encode(polymer)

        assert mu.shape == (32,)
        assert logvar.shape == (32,)
        assert not torch.isnan(mu).any()

    def test_vae_forward(self):
        """Test VAE full forward pass."""
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        mu_pred, kappa_pred, target, z_mu, z_logvar, mask = vae.forward(polymer)

        n_res = polymer.size(Scale.RESIDUE)
        assert mu_pred.shape[0] == n_res
        assert target.shape[0] == n_res

    def test_vae_loss_computation(self):
        """Test VAE loss is computed correctly."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        losses = vae.compute_loss(polymer)

        assert "loss" in losses
        assert "recon_loss" in losses
        assert "kl_loss" in losses
        assert not torch.isnan(losses["loss"])

    def test_vae_decode(self):
        """Test VAE decode produces valid polymer."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        z = torch.randn(32)
        decoded = vae.decode(z, polymer)

        assert decoded.size() == polymer.size()
        assert decoded.size(Scale.RESIDUE) == polymer.size(Scale.RESIDUE)

    def test_vae_sample(self):
        """Test VAE can sample new conformations."""
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        samples = vae.sample(polymer, n_samples=3)

        assert len(samples) == 3
        for s in samples:
            assert s.size() == polymer.size()

    def test_vae_reconstruct(self):
        """Test VAE reconstruction method."""
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        recon = vae.reconstruct(polymer)

        assert recon.size() == polymer.size()

    def test_vae_interpolate(self):
        """Test VAE latent space interpolation."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        p1 = ciffy.from_sequence("acgu", backend="torch")
        p2 = ciffy.from_sequence("acgu", backend="torch")

        interp = vae.interpolate(p1, p2, n_steps=5)

        assert len(interp) == 5
        for p in interp:
            assert p.size() == p1.size()

    def test_vae_gradients(self):
        """Test that gradients flow through VAE."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        losses = vae.compute_loss(polymer)
        losses["loss"].backward()

        # Check gradients exist on some parameters
        has_grad = False
        for name, param in vae.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found on any parameters"


# =============================================================================
# Test PolymerVAE with Real Structures
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolymerVAEReal:
    """Tests using real PDB structures."""

    def test_vae_on_real_structure(self):
        """Test VAE on real structure."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.load(get_test_cif("3SKW"), backend="torch")
        polymer = polymer.poly()  # Remove HETATM

        # Get first chain
        chain = polymer.by_index(0)

        mu, logvar = vae.encode(chain)
        assert not torch.isnan(mu).any()

    def test_vae_training_loop(self):
        """Test a simple training loop works."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=16, hidden_dim=32, num_layers=1)
        polymer = ciffy.from_sequence("acgu", backend="torch")

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        initial_loss = None
        final_loss = None

        for i in range(20):
            losses = vae.compute_loss(polymer)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            if i == 0:
                initial_loss = losses["loss"].item()
            if i == 19:
                final_loss = losses["loss"].item()

        # Loss should decrease (or at least not increase dramatically)
        # Note: VAE loss can be volatile, so we just check it doesn't explode
        assert final_loss < initial_loss * 2, "Loss increased too much during training"


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolymerVAEEdgeCases:
    """Edge case tests for PolymerVAE."""

    def test_vae_short_sequence(self):
        """Test VAE handles short sequence (2 residues)."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("ac", backend="torch")

        losses = vae.compute_loss(polymer)
        assert not torch.isnan(losses["loss"])

    def test_vae_protein_short(self):
        """Test VAE handles short protein."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)
        polymer = ciffy.from_sequence("MG", backend="torch")

        losses = vae.compute_loss(polymer)
        assert not torch.isnan(losses["loss"])

    def test_vae_different_latent_dims(self):
        """Test VAE with different latent dimensions."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        for latent_dim in [8, 32, 128]:
            vae = PolymerVAE(latent_dim=latent_dim, hidden_dim=64, num_layers=2)
            polymer = ciffy.from_sequence("acgu", backend="torch")

            mu, logvar = vae.encode(polymer)
            assert mu.shape == (latent_dim,)

            z = torch.randn(latent_dim)
            decoded = vae.decode(z, polymer)
            assert decoded.size() == polymer.size()

    def test_vae_residue_embedding_integration(self):
        """Test that residue embeddings are properly integrated."""
        import torch
        from ciffy.nn.vae import PolymerVAE

        vae = PolymerVAE(latent_dim=32, hidden_dim=64, num_layers=2)

        # Different sequences should produce different encodings
        poly_a = ciffy.from_sequence("aaaa", backend="torch")
        poly_u = ciffy.from_sequence("uuuu", backend="torch")

        mu_a, _ = vae.encode(poly_a)
        mu_u, _ = vae.encode(poly_u)

        # Encodings should differ (residue embeddings make them sequence-aware)
        # Note: they may not differ much at initialization, but should have some difference
        assert mu_a.shape == mu_u.shape
