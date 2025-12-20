"""Tests for diffusion process utilities.

Tests cover:
- FixedSinusoidalEmbedding: Sinusoidal positional embeddings
- NoiseSchedule: Base noise schedule functionality
- LinearNoiseSchedule: Linear beta interpolation
- CosineNoiseSchedule: Cosine schedule
- DiffusionProcess: Forward and reverse diffusion
- TimestepEmbedding: Learnable timestep embeddings
"""
from __future__ import annotations

import pytest
import torch

from ciffy.nn.diffusion import (
    FixedSinusoidalEmbedding,
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    DiffusionProcess,
    TimestepEmbedding,
)


# ============================================================================
# FIXED SINUSOIDAL EMBEDDING TESTS
# ============================================================================


class TestFixedSinusoidalEmbedding:
    """Tests for FixedSinusoidalEmbedding class."""

    def test_construction(self):
        """Test basic construction."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=128)
        assert embed.max_index == 1000
        assert embed.embedding_dim == 128

    def test_embedding_shape(self):
        """Test embedding output shape."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=128)
        assert embed.embeddings.shape == (1000, 128)

    def test_forward_single_index(self):
        """Test forward with single integer index."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=64)
        result = embed(0)
        assert result.shape == (64,)

    def test_forward_tensor_index(self):
        """Test forward with tensor of indices."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=64)
        indices = torch.tensor([0, 100, 500, 999])
        result = embed(indices)
        assert result.shape == (4, 64)

    def test_forward_with_shape(self):
        """Test forward with shape broadcasting."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=64)
        result = embed(50, shape=torch.Size([8, 16]))
        assert result.shape == (8, 16, 64)

    def test_different_indices_different_embeddings(self):
        """Test that different indices produce different embeddings."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=64)
        e0 = embed(0)
        e100 = embed(100)
        assert not torch.allclose(e0, e100)

    def test_same_index_same_embedding(self):
        """Test that same index produces same embedding."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=64)
        e1 = embed(50)
        e2 = embed(50)
        assert torch.allclose(e1, e2)

    def test_no_nan(self):
        """Test embeddings contain no NaN values."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=128)
        assert not torch.isnan(embed.embeddings).any()

    def test_bounded_values(self):
        """Test embedding values are in [-1, 1] (sine/cosine range)."""
        embed = FixedSinusoidalEmbedding(max_index=1000, embedding_dim=128)
        assert (embed.embeddings >= -1).all()
        assert (embed.embeddings <= 1).all()

    def test_invalid_max_index(self):
        """Test that invalid max_index raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            FixedSinusoidalEmbedding(max_index=0, embedding_dim=64)

    def test_invalid_embedding_dim(self):
        """Test that invalid embedding_dim raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            FixedSinusoidalEmbedding(max_index=1000, embedding_dim=0)


# ============================================================================
# NOISE SCHEDULE TESTS
# ============================================================================


class TestNoiseSchedule:
    """Tests for base NoiseSchedule class."""

    def test_construction(self):
        """Test basic construction with valid betas."""
        betas = torch.linspace(1e-4, 2e-2, 100)
        schedule = NoiseSchedule(betas)
        assert len(schedule) == 100

    def test_alpha_property(self):
        """Test alpha = 1 - beta."""
        betas = torch.tensor([0.1, 0.2, 0.3])
        schedule = NoiseSchedule(betas)
        t = torch.tensor([0, 1, 2])
        expected = torch.tensor([0.9, 0.8, 0.7])
        assert torch.allclose(schedule.alpha(t), expected)

    def test_alphabar_cumulative(self):
        """Test alphabar is cumulative product of alpha."""
        betas = torch.tensor([0.1, 0.2, 0.3])
        schedule = NoiseSchedule(betas)
        # alphabar[0] = 0.9, alphabar[1] = 0.9*0.8, alphabar[2] = 0.9*0.8*0.7
        expected = torch.tensor([0.9, 0.72, 0.504])
        t = torch.tensor([0, 1, 2])
        assert torch.allclose(schedule.alphabar(t), expected)

    def test_beta_property(self):
        """Test beta returns original values."""
        betas = torch.tensor([0.1, 0.2, 0.3])
        schedule = NoiseSchedule(betas)
        t = torch.tensor([0, 1, 2])
        assert torch.allclose(schedule.beta(t), betas)

    def test_sigma_property(self):
        """Test sigma = sqrt(beta)."""
        betas = torch.tensor([0.04, 0.09, 0.16])
        schedule = NoiseSchedule(betas)
        t = torch.tensor([0, 1, 2])
        expected = torch.tensor([0.2, 0.3, 0.4])
        assert torch.allclose(schedule.sigma(t), expected)

    def test_timesteps_reverse_order(self):
        """Test timesteps returns indices in reverse order."""
        betas = torch.linspace(1e-4, 2e-2, 100)
        schedule = NoiseSchedule(betas)
        timesteps = schedule.timesteps()
        assert timesteps[0] == 99
        assert timesteps[-1] == 0
        assert len(timesteps) == 100

    def test_random_timestep_shape(self):
        """Test random_timestep returns correct shape."""
        betas = torch.linspace(1e-4, 2e-2, 100)
        schedule = NoiseSchedule(betas)
        t = schedule.random_timestep((32,))
        assert t.shape == (32,)

    def test_random_timestep_range(self):
        """Test random_timestep values are in valid range."""
        betas = torch.linspace(1e-4, 2e-2, 100)
        schedule = NoiseSchedule(betas)
        t = schedule.random_timestep((1000,))
        assert (t >= 0).all()
        assert (t < 100).all()

    def test_invalid_betas_negative(self):
        """Test that negative betas raise error."""
        betas = torch.tensor([-0.1, 0.1, 0.2])
        with pytest.raises(ValueError, match="range"):
            NoiseSchedule(betas)

    def test_invalid_betas_too_large(self):
        """Test that betas >= 1 raise error."""
        betas = torch.tensor([0.1, 0.5, 1.0])
        with pytest.raises(ValueError, match="range"):
            NoiseSchedule(betas)


# ============================================================================
# LINEAR NOISE SCHEDULE TESTS
# ============================================================================


class TestLinearNoiseSchedule:
    """Tests for LinearNoiseSchedule class."""

    def test_construction(self):
        """Test basic construction."""
        schedule = LinearNoiseSchedule(num_timesteps=1000)
        assert len(schedule) == 1000

    def test_linear_interpolation(self):
        """Test betas are linearly interpolated."""
        schedule = LinearNoiseSchedule(num_timesteps=100, beta_range=(0.0001, 0.02))
        t = torch.tensor([0, 99])
        betas = schedule.beta(t)
        assert torch.isclose(betas[0], torch.tensor(0.0001), atol=1e-6)
        assert torch.isclose(betas[1], torch.tensor(0.02), atol=1e-6)

    def test_custom_beta_range(self):
        """Test custom beta range."""
        schedule = LinearNoiseSchedule(num_timesteps=100, beta_range=(0.001, 0.1))
        assert torch.isclose(schedule.beta(torch.tensor(0)), torch.tensor(0.001), atol=1e-6)

    def test_invalid_num_timesteps(self):
        """Test that invalid num_timesteps raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            LinearNoiseSchedule(num_timesteps=0)


# ============================================================================
# COSINE NOISE SCHEDULE TESTS
# ============================================================================


class TestCosineNoiseSchedule:
    """Tests for CosineNoiseSchedule class."""

    def test_construction(self):
        """Test basic construction."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        assert len(schedule) == 1000

    def test_alphabar_starts_near_one(self):
        """Test alphabar starts near 1 (little noise at start)."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        alphabar_0 = schedule.alphabar(torch.tensor(0))
        assert alphabar_0 > 0.99

    def test_alphabar_ends_near_zero(self):
        """Test alphabar ends near 0 (mostly noise at end)."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        alphabar_end = schedule.alphabar(torch.tensor(999))
        assert alphabar_end < 0.01

    def test_alphabar_monotonic_decreasing(self):
        """Test alphabar is monotonically decreasing."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        t = torch.arange(100)
        alphabar = schedule.alphabar(t)
        diffs = alphabar[1:] - alphabar[:-1]
        assert (diffs <= 0).all()

    def test_betas_clipped(self):
        """Test betas are clipped to valid range."""
        schedule = CosineNoiseSchedule(num_timesteps=1000, beta_clip=0.999)
        t = torch.arange(1000)
        betas = schedule.beta(t)
        assert (betas >= 0).all()
        assert (betas < 1).all()

    def test_invalid_num_timesteps(self):
        """Test that invalid num_timesteps raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            CosineNoiseSchedule(num_timesteps=0)


# ============================================================================
# DIFFUSION PROCESS TESTS
# ============================================================================


class TestDiffusionProcess:
    """Tests for DiffusionProcess class."""

    def test_construction(self):
        """Test basic construction."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        assert len(process) == 100

    def test_timesteps(self):
        """Test timesteps returns schedule timesteps."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        timesteps = process.timesteps()
        assert len(timesteps) == 100
        assert timesteps[0] == 99
        assert timesteps[-1] == 0

    def test_forward_diffusion_output_shapes(self):
        """Test forward diffusion returns correct shapes."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(32, 3, 64, 64)
        t = schedule.random_timestep((32,))
        noise, noisy_x = process.forward_diffusion(x, t)
        assert noise.shape == x.shape
        assert noisy_x.shape == x.shape

    def test_forward_diffusion_no_nan(self):
        """Test forward diffusion produces no NaN."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(32, 3, 64, 64)
        t = schedule.random_timestep((32,))
        noise, noisy_x = process.forward_diffusion(x, t)
        assert not torch.isnan(noise).any()
        assert not torch.isnan(noisy_x).any()

    def test_forward_diffusion_t0_minimal_noise(self):
        """Test that t=0 adds minimal noise."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 8, 8)
        t = torch.zeros(4, dtype=torch.long)
        _, noisy_x = process.forward_diffusion(x, t)
        # At t=0, alphabar is close to 1, so noisy_x should be close to x
        assert torch.allclose(noisy_x, x, atol=0.1)

    def test_reverse_diffusion_output_shape(self):
        """Test reverse diffusion returns correct shape."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(32, 3, 64, 64)
        predicted_noise = torch.randn_like(x)
        denoised = process.reverse_diffusion(x, predicted_noise, timestep=50)
        assert denoised.shape == x.shape

    def test_reverse_diffusion_no_nan(self):
        """Test reverse diffusion produces no NaN."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(32, 3, 64, 64)
        predicted_noise = torch.randn_like(x)
        denoised = process.reverse_diffusion(x, predicted_noise, timestep=50)
        assert not torch.isnan(denoised).any()

    def test_reverse_diffusion_t0_no_extra_noise(self):
        """Test that reverse diffusion at t=0 adds no extra noise."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        torch.manual_seed(42)
        x = torch.randn(4, 3, 8, 8)
        predicted_noise = torch.randn_like(x)
        # Run twice with same seed - should get same result at t=0
        torch.manual_seed(123)
        denoised1 = process.reverse_diffusion(x, predicted_noise, timestep=0)
        torch.manual_seed(456)  # Different seed
        denoised2 = process.reverse_diffusion(x, predicted_noise, timestep=0)
        assert torch.allclose(denoised1, denoised2)

    def test_ddpm_step_alias(self):
        """Test ddpm_step is equivalent to reverse_diffusion."""
        schedule = LinearNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        torch.manual_seed(42)
        x = torch.randn(4, 3, 8, 8)
        predicted_noise = torch.randn_like(x)
        torch.manual_seed(123)
        result1 = process.reverse_diffusion(x, predicted_noise, timestep=50)
        torch.manual_seed(123)
        result2 = process.ddpm_step(x, predicted_noise, timestep=50)
        assert torch.allclose(result1, result2)


# ============================================================================
# DDIM SAMPLING TESTS
# ============================================================================


class TestDDIMSampling:
    """Tests for DDIM sampling functionality."""

    def test_get_sampling_timesteps_uniform(self):
        """Test uniform timestep spacing."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        process = DiffusionProcess(schedule)
        timesteps = process.get_sampling_timesteps(num_steps=50)
        assert len(timesteps) == 50
        assert timesteps[0] > timesteps[-1]  # Descending order
        assert timesteps[0] <= 999
        assert timesteps[-1] >= 0

    def test_get_sampling_timesteps_count(self):
        """Test different step counts."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        process = DiffusionProcess(schedule)
        for n in [10, 50, 100, 200]:
            timesteps = process.get_sampling_timesteps(num_steps=n)
            assert len(timesteps) == n

    def test_get_sampling_timesteps_invalid(self):
        """Test invalid step counts raise errors."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        with pytest.raises(ValueError, match="cannot exceed"):
            process.get_sampling_timesteps(num_steps=200)
        with pytest.raises(ValueError, match="positive"):
            process.get_sampling_timesteps(num_steps=0)

    def test_get_sampling_timesteps_spacing_options(self):
        """Test different spacing options."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        process = DiffusionProcess(schedule)
        uniform = process.get_sampling_timesteps(50, spacing="uniform")
        trailing = process.get_sampling_timesteps(50, spacing="trailing")
        leading = process.get_sampling_timesteps(50, spacing="leading")
        assert len(uniform) == len(trailing) == len(leading) == 50

    def test_ddim_step_output_shape(self):
        """Test DDIM step returns correct shape."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 32, 32)
        predicted_noise = torch.randn_like(x)
        result = process.ddim_step(x, predicted_noise, timestep=50, timestep_prev=40)
        assert result.shape == x.shape

    def test_ddim_step_no_nan(self):
        """Test DDIM step produces no NaN."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 32, 32)
        predicted_noise = torch.randn_like(x)
        result = process.ddim_step(x, predicted_noise, timestep=50, timestep_prev=40)
        assert not torch.isnan(result).any()

    def test_ddim_deterministic_eta0(self):
        """Test DDIM is deterministic when eta=0."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 16, 16)
        predicted_noise = torch.randn_like(x)
        # Run twice with different seeds - should get same result
        torch.manual_seed(123)
        result1 = process.ddim_step(x, predicted_noise, timestep=50, timestep_prev=40, eta=0.0)
        torch.manual_seed(456)
        result2 = process.ddim_step(x, predicted_noise, timestep=50, timestep_prev=40, eta=0.0)
        assert torch.allclose(result1, result2)

    def test_ddim_stochastic_eta1(self):
        """Test DDIM is stochastic when eta>0."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 16, 16)
        predicted_noise = torch.randn_like(x)
        # Run twice with different seeds - should get different results
        torch.manual_seed(123)
        result1 = process.ddim_step(x, predicted_noise, timestep=50, timestep_prev=40, eta=1.0)
        torch.manual_seed(456)
        result2 = process.ddim_step(x, predicted_noise, timestep=50, timestep_prev=40, eta=1.0)
        assert not torch.allclose(result1, result2)

    def test_ddim_step_skipping(self):
        """Test DDIM can skip multiple timesteps."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 16, 16)
        predicted_noise = torch.randn_like(x)
        # Skip from 900 to 800 (100 step skip)
        result = process.ddim_step(x, predicted_noise, timestep=900, timestep_prev=800, eta=0.0)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_ddim_final_step(self):
        """Test DDIM handles final step (t_prev=0) correctly."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 16, 16)
        predicted_noise = torch.randn_like(x)
        result = process.ddim_step(x, predicted_noise, timestep=10, timestep_prev=0, eta=0.0)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_sample_step_ddpm(self):
        """Test unified sample_step with DDPM method."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 16, 16)
        predicted_noise = torch.randn_like(x)
        torch.manual_seed(42)
        result1 = process.sample_step(x, predicted_noise, timestep=50, method="ddpm")
        torch.manual_seed(42)
        result2 = process.ddpm_step(x, predicted_noise, timestep=50)
        assert torch.allclose(result1, result2)

    def test_sample_step_ddim(self):
        """Test unified sample_step with DDIM method."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 16, 16)
        predicted_noise = torch.randn_like(x)
        result1 = process.sample_step(x, predicted_noise, timestep=50, timestep_prev=40, method="ddim", eta=0.0)
        result2 = process.ddim_step(x, predicted_noise, timestep=50, timestep_prev=40, eta=0.0)
        assert torch.allclose(result1, result2)

    def test_sample_step_invalid_method(self):
        """Test sample_step raises error for invalid method."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        x = torch.randn(4, 3, 16, 16)
        predicted_noise = torch.randn_like(x)
        with pytest.raises(ValueError, match="Unknown method"):
            process.sample_step(x, predicted_noise, timestep=50, method="invalid")

    def test_predict_x0(self):
        """Test x0 prediction from noisy sample."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)
        # Create known x0 and add noise at t=50
        x0 = torch.randn(4, 3, 16, 16)
        t = torch.tensor([50, 50, 50, 50])
        noise, x_t = process.forward_diffusion(x0, t)
        # Predict x0 using the actual noise (perfect model)
        x0_pred = process._predict_x0(x_t, noise, t[0])
        # Should recover original x0 closely
        assert torch.allclose(x0_pred, x0, atol=1e-5)


# ============================================================================
# TIMESTEP EMBEDDING TESTS
# ============================================================================


class TestTimestepEmbedding:
    """Tests for TimestepEmbedding class."""

    def test_construction(self):
        """Test basic construction."""
        embed = TimestepEmbedding(max_index=1000, embedding_dim=256)
        assert embed.embedding.max_index == 1000
        assert embed.embedding.embedding_dim == 256

    def test_forward_single_index(self):
        """Test forward with single integer index."""
        embed = TimestepEmbedding(max_index=1000, embedding_dim=64)
        result = embed(50)
        assert result.shape == (64,)

    def test_forward_tensor_index(self):
        """Test forward with tensor of indices."""
        embed = TimestepEmbedding(max_index=1000, embedding_dim=64)
        indices = torch.tensor([0, 100, 500])
        result = embed(indices)
        assert result.shape == (3, 64)

    def test_forward_no_nan(self):
        """Test forward produces no NaN."""
        embed = TimestepEmbedding(max_index=1000, embedding_dim=64)
        indices = torch.arange(100)
        result = embed(indices)
        assert not torch.isnan(result).any()

    def test_learnable_parameters(self):
        """Test that the embedding has learnable parameters."""
        embed = TimestepEmbedding(max_index=1000, embedding_dim=64)
        params = list(embed.parameters())
        assert len(params) > 0  # Should have MLP parameters

    def test_backward_gradients(self):
        """Test gradients flow correctly."""
        embed = TimestepEmbedding(max_index=1000, embedding_dim=64)
        indices = torch.tensor([0, 50, 100])
        result = embed(indices)
        loss = result.sum()
        loss.backward()
        # Check that gradients exist
        for param in embed.parameters():
            assert param.grad is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestDiffusionIntegration:
    """Integration tests for diffusion utilities."""

    def test_training_loop_simulation(self):
        """Simulate a training loop with diffusion."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)

        # Simulate training data
        batch_size = 16
        x = torch.randn(batch_size, 3, 32, 32)

        # Sample random timesteps
        t = schedule.random_timestep((batch_size,))

        # Forward diffusion
        noise, noisy_x = process.forward_diffusion(x, t)

        # Check outputs
        assert noise.shape == x.shape
        assert noisy_x.shape == x.shape
        assert not torch.isnan(noise).any()
        assert not torch.isnan(noisy_x).any()

    def test_ddpm_sampling_loop_simulation(self):
        """Simulate a DDPM sampling loop (all steps, stochastic)."""
        schedule = CosineNoiseSchedule(num_timesteps=100)
        process = DiffusionProcess(schedule)

        # Start from noise
        x = torch.randn(4, 3, 16, 16)

        # Simulate sampling (reverse diffusion) - just first 10 steps
        for t in process.timesteps()[:10]:
            predicted_noise = torch.randn_like(x)  # Fake model prediction
            x = process.ddpm_step(x, predicted_noise, t)
            assert not torch.isnan(x).any()

    def test_ddim_sampling_loop_simulation(self):
        """Simulate a DDIM sampling loop (fewer steps, deterministic)."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        process = DiffusionProcess(schedule)

        # Start from noise
        x = torch.randn(4, 3, 16, 16)

        # Get 50 timesteps instead of 1000
        timesteps = process.get_sampling_timesteps(num_steps=50)

        # Simulate DDIM sampling
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            predicted_noise = torch.randn_like(x)  # Fake model prediction
            x = process.ddim_step(x, predicted_noise, t, t_prev, eta=0.0)
            assert not torch.isnan(x).any()

    def test_ddim_vs_ddpm_same_training(self):
        """Verify DDPM and DDIM use the same training (forward diffusion)."""
        schedule = CosineNoiseSchedule(num_timesteps=1000)
        process = DiffusionProcess(schedule)

        # Training is identical for both
        x = torch.randn(32, 3, 32, 32)
        t = schedule.random_timestep((32,))
        noise, noisy_x = process.forward_diffusion(x, t)

        # Both methods can denoise from the same noisy sample
        ddpm_result = process.ddpm_step(noisy_x[0:1], noise[0:1], t[0])
        ddim_result = process.ddim_step(noisy_x[0:1], noise[0:1], t[0], max(0, t[0]-1), eta=0.0)

        # Both should produce valid outputs (not necessarily identical)
        assert not torch.isnan(ddpm_result).any()
        assert not torch.isnan(ddim_result).any()

    def test_timestep_embedding_with_process(self):
        """Test timestep embedding works with diffusion process."""
        schedule = LinearNoiseSchedule(num_timesteps=1000)
        embed = TimestepEmbedding(max_index=1000, embedding_dim=128)

        t = schedule.random_timestep((32,))
        embeddings = embed(t)

        assert embeddings.shape == (32, 128)
        assert not torch.isnan(embeddings).any()

    def test_schedule_comparison(self):
        """Compare linear and cosine schedules."""
        linear = LinearNoiseSchedule(num_timesteps=1000)
        cosine = CosineNoiseSchedule(num_timesteps=1000)

        t = torch.arange(1000)

        linear_alphabar = linear.alphabar(t)
        cosine_alphabar = cosine.alphabar(t)

        # Both should start near 1 and end near 0
        assert linear_alphabar[0] > 0.99
        assert cosine_alphabar[0] > 0.99
        assert linear_alphabar[-1] < 0.5
        assert cosine_alphabar[-1] < 0.01

        # Cosine should preserve more signal in early steps
        # (higher alphabar in the middle)
        mid = 500
        assert cosine_alphabar[mid] > linear_alphabar[mid]
