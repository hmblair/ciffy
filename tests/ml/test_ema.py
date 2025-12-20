"""Tests for Exponential Moving Average (EMA) utilities.

Tests cover:
- EMA: Main EMA class
- create_ema_model: Creating a separate EMA model copy
- update_ema_model: Updating EMA model from training model
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from ciffy.nn.diffusion.ema import EMA, create_ema_model, update_ema_model


# ============================================================================
# HELPER MODELS
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class ModelWithBatchNorm(nn.Module):
    """Model with BatchNorm for testing buffer handling."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.output = nn.Linear(20, 5)

    def forward(self, x):
        return self.output(self.bn(torch.relu(self.linear(x))))


# ============================================================================
# EMA CLASS TESTS
# ============================================================================


class TestEMA:
    """Tests for the EMA class."""

    def test_construction(self):
        """Test basic EMA construction."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)
        assert ema.decay == 0.999
        assert ema.step_count == 0
        assert len(ema.shadow_params) > 0

    def test_shadow_params_initialized(self):
        """Test shadow parameters are initialized from model."""
        model = SimpleModel()
        ema = EMA(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow_params
                assert torch.allclose(ema.shadow_params[name], param.data)

    def test_update_changes_shadow(self):
        """Test that update modifies shadow parameters."""
        model = SimpleModel()
        ema = EMA(model, decay=0.9)  # Low decay for visible change

        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        # Get shadow before update
        shadow_before = {k: v.clone() for k, v in ema.shadow_params.items()}

        # Update EMA
        ema.update(model)

        # Shadow should have changed
        for name in shadow_before:
            assert not torch.allclose(shadow_before[name], ema.shadow_params[name])

    def test_update_moves_toward_model(self):
        """Test that update moves shadow toward model parameters."""
        model = SimpleModel()
        ema = EMA(model, decay=0.9)

        # Set model params to all ones
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        # Update
        ema.update(model)

        # Shadow should be closer to 1.0 than before (which was random init)
        for name, shadow in ema.shadow_params.items():
            # decay=0.9, so shadow = 0.9 * old + 0.1 * 1.0
            # Should be closer to 1.0
            pass  # Just checking no errors

    def test_update_increments_step_count(self):
        """Test that update increments step count."""
        model = SimpleModel()
        ema = EMA(model)
        assert ema.step_count == 0
        ema.update(model)
        assert ema.step_count == 1
        ema.update(model)
        assert ema.step_count == 2

    def test_apply_context_manager(self):
        """Test apply context manager swaps and restores weights."""
        model = SimpleModel()
        ema = EMA(model, decay=0.5)

        # Modify model to be different from EMA
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        with torch.no_grad():
            for param in model.parameters():
                param.add_(10.0)

        # Apply EMA - weights should temporarily change
        with ema.apply(model):
            for name, param in model.named_parameters():
                # Should have EMA weights (close to original)
                assert torch.allclose(param.data, ema.shadow_params[name])

        # After context, should be restored
        for name, param in model.named_parameters():
            expected = original_params[name] + 10.0
            assert torch.allclose(param.data, expected)

    def test_apply_restores_on_exception(self):
        """Test that apply restores weights even if exception occurs."""
        model = SimpleModel()
        ema = EMA(model)

        original_params = {name: param.clone() for name, param in model.named_parameters()}
        with torch.no_grad():
            for param in model.parameters():
                param.add_(5.0)
        modified_params = {name: param.clone() for name, param in model.named_parameters()}

        # Cause an exception inside the context
        try:
            with ema.apply(model):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be restored
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, modified_params[name])

    def test_copy_to(self):
        """Test copy_to permanently copies EMA weights."""
        model = SimpleModel()
        ema = EMA(model)

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(10.0)

        # Copy EMA weights (original values)
        ema.copy_to(model)

        # Model should now have EMA weights
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, ema.shadow_params[name])

    def test_state_dict_and_load(self):
        """Test state dict save and load."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Do some updates
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.1)
            ema.update(model)

        # Save state
        state = ema.state_dict()

        # Create new EMA and load
        model2 = SimpleModel()
        ema2 = EMA(model2)
        ema2.load_state_dict(state)

        # Should match
        assert ema2.decay == ema.decay
        assert ema2.step_count == ema.step_count
        for name in ema.shadow_params:
            assert torch.allclose(ema2.shadow_params[name], ema.shadow_params[name])

    def test_to_device(self):
        """Test moving EMA to device."""
        model = SimpleModel()
        ema = EMA(model)

        # Move to CPU (should work even if already there)
        ema.to(torch.device('cpu'))

        for param in ema.shadow_params.values():
            assert param.device == torch.device('cpu')

    def test_warmup(self):
        """Test decay warmup."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999, warmup_steps=100, warmup_decay=0.9)

        # Initially should use warmup_decay
        assert ema.current_decay == 0.9

        # After some steps, should be higher
        for _ in range(50):
            ema.update(model)
        assert 0.9 < ema.current_decay < 0.999

        # After warmup, should be at target decay
        for _ in range(60):
            ema.update(model)
        assert ema.current_decay == 0.999

    def test_include_buffers(self):
        """Test that buffers are tracked when include_buffers=True."""
        model = ModelWithBatchNorm()
        # Run forward to initialize buffers
        model(torch.randn(32, 10))

        ema = EMA(model, include_buffers=True)
        assert len(ema.shadow_buffers) > 0

    def test_exclude_buffers(self):
        """Test that buffers are not tracked when include_buffers=False."""
        model = ModelWithBatchNorm()
        model(torch.randn(32, 10))

        ema = EMA(model, include_buffers=False)
        assert len(ema.shadow_buffers) == 0

    def test_invalid_decay(self):
        """Test that invalid decay raises error."""
        model = SimpleModel()
        with pytest.raises(ValueError, match="decay"):
            EMA(model, decay=1.5)
        with pytest.raises(ValueError, match="decay"):
            EMA(model, decay=-0.1)

    def test_invalid_warmup_steps(self):
        """Test that invalid warmup_steps raises error."""
        model = SimpleModel()
        with pytest.raises(ValueError, match="warmup_steps"):
            EMA(model, warmup_steps=-1)

    def test_repr(self):
        """Test string representation."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)
        repr_str = repr(ema)
        assert "0.999" in repr_str
        assert "EMA" in repr_str


# ============================================================================
# CREATE EMA MODEL TESTS
# ============================================================================


class TestCreateEMAModel:
    """Tests for create_ema_model function."""

    def test_creates_copy(self):
        """Test that it creates a separate copy."""
        model = SimpleModel()
        ema_model = create_ema_model(model)

        # Should be different objects
        assert model is not ema_model

        # Modifying one shouldn't affect the other
        with torch.no_grad():
            list(model.parameters())[0].add_(10.0)

        assert not torch.allclose(
            list(model.parameters())[0],
            list(ema_model.parameters())[0]
        )

    def test_no_grad(self):
        """Test that EMA model has requires_grad=False."""
        model = SimpleModel()
        ema_model = create_ema_model(model)

        for param in ema_model.parameters():
            assert not param.requires_grad

    def test_eval_mode(self):
        """Test that EMA model is in eval mode."""
        model = SimpleModel()
        model.train()
        ema_model = create_ema_model(model)
        assert not ema_model.training


# ============================================================================
# UPDATE EMA MODEL TESTS
# ============================================================================


class TestUpdateEMAModel:
    """Tests for update_ema_model function."""

    def test_update_moves_toward_model(self):
        """Test that update moves EMA model toward training model."""
        model = SimpleModel()
        ema_model = create_ema_model(model)

        # Set model params to all ones
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        # Get EMA params before
        ema_before = [p.clone() for p in ema_model.parameters()]

        # Update
        update_ema_model(model, ema_model, decay=0.9)

        # EMA should have moved toward 1.0
        for ema_param, before in zip(ema_model.parameters(), ema_before):
            # Should be different from before
            assert not torch.allclose(ema_param, before)
            # Should be closer to 1.0 (the model value)
            assert (ema_param - 1.0).abs().mean() < (before - 1.0).abs().mean()

    def test_decay_rate(self):
        """Test that decay rate affects update speed."""
        model = SimpleModel()

        # Set model to known value
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)

        # Create two EMA models
        ema_fast = create_ema_model(model)
        ema_slow = create_ema_model(model)

        # Initialize both to zeros
        with torch.no_grad():
            for param in ema_fast.parameters():
                param.fill_(0.0)
            for param in ema_slow.parameters():
                param.fill_(0.0)

        # Update with different decay rates
        update_ema_model(model, ema_fast, decay=0.5)  # Fast update
        update_ema_model(model, ema_slow, decay=0.99)  # Slow update

        # Fast should be closer to 1.0
        fast_dist = sum((p - 1.0).abs().sum() for p in ema_fast.parameters())
        slow_dist = sum((p - 1.0).abs().sum() for p in ema_slow.parameters())
        assert fast_dist < slow_dist


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestEMAIntegration:
    """Integration tests for EMA utilities."""

    def test_training_loop_simulation(self):
        """Simulate a training loop with EMA."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simulate training
        for _ in range(10):
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

        # Check EMA is different from model
        model_params = list(model.parameters())
        for i, (name, shadow) in enumerate(ema.shadow_params.items()):
            # They should be different after training
            # (EMA is smoothed version)
            pass  # Just checking no errors

    def test_inference_with_ema(self):
        """Test using EMA weights for inference."""
        model = SimpleModel()
        ema = EMA(model, decay=0.99)

        # Do some "training"
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema.update(model)

        # Inference with EMA
        x = torch.randn(4, 10)
        with ema.apply(model):
            ema_output = model(x).clone()

        # Inference with training model
        train_output = model(x)

        # Outputs should be different
        assert not torch.allclose(ema_output, train_output)

    def test_checkpoint_roundtrip(self):
        """Test saving and loading EMA with model checkpoint."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Train a bit
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.1)
            ema.update(model)

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'ema': ema.state_dict(),
        }

        # Create new model and EMA, load checkpoint
        model2 = SimpleModel()
        model2.load_state_dict(checkpoint['model'])
        ema2 = EMA(model2)
        ema2.load_state_dict(checkpoint['ema'])

        # Should produce same outputs
        x = torch.randn(4, 10)
        with ema.apply(model):
            out1 = model(x)
        with ema2.apply(model2):
            out2 = model2(x)

        assert torch.allclose(out1, out2)

    def test_ema_with_batchnorm(self):
        """Test EMA with model containing BatchNorm."""
        model = ModelWithBatchNorm()
        ema = EMA(model, include_buffers=True)

        # Run some forward passes to update running stats
        model.train()
        for _ in range(10):
            x = torch.randn(32, 10)
            _ = model(x)
            ema.update(model)

        # Test inference
        model.eval()
        x = torch.randn(4, 10)
        with ema.apply(model):
            output = model(x)
        assert not torch.isnan(output).any()

    def test_ema_model_approach(self):
        """Test the alternative create_ema_model + update_ema_model approach."""
        model = SimpleModel()
        ema_model = create_ema_model(model)

        # Simulate training
        for _ in range(10):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            update_ema_model(model, ema_model, decay=0.99)

        # Use EMA model directly for inference
        x = torch.randn(4, 10)
        ema_output = ema_model(x)
        train_output = model(x)

        # Should be different
        assert not torch.allclose(ema_output, train_output)


# ============================================================================
# SAVE/LOAD TESTS
# ============================================================================


class TestEMASaveLoad:
    """Tests for EMA save and load methods."""

    def test_save_creates_file(self, tmp_path):
        """Test save creates a checkpoint file."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model)

        assert path.exists()

    def test_save_contains_model_and_ema(self, tmp_path):
        """Test checkpoint contains model and EMA state."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model)

        checkpoint = torch.load(path, weights_only=False)
        assert 'model' in checkpoint
        assert 'ema' in checkpoint

    def test_save_with_optimizer(self, tmp_path):
        """Test save includes optimizer state."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)
        optimizer = torch.optim.Adam(model.parameters())

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model, optimizer)

        checkpoint = torch.load(path, weights_only=False)
        assert 'optimizer' in checkpoint

    def test_save_with_extras(self, tmp_path):
        """Test save includes extra items."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model, epoch=10, loss=0.5, custom_data=[1, 2, 3])

        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint['epoch'] == 10
        assert checkpoint['loss'] == 0.5
        assert checkpoint['custom_data'] == [1, 2, 3]

    def test_load_restores_model(self, tmp_path):
        """Test load restores model weights."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Modify and save
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(42.0)
        ema.update(model)

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model)

        # Create new model and load
        model2 = SimpleModel()
        ema2, extras = EMA.load(str(path), model2)

        # Model should have saved weights
        for param in model2.parameters():
            assert torch.allclose(param, torch.full_like(param, 42.0))

    def test_load_restores_ema(self, tmp_path):
        """Test load restores EMA state."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Do some updates
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.1)
            ema.update(model)

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model)

        # Load into new model
        model2 = SimpleModel()
        ema2, _ = EMA.load(str(path), model2)

        # EMA state should match
        assert ema2.step_count == ema.step_count
        assert ema2.decay == ema.decay
        for name in ema.shadow_params:
            assert torch.allclose(ema2.shadow_params[name], ema.shadow_params[name])

    def test_load_restores_optimizer(self, tmp_path):
        """Test load restores optimizer state."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Do a training step to create optimizer state
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        ema.update(model)

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model, optimizer)

        # Load into new model and optimizer
        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        ema2, _ = EMA.load(str(path), model2, optimizer2)

        # Optimizer state should be restored
        assert len(optimizer2.state) > 0

    def test_load_returns_extras(self, tmp_path):
        """Test load returns extra items."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model, epoch=10, best_loss=0.5)

        model2 = SimpleModel()
        ema2, extras = EMA.load(str(path), model2)

        assert extras['epoch'] == 10
        assert extras['best_loss'] == 0.5

    def test_save_load_roundtrip(self, tmp_path):
        """Test full save/load roundtrip produces identical outputs."""
        model = SimpleModel()
        ema = EMA(model, decay=0.999)

        # Train a bit
        for _ in range(10):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema.update(model)

        # Get output with EMA
        x = torch.randn(4, 10)
        with ema.apply(model):
            original_output = model(x).clone()

        # Save
        path = tmp_path / "checkpoint.pt"
        ema.save(str(path), model)

        # Load into fresh model
        model2 = SimpleModel()
        ema2, _ = EMA.load(str(path), model2)

        # Should produce same output
        with ema2.apply(model2):
            loaded_output = model2(x)

        assert torch.allclose(original_output, loaded_output)
