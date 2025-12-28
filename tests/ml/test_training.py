"""Tests for ciffy.nn.training module."""

import pytest
import tempfile
from pathlib import Path

from tests.utils import TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_cpu(self):
        """Test CPU device selection."""
        import torch
        from ciffy.nn.training import get_device

        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_get_device_auto(self):
        """Test auto device selection returns valid device."""
        import torch
        from ciffy.nn.training import get_device

        device = get_device("auto")
        assert isinstance(device, torch.device)
        # Should be one of: cuda, mps, or cpu
        assert device.type in ("cuda", "mps", "cpu")

    def test_get_device_invalid(self):
        """Test invalid device raises error."""
        from ciffy.nn.training import get_device

        with pytest.raises(RuntimeError):
            get_device("invalid_device_xyz")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCheckpoint:
    """Tests for save_checkpoint and load_checkpoint functions."""

    def test_save_load_checkpoint_roundtrip(self):
        """Checkpoint save/load preserves state."""
        import torch
        import torch.nn as nn
        from ciffy.nn.training import save_checkpoint, load_checkpoint

        # Create simple model
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Do a forward/backward pass to initialize optimizer state
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Save original state
        original_params = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"

            # Save checkpoint
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=5,
                metrics={"loss": 0.5},
                config={"lr": 0.001},
                custom_key="custom_value",
            )

            assert path.exists()

            # Modify model
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(999.0)

            # Create new model and load
            model2 = nn.Linear(10, 5)
            optimizer2 = torch.optim.Adam(model2.parameters())

            ckpt = load_checkpoint(path, model2, optimizer2)

            # Verify loaded state matches original
            for k, v in model2.state_dict().items():
                assert torch.allclose(v, original_params[k])

            # Verify metadata
            assert ckpt["epoch"] == 5
            assert ckpt["metrics"]["loss"] == 0.5
            assert ckpt["config"]["lr"] == 0.001
            assert ckpt["custom_key"] == "custom_value"

    def test_save_checkpoint_creates_dirs(self):
        """save_checkpoint creates parent directories."""
        import torch.nn as nn
        from ciffy.nn.training import save_checkpoint

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dirs" / "test.pt"

            save_checkpoint(path, model)

            assert path.exists()

    def test_load_checkpoint_not_found(self):
        """load_checkpoint raises FileNotFoundError for missing file."""
        import torch.nn as nn
        from ciffy.nn.training import load_checkpoint

        model = nn.Linear(10, 5)

        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.pt", model)
