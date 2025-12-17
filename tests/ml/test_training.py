"""Tests for ciffy.nn.training module."""

import pytest
import random
import tempfile
from pathlib import Path

from tests.utils import TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_reproducibility(self):
        """Verify set_seed produces reproducible results."""
        import torch
        from ciffy.nn.training import set_seed

        # Set seed and generate random values
        set_seed(42)
        vals1 = [random.random() for _ in range(5)]
        torch_vals1 = torch.rand(5).tolist()

        # Reset seed and generate again
        set_seed(42)
        vals2 = [random.random() for _ in range(5)]
        torch_vals2 = torch.rand(5).tolist()

        assert vals1 == vals2
        assert torch_vals1 == torch_vals2

    def test_set_seed_different_seeds(self):
        """Different seeds produce different values."""
        import torch
        from ciffy.nn.training import set_seed

        set_seed(42)
        vals1 = torch.rand(5).tolist()

        set_seed(123)
        vals2 = torch.rand(5).tolist()

        assert vals1 != vals2


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


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainEpoch:
    """Tests for train_epoch function."""

    def test_train_epoch_basic(self):
        """Basic train_epoch execution."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from ciffy.nn.training import train_epoch

        # Simple model
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simple dataset
        X = torch.randn(20, 10)
        y = torch.randn(20, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4)

        # Loss function
        def loss_fn(model, batch):
            x, y = batch
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            return {"loss": loss}

        # Train
        metrics = train_epoch(model, loader, loss_fn, optimizer, progress_bar=False)

        assert "loss" in metrics
        assert "n_samples" in metrics
        assert "n_skipped" in metrics
        assert metrics["n_samples"] > 0
        assert isinstance(metrics["loss"], float)

    def test_train_epoch_with_none_samples(self):
        """train_epoch handles None samples correctly."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from ciffy.nn.training import train_epoch, polymer_collate_fn

        # Simple model
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Dataset that returns None for some items
        class NoneDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                if idx % 2 == 0:
                    return None
                return torch.randn(10)

        dataset = NoneDataset()
        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=polymer_collate_fn,
        )

        # Loss function
        def loss_fn(model, x):
            pred = model(x.unsqueeze(0))
            return {"loss": pred.sum()}

        # Train
        metrics = train_epoch(model, loader, loss_fn, optimizer, progress_bar=False)

        # Should have processed 5 samples (odd indices)
        assert metrics["n_samples"] == 5
        # Should have skipped 5 samples (even indices, None)
        assert metrics["n_skipped"] == 5

    def test_train_epoch_with_nan_loss(self):
        """train_epoch skips samples with NaN loss."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from ciffy.nn.training import train_epoch

        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        X = torch.randn(10, 10)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=1)

        # Loss function that returns NaN for half the samples
        call_count = [0]

        def loss_fn(model, batch):
            x = batch[0]
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return {"loss": torch.tensor(float("nan"))}
            return {"loss": model(x).sum()}

        metrics = train_epoch(model, loader, loss_fn, optimizer, progress_bar=False)

        # Half should be skipped due to NaN
        assert metrics["n_skipped"] == 5
        assert metrics["n_samples"] == 5

    def test_train_epoch_with_grad_clip(self):
        """train_epoch applies gradient clipping."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from ciffy.nn.training import train_epoch

        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Large values to create large gradients
        X = torch.randn(10, 10) * 1000
        y = torch.randn(10, 1) * 1000
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)

        def loss_fn(model, batch):
            x, y = batch
            pred = model(x)
            return {"loss": nn.functional.mse_loss(pred, y)}

        # Should not raise with grad clipping
        metrics = train_epoch(
            model, loader, loss_fn, optimizer,
            grad_clip=1.0,
            progress_bar=False,
        )

        assert metrics["n_samples"] > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCollateFn:
    """Tests for polymer_collate_fn."""

    def test_polymer_collate_fn_single_item(self):
        """collate_fn returns single item for batch_size=1."""
        from ciffy.nn.training import polymer_collate_fn

        batch = ["item"]
        result = polymer_collate_fn(batch)
        assert result == "item"

    def test_polymer_collate_fn_filters_none(self):
        """collate_fn filters None values."""
        from ciffy.nn.training import polymer_collate_fn

        batch = [None, "item", None]
        result = polymer_collate_fn(batch)
        assert result == "item"

    def test_polymer_collate_fn_all_none(self):
        """collate_fn returns None if all items are None."""
        from ciffy.nn.training import polymer_collate_fn

        batch = [None, None]
        result = polymer_collate_fn(batch)
        assert result is None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWorkerInitFn:
    """Tests for get_worker_init_fn."""

    def test_worker_init_fn_with_seed(self):
        """Worker init function sets reproducible seed."""
        import torch
        from ciffy.nn.training import get_worker_init_fn

        init_fn = get_worker_init_fn(base_seed=42)

        # Call for worker 0
        init_fn(0)
        vals1 = [random.random() for _ in range(3)]
        torch_vals1 = torch.rand(3).tolist()

        # Call again for worker 0 (same seed)
        init_fn(0)
        vals2 = [random.random() for _ in range(3)]
        torch_vals2 = torch.rand(3).tolist()

        assert vals1 == vals2
        assert torch_vals1 == torch_vals2

    def test_worker_init_fn_different_workers(self):
        """Different workers get different seeds."""
        import torch
        from ciffy.nn.training import get_worker_init_fn

        init_fn = get_worker_init_fn(base_seed=42)

        init_fn(0)
        vals0 = torch.rand(5).tolist()

        init_fn(1)
        vals1 = torch.rand(5).tolist()

        assert vals0 != vals1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBetaScheduler:
    """Tests for BetaScheduler class."""

    def test_constant_schedule(self):
        """Constant schedule returns target_beta for all epochs."""
        from ciffy.nn.training import BetaScheduler

        scheduler = BetaScheduler(schedule="constant", target_beta=0.5)

        assert scheduler.get_beta(0) == 0.5
        assert scheduler.get_beta(50) == 0.5
        assert scheduler.get_beta(100) == 0.5

    def test_linear_schedule(self):
        """Linear schedule increases from 0 to target_beta."""
        from ciffy.nn.training import BetaScheduler

        scheduler = BetaScheduler(
            schedule="linear",
            target_beta=1.0,
            warmup_epochs=10,
            start_beta=0.0,
        )

        # Start at 0
        assert scheduler.get_beta(0) == 0.0

        # Midpoint
        assert scheduler.get_beta(5) == pytest.approx(0.5, rel=0.01)

        # At warmup end
        assert scheduler.get_beta(10) == 1.0

        # After warmup, stays at target
        assert scheduler.get_beta(50) == 1.0

    def test_linear_schedule_with_start_beta(self):
        """Linear schedule respects start_beta."""
        from ciffy.nn.training import BetaScheduler

        scheduler = BetaScheduler(
            schedule="linear",
            target_beta=1.0,
            warmup_epochs=10,
            start_beta=0.5,
        )

        # Start at 0.5
        assert scheduler.get_beta(0) == 0.5

        # Midpoint between 0.5 and 1.0
        assert scheduler.get_beta(5) == pytest.approx(0.75, rel=0.01)

        # At warmup end
        assert scheduler.get_beta(10) == 1.0

    def test_cosine_schedule(self):
        """Cosine schedule follows cosine curve."""
        from ciffy.nn.training import BetaScheduler
        import math

        scheduler = BetaScheduler(
            schedule="cosine",
            target_beta=1.0,
            warmup_epochs=10,
            start_beta=0.0,
        )

        # Start at 0
        assert scheduler.get_beta(0) == 0.0

        # At warmup end
        assert scheduler.get_beta(10) == 1.0

        # After warmup, stays at target
        assert scheduler.get_beta(50) == 1.0

        # Midpoint should be less than linear (cosine starts slow)
        midpoint = scheduler.get_beta(5)
        linear_midpoint = 0.5
        assert midpoint == pytest.approx(0.5, rel=0.01)  # Cosine midpoint is also 0.5

    def test_cyclical_schedule(self):
        """Cyclical schedule repeats warmup cycles."""
        from ciffy.nn.training import BetaScheduler

        scheduler = BetaScheduler(
            schedule="cyclical",
            target_beta=1.0,
            total_epochs=100,
            n_cycles=4,
            start_beta=0.0,
        )

        # Each cycle is 25 epochs, warmup is first half (12.5 epochs)
        # Start of cycle 1
        assert scheduler.get_beta(0) == 0.0

        # End of warmup in cycle 1 (~12.5 epochs)
        assert scheduler.get_beta(12) == pytest.approx(1.0, rel=0.1)

        # Start of cycle 2 (epoch 25)
        assert scheduler.get_beta(25) == pytest.approx(0.0, rel=0.1)

    def test_invalid_schedule(self):
        """Invalid schedule raises ValueError."""
        from ciffy.nn.training import BetaScheduler

        with pytest.raises(ValueError, match="schedule must be one of"):
            BetaScheduler(schedule="invalid")

    def test_repr(self):
        """repr returns useful string."""
        from ciffy.nn.training import BetaScheduler

        scheduler = BetaScheduler(
            schedule="linear",
            target_beta=1.0,
            warmup_epochs=50,
        )

        repr_str = repr(scheduler)
        assert "linear" in repr_str
        assert "1.0" in repr_str
        assert "50" in repr_str
