"""Tests for ciffy.nn.config training utilities."""

import pytest

from tests.utils import TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_cpu(self):
        """Test CPU device selection."""
        from ciffy.nn.config import get_device

        device = get_device("cpu")
        assert device == "cpu"

    def test_get_device_auto(self):
        """Test auto device selection returns valid device."""
        from ciffy.nn.config import get_device

        device = get_device("auto")
        # Should be one of: cuda, mps, or cpu
        assert device in ("cuda", "mps", "cpu")

    def test_get_device_passthrough(self):
        """Test explicit device is passed through."""
        from ciffy.nn.config import get_device

        assert get_device("cuda:0") == "cuda:0"
        assert get_device("mps") == "mps"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLightningImports:
    """Tests that Lightning components can be imported."""

    def test_import_modules(self):
        """Test LightningModule imports."""
        from ciffy.nn.lightning import (
            BaseCiffyModule,
            LatentDiffusionModule,
            CoordinateDiffusionModule,
        )

    def test_import_data_modules(self):
        """Test LightningDataModule imports."""
        from ciffy.nn.lightning import (
            LatentDiffusionDataModule,
            CoordinateDiffusionDataModule,
        )

    def test_import_callbacks(self):
        """Test Callback imports."""
        from ciffy.nn.lightning import (
            EMACallback,
            SampleGenerationCallback,
        )

    def test_import_configs(self):
        """Test config imports."""
        from ciffy.nn.lightning import (
            LatentDiffusionFullConfig,
            CoordinateDiffusionFullConfig,
        )
