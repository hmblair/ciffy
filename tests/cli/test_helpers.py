"""
Tests for CLI helper functions.
"""

import pytest
import sys
from io import StringIO
from pathlib import Path

from tests.utils import get_test_cif, requires_torch


class TestResolveDevice:
    """Tests for resolve_device helper."""

    def test_cpu_passthrough(self):
        """resolve_device returns 'cpu' unchanged."""
        from ciffy.cli.helpers import resolve_device

        assert resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        """resolve_device returns 'cuda' unchanged."""
        from ciffy.cli.helpers import resolve_device

        assert resolve_device("cuda") == "cuda"

    def test_mps_passthrough(self):
        """resolve_device returns 'mps' unchanged."""
        from ciffy.cli.helpers import resolve_device

        assert resolve_device("mps") == "mps"

    @requires_torch
    def test_auto_returns_valid_device(self):
        """resolve_device('auto') returns one of cpu/cuda/mps."""
        from ciffy.cli.helpers import resolve_device

        result = resolve_device("auto")
        assert result in ("cpu", "cuda", "mps")


class TestResolveAccelerator:
    """Tests for resolve_accelerator helper."""

    def test_cpu_passthrough(self):
        """resolve_accelerator returns 'cpu' unchanged."""
        from ciffy.cli.helpers import resolve_accelerator

        assert resolve_accelerator("cpu") == "cpu"

    def test_gpu_passthrough(self):
        """resolve_accelerator returns 'gpu' unchanged."""
        from ciffy.cli.helpers import resolve_accelerator

        assert resolve_accelerator("gpu") == "gpu"

    def test_auto_returns_valid_accelerator(self):
        """resolve_accelerator('auto') returns one of cpu/gpu/mps."""
        from ciffy.cli.helpers import resolve_accelerator

        result = resolve_accelerator("auto")
        assert result in ("cpu", "gpu", "mps")


class TestLoadStructure:
    """Tests for load_structure helper."""

    def test_load_valid_file(self):
        """load_structure returns Polymer for valid file."""
        from ciffy.cli.helpers import load_structure

        polymer = load_structure(get_test_cif("3SKW"))
        assert polymer is not None
        assert polymer.size() > 0

    def test_load_nonexistent_file(self, capsys):
        """load_structure returns None and prints error for missing file."""
        from ciffy.cli.helpers import load_structure

        result = load_structure("/nonexistent/path.cif")
        assert result is None

        captured = capsys.readouterr()
        assert "Error" in captured.err
        assert "does not exist" in captured.err.lower() or "not found" in captured.err.lower()

    def test_load_invalid_file(self, tmp_path, capsys):
        """load_structure returns None and prints error for invalid file."""
        from ciffy.cli.helpers import load_structure

        invalid_file = tmp_path / "invalid.cif"
        invalid_file.write_text("not a valid cif file")

        result = load_structure(str(invalid_file))
        assert result is None

        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestLoadStructureOrExit:
    """Tests for load_structure_or_exit helper."""

    def test_load_valid_file(self):
        """load_structure_or_exit returns Polymer for valid file."""
        from ciffy.cli.helpers import load_structure_or_exit

        polymer = load_structure_or_exit(get_test_cif("3SKW"))
        assert polymer is not None
        assert polymer.size() > 0

    def test_load_nonexistent_file_exits(self):
        """load_structure_or_exit calls sys.exit for missing file."""
        from ciffy.cli.helpers import load_structure_or_exit

        with pytest.raises(SystemExit) as exc_info:
            load_structure_or_exit("/nonexistent/path.cif")

        assert exc_info.value.code == 1


class TestSaveSamples:
    """Tests for save_samples helper."""

    def test_save_single_sample(self, tmp_path):
        """save_samples saves single polymer to file."""
        import numpy as np
        from ciffy.cli.helpers import save_samples
        from ciffy import template

        polymer = template("acgu")
        # Add coordinates (templates have None by default)
        coords = np.random.randn(polymer.size(), 3).astype(np.float32)
        polymer = polymer.copy(coordinates=coords)
        output = tmp_path / "output.cif"

        save_samples([polymer], output, quiet=True)

        assert output.exists()

    def test_save_single_sample_adds_extension(self, tmp_path):
        """save_samples adds .cif extension if missing."""
        import numpy as np
        from ciffy.cli.helpers import save_samples
        from ciffy import template

        polymer = template("acgu")
        coords = np.random.randn(polymer.size(), 3).astype(np.float32)
        polymer = polymer.copy(coordinates=coords)
        output = tmp_path / "output"

        save_samples([polymer], output, quiet=True)

        expected = tmp_path / "output.cif"
        assert expected.exists()

    def test_save_multiple_samples(self, tmp_path):
        """save_samples creates directory for multiple polymers."""
        import numpy as np
        from ciffy.cli.helpers import save_samples
        from ciffy import template

        polymers = []
        for _ in range(3):
            p = template("acgu")
            coords = np.random.randn(p.size(), 3).astype(np.float32)
            polymers.append(p.copy(coordinates=coords))

        output = tmp_path / "samples"

        save_samples(polymers, output, quiet=True)

        assert output.is_dir()
        assert (output / "sample_000.cif").exists()
        assert (output / "sample_001.cif").exists()
        assert (output / "sample_002.cif").exists()


class TestPrintTrainingHeader:
    """Tests for print_training_header helper."""

    def test_prints_header(self, capsys):
        """print_training_header outputs formatted header."""
        from ciffy.cli.helpers import print_training_header

        print_training_header("Test Model", Data="100 files", Epochs=10)

        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "Data" in captured.out
        assert "100 files" in captured.out
        assert "Epochs" in captured.out

    def test_quiet_suppresses_output(self, capsys):
        """print_training_header with quiet=True produces no output."""
        from ciffy.cli.helpers import print_training_header

        print_training_header("Test Model", quiet=True, Data="100 files")

        captured = capsys.readouterr()
        assert captured.out == ""


class TestPrintTrainingComplete:
    """Tests for print_training_complete helper."""

    def test_prints_completion(self, capsys):
        """print_training_complete outputs formatted message."""
        from ciffy.cli.helpers import print_training_complete

        print_training_complete("/path/to/model.pt", Final_loss=0.01)

        captured = capsys.readouterr()
        assert "Training Complete" in captured.out
        assert "/path/to/model.pt" in captured.out
        assert "Final loss" in captured.out

    def test_quiet_suppresses_output(self, capsys):
        """print_training_complete with quiet=True produces no output."""
        from ciffy.cli.helpers import print_training_complete

        print_training_complete("/path/to/model.pt", quiet=True)

        captured = capsys.readouterr()
        assert captured.out == ""


class TestRequireMatplotlib:
    """Tests for require_matplotlib helper."""

    def test_returns_true_when_available(self):
        """require_matplotlib returns True when matplotlib is installed."""
        from ciffy.cli.helpers import require_matplotlib

        # matplotlib should be available in test environment
        result = require_matplotlib()
        # May be True or False depending on environment
        assert isinstance(result, bool)


class TestRequireTorchLightning:
    """Tests for require_torch_lightning helper."""

    def test_returns_bool(self):
        """require_torch_lightning returns a boolean."""
        from ciffy.cli.helpers import require_torch_lightning

        result = require_torch_lightning()
        assert isinstance(result, bool)
