"""
Tests for CLI commands.

Tests the CLI commands by calling their handler functions directly.
"""

import argparse
import pytest
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.utils import get_test_cif


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_single_file(self, capsys):
        """info command prints polymer summary for single file."""
        from ciffy.cli.__main__ import _info_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            atoms=False,
            sequence=False,
            desc=False,
        )

        _info_command(args)

        captured = capsys.readouterr()
        assert "3SKW" in captured.out

    def test_info_with_sequence(self, capsys):
        """info command with --sequence shows sequence."""
        from ciffy.cli.__main__ import _info_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            atoms=False,
            sequence=True,
            desc=False,
        )

        _info_command(args)

        captured = capsys.readouterr()
        assert "Sequence" in captured.out

    def test_info_with_atoms(self, capsys):
        """info command with --atoms shows atom counts."""
        from ciffy.cli.__main__ import _info_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            atoms=True,
            sequence=False,
            desc=False,
        )

        _info_command(args)

        captured = capsys.readouterr()
        assert "Atoms per residue" in captured.out

    def test_info_nonexistent_file(self, capsys):
        """info command handles nonexistent file gracefully."""
        from ciffy.cli.__main__ import _info_command

        args = argparse.Namespace(
            files=["/nonexistent/file.cif"],
            atoms=False,
            sequence=False,
            desc=False,
        )

        _info_command(args)

        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_info_multiple_files(self, capsys):
        """info command handles multiple files."""
        from ciffy.cli.__main__ import _info_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW"), get_test_cif("1ZEW")],
            atoms=False,
            sequence=False,
            desc=False,
        )

        _info_command(args)

        captured = capsys.readouterr()
        # Should show table output for multiple files
        assert "3SKW" in captured.out or "1ZEW" in captured.out


class TestGeometryCommand:
    """Tests for the geometry command."""

    def test_geometry_basic(self, capsys):
        """geometry command computes Rg and clashes."""
        from ciffy.cli.__main__ import _geometry_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            per_chain=False,
        )

        _geometry_command(args)

        captured = capsys.readouterr()
        assert "Radius of gyration" in captured.out
        assert "Clashes" in captured.out

    def test_geometry_per_chain(self, capsys):
        """geometry command with --per-chain shows per-chain Rg."""
        from ciffy.cli.__main__ import _geometry_command

        # Use a multi-chain structure
        args = argparse.Namespace(
            files=[get_test_cif("9GCM")],
            per_chain=True,
        )

        _geometry_command(args)

        captured = capsys.readouterr()
        assert "Per-chain Rg" in captured.out

    def test_geometry_nonexistent_file(self, capsys):
        """geometry command handles nonexistent file gracefully."""
        from ciffy.cli.__main__ import _geometry_command

        args = argparse.Namespace(
            files=["/nonexistent/file.cif"],
            per_chain=False,
        )

        _geometry_command(args)

        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestSplitCommand:
    """Tests for the split command."""

    def test_split_creates_chain_files(self, tmp_path):
        """split command creates separate files for each chain."""
        from ciffy.cli.__main__ import _split_command

        args = argparse.Namespace(
            file=get_test_cif("9GCM"),
            output=str(tmp_path),
        )

        _split_command(args)

        # Check that files were created
        cif_files = list(tmp_path.glob("*.cif"))
        assert len(cif_files) > 0

    def test_split_prints_output(self, tmp_path, capsys):
        """split command prints output paths."""
        from ciffy.cli.__main__ import _split_command

        args = argparse.Namespace(
            file=get_test_cif("3SKW"),
            output=str(tmp_path),
        )

        _split_command(args)

        captured = capsys.readouterr()
        assert "Wrote" in captured.out
        assert "Split into" in captured.out

    def test_split_nonexistent_file(self):
        """split command exits for nonexistent file."""
        from ciffy.cli.__main__ import _split_command

        args = argparse.Namespace(
            file="/nonexistent/file.cif",
            output=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            _split_command(args)

        assert exc_info.value.code == 1


class TestMapCommand:
    """Tests for the map command."""

    def test_map_saves_to_file(self, tmp_path):
        """map command saves figure to file."""
        pytest.importorskip("matplotlib")
        from ciffy.cli.__main__ import _map_command
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        output_path = tmp_path / "map.png"
        args = argparse.Namespace(
            file=get_test_cif("3SKW"),
            scale="residue",
            power=2.0,
            chain=None,
            cmap="RdPu",
            output=str(output_path),
            dpi=150,
        )

        _map_command(args)

        assert output_path.exists()

    def test_map_nonexistent_file(self):
        """map command exits for nonexistent file."""
        pytest.importorskip("matplotlib")
        from ciffy.cli.__main__ import _map_command

        args = argparse.Namespace(
            file="/nonexistent/file.cif",
            scale="residue",
            power=2.0,
            chain=None,
            cmap="RdPu",
            output=None,
            dpi=150,
        )

        with pytest.raises(SystemExit) as exc_info:
            _map_command(args)

        assert exc_info.value.code == 1


class TestDownloadCommand:
    """Tests for the download command."""

    def test_download_list_presets(self, capsys):
        """download --list-presets shows available presets."""
        from ciffy.cli.__main__ import _download_command

        args = argparse.Namespace(
            pdb_ids=[],
            preset=None,
            type=None,
            output_dir=".",
            max_count=None,
            max_resolution=None,
            min_resolution=None,
            min_length=None,
            max_length=None,
            method=None,
            released_after=None,
            released_before=None,
            overwrite=False,
            max_workers=4,
            search_only=False,
            list_ids=False,
            list_presets=True,
            quiet=False,
        )

        _download_command(args)

        captured = capsys.readouterr()
        # Should list presets without error
        # Output depends on download_cli implementation


class TestMainEntrypoint:
    """Tests for the main CLI entrypoint."""

    def test_main_no_args_shows_help(self, capsys):
        """main() with no command shows help."""
        from ciffy.cli.__main__ import main

        with patch("sys.argv", ["ciffy"]):
            main()

        captured = capsys.readouterr()
        # Should show help or usage info
        assert "usage" in captured.out.lower() or "ciffy" in captured.out.lower()

    def test_main_info_subcommand(self, capsys):
        """main() routes to info command."""
        from ciffy.cli.__main__ import main

        with patch("sys.argv", ["ciffy", "info", get_test_cif("3SKW")]):
            main()

        captured = capsys.readouterr()
        assert "3SKW" in captured.out

    def test_main_geometry_subcommand(self, capsys):
        """main() routes to geometry command."""
        from ciffy.cli.__main__ import main

        with patch("sys.argv", ["ciffy", "geometry", get_test_cif("3SKW")]):
            main()

        captured = capsys.readouterr()
        assert "Radius of gyration" in captured.out


class TestClusterCommand:
    """Tests for the cluster command (requires mmseqs2)."""

    def test_cluster_basic(self, capsys):
        """cluster command runs without error."""
        from ciffy.cli.__main__ import _cluster_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            threshold=0.5,
            coverage=0.8,
            output=None,
            split=None,
            seed=42,
            copy=False,
            threads=4,
            verbose=False,
            quiet=False,
        )

        # Should complete without error (may use fallback if mmseqs unavailable)
        _cluster_command(args)

        captured = capsys.readouterr()
        assert "cluster" in captured.out.lower()

    def test_cluster_split_requires_output(self, capsys):
        """cluster --split requires --output."""
        from ciffy.cli.__main__ import _cluster_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            threshold=0.5,
            coverage=0.8,
            output=None,  # Missing!
            split="0.8,0.1,0.1",
            seed=42,
            copy=False,
            threads=4,
            verbose=False,
            quiet=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            _cluster_command(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--split requires --output" in captured.err

    def test_cluster_invalid_split_format(self, capsys):
        """cluster --split with invalid format shows error."""
        from ciffy.cli.__main__ import _cluster_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            threshold=0.5,
            coverage=0.8,
            output="/tmp/out",
            split="invalid",
            seed=42,
            copy=False,
            threads=4,
            verbose=False,
            quiet=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            _cluster_command(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Invalid --split format" in captured.err

    def test_cluster_split_sum_not_one(self, capsys):
        """cluster --split ratios must sum to 1."""
        from ciffy.cli.__main__ import _cluster_command

        args = argparse.Namespace(
            files=[get_test_cif("3SKW")],
            threshold=0.5,
            coverage=0.8,
            output="/tmp/out",
            split="0.5,0.5,0.5",  # Sums to 1.5
            seed=42,
            copy=False,
            threads=4,
            verbose=False,
            quiet=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            _cluster_command(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "sum to 1" in captured.err.lower()

    def test_cluster_no_files_found(self, capsys):
        """cluster with no matching files shows error."""
        from ciffy.cli.__main__ import _cluster_command

        args = argparse.Namespace(
            files=["/nonexistent/pattern/*.cif"],
            threshold=0.5,
            coverage=0.8,
            output=None,
            split=None,
            seed=42,
            copy=False,
            threads=4,
            verbose=False,
            quiet=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            _cluster_command(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No structure files found" in captured.err
