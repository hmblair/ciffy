"""
Tests for I/O error handling and edge cases.

Tests error conditions for loading and saving CIF files.
"""

import pytest
import numpy as np

from tests.utils import get_test_cif, BACKENDS


class TestLoadErrors:
    """Test error handling during CIF loading."""

    def test_load_non_cif_file(self, tmp_path):
        """load raises error for non-CIF text file."""
        import ciffy

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is not a CIF file\nJust plain text.")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(txt_file))

    def test_load_truncated_cif(self, tmp_path):
        """load raises error for truncated/incomplete CIF."""
        import ciffy

        cif_file = tmp_path / "truncated.cif"
        # Write partial CIF header without atom data
        cif_file.write_text("data_TEST\n_entry.id TEST\n")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_load_empty_file(self, tmp_path):
        """load raises error for empty file."""
        import ciffy

        empty_file = tmp_path / "empty.cif"
        empty_file.write_text("")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(empty_file))

    def test_load_nonexistent_file(self):
        """load raises OSError for non-existent file."""
        import ciffy

        with pytest.raises(OSError):
            ciffy.load("/nonexistent/path/to/file.cif")

    def test_load_directory(self, tmp_path):
        """load raises OSError when given a directory."""
        import ciffy

        with pytest.raises(OSError):
            ciffy.load(str(tmp_path))

    def test_load_invalid_backend(self, backend):
        """load raises ValueError for invalid backend string."""
        import ciffy

        cif_path = get_test_cif("3SKW")
        with pytest.raises(ValueError):
            ciffy.load(cif_path, backend="invalid_backend")

    def test_load_binary_file(self, tmp_path):
        """load raises error for binary file."""
        import ciffy

        binary_file = tmp_path / "binary.cif"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(binary_file))


class TestWriteErrors:
    """Test error handling during CIF writing."""

    def test_write_empty_polymer(self, tmp_path):
        """write raises ValueError for empty polymer."""
        import ciffy

        template = ciffy.from_sequence("a")
        empty = template[template.atoms < 0]  # Empty mask

        assert empty.empty()

        with pytest.raises(ValueError, match="empty"):
            empty.write(str(tmp_path / "test.cif"))

    def test_write_invalid_extension_pdb(self, tmp_path):
        """write raises ValueError for .pdb extension."""
        import ciffy

        p = ciffy.from_sequence("acgu")

        with pytest.raises(ValueError, match=".cif"):
            p.write(str(tmp_path / "test.pdb"))

    def test_write_invalid_extension_txt(self, tmp_path):
        """write raises ValueError for .txt extension."""
        import ciffy

        p = ciffy.from_sequence("acgu")

        with pytest.raises(ValueError, match=".cif"):
            p.write(str(tmp_path / "test.txt"))

    def test_write_no_extension(self, tmp_path):
        """write raises ValueError when no extension provided."""
        import ciffy

        p = ciffy.from_sequence("acgu")

        with pytest.raises(ValueError, match=".cif"):
            p.write(str(tmp_path / "test"))

    def test_write_invalid_directory(self):
        """write raises error for non-existent directory."""
        import ciffy

        p = ciffy.from_sequence("acgu")

        with pytest.raises((OSError, IOError)):
            p.write("/nonexistent/directory/test.cif")


class TestRoundTripEdgeCases:
    """Test round-trip (load -> save -> load) edge cases."""

    def test_round_trip_single_residue(self, backend, tmp_path):
        """Round-trip preserves single-residue structure."""
        import ciffy
        from ciffy import Scale

        p = ciffy.from_sequence("a", backend=backend)
        # Attach non-zero coordinates (template has zeros)
        coords = np.random.randn(p.size(), 3).astype(np.float32) * 10
        if backend == "torch":
            import torch
            p.coordinates = torch.from_numpy(coords)
        else:
            p.coordinates = coords

        out_path = tmp_path / "single_residue.cif"
        p.write(str(out_path))

        reloaded = ciffy.load(str(out_path), backend=backend)
        assert reloaded.size(Scale.RESIDUE) == 1
        assert not reloaded.empty()

    def test_round_trip_preserves_coordinates(self, backend, tmp_path):
        """Round-trip preserves coordinate values within tolerance."""
        import ciffy

        p = ciffy.load(get_test_cif("3SKW"), backend=backend)
        # Only test polymer portion (writer only writes polymer atoms)
        polymer_coords = np.asarray(p.coordinates[:p.polymer_count])

        out_path = tmp_path / "roundtrip.cif"
        p.write(str(out_path))

        reloaded = ciffy.load(str(out_path), backend=backend)
        reloaded_coords = np.asarray(reloaded.coordinates)

        assert np.allclose(polymer_coords, reloaded_coords, atol=0.001)

    def test_round_trip_chain_count(self, backend, tmp_path):
        """Round-trip preserves chain count."""
        import ciffy
        from ciffy import Scale

        p = ciffy.load(get_test_cif("9GCM"), backend=backend)
        original_chains = p.size(Scale.CHAIN)

        out_path = tmp_path / "chains.cif"
        p.write(str(out_path))

        reloaded = ciffy.load(str(out_path), backend=backend)
        assert reloaded.size(Scale.CHAIN) == original_chains


class TestMetadataLoading:
    """Test metadata-only loading edge cases."""

    def test_load_metadata_has_id(self):
        """load_metadata returns dict with valid ID."""
        import ciffy

        meta = ciffy.load_metadata(get_test_cif("3SKW"))
        # load_metadata returns a dict, not a Polymer
        assert isinstance(meta, dict)
        assert meta["id"] == "3SKW"

    def test_load_metadata_nonexistent_file(self):
        """load_metadata raises OSError for non-existent file."""
        import ciffy

        with pytest.raises(OSError):
            ciffy.load_metadata("/nonexistent/file.cif")
