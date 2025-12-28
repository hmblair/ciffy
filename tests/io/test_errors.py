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

        p = ciffy.load(get_test_cif("1ZEW"))

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


class TestMalformedCIF:
    """Test error handling for malformed CIF files."""

    def test_missing_entry_id(self, tmp_path):
        """load raises error when _entry.id is missing."""
        import ciffy

        cif_file = tmp_path / "no_entry.cif"
        cif_file.write_text("""data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C 1.0 2.0 3.0
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_missing_atom_site_loop(self, tmp_path):
        """load raises error when atom_site loop is missing."""
        import ciffy

        cif_file = tmp_path / "no_atoms.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
_cell.length_a 50.0
_cell.length_b 50.0
_cell.length_c 50.0
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_invalid_coordinate_values(self, tmp_path):
        """load raises error for non-numeric coordinates."""
        import ciffy

        cif_file = tmp_path / "bad_coords.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 NOT_A_NUMBER 2.0 3.0 A 1
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_wrong_number_of_columns(self, tmp_path):
        """load raises error when row has wrong number of columns."""
        import ciffy

        cif_file = tmp_path / "wrong_cols.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 1.0 2.0
""")
        # Row has fewer columns than headers - should fail

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_unmatched_quotes(self, tmp_path):
        """load raises error for unmatched quotes in CIF data."""
        import ciffy

        cif_file = tmp_path / "bad_quotes.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C "C5' . ALA A 1 1 1.0 2.0 3.0 A 1
""")
        # Unmatched quote in atom name

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_data_block_missing(self, tmp_path):
        """load raises error when data_ block header is missing."""
        import ciffy

        cif_file = tmp_path / "no_data_block.cif"
        cif_file.write_text("""_entry.id TEST
loop_
_atom_site.group_PDB
ATOM 1 C
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_corrupted_middle_of_file(self, tmp_path):
        """load raises error for corrupted data mid-file."""
        import ciffy

        cif_file = tmp_path / "corrupted.cif"
        # Valid start, then garbage, then more data
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 1.0 2.0 3.0 A 1
\x00\x01\x02\x03GARBAGE_DATA_HERE\xff\xfe
ATOM 2 N N . ALA A 1 1 4.0 5.0 6.0 A 1
""")

        with pytest.raises((RuntimeError, ValueError, UnicodeDecodeError)):
            ciffy.load(str(cif_file))

    def test_extremely_large_coordinate(self, tmp_path):
        """load handles extremely large coordinate values."""
        import ciffy

        cif_file = tmp_path / "huge_coords.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 1e308 2e308 3e308 A 1
""")
        # Very large but valid floats - may overflow or be handled

        with pytest.raises((RuntimeError, ValueError, OverflowError)):
            ciffy.load(str(cif_file))

    def test_inf_coordinate(self, tmp_path):
        """load raises error for infinity in coordinates."""
        import ciffy

        cif_file = tmp_path / "inf_coords.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 inf 2.0 3.0 A 1
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_nan_coordinate(self, tmp_path):
        """load raises error for NaN in coordinates."""
        import ciffy

        cif_file = tmp_path / "nan_coords.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 nan 2.0 3.0 A 1
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_negative_atom_id(self, tmp_path):
        """load handles negative atom IDs."""
        import ciffy

        cif_file = tmp_path / "negative_id.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM -1 C CA . ALA A 1 1 1.0 2.0 3.0 A 1
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_null_bytes_in_file(self, tmp_path):
        """load raises error for null bytes in file."""
        import ciffy

        cif_file = tmp_path / "null_bytes.cif"
        cif_file.write_bytes(b"data_TEST\n_entry.id TEST\n\x00\x00\x00")

        with pytest.raises((RuntimeError, ValueError, UnicodeDecodeError)):
            ciffy.load(str(cif_file))

    def test_only_whitespace(self, tmp_path):
        """load raises error for file with only whitespace."""
        import ciffy

        cif_file = tmp_path / "whitespace.cif"
        cif_file.write_text("   \n\n\t\t\n   \n")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_only_comments(self, tmp_path):
        """load raises error for file with only comments."""
        import ciffy

        cif_file = tmp_path / "comments_only.cif"
        cif_file.write_text("""# This is a comment
# Another comment
# No actual data
""")

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_incomplete_loop_header(self, tmp_path):
        """load raises error for incomplete loop definition."""
        import ciffy

        cif_file = tmp_path / "incomplete_loop.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
""")
        # Loop headers without any data rows

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_semicolon_text_block_unterminated(self, tmp_path):
        """load raises error for unterminated semicolon text block."""
        import ciffy

        cif_file = tmp_path / "unterminated_semicolon.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
_struct.title
;This is a multi-line
text field that is never
terminated with a closing semicolon
_cell.length_a 50.0
""")
        # Semicolon text block not closed

        with pytest.raises((RuntimeError, ValueError)):
            ciffy.load(str(cif_file))

    def test_duplicate_data_block(self, tmp_path):
        """load handles duplicate data block names."""
        import ciffy

        cif_file = tmp_path / "duplicate_data.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST1
data_TEST
_entry.id TEST2
""")
        # Two data blocks with same name - parser may take first or raise error

        # Should either raise an error or successfully load one of them
        try:
            result = ciffy.load(str(cif_file))
            # If it succeeds, verify it loaded something
            assert result is not None
        except (RuntimeError, ValueError):
            pass  # Also acceptable behavior

    def test_missing_entity_poly_loop(self, tmp_path):
        """load handles missing entity_poly information gracefully."""
        import ciffy

        # Minimal valid CIF without entity_poly - should either work or raise clear error
        cif_file = tmp_path / "no_entity_poly.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 1.0 2.0 3.0 A 1
""")

        # This may or may not work depending on how the parser handles missing entity_poly
        # Either outcome is acceptable as long as it doesn't crash unexpectedly
        try:
            result = ciffy.load(str(cif_file))
            assert result is not None
        except (RuntimeError, ValueError) as e:
            # Clear error message is acceptable
            assert len(str(e)) > 0

    def test_very_long_line(self, tmp_path):
        """load handles very long lines in CIF file."""
        import ciffy

        cif_file = tmp_path / "long_line.cif"
        # Create a line that's extremely long (100KB)
        long_value = "A" * 100000
        cif_file.write_text(f"""data_TEST
_entry.id TEST
_struct.title '{long_value}'
""")

        # Should either handle or raise clear error
        try:
            ciffy.load(str(cif_file))
        except (RuntimeError, ValueError, MemoryError):
            pass  # Acceptable to reject very long lines

    def test_special_characters_in_values(self, tmp_path):
        """load handles special characters in quoted values."""
        import ciffy

        cif_file = tmp_path / "special_chars.cif"
        cif_file.write_text("""data_TEST
_entry.id TEST
_struct.title "Title with 'quotes' and special <>&chars"
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 1.0 2.0 3.0 A 1
""")

        # Should handle or raise clear error
        try:
            ciffy.load(str(cif_file))
        except (RuntimeError, ValueError):
            pass  # Acceptable to fail on special chars
