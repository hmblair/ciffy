"""Tests for the skip parameter in load() and load_metadata()."""

import pytest
import numpy as np
import ciffy


@pytest.fixture
def test_file():
    """Path to test CIF file."""
    return "tests/data/9MDS.cif"


class TestSkipNone:
    """Test that skip=None loads all fields."""

    def test_skip_none_loads_all_fields(self, test_file):
        """skip=None should load all fields."""
        polymer = ciffy.load(test_file)

        # All fields should be present
        assert polymer.coordinates is not None
        assert polymer._bfactors is not None
        assert polymer._atoms is not None
        assert polymer._elements is not None
        assert polymer._sequence is not None

    def test_skip_empty_list_loads_all(self, test_file):
        """skip=[] should load all fields."""
        polymer = ciffy.load(test_file, skip=[])

        assert polymer.coordinates is not None
        assert polymer._bfactors is not None


class TestSkipSingleField:
    """Test skipping individual fields."""

    def test_skip_bfactors(self, test_file):
        """skip='bfactors' should skip B-factors."""
        polymer = ciffy.load(test_file, skip='bfactors')

        assert polymer.coordinates is not None
        assert polymer._bfactors is None

    def test_skip_resolution(self, test_file):
        """skip='resolution' should skip resolution."""
        polymer = ciffy.load(test_file, skip='resolution')

        assert polymer.coordinates is not None
        assert polymer._resolution is None


class TestSkipMultipleFields:
    """Test skipping multiple fields."""

    def test_skip_list(self, test_file):
        """skip=['bfactors', 'resolution'] should skip both."""
        polymer = ciffy.load(test_file, skip=['bfactors', 'resolution'])

        assert polymer.coordinates is not None
        assert polymer._bfactors is None
        assert polymer._resolution is None


class TestSkipMetadataPreset:
    """Test the 'metadata' preset."""

    def test_skip_metadata_matches_load_metadata(self, test_file):
        """skip='metadata' should match load_metadata() behavior."""
        # Load with skip='metadata' via _load directly
        from ciffy._c import _load
        data = _load(test_file, skip='metadata')

        # Should have chain structure
        assert 'atoms_per_chain' in data
        assert 'molecule_types' in data

        # Should NOT have heavy fields
        assert data.get('coordinates') is None
        assert data.get('bfactors') is None

    def test_load_metadata_uses_skip_metadata(self, test_file):
        """load_metadata() should work the same as before."""
        meta = ciffy.load_metadata(test_file)

        assert 'atoms' in meta
        assert 'chains' in meta
        assert 'atoms_per_chain' in meta
        assert 'molecule_types' in meta
        assert meta['atoms'] > 0


class TestSkipCoreFieldsRejected:
    """Test that core fields cannot be skipped."""

    @pytest.mark.parametrize("field", [
        "chains",
        "models",
        "names",
        "chain_names",
        "strands",
        "strand_names",
        "molecule_types",
    ])
    def test_cannot_skip_core_field(self, test_file, field):
        """Core fields should raise ValueError when skipped."""
        with pytest.raises(ValueError, match="Cannot skip core field"):
            ciffy.load(test_file, skip=field)


class TestSkipInvalidFieldRejected:
    """Test that invalid field names raise errors."""

    def test_invalid_field_name(self, test_file):
        """Unknown field name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown field name"):
            ciffy.load(test_file, skip='invalid_field_xyz')

    def test_invalid_field_in_list(self, test_file):
        """Unknown field name in list should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown field name"):
            ciffy.load(test_file, skip=['bfactors', 'invalid_field_xyz'])


class TestSkipTypeErrors:
    """Test type validation for skip parameter."""

    def test_skip_non_string_in_list(self, test_file):
        """Non-string items in skip list should raise TypeError."""
        with pytest.raises(TypeError, match="skip list must contain strings"):
            ciffy.load(test_file, skip=[123])

    def test_skip_wrong_type(self, test_file):
        """Non-string, non-list skip should raise TypeError."""
        with pytest.raises(TypeError, match="skip must be"):
            ciffy.load(test_file, skip=123)
