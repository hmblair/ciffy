"""Tests for the skip parameter in load() and load_metadata()."""

import pytest
import numpy as np
import ciffy


class TestSkipNone:
    """Test that skip=None loads all fields."""

    def test_skip_none_loads_all_fields(self, cif_9mds):
        """skip=None should load all fields."""
        polymer = ciffy.load(cif_9mds)

        # All fields should be present (use _get_field_data for checking availability)
        assert polymer._get_field_data('coordinates') is not None
        assert polymer._get_field_data('bfactors') is not None
        assert polymer._get_field_data('atoms') is not None
        assert polymer._get_field_data('elements') is not None
        assert polymer._get_field_data('sequence') is not None

    def test_skip_empty_list_loads_all(self, cif_9mds):
        """skip=[] should load all fields."""
        polymer = ciffy.load(cif_9mds, skip=[])

        assert polymer._get_field_data('coordinates') is not None
        assert polymer._get_field_data('bfactors') is not None


class TestSkipSingleField:
    """Test skipping individual fields."""

    def test_skip_bfactors(self, cif_9mds):
        """skip='bfactors' should skip B-factors."""
        polymer = ciffy.load(cif_9mds, skip='bfactors')

        assert polymer._get_field_data('coordinates') is not None
        assert polymer._get_field_data('bfactors') is None

    def test_skip_resolution(self, cif_9mds):
        """skip='resolution' should skip resolution."""
        polymer = ciffy.load(cif_9mds, skip='resolution')

        assert polymer._get_field_data('coordinates') is not None
        assert polymer.resolution is None


class TestSkipMultipleFields:
    """Test skipping multiple fields."""

    def test_skip_list(self, cif_9mds):
        """skip=['bfactors', 'resolution'] should skip both."""
        polymer = ciffy.load(cif_9mds, skip=['bfactors', 'resolution'])

        assert polymer._get_field_data('coordinates') is not None
        assert polymer._get_field_data('bfactors') is None
        assert polymer.resolution is None


class TestSkipMetadataPreset:
    """Test the 'metadata' preset."""

    def test_skip_metadata_matches_load_metadata(self, cif_9mds):
        """skip='metadata' should match load_metadata() behavior."""
        # Load with skip='metadata' via _load directly
        from ciffy._c import _load
        data = _load(cif_9mds, skip='metadata')

        # Should have chain structure
        assert 'atoms_per_chain' in data
        assert 'molecule_types' in data

        # Should NOT have heavy fields
        assert data.get('coordinates') is None
        assert data.get('bfactors') is None

    def test_load_metadata_uses_skip_metadata(self, cif_9mds):
        """load_metadata() should work the same as before."""
        meta = ciffy.load_metadata(cif_9mds)

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
    def test_cannot_skip_core_field(self, cif_9mds, field):
        """Core fields should raise ValueError when skipped."""
        with pytest.raises(ValueError, match="Cannot skip core field"):
            ciffy.load(cif_9mds, skip=field)


class TestSkipInvalidFieldRejected:
    """Test that invalid field names raise errors."""

    def test_invalid_field_name(self, cif_9mds):
        """Unknown field name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown field name"):
            ciffy.load(cif_9mds, skip='invalid_field_xyz')

    def test_invalid_field_in_list(self, cif_9mds):
        """Unknown field name in list should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown field name"):
            ciffy.load(cif_9mds, skip=['bfactors', 'invalid_field_xyz'])


class TestSkipTypeErrors:
    """Test type validation for skip parameter."""

    def test_skip_non_string_in_list(self, cif_9mds):
        """Non-string items in skip list should raise TypeError."""
        with pytest.raises(TypeError, match="skip list must contain strings"):
            ciffy.load(cif_9mds, skip=[123])

    def test_skip_wrong_type(self, cif_9mds):
        """Non-string, non-list skip should raise TypeError."""
        with pytest.raises(TypeError, match="skip must be"):
            ciffy.load(cif_9mds, skip=123)
