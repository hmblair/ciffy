"""Reusable test patterns for common ciffy test scenarios.

These patterns encapsulate common testing workflows so tests can focus
on what they're testing rather than boilerplate.
"""

from typing import Optional
import numpy as np

from .tolerances import get_tolerances


def assert_cif_roundtrip(
    polymer,
    tmp_path,
    check_coordinates: bool = True,
    check_sequence: bool = True,
) -> "Polymer":
    """Test CIF write/read roundtrip.

    Args:
        polymer: Polymer to write and reload
        tmp_path: pytest tmp_path fixture
        check_coordinates: Whether to verify coordinates match
        check_sequence: Whether to verify sequence exists

    Returns:
        The reloaded polymer

    Example:
        def test_cif_roundtrip(self, tmp_path):
            polymer = from_sequence("acgu")
            reloaded = assert_cif_roundtrip(polymer, tmp_path)
    """
    from ciffy import load

    tol = get_tolerances()
    output_path = tmp_path / "test.cif"

    # Get polymer count before writing
    polymer_count = getattr(polymer, "polymer_count", polymer.size())

    # Write
    polymer.write(str(output_path))

    # Reload
    reloaded = load(str(output_path), backend=polymer.backend)

    # Verify
    if check_coordinates and polymer_count > 0:
        orig_coords = np.asarray(polymer.coordinates[:polymer_count])
        reload_coords = np.asarray(reloaded.coordinates)
        assert np.allclose(orig_coords, reload_coords, atol=tol.coord_roundtrip), (
            "Coordinates not preserved in CIF roundtrip"
        )

    if check_sequence:
        assert len(reloaded.sequence) > 0, "Reloaded polymer has no sequence"

    return reloaded
