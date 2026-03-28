"""
Tests for Polymer.append() and _append() - chain building operations.
"""

import numpy as np
import pytest

import ciffy
from ciffy import Scale, template
from ciffy.backend import ops
from ciffy.biochemistry import Residue
from tests.utils import get_like


# Identity quaternion (w=1) + Z-axis translation for ideal backbone spacing
# Format: [w, x, y, z, tx, ty, tz] - 7D quaternion format
LINEAR_EXTEND_TRANSFORM = np.array([1, 0, 0, 0, 0, 0, 6.0], dtype=np.float32)


def _template_with_coords(sequence: str, backend: str = "numpy") -> ciffy.Polymer:
    """Create a _template with ideal coordinates for testing."""
    _template = template(sequence, backend=backend)
    if _template.empty():
        return _template

    poly = ciffy.Polymer()
    if backend == "torch":
        poly = poly.torch()

    residue_indices = list(_template.sequence[:len(sequence)])
    for i, res_idx in enumerate(residue_indices):
        residue = Residue.from_index(int(res_idx))
        is_first = (i == 0)
        is_last = (i == len(residue_indices) - 1)

        atom_group = residue.terminal(start=is_first, end=is_last)
        atoms = atom_group.index()
        elements = atom_group.elements()
        coords = atom_group.ideal

        ref = get_like(backend)
        coords = ops.to_backend(coords, like=ref)
        atoms = ops.to_backend(atoms, like=ref)
        elements = ops.to_backend(elements, like=ref)

        if poly.empty():
            poly = poly._append(residue, coords, atoms=atoms, elements=elements)
        else:
            poly = poly._append(
                residue, coords, LINEAR_EXTEND_TRANSFORM, atoms=atoms, elements=elements
            )

    return poly


class TestAppend:
    """Tests for Polymer.append() method."""

    def test_append__template_mode(self):
        """append() creates _template without coordinates."""
        p = ciffy.Polymer()
        p = p.append(Residue.A)

        assert p.size() == Residue.A.n_atoms
        assert p.size(Scale.RESIDUE) == 1
        assert hasattr(p, 'atoms')
        assert hasattr(p, 'elements')
        # Template mode: no coordinates
        assert not hasattr(p, '_coordinates') or p._coordinates is None

    def test_append_with_coordinates(self):
        """append() with coordinates creates polymer with coords."""
        p = ciffy.Polymer()
        coords = Residue.A.ideal
        p = p.append(Residue.A, coords)

        assert p.size() == Residue.A.n_atoms
        assert p.coordinates is not None
        assert p.coordinates.shape == coords.shape

    def test_append_multi_residue__template(self):
        """append() builds multi-residue _template."""
        p = ciffy.Polymer()
        for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
            p = p.append(res)

        assert p.size(Scale.RESIDUE) == 4
        expected_atoms = sum(r.n_atoms for r in [Residue.A, Residue.C, Residue.G, Residue.U])
        assert p.size() == expected_atoms
        assert p.sequence_str() == "acgu"

    def test_append_with_local_coordinates(self):
        """append() positions residue using LocalCoordinates."""
        from ciffy.geometry import LocalCoordinates
        p = ciffy.Polymer()
        p = p.append(Residue.A, Residue.A.ideal)
        # 7D quaternion format: [w=1, x=0, y=0, z=0, tx=0, ty=0, tz=0] = identity
        identity_transform = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        p = p.append(Residue.C, LocalCoordinates(Residue.C.ideal, identity_transform))

        assert p.size(Scale.RESIDUE) == 2
        assert p.coordinates is not None
        expected = Residue.A.n_atoms + Residue.C.n_atoms
        assert p.coordinates.shape == (expected, 3)

    def test_append_with_subset_requires_residue(self):
        """append() with subset requires explicit residue parameter."""
        subset = Residue.A.subset({2, 3, 5, 6, 7})
        p = ciffy.Polymer()

        # Should raise without residue=
        with pytest.raises(ValueError, match="residue"):
            p.append(subset)

        # Should work with residue=
        p = p.append(subset, residue=Residue.A)
        assert p.size() == 5

    def test_append_preserves_sequence(self):
        """append() correctly sets sequence field."""
        p = ciffy.Polymer()
        p = p.append(Residue.G)
        p = p.append(Residue.C)
        p = p.append(Residue.A)
        p = p.append(Residue.U)

        assert p.sequence_str() == "gcau"

    def test_append_atoms_match_atomgroup(self):
        """append() atoms field matches AtomGroup.index()."""
        p = ciffy.Polymer()
        p = p.append(Residue.A)

        np.testing.assert_array_equal(
            np.asarray(p.atoms),
            Residue.A.index()
        )

    def test_append_elements_match_atomgroup(self):
        """append() elements field matches AtomGroup.elements()."""
        p = ciffy.Polymer()
        p = p.append(Residue.A)

        np.testing.assert_array_equal(
            np.asarray(p.elements),
            Residue.A.elements()
        )
