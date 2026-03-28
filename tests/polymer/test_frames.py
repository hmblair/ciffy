"""
Tests for frame operations.

Primary API:
- decompose(polymer, source, target) -> Transforms
- compose(polymer, transforms) -> Polymer
"""

import numpy as np
import pytest

import ciffy
from ciffy import Scale, operations
from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME, O3P_FRAME, P_FRAME
from tests.utils import get_test_cif

CIF_9MDS = get_test_cif("9MDS")


class TestDecomposeCompose:
    """Tests for the decompose/compose API."""

    def test_roundtrip_default_frames(self):
        """decompose -> compose roundtrip preserves structure."""
        p = ciffy.load(CIF_9MDS).chain(0).strip()
        p = p.residue(list(range(min(5, p.size(Scale.RESIDUE)))))

        transforms = operations.decompose(p)
        rebuilt = operations.compose(p, transforms)

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01, f"Roundtrip RMSD should be ~0, got {rmsd}"

    def test_roundtrip_glycosidic_frames(self):
        """Roundtrip with GLYCOSIDIC->GLYCOSIDIC frames."""
        p = ciffy.load(CIF_9MDS).chain(0).strip()
        p = p.residue(list(range(min(6, p.size(Scale.RESIDUE)))))

        transforms = operations.decompose(p, source=GLYCOSIDIC_FRAME, target=GLYCOSIDIC_FRAME)
        rebuilt = operations.compose(p, transforms)

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01, f"Roundtrip RMSD should be ~0, got {rmsd}"

    def test_roundtrip_o3p_p_frames(self):
        """Roundtrip with O3P->P frames (default)."""
        p = ciffy.load(CIF_9MDS).chain(0).strip()
        p = p.residue(list(range(min(5, p.size(Scale.RESIDUE)))))

        transforms = operations.decompose(p, source=O3P_FRAME, target=P_FRAME)
        rebuilt = operations.compose(p, transforms)

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01, f"Roundtrip RMSD should be ~0, got {rmsd}"

    def test_transforms_shape(self):
        """Transforms has correct shape and identity first element."""
        p = ciffy.load(CIF_9MDS).chain(0).strip()
        p = p.residue(list(range(min(5, p.size(Scale.RESIDUE)))))
        n_res = p.size(Scale.RESIDUE)

        transforms = operations.decompose(p)

        assert transforms.data.shape == (n_res, 7)
        assert len(transforms) == n_res
        # First transform should be identity quaternion [1,0,0,0] + zero translation
        assert np.allclose(transforms.data[0, 0], 1, atol=1e-6)  # w=1
        assert np.allclose(transforms.data[0, 1:], 0, atol=1e-6)  # rest=0

    def test_transforms_carries_frame_metadata(self):
        """Transforms object stores source and target frames."""
        p = ciffy.load(CIF_9MDS).chain(0).strip()
        p = p.residue([0, 1, 2])

        transforms = operations.decompose(p, source=O3P_FRAME, target=P_FRAME)

        assert transforms.source is O3P_FRAME
        assert transforms.target is P_FRAME

    def test_invariant_to_global_rotation(self):
        """Transforms are invariant to global rotation."""
        pytest.importorskip("torch")

        from ciffy.backend import ops
        from ciffy.geometry import quaternion_to_rotation_matrix

        p = ciffy.load(CIF_9MDS).chain(0).strip().torch()
        p = p.residue(list(range(min(4, p.size(Scale.RESIDUE)))))

        # Create a random rotation using numpy, then convert
        angle = 0.5
        axis = np.random.randn(3).astype(np.float32)
        axis = axis / np.linalg.norm(axis)
        quat_np = np.zeros(4, dtype=np.float32)
        quat_np[0] = np.cos(angle / 2)
        quat_np[1:] = axis * np.sin(angle / 2)
        quat = ops.to_backend(quat_np, like=p.coordinates)
        R = quaternion_to_rotation_matrix(ops.unsqueeze(quat, 0)).squeeze(0)

        # Rotate the polymer
        rotated = p.copy(coordinates=p.coordinates @ R.T)

        # Compute transforms
        t_original = operations.decompose(p)
        t_rotated = operations.decompose(rotated)

        # Should be the same
        diff = (t_original.data - t_rotated.data).abs().max().item()
        assert diff < 1e-3, f"Transforms should be invariant to rotation, max diff={diff}"

    def test_invariant_to_global_translation(self):
        """Transforms are invariant to global translation."""
        pytest.importorskip("torch")

        from ciffy.backend import ops

        p = ciffy.load(CIF_9MDS).chain(0).strip().torch()
        p = p.residue(list(range(min(4, p.size(Scale.RESIDUE)))))

        # Translate the polymer
        translation = ops.randn((3,), like=p.coordinates) * 100
        translated = p.copy(coordinates=p.coordinates + translation)

        # Compute transforms
        t_original = operations.decompose(p)
        t_translated = operations.decompose(translated)

        # Should be the same
        diff = (t_original.data - t_translated.data).abs().max().item()
        assert diff < 1e-3, f"Transforms should be invariant to translation, max diff={diff}"

    def test_backend_preserved(self, backend):
        """decompose/compose work with both backends."""
        p = ciffy.load(CIF_9MDS, backend=backend).chain(0).strip().residue([0, 1, 2])

        transforms = operations.decompose(p)
        rebuilt = operations.compose(p, transforms)

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01

    def test_single_residue(self):
        """Works with single residue polymer."""
        p = ciffy.load(CIF_9MDS).chain(0).strip().residue([0])

        transforms = operations.decompose(p)
        assert len(transforms) == 1
        assert np.allclose(transforms.data[0, 0], 1, atol=1e-6)  # identity quat

        rebuilt = operations.compose(p, transforms)
        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01

    def test_compose_with_different_polymer(self):
        """compose works with a different polymer (template)."""
        p = ciffy.load(CIF_9MDS).chain(0).strip()
        p = p.residue(list(range(min(4, p.size(Scale.RESIDUE)))))

        # Extract transforms from original
        transforms = operations.decompose(p)

        # Use same polymer as template (simplest case)
        rebuilt = operations.compose(p, transforms)

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01

    def test_compose_size_mismatch_raises(self):
        """compose raises if transform count doesn't match residue count."""
        p = ciffy.load(CIF_9MDS).chain(0).strip()
        p = p.residue([0, 1, 2])

        transforms = operations.decompose(p)
        p_short = p.residue([0, 1])

        with pytest.raises(ValueError, match="transforms has 3 entries"):
            operations.compose(p_short, transforms)


class TestTransformsDataclass:
    """Tests for the Transforms dataclass."""

    def test_len(self):
        """len(transforms) returns number of residues."""
        p = ciffy.load(CIF_9MDS).chain(0).strip().residue([0, 1, 2])
        transforms = operations.decompose(p)
        assert len(transforms) == 3

    def test_data_access(self):
        """Can access transform data directly."""
        p = ciffy.load(CIF_9MDS).chain(0).strip().residue([0, 1])
        transforms = operations.decompose(p)

        # Access quaternion and translation
        quat = transforms.data[1, :4]
        trans = transforms.data[1, 4:]
        assert quat.shape == (4,)
        assert trans.shape == (3,)

    def test_create_manually(self):
        """Can create Transforms manually for predictions."""
        import numpy as np
        from ciffy.operations import Transforms

        # Create identity transforms
        data = np.zeros((3, 7), dtype=np.float32)
        data[:, 0] = 1.0  # identity quaternions

        transforms = Transforms(data=data, source=O3P_FRAME, target=P_FRAME)
        assert len(transforms) == 3
        assert transforms.source is O3P_FRAME
