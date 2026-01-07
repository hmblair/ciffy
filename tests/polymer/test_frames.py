"""
Tests for frame paradigm methods on Polymer.

Two paradigms exist for frame operations:
1. Global Frames (single FrameDefinition): align() / unalign()
2. Local Frames (pair of FrameDefinitions): local_transforms() / apply_local_transforms()
"""

import numpy as np
import pytest

import ciffy
from ciffy import Scale, Molecule, Polymer
from ciffy.biochemistry import Residue
from ciffy.biochemistry.linking import GLYCOSIDIC_FRAME, NUCLEIC_ACID_LINK
from ciffy.geometry import LocalCoordinates

from tests.utils import BACKENDS


class TestGlobalFrames:
    """Tests for Paradigm 1: Global frame operations (align/unalign)."""

    def test_align_unalign_roundtrip(self):
        """align() -> unalign() restores original coordinates."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue(list(range(min(5, p.size(Scale.RESIDUE)))))

        aligned, Rs, origins = p.align(return_origins=True)
        restored = aligned.unalign(Rs, origins)

        rmsd = ciffy.rmsd(restored, p).item()
        assert rmsd < 0.001, f"Roundtrip RMSD should be ~0, got {rmsd}"

    def test_align_unalign_with_custom_frame(self):
        """Roundtrip works with non-default frame."""
        # Use P_FRAME instead of default GLYCOSIDIC_FRAME
        from ciffy.biochemistry.linking import P_FRAME

        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue(list(range(1, min(5, p.size(Scale.RESIDUE)))))  # Skip first (no P)

        aligned, Rs, origins = p.align(frame=P_FRAME, return_origins=True)
        restored = aligned.unalign(Rs, origins)

        rmsd = ciffy.rmsd(restored, p).item()
        assert rmsd < 0.001, f"Roundtrip RMSD should be ~0, got {rmsd}"

    def test_align_returns_origins_optionally(self):
        """align() returns origins only when return_origins=True."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue([0, 1, 2])

        # Default: no origins
        result = p.align()
        assert len(result) == 2
        aligned, Rs = result
        assert Rs.shape == (3, 3, 3)

        # With origins
        result = p.align(return_origins=True)
        assert len(result) == 3
        aligned, Rs, origins = result
        assert Rs.shape == (3, 3, 3)
        assert origins.shape == (3, 3)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_align_unalign_backend_preserved(self, backend):
        """align/unalign work with both backends."""
        if backend == "torch":
            pytest.importorskip("torch")

        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip().residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()

        aligned, Rs, origins = p.align(return_origins=True)
        restored = aligned.unalign(Rs, origins)

        rmsd = ciffy.rmsd(restored, p).item()
        assert rmsd < 0.001


class TestLocalFrames:
    """Tests for Paradigm 2: Local frame operations."""

    def test_local_transforms_requires_both_frames(self):
        """Raises TypeError if either frame is None."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue([0, 1, 2])

        with pytest.raises(TypeError, match="source_frame is required"):
            p.local_transforms(None, GLYCOSIDIC_FRAME)

        with pytest.raises(TypeError, match="target_frame is required"):
            p.local_transforms(GLYCOSIDIC_FRAME, None)

    def test_apply_local_transforms_requires_both_frames(self):
        """Raises TypeError if either frame is None."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue([0, 1, 2])
        transforms = p.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        aligned, _ = p.align()

        with pytest.raises(TypeError, match="source_frame is required"):
            aligned.apply_local_transforms(transforms, None, GLYCOSIDIC_FRAME)

        with pytest.raises(TypeError, match="target_frame is required"):
            aligned.apply_local_transforms(transforms, GLYCOSIDIC_FRAME, None)

    def test_roundtrip_same_frame(self):
        """Roundtrip with GLYCOSIDIC->GLYCOSIDIC preserves structure."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue(list(range(min(6, p.size(Scale.RESIDUE)))))

        transforms = p.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        aligned, _ = p.align(frame=GLYCOSIDIC_FRAME)
        rebuilt = aligned.apply_local_transforms(
            transforms, GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME
        )

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01, f"Roundtrip RMSD should be ~0, got {rmsd}"

    def test_roundtrip_different_frames(self):
        """Roundtrip with O3P->P frames preserves structure."""
        O3P_FRAME = NUCLEIC_ACID_LINK.prev_frame
        P_FRAME = NUCLEIC_ACID_LINK.next_frame

        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        # Skip first and last residues (may not have O3' or P)
        n_res = p.size(Scale.RESIDUE)
        if n_res > 4:
            p = p.residue(list(range(1, min(5, n_res - 1))))
        else:
            pytest.skip("Need at least 5 residues for this test")

        transforms = p.local_transforms(O3P_FRAME, P_FRAME)
        aligned, _ = p.align(frame=P_FRAME)  # Align to TARGET frame
        rebuilt = aligned.apply_local_transforms(transforms, O3P_FRAME, P_FRAME)

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01, f"Roundtrip RMSD should be ~0, got {rmsd}"

    def test_invariant_to_global_rotation(self):
        """Local transforms are invariant to global rotation."""
        pytest.importorskip("torch")
        import torch

        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip().torch()
        p = p.residue(list(range(min(4, p.size(Scale.RESIDUE)))))

        # Create a random rotation
        axis = torch.randn(3)
        axis = axis / axis.norm()
        angle = torch.tensor(0.5)  # ~30 degrees
        axis_angle = axis * angle

        from ciffy.geometry.transforms import rodrigues
        R = rodrigues(axis_angle)

        # Rotate the polymer
        rotated_coords = p.coordinates @ R.T
        rotated = p.copy(coordinates=rotated_coords)

        # Compute transforms
        t_original = p.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        t_rotated = rotated.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)

        # Should be the same (within numerical precision)
        diff = (t_original - t_rotated).abs().max().item()
        assert diff < 1e-3, f"Transforms should be invariant to rotation, max diff={diff}"

    def test_invariant_to_global_translation(self):
        """Local transforms are invariant to global translation."""
        pytest.importorskip("torch")
        import torch

        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip().torch()
        p = p.residue(list(range(min(4, p.size(Scale.RESIDUE)))))

        # Translate the polymer
        translation = torch.randn(3) * 100  # Large translation
        translated = p.copy(coordinates=p.coordinates + translation)

        # Compute transforms
        t_original = p.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        t_translated = translated.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)

        # Should be the same
        diff = (t_original - t_translated).abs().max().item()
        assert diff < 1e-3, f"Transforms should be invariant to translation, max diff={diff}"

    def test_transforms_shape(self):
        """local_transforms returns correct shape."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue(list(range(min(5, p.size(Scale.RESIDUE)))))
        n_res = p.size(Scale.RESIDUE)

        transforms = p.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)

        assert transforms.shape == (n_res, 6)
        # Last transform should be zeros (no successor residue)
        assert np.allclose(transforms[-1], 0, atol=1e-6)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_local_transforms_backend_preserved(self, backend):
        """local_transforms/apply work with both backends."""
        if backend == "torch":
            pytest.importorskip("torch")

        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip().residue([0, 1, 2])
        if backend == "torch":
            p = p.torch()

        transforms = p.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        aligned, _ = p.align()
        rebuilt = aligned.apply_local_transforms(
            transforms, GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME
        )

        rmsd = ciffy.rmsd(rebuilt, p).item()
        assert rmsd < 0.01


class TestAppendWithFrames:
    """Tests for append() with explicit frame parameters."""

    def test_append_default_frames_backward_compatible(self):
        """Default frames use LINKING_BY_TYPE (backward compatible)."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue([0, 1])

        # Build first residue
        res0 = p.residue(0)
        res_type = Residue.from_index(p.sequence[0])
        built = Polymer().append(
            res_type.subset(res0.atoms.tolist()),
            res0.coordinates,
            residue=res_type,
        )

        # Should work without specifying frames
        O3P_FRAME = NUCLEIC_ACID_LINK.prev_frame
        P_FRAME = NUCLEIC_ACID_LINK.next_frame
        transforms = p.local_transforms(O3P_FRAME, P_FRAME)
        aligned, _ = p.align(frame=P_FRAME)

        res1 = p.residue(1)
        res_type1 = Residue.from_index(p.sequence[1])
        n_atoms0 = res0.size()

        built = built.append(
            res_type1.subset(res1.atoms.tolist()),
            LocalCoordinates(aligned.coordinates[n_atoms0:], transforms[0]),
            residue=res_type1,
            # No explicit frames - uses defaults
        )

        assert built.size(Scale.RESIDUE) == 2

    def test_append_explicit_glycosidic_frames(self):
        """Explicit GLYCOSIDIC frames work for ML-style transforms."""
        p = ciffy.load("tests/data/9MDS.cif").chain(0).strip()
        p = p.residue(list(range(min(4, p.size(Scale.RESIDUE)))))

        # Get GLYCOSIDIC transforms
        transforms = p.local_transforms(GLYCOSIDIC_FRAME, GLYCOSIDIC_FRAME)
        aligned, _ = p.align(frame=GLYCOSIDIC_FRAME)

        counts = p.counts(Scale.RESIDUE)
        sequence = p.sequence
        n_res = p.size(Scale.RESIDUE)

        # Build first residue
        res0 = p.residue(0)
        res_type = Residue.from_index(sequence[0])
        built = Polymer().append(
            res_type.subset(res0.atoms.tolist()),
            aligned.coordinates[: int(counts[0])],
            residue=res_type,
        )

        # Build subsequent residues with explicit frames
        offset = int(counts[0])
        for i in range(1, n_res):
            n_atoms = int(counts[i])
            res_i = p.residue(i)
            res_type = Residue.from_index(sequence[i])

            built = built.append(
                res_type.subset(res_i.atoms.tolist()),
                LocalCoordinates(
                    aligned.coordinates[offset : offset + n_atoms], transforms[i - 1]
                ),
                residue=res_type,
                source_frame=GLYCOSIDIC_FRAME,
                target_frame=GLYCOSIDIC_FRAME,
            )
            offset += n_atoms

        rmsd = ciffy.rmsd(built, p).item()
        assert rmsd < 0.01, f"Append with GLYCOSIDIC frames should preserve structure, got RMSD={rmsd}"
