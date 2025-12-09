"""
Tests for loading and saving molecular structures.
"""

import glob
import os
import tempfile
import pytest

TESTS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTS_DIR, "data")
CIF_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.cif")))


class TestLoad:
    """Test CIF file loading."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_file(self, cif_file):
        from ciffy import load

        polymer = load(cif_file, backend="numpy")
        assert polymer is not None
        assert not polymer.empty()
        assert polymer.size() > 0

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_coordinates(self, cif_file):
        from ciffy import load

        polymer = load(cif_file, backend="numpy")
        assert polymer.coordinates.shape[0] == polymer.size()
        assert polymer.coordinates.shape[1] == 3

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_atoms(self, cif_file):
        from ciffy import load

        polymer = load(cif_file, backend="numpy")
        assert polymer.atoms.shape[0] == polymer.size()
        assert polymer.elements.shape[0] == polymer.size()

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_chains(self, cif_file):
        from ciffy import load, Scale

        polymer = load(cif_file, backend="numpy")
        assert polymer.size(Scale.CHAIN) > 0
        assert len(polymer.names) == polymer.size(Scale.CHAIN)
        assert len(polymer.strands) == polymer.size(Scale.CHAIN)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_residues(self, cif_file):
        from ciffy import load, Scale

        polymer = load(cif_file, backend="numpy")
        assert polymer.size(Scale.RESIDUE) > 0
        assert polymer.sequence.shape[0] == polymer.size(Scale.RESIDUE)

    def test_load_nonexistent_file(self):
        from ciffy import load

        with pytest.raises(OSError):
            load("nonexistent_file.cif", backend="numpy")


class TestSave:
    """Test PDB file saving."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_save_file(self, cif_file):
        from ciffy import load, RNA

        polymer = load(cif_file, backend="numpy")

        # Get RNA chains only (PDB writer currently supports RNA)
        rna = polymer.subset(RNA)
        if rna.empty():
            pytest.skip("No RNA chains in structure")

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            output_path = f.name

        try:
            rna.write(output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_save_and_check_content(self, cif_file):
        from ciffy import load, RNA

        polymer = load(cif_file, backend="numpy")

        rna = polymer.subset(RNA)
        if rna.empty():
            pytest.skip("No RNA chains in structure")

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            output_path = f.name

        try:
            rna.write(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            # Check PDB format
            lines = content.strip().split('\n')
            assert len(lines) > 0

            for line in lines:
                assert line.startswith("ATOM")
                # Check line has expected PDB format length
                assert len(line) >= 54  # Minimum PDB ATOM record length

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestRoundTrip:
    """Test loading, manipulating, and saving structures."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_center_save(self, cif_file):
        from ciffy import load, RNA, MOLECULE

        polymer = load(cif_file, backend="numpy")

        rna = polymer.subset(RNA)
        if rna.empty():
            pytest.skip("No RNA chains in structure")

        # Center the structure
        centered, means = rna.center(MOLECULE)
        assert centered.size() == rna.size()

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            output_path = f.name

        try:
            centered.write(output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_select_chain_save(self, cif_file):
        from ciffy import load, RNA

        polymer = load(cif_file, backend="numpy")

        rna = polymer.subset(RNA)
        if rna.empty():
            pytest.skip("No RNA chains in structure")

        # Select first chain
        first_chain = rna.select(0)
        assert first_chain.size() > 0

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            output_path = f.name

        try:
            first_chain.write(output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
