"""
Tests for load() options and load_metadata() function.

Tests:
- load_metadata() for fast metadata extraction
- load() with molecule_types filter
- load() with chains filter
- load() with skip=None to load descriptions
"""

import glob
import pytest
import numpy as np

from tests.utils import DATA_DIR, get_test_cif


CIF_FILES = sorted(glob.glob(str(DATA_DIR / "*.cif")))


class TestLoadMetadata:
    """Test load_metadata() fast path for indexing."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_metadata_returns_dict(self, cif_file):
        """load_metadata returns a dictionary."""
        from ciffy import load_metadata

        meta = load_metadata(cif_file)
        assert isinstance(meta, dict)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_metadata_has_required_keys(self, cif_file):
        """load_metadata returns dict with required keys."""
        from ciffy import load_metadata

        meta = load_metadata(cif_file)

        required_keys = ["id", "atoms", "chains", "atoms_per_chain", "molecule_types"]
        for key in required_keys:
            assert key in meta, f"Missing key: {key}"

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_metadata_atoms_count_matches(self, cif_file):
        """load_metadata atoms count matches sum of atoms_per_chain."""
        from ciffy import load_metadata

        meta = load_metadata(cif_file)

        assert meta["atoms"] == meta["atoms_per_chain"].sum()

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_metadata_chains_count_matches(self, cif_file):
        """load_metadata chains count matches length of atoms_per_chain."""
        from ciffy import load_metadata

        meta = load_metadata(cif_file)

        assert meta["chains"] == len(meta["atoms_per_chain"])
        assert meta["chains"] == len(meta["molecule_types"])

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_metadata_matches_full_load(self, cif_file):
        """load_metadata returns consistent data with full load."""
        from ciffy import load, load_metadata, Scale

        meta = load_metadata(cif_file)
        polymer = load(cif_file, backend="numpy")

        assert meta["atoms"] == polymer.size()
        assert meta["chains"] == polymer.size(Scale.CHAIN)
        assert np.array_equal(meta["atoms_per_chain"], polymer.counts(Scale.CHAIN))
        assert np.array_equal(meta["molecule_types"], polymer.molecule_types)

    def test_load_metadata_nonexistent_file_raises(self):
        """load_metadata raises OSError for nonexistent file."""
        from ciffy import load_metadata

        with pytest.raises(OSError):
            load_metadata("nonexistent_file.cif")


class TestLoadWithMoleculeTypesFilter:
    """Test load() with molecule_types parameter."""

    def test_load_molecule_types_single_rna(self, backend):
        """load with molecule_types=Molecule.RNA returns only RNA chains."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")  # Has RNA and protein
        polymer = load(cif, backend=backend, molecule_types=Molecule.RNA)

        # All chains should be RNA
        for mol_type in polymer.molecule_types:
            val = int(mol_type.item() if hasattr(mol_type, 'item') else mol_type)
            assert val == Molecule.RNA.value

    def test_load_molecule_types_single_protein(self, backend):
        """load with molecule_types=Molecule.PROTEIN returns only protein chains."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")  # Has RNA and protein
        polymer = load(cif, backend=backend, molecule_types=Molecule.PROTEIN)

        # All chains should be protein
        for mol_type in polymer.molecule_types:
            val = int(mol_type.item() if hasattr(mol_type, 'item') else mol_type)
            assert val == Molecule.PROTEIN.value

    def test_load_molecule_types_list(self, backend):
        """load with molecule_types=[...] returns matching chains."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")  # Has RNA and protein
        polymer = load(cif, backend=backend, molecule_types=[Molecule.RNA, Molecule.PROTEIN])

        # All chains should be RNA or protein
        for mol_type in polymer.molecule_types:
            val = int(mol_type.item() if hasattr(mol_type, 'item') else mol_type)
            assert val in (Molecule.RNA.value, Molecule.PROTEIN.value)

        # Should have at least one of each (9GCM has both)
        types = [int(m.item() if hasattr(m, 'item') else m) for m in polymer.molecule_types]
        assert Molecule.RNA.value in types
        assert Molecule.PROTEIN.value in types

    def test_load_molecule_types_no_match_empty(self, backend):
        """load with non-matching molecule_types returns empty polymer."""
        from ciffy import load, Molecule

        cif = get_test_cif("9MDS")  # All RNA
        polymer = load(cif, backend=backend, molecule_types=Molecule.DNA)

        assert polymer.empty()

    def test_load_molecule_types_all_match(self, backend):
        """load with matching molecule_types returns same as unfiltered."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9MDS")  # All RNA
        filtered = load(cif, backend=backend, molecule_types=Molecule.RNA)
        unfiltered = load(cif, backend=backend)

        assert filtered.size() == unfiltered.size()
        assert filtered.size(Scale.CHAIN) == unfiltered.size(Scale.CHAIN)


class TestLoadWithChainsFilter:
    """Test load() with chains parameter."""

    def test_load_chains_single(self, backend):
        """load with chains='A' returns only chain A."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")  # Has chains A, B, C, D
        polymer = load(cif, backend=backend, chains="A")

        assert polymer.size(Scale.CHAIN) == 1
        assert polymer.names[0] == "A"

    def test_load_chains_list(self, backend):
        """load with chains=['A', 'B'] returns those chains."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")  # Has chains A, B, C, D
        polymer = load(cif, backend=backend, chains=["A", "B"])

        assert polymer.size(Scale.CHAIN) == 2
        assert set(polymer.names) == {"A", "B"}

    def test_load_chains_nonexistent_empty(self, backend):
        """load with nonexistent chain returns empty polymer."""
        from ciffy import load

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend, chains="Z")

        assert polymer.empty()

    def test_load_chains_partial_match(self, backend):
        """load with some matching chains returns only those."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")  # Has chains A, B, C, D
        polymer = load(cif, backend=backend, chains=["A", "Z"])  # Z doesn't exist

        assert polymer.size(Scale.CHAIN) == 1
        assert polymer.names[0] == "A"

    def test_load_chains_all_returns_same(self, backend):
        """load with all chain names returns same as unfiltered."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")
        unfiltered = load(cif, backend=backend)
        all_chains = list(unfiltered.names)

        filtered = load(cif, backend=backend, chains=all_chains)

        assert filtered.size() == unfiltered.size()
        assert filtered.size(Scale.CHAIN) == unfiltered.size(Scale.CHAIN)


class TestLoadCombinedFilters:
    """Test load() with both molecule_types and chains filters."""

    def test_load_combined_filters(self, backend):
        """load with both molecule_types and chains applies both filters."""
        from ciffy import load, Molecule, Scale

        cif = get_test_cif("9GCM")  # A=RNA, B/C/D=protein

        # Filter to protein chains named B or C
        polymer = load(
            cif,
            backend=backend,
            molecule_types=Molecule.PROTEIN,
            chains=["B", "C"],
        )

        assert polymer.size(Scale.CHAIN) == 2
        assert set(polymer.names) == {"B", "C"}

        # All should be protein
        for mol_type in polymer.molecule_types:
            val = int(mol_type.item() if hasattr(mol_type, 'item') else mol_type)
            assert val == Molecule.PROTEIN.value

    def test_load_combined_filters_no_overlap(self, backend):
        """load with non-overlapping filters returns empty."""
        from ciffy import load, Molecule

        cif = get_test_cif("9GCM")  # A=RNA, B/C/D=protein

        # Ask for RNA chain B (but B is protein)
        polymer = load(
            cif,
            backend=backend,
            molecule_types=Molecule.RNA,
            chains=["B"],
        )

        assert polymer.empty()


class TestLoadDescriptions:
    """Test load() with skip parameter for descriptions."""

    def test_descriptions_skipped_by_default(self, backend):
        """load with default skip='descriptions' has None descriptions."""
        from ciffy import load

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend)

        assert polymer.descriptions is None

    def test_skip_none_loads_descriptions(self, backend):
        """load with skip=None loads descriptions."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend, skip=None)

        assert polymer.descriptions is not None
        assert isinstance(polymer.descriptions, list)

    def test_descriptions_per_chain(self, backend):
        """descriptions has one entry per chain."""
        from ciffy import load, Scale

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend, skip=None)

        assert len(polymer.descriptions) == polymer.size(Scale.CHAIN)

    def test_descriptions_are_strings(self, backend):
        """descriptions entries are strings."""
        from ciffy import load

        cif = get_test_cif("9GCM")
        polymer = load(cif, backend=backend, skip=None)

        for desc in polymer.descriptions:
            assert isinstance(desc, str)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_descriptions_preserved_after_chain_selection(self, cif_file, backend):
        """descriptions are preserved after selecting a chain."""
        from ciffy import load, Scale

        polymer = load(cif_file, backend=backend, skip=None)

        if polymer.size(Scale.CHAIN) > 0:
            chain = polymer.chain(0)

            # Chain should still have descriptions
            assert chain.descriptions is not None
            assert len(chain.descriptions) == 1
