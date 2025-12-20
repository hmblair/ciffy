"""
Test that documentation examples work correctly.

This test suite validates code snippets from documentation to catch
API drift early. Run with: pytest tests/test_documentation.py
"""

import pytest
import ciffy
from ciffy.biochemistry import Residue


class TestREADMEExamples:
    """Validate README.md code examples."""

    def test_basic_load_and_access(self):
        """Test basic load and access pattern from README."""
        polymer = ciffy.load("tests/data/9MDS.cif", backend="numpy")

        # From README: basic information
        coords = polymer.coordinates
        atoms = polymer.atoms
        sequence = polymer.sequence_str()

        assert coords.shape[1] == 3
        assert len(atoms) == len(coords)
        assert isinstance(sequence, str)

    def test_molecule_type_selection(self):
        """Test that by_type works (not subset)."""
        polymer = ciffy.load("tests/data/9MDS.cif")

        # Correct API is by_type, not subset
        rna_chains = polymer.by_type(ciffy.RNA)

        # Verify subset doesn't exist (docs should not reference it)
        assert not hasattr(polymer, "subset"), "API changed: 'subset' method now exists"

    def test_geometric_operations(self):
        """Test geometric operations from README."""
        polymer = ciffy.load("tests/data/9MDS.cif", backend="numpy")

        # From README: geometric operations
        centered, means = polymer.center(ciffy.MOLECULE)
        distances = polymer.pairwise_distances(ciffy.RESIDUE)

        assert centered.coordinates is not None
        assert distances is not None


class TestBiochemistryAPI:
    """Validate biochemistry module API used in docs."""

    def test_residue_atom_access(self):
        """Test hierarchical atom access via Residue."""
        # Documented pattern: Residue.A.N1
        n1_index = Residue.A.N1.value
        c1p_index = Residue.A.C1p.value

        assert isinstance(n1_index, int)
        assert isinstance(c1p_index, int)

    def test_no_standalone_nucleotide_classes(self):
        """Ensure Adenosine/Guanosine classes don't exist (docs should use Residue.A/G)."""
        from ciffy import biochemistry

        assert not hasattr(
            biochemistry, "Adenosine"
        ), "Adenosine class exists - update docs/guides/selection.md"
        assert not hasattr(
            biochemistry, "Guanosine"
        ), "Guanosine class exists - update docs/guides/selection.md"
        assert not hasattr(
            biochemistry, "Cytosine"
        ), "Cytosine class exists - update docs/guides/selection.md"
        assert not hasattr(
            biochemistry, "Uridine"
        ), "Uridine class exists - update docs/guides/selection.md"

    def test_by_atom_requires_value(self):
        """Test that by_atom requires .value for enum access."""
        polymer = ciffy.load("tests/data/9GCM.cif")
        protein = polymer.by_type(ciffy.PROTEIN)

        if protein.size() > 0:
            # by_atom with integer works
            result = protein.by_atom(Residue.ALA.CA.value)
            assert result.size() >= 0

            # by_atom with raw enum doesn't work (returns 0)
            result_enum = protein.by_atom(Residue.ALA.CA)
            assert result_enum.size() == 0, "by_atom now accepts enum directly - update docs"


class TestAllExportsDocumented:
    """Verify all __all__ exports exist and are importable."""

    def test_all_exports_exist(self):
        """Verify all __all__ items are importable."""
        for name in ciffy.__all__:
            assert hasattr(ciffy, name), f"Missing export: {name}"

    def test_core_functions_exist(self):
        """Verify core functions documented in api.md exist."""
        # From api.md Core section
        assert callable(ciffy.load)
        assert ciffy.Polymer is not None

    def test_operations_exist(self):
        """Verify operations documented in api.md exist."""
        assert callable(ciffy.rmsd)
        assert callable(ciffy.align)
        assert callable(ciffy.kabsch_rotation)
        assert callable(ciffy.kabsch_align)
        assert callable(ciffy.tm_score)
        assert callable(ciffy.lddt)
        assert ciffy.Reduction is not None

    def test_types_exist(self):
        """Verify types documented in api.md exist."""
        assert ciffy.Scale is not None
        assert ciffy.Molecule is not None
        assert ciffy.DihedralType is not None

    def test_io_functions_exist(self):
        """Verify I/O functions documented in api.md exist."""
        assert callable(ciffy.write_cif)
        assert callable(ciffy.load_metadata)
        assert callable(ciffy.from_sequence)
        assert callable(ciffy.from_extract)

    def test_sampling_exists(self):
        """Verify sampling functions documented in api.md exist."""
        assert callable(ciffy.randomize_backbone)

    def test_ensemble_exists(self):
        """Verify Ensemble class documented in api.md exists."""
        assert ciffy.Ensemble is not None

    def test_dihedral_constants_exist(self):
        """Verify dihedral constants documented in api.md exist."""
        assert ciffy.PROTEIN_BACKBONE is not None
        assert ciffy.RNA_BACKBONE is not None
        assert ciffy.RNA_GLYCOSIDIC is not None
        assert ciffy.DIHEDRAL_ATOMS is not None
        assert ciffy.DIHEDRAL_NAME_TO_TYPE is not None

    def test_visualization_functions_exist(self):
        """Verify visualization functions documented in api.md exist."""
        assert callable(ciffy.to_defattr)
        assert callable(ciffy.plot_profile)
        assert callable(ciffy.contact_map)

    def test_convenience_aliases_exist(self):
        """Verify convenience aliases exist."""
        # Scale aliases
        assert ciffy.ATOM == ciffy.Scale.ATOM
        assert ciffy.RESIDUE == ciffy.Scale.RESIDUE
        assert ciffy.CHAIN == ciffy.Scale.CHAIN
        assert ciffy.MOLECULE == ciffy.Scale.MOLECULE

        # Molecule aliases
        assert ciffy.PROTEIN == ciffy.Molecule.PROTEIN
        assert ciffy.RNA == ciffy.Molecule.RNA
        assert ciffy.DNA == ciffy.Molecule.DNA


class TestPolymerMethods:
    """Verify Polymer class has all documented methods."""

    @pytest.fixture
    def polymer(self):
        return ciffy.load("tests/data/9MDS.cif")

    def test_documented_methods_exist(self, polymer):
        """Verify all methods listed in api.md Polymer section exist."""
        documented_methods = [
            "size",
            "sizes",
            "per",
            "molecule_type",
            "istype",
            "reduce",
            "rreduce",
            "expand",
            "count",
            "index",
            "center",
            "pairwise_distances",
            "align",
            "moment",
            "mask",
            "__getitem__",
            "by_index",
            "by_atom",
            "by_residue",
            "by_type",
            "poly",
            "hetero",
            "chains",
            "resolved",
            "strip",
            "backbone",
            "atom_names",
            "numpy",
            "torch",
            "to",
            "write",
            "with_coordinates",
        ]

        for method in documented_methods:
            assert hasattr(polymer, method), f"Missing Polymer method: {method}"

    def test_internal_coordinate_properties_exist(self, polymer):
        """Verify internal coordinate properties exist."""
        # These are documented in api.md
        assert hasattr(polymer, "distances")
        assert hasattr(polymer, "angles")
        assert hasattr(polymer, "dihedrals")
        assert hasattr(polymer, "dihedral")
        assert hasattr(polymer, "set_dihedral")
