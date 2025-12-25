"""Tests for Atom and AtomGroup classes (2-class design)."""

import numpy as np
import pytest

from ciffy.biochemistry.atom import Atom, AtomGroup, build_atom_group


class TestAtom:
    """Tests for Atom (int subclass)."""

    def test_creation(self):
        atom = Atom("P", 2, 0)
        assert atom.name == "P"
        assert atom.value == 2  # .value property for compatibility
        assert int(atom) == 2   # IS an int
        assert atom.local == 0

    def test_is_int(self):
        atom = Atom("P", 2, 0)
        assert isinstance(atom, int)
        assert atom == 2
        assert atom + 1 == 3
        assert 2 + atom == 4

    def test_works_in_array_indexing(self):
        atom = Atom("P", 2, 0)
        arr = np.zeros(10)
        arr[atom] = 1.0  # Works directly without .value
        assert arr[2] == 1.0

    def test_repr(self):
        atom = Atom("P", 2, 0)
        assert repr(atom) == "Atom(P, 2)"

    def test_equality(self):
        atom1 = Atom("P", 2, 0)
        atom2 = Atom("P", 2, 0)
        atom3 = Atom("OP1", 3, 1)
        # Atoms compare as ints
        assert atom1 == atom2
        assert atom1 == 2
        assert atom1 != atom3
        assert atom1 != 3

    def test_hashable(self):
        atom = Atom("P", 2, 0)
        # Should be hashable for use in sets/dicts (as int)
        assert hash(atom) == hash(2)
        s = {atom}
        assert 2 in s


class TestAtomGroup:
    """Tests for AtomGroup class."""

    @pytest.fixture
    def simple_group(self) -> AtomGroup:
        """A simple flat AtomGroup."""
        return AtomGroup("A", {
            "P": Atom("P", 2, 0),
            "OP1": Atom("OP1", 3, 1),
            "OP2": Atom("OP2", 4, 2),
            "C5p": Atom("C5p", 5, 3),
        })

    @pytest.fixture
    def hierarchical_group(self) -> AtomGroup:
        """A hierarchical AtomGroup (like Sugar.C5p)."""
        return AtomGroup("Sugar", {
            "C5p": AtomGroup("C5p", {
                "A": Atom("A", 5, 0),
                "G": Atom("G", 45, 0),
            }),
            "C4p": AtomGroup("C4p", {
                "A": Atom("A", 6, 1),
                "G": Atom("G", 46, 1),
            }),
        })

    def test_attribute_access(self, simple_group: AtomGroup):
        atom = simple_group.P
        assert isinstance(atom, Atom)
        assert int(atom) == 2

    def test_attribute_access_nested(self, hierarchical_group: AtomGroup):
        # Access nested group
        c5p = hierarchical_group.C5p
        assert isinstance(c5p, AtomGroup)

        # Access atom through nested group
        atom = hierarchical_group.C5p.A
        assert isinstance(atom, Atom)
        assert int(atom) == 5

    def test_attribute_error(self, simple_group: AtomGroup):
        with pytest.raises(AttributeError, match="has no member 'XYZ'"):
            _ = simple_group.XYZ

    def test_index_flat(self, simple_group: AtomGroup):
        idx = simple_group.index()
        assert isinstance(idx, np.ndarray)
        assert idx.dtype == np.int64
        np.testing.assert_array_equal(idx, [2, 3, 4, 5])

    def test_index_hierarchical(self, hierarchical_group: AtomGroup):
        # Index should flatten and sort all nested values
        idx = hierarchical_group.index()
        np.testing.assert_array_equal(idx, [5, 6, 45, 46])

        # Nested group index
        c5p_idx = hierarchical_group.C5p.index()
        np.testing.assert_array_equal(c5p_idx, [5, 45])

    def test_iteration(self, simple_group: AtomGroup):
        atoms = list(simple_group)
        assert len(atoms) == 4
        assert all(isinstance(a, Atom) for a in atoms)
        assert atoms[0].name == "P"

    def test_iteration_hierarchical_skips_groups(self, hierarchical_group: AtomGroup):
        # Iteration should yield nothing for pure hierarchy
        atoms = list(hierarchical_group)
        assert len(atoms) == 0

        # Nested group iteration should yield atoms
        c5p_atoms = list(hierarchical_group.C5p)
        assert len(c5p_atoms) == 2

    def test_len(self, simple_group: AtomGroup):
        assert len(simple_group) == 4

    def test_contains_string(self, simple_group: AtomGroup):
        assert "P" in simple_group
        assert "XYZ" not in simple_group

    def test_contains_int(self, simple_group: AtomGroup):
        assert 2 in simple_group
        assert 999 not in simple_group

    def test_contains_atom(self, simple_group: AtomGroup):
        atom = Atom("P", 2, 0)
        assert atom in simple_group

    def test_repr(self, simple_group: AtomGroup, hierarchical_group: AtomGroup):
        assert "4 atoms" in repr(simple_group)
        assert "subgroups" in repr(hierarchical_group)

    def test_items(self, simple_group: AtomGroup):
        items = list(simple_group.items())
        assert len(items) == 4
        name, atom = items[0]
        assert name == "P"
        assert int(atom) == 2

    def test_dict(self, simple_group: AtomGroup):
        d = simple_group.dict()
        assert d == {"P": 2, "OP1": 3, "OP2": 4, "C5p": 5}

    def test_list(self, simple_group: AtomGroup):
        names = simple_group.list()
        assert names == ["P", "OP1", "OP2", "C5p"]


class TestAtomGroupFiltering:
    """Tests for AtomGroup filtering methods."""

    @pytest.fixture
    def group(self) -> AtomGroup:
        return AtomGroup("A", {
            "P": Atom("P", 2, 0),
            "OP1": Atom("OP1", 3, 1),
            "OP2": Atom("OP2", 4, 2),
            "C5p": Atom("C5p", 5, 3),
            "H5p": Atom("H5p", 6, 4),
        })

    def test_subset(self, group: AtomGroup):
        subset = group.subset({2, 3, 5})
        assert isinstance(subset, AtomGroup)
        assert len(subset) == 3
        assert "P" in subset
        assert "OP1" in subset
        assert "C5p" in subset
        assert "H5p" not in subset

    def test_subset_renumbers_locals(self, group: AtomGroup):
        subset = group.subset({3, 5})  # OP1 and C5p
        assert subset.OP1.local == 0
        assert subset.C5p.local == 1

    def test_exclude(self, group: AtomGroup):
        subset = group.exclude({"H5p"})
        assert len(subset) == 4
        assert "H5p" not in subset
        assert "P" in subset

    def test_exclude_renumbers_locals(self, group: AtomGroup):
        subset = group.exclude({"OP1", "OP2"})
        assert subset.P.local == 0
        assert subset.C5p.local == 1
        assert subset.H5p.local == 2

    def test_filter_predicate(self, group: AtomGroup):
        # Filter to atoms with value < 5
        subset = group.filter(lambda a: int(a) < 5)
        assert len(subset) == 3
        np.testing.assert_array_equal(subset.index(), [2, 3, 4])

    def test_filter_empty_result(self, group: AtomGroup):
        subset = group.subset(set())
        assert len(subset) == 0
        assert len(subset.index()) == 0


class TestAtomGroupWithGeometry:
    """Tests for AtomGroup with geometry (replaces ResidueInfo)."""

    @pytest.fixture
    def residue(self) -> AtomGroup:
        """AtomGroup with geometry (like Residue.A)."""
        ideal = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ], dtype=np.float32)
        bonds = np.array([
            [0, 1],  # P-OP1
            [0, 2],  # P-OP2
            [0, 3],  # P-C5p
        ], dtype=np.int64)
        return AtomGroup(
            "A",
            {
                "P": Atom("P", 2, 0),
                "OP1": Atom("OP1", 3, 1),
                "OP2": Atom("OP2", 4, 2),
                "C5p": Atom("C5p", 5, 3),
            },
            value=0,
            molecule_type=1,  # RNA
            abbrev="a",
            ideal=ideal,
            bonds=bonds,
        )

    def test_attribute_delegation(self, residue: AtomGroup):
        # Access atoms through residue
        atom = residue.P
        assert isinstance(atom, Atom)
        assert int(atom) == 2

    def test_residue_metadata(self, residue: AtomGroup):
        assert residue.value == 0
        assert residue.molecule_type == 1
        assert residue.abbrev == "a"

    def test_iteration(self, residue: AtomGroup):
        atoms = list(residue)
        assert len(atoms) == 4

    def test_len(self, residue: AtomGroup):
        assert len(residue) == 4

    def test_index(self, residue: AtomGroup):
        np.testing.assert_array_equal(residue.index(), [2, 3, 4, 5])

    def test_repr(self, residue: AtomGroup):
        assert repr(residue) == "Residue.A"

    def test_ideal_access(self, residue: AtomGroup):
        assert residue.ideal.shape == (4, 3)
        assert residue.ideal.dtype == np.float32

    def test_bonds_access(self, residue: AtomGroup):
        assert residue.bonds.shape == (3, 2)
        assert residue.bonds.dtype == np.int64

    def test_n_atoms(self, residue: AtomGroup):
        assert residue.n_atoms == 4

    def test_n_bonds(self, residue: AtomGroup):
        assert residue.n_bonds == 3

    def test_bond_lengths_computed(self, residue: AtomGroup):
        lengths = residue.bond_lengths
        assert lengths is not None
        assert len(lengths) == 3
        # P-OP1 length should be 1.5
        assert abs(lengths[0] - 1.5) < 1e-6

    def test_subset_preserves_geometry(self, residue: AtomGroup):
        subset = residue.subset({2, 3, 5})  # P, OP1, C5p
        assert isinstance(subset, AtomGroup)
        assert subset.n_atoms == 3
        assert subset.ideal is not None
        assert subset.ideal.shape == (3, 3)

    def test_subset_filters_bonds(self, residue: AtomGroup):
        subset = residue.subset({2, 3, 5})  # P, OP1, C5p
        # Should have P-OP1 and P-C5p bonds (not P-OP2)
        assert subset.n_bonds == 2


class TestSubsetGeometry:
    """Tests for geometry filtering in subsets."""

    @pytest.fixture
    def residue(self) -> AtomGroup:
        ideal = np.array([
            [0.0, 0.0, 0.0],   # P
            [1.5, 0.0, 0.0],   # OP1
            [-1.5, 0.0, 0.0],  # OP2
            [0.0, 1.5, 0.0],   # C5p
            [0.0, 2.5, 0.0],   # H5p
        ], dtype=np.float32)
        bonds = np.array([
            [0, 1],  # P-OP1
            [0, 2],  # P-OP2
            [0, 3],  # P-C5p
            [3, 4],  # C5p-H5p
        ], dtype=np.int64)
        return AtomGroup(
            "A",
            {
                "P": Atom("P", 2, 0),
                "OP1": Atom("OP1", 3, 1),
                "OP2": Atom("OP2", 4, 2),
                "C5p": Atom("C5p", 5, 3),
                "H5p": Atom("H5p", 6, 4),
            },
            value=0,
            molecule_type=1,
            abbrev="a",
            ideal=ideal,
            bonds=bonds,
        )

    def test_ideal_coordinates_filtered(self, residue: AtomGroup):
        subset = residue.subset({2, 3, 5})  # P, OP1, C5p
        assert subset.ideal.shape == (3, 3)
        # Check coordinates match original (by local index)
        np.testing.assert_array_equal(subset.ideal[0], [0.0, 0.0, 0.0])   # P
        np.testing.assert_array_equal(subset.ideal[1], [1.5, 0.0, 0.0])   # OP1
        np.testing.assert_array_equal(subset.ideal[2], [0.0, 1.5, 0.0])   # C5p

    def test_bonds_filtered_and_renumbered(self, residue: AtomGroup):
        # Include P, OP1, C5p - should get P-OP1, P-C5p bonds
        subset = residue.subset({2, 3, 5})
        assert subset.n_bonds == 2
        # Bonds should be renumbered to local indices
        bonds_list = subset.bonds.tolist()
        assert [0, 1] in bonds_list  # P-OP1 (local 0-1)
        assert [0, 2] in bonds_list  # P-C5p (local 0-2)

    def test_bond_lengths_from_subset(self, residue: AtomGroup):
        subset = residue.subset({2, 3, 5})
        lengths = subset.bond_lengths
        assert lengths is not None
        assert len(lengths) == 2
        # P-OP1 length should be 1.5
        assert abs(lengths[0] - 1.5) < 1e-6

    def test_atom_to_local(self, residue: AtomGroup):
        subset = residue.subset({2, 5, 6})  # P, C5p, H5p
        mapping = subset.atom_to_local
        assert mapping[2] == 0  # P
        assert mapping[5] == 1  # C5p
        assert mapping[6] == 2  # H5p

    def test_empty_subset(self, residue: AtomGroup):
        subset = residue.subset(set())
        assert subset.n_atoms == 0
        assert subset.n_bonds == 0
        assert subset.ideal is None or len(subset.ideal) == 0


class TestIntegration:
    """Integration tests for the complete atom system."""

    def test_hierarchical_access_pattern(self):
        """Test Sugar.C5p.A pattern."""
        # Create a Sugar-like hierarchical group
        sugar = AtomGroup("Sugar", {
            "C5p": AtomGroup("C5p", {
                "A": Atom("A", 5, 0),
                "G": Atom("G", 45, 0),
                "C": Atom("C", 85, 0),
                "U": Atom("U", 125, 0),
            }),
            "C4p": AtomGroup("C4p", {
                "A": Atom("A", 6, 1),
                "G": Atom("G", 46, 1),
            }),
        })

        # Access patterns - atoms are ints now
        assert sugar.C5p.A == 5
        assert sugar.C5p.G == 45
        np.testing.assert_array_equal(sugar.C5p.index(), [5, 45, 85, 125])
        np.testing.assert_array_equal(sugar.index(), [5, 6, 45, 46, 85, 125])

    def test_ml_workflow(self):
        """Test typical ML workflow: create subset, get geometry."""
        ideal = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.3, 1.2, 0.0],
            [2.3, 2.4, 0.0],
            [1.5, -1.5, 0.0],
        ], dtype=np.float32)
        bonds = np.array([
            [0, 1],  # N-CA
            [1, 2],  # CA-C
            [2, 3],  # C-O
            [1, 4],  # CA-CB
        ], dtype=np.int64)

        residue = AtomGroup(
            "ALA",
            {
                "N": Atom("N", 100, 0),
                "CA": Atom("CA", 101, 1),
                "C": Atom("C", 102, 2),
                "O": Atom("O", 103, 3),
                "CB": Atom("CB", 104, 4),
            },
            value=28,
            molecule_type=2,  # PROTEIN
            abbrev="A",
            ideal=ideal,
            bonds=bonds,
        )

        # ML workflow: get backbone subset
        backbone_indices = {100, 101, 102, 103}  # N, CA, C, O
        subset = residue.subset(backbone_indices)

        # Use in model
        assert subset.n_atoms == 4
        assert subset.ideal.shape == (4, 3)
        assert subset.n_bonds == 3  # N-CA, CA-C, C-O

        # Get local indices for model input
        atom_to_local = subset.atom_to_local
        assert atom_to_local[100] == 0  # N is first
        assert atom_to_local[101] == 1  # CA is second


class TestBuildAtomGroup:
    """Tests for build_atom_group function."""

    def test_basic_creation(self):
        """Test basic hierarchical group creation."""
        # Create mock residue AtomGroups
        residue_a = AtomGroup("A", {
            "N1": Atom("N1", 10, 0),
            "C2": Atom("C2", 11, 1),
            "N3": Atom("N3", 12, 2),
        })
        residue_g = AtomGroup("G", {
            "N1": Atom("N1", 50, 0),
            "C2": Atom("C2", 51, 1),
            "N3": Atom("N3", 52, 2),
        })

        # Build hierarchical group
        purines = build_atom_group("Purines", [("A", residue_a), ("G", residue_g)])

        # Check structure
        assert purines.name == "Purines"
        assert hasattr(purines, "N1")
        assert hasattr(purines, "C2")
        assert hasattr(purines, "N3")

    def test_hierarchical_access(self):
        """Test hierarchical access patterns."""
        residue_a = AtomGroup("A", {
            "N1": Atom("N1", 10, 0),
            "C5": Atom("C5", 15, 5),
        })
        residue_g = AtomGroup("G", {
            "N1": Atom("N1", 50, 0),
            "C5": Atom("C5", 55, 5),
        })

        group = build_atom_group("Group", [("A", residue_a), ("G", residue_g)])

        # Access specific atom - atoms are ints
        assert group.N1.A == 10
        assert group.N1.G == 50
        assert group.C5.A == 15

        # Get all values for an atom
        np.testing.assert_array_equal(group.N1.index(), [10, 50])
        np.testing.assert_array_equal(group.C5.index(), [15, 55])

    def test_top_level_index(self):
        """Test that top-level index() returns all values."""
        residue_a = AtomGroup("A", {
            "N1": Atom("N1", 10, 0),
            "C2": Atom("C2", 11, 1),
        })
        residue_g = AtomGroup("G", {
            "N1": Atom("N1", 50, 0),
            "C2": Atom("C2", 51, 1),
        })

        group = build_atom_group("Group", [("A", residue_a), ("G", residue_g)])
        np.testing.assert_array_equal(group.index(), [10, 11, 50, 51])

    def test_atom_filter(self):
        """Test filtering to specific atoms."""
        residue = AtomGroup("A", {
            "N1": Atom("N1", 10, 0),
            "C2": Atom("C2", 11, 1),
            "N3": Atom("N3", 12, 2),
            "C4": Atom("C4", 13, 3),
        })

        # Only include N atoms
        group = build_atom_group("NAtoms", [("A", residue)], {"N1", "N3"})

        assert hasattr(group, "N1")
        assert hasattr(group, "N3")
        assert not hasattr(group, "C2")
        assert not hasattr(group, "C4")

    def test_with_geometry_source(self):
        """Test that AtomGroup sources with geometry work."""
        ideal = np.zeros((2, 3), dtype=np.float32)
        bonds = np.array([[0, 1]], dtype=np.int64)

        residue_a = AtomGroup(
            "A",
            {
                "P": Atom("P", 2, 0),
                "O5p": Atom("O5p", 3, 1),
            },
            value=0,
            molecule_type=1,
            abbrev="a",
            ideal=ideal,
            bonds=bonds,
        )

        residue_g = AtomGroup(
            "G",
            {
                "P": Atom("P", 42, 0),
                "O5p": Atom("O5p", 43, 1),
            },
            value=1,
            molecule_type=1,
            abbrev="g",
            ideal=ideal,
            bonds=bonds,
        )

        group = build_atom_group("Phosphate", [("A", residue_a), ("G", residue_g)])

        assert group.P.A == 2
        assert group.P.G == 42
        np.testing.assert_array_equal(group.O5p.index(), [3, 43])

    def test_values_match_source(self):
        """Test that values are preserved from source."""
        source = AtomGroup("A", {
            "X": Atom("X", 999, 0),  # Unusual value
        })

        group = build_atom_group("G", [("A", source)])
        assert group.X.A == 999  # Value preserved
