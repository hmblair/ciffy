"""Tests for from_biotite conversion."""

import numpy as np
import pytest

try:
    import biotite.structure
    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False

pytestmark = pytest.mark.skipif(not HAS_BIOTITE, reason="biotite not installed")


class TestFromBiotiteVsCiffyLoad:
    """Cross-loader: biotite.load vs ciffy.load must agree on shared atoms."""

    def test_atoms_match_native_loader(self, any_cif, backend):
        """Every atom biotite considers non-HETATM must match ciffy's encoding."""
        from biotite.structure.io import load_structure
        import ciffy
        from ciffy.backend import to_numpy

        bt = load_structure(any_cif)
        p_bt = ciffy.from_biotite(bt, backend=backend)
        p_ciffy = ciffy.load(any_cif, backend=backend)

        # biotite marks some polymer residues (e.g. GTP) as HETATM,
        # so p_bt is a subset of p_ciffy. Match by exact coordinates.
        ciffy_by_coord = {
            tuple(row): i
            for i, row in enumerate(to_numpy(p_ciffy.coordinates))
        }

        coords_bt = to_numpy(p_bt.coordinates)
        atoms_bt = to_numpy(p_bt.atoms)
        elements_bt = to_numpy(p_bt.elements)
        atoms_ciffy = to_numpy(p_ciffy.atoms)
        elements_ciffy = to_numpy(p_ciffy.elements)
        names_bt = p_bt.atom_names()
        names_ciffy = p_ciffy.atom_names()

        for i in range(p_bt.size()):
            j = ciffy_by_coord[tuple(coords_bt[i])]
            assert names_bt[i] == names_ciffy[j]
            assert atoms_bt[i] == atoms_ciffy[j]
            assert elements_bt[i] == elements_ciffy[j]

    def test_biotite_export_matches_native_loader(self, any_cif):
        """ciffy.load -> .biotite() must match biotite's own loader atom-for-atom."""
        from biotite.structure.io import load_structure
        import ciffy

        bt_native = load_structure(any_cif)
        bt_native_poly = bt_native[~bt_native.hetero]

        p = ciffy.load(any_cif)
        bt_ciffy = p.biotite()

        ciffy_by_coord = {
            tuple(bt_ciffy.coord[i]): i
            for i in range(len(bt_ciffy))
        }

        for i in range(len(bt_native_poly)):
            key = tuple(bt_native_poly.coord[i])
            assert key in ciffy_by_coord
            j = ciffy_by_coord[key]
            assert bt_native_poly.atom_name[i] == bt_ciffy.atom_name[j]
            assert bt_native_poly.element[i] == bt_ciffy.element[j]
            assert bt_native_poly.res_name[i] == bt_ciffy.res_name[j]


class TestFromBiotiteRoundtrip:
    """Roundtrip: Polymer -> biotite -> Polymer."""

    def test_sequence_preserved(self, rna_polymer):
        bt = rna_polymer.biotite()
        p2 = ciffy.from_biotite(bt)

        assert (p2.sequence == rna_polymer.sequence).all()
        assert p2.sequence_str() == rna_polymer.sequence_str()

    def test_atoms_preserved(self, rna_polymer):
        bt = rna_polymer.biotite()
        p2 = ciffy.from_biotite(bt)

        assert p2.size() == rna_polymer.size()
        assert (p2.atoms == rna_polymer.atoms).all()
        assert (p2.elements == rna_polymer.elements).all()

    def test_protein_roundtrip(self, protein_polymer):
        bt = protein_polymer.biotite()
        p2 = ciffy.from_biotite(bt)

        assert p2.size() == protein_polymer.size()
        assert p2.sequence_str() == protein_polymer.sequence_str()
        assert (p2.atoms == protein_polymer.atoms).all()

    def test_coordinates_preserved(self, any_cif, backend):
        import ciffy
        from ciffy.backend import to_numpy

        p = ciffy.load(any_cif, backend=backend)
        bt = p.biotite()
        p2 = ciffy.from_biotite(bt, backend=backend)

        # Only non-HETATM residues survive the roundtrip, but for CIFs
        # where ciffy and biotite agree on all residues, sizes match.
        # Compare by coordinate matching for safety.
        ciffy_coords = to_numpy(p.coordinates)
        bt_coords = to_numpy(p2.coordinates)
        ciffy_by_coord = {tuple(row): i for i, row in enumerate(ciffy_coords)}

        for i in range(p2.size()):
            j = ciffy_by_coord[tuple(bt_coords[i])]
            assert np.array_equal(bt_coords[i], ciffy_coords[j])

    def test_multi_chain(self, multi_chain_polymer):
        import ciffy
        bt = multi_chain_polymer.biotite()
        p2 = ciffy.from_biotite(bt)

        assert p2.size(ciffy.Scale.CHAIN) == multi_chain_polymer.size(ciffy.Scale.CHAIN)
        assert (p2.molecule_types == multi_chain_polymer.molecule_types).all()

    def test_chain_names_preserved(self, any_cif, backend):
        import ciffy
        p = ciffy.load(any_cif, backend=backend)
        bt = p.biotite()
        p2 = ciffy.from_biotite(bt, backend=backend)
        assert p2.names == p.names

    def test_custom_annotations_imported(self, rna_polymer):
        bt = rna_polymer.biotite()
        bt.set_annotation("occupancy", np.ones(len(bt), dtype=np.float32))
        bt.set_annotation("charge", np.zeros(len(bt), dtype=np.int32))

        p2 = ciffy.from_biotite(bt)
        assert (p2.occupancy == 1.0).all()
        assert (p2.charge == 0).all()


class TestFromBiotiteEdgeCases:
    """Edge cases and error handling."""

    def test_empty_array(self):
        from biotite.structure import AtomArray
        p = ciffy.from_biotite(AtomArray(0))
        assert p.empty()

    def test_all_hetatm_returns_empty(self):
        from biotite.structure import AtomArray
        arr = AtomArray(1)
        arr.coord = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        arr.atom_name = np.array(["MG"], dtype="U6")
        arr.element = np.array(["MG"], dtype="U2")
        arr.res_name = np.array(["MG"], dtype="U5")
        arr.res_id = np.array([1], dtype=np.int32)
        arr.chain_id = np.array(["A"], dtype="U4")
        arr.hetero = np.array([True])

        p = ciffy.from_biotite(arr)
        assert p.empty()

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="AtomArray"):
            ciffy.from_biotite("not an array")

    def test_unknown_residue_raises(self):
        from biotite.structure import AtomArray
        arr = AtomArray(1)
        arr.coord = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        arr.atom_name = np.array(["CA"], dtype="U6")
        arr.element = np.array(["C"], dtype="U2")
        arr.res_name = np.array(["ZZZ"], dtype="U5")
        arr.res_id = np.array([1], dtype=np.int32)
        arr.chain_id = np.array(["A"], dtype="U4")
        with pytest.raises(ValueError, match="Unknown residue"):
            ciffy.from_biotite(arr)

    def test_hetatm_filtered(self):
        from biotite.structure import AtomArray
        arr = AtomArray(2)
        arr.coord = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        arr.atom_name = np.array(["P", "MG"], dtype="U6")
        arr.element = np.array(["P", "MG"], dtype="U2")
        arr.res_name = np.array(["A", "MG"], dtype="U5")
        arr.res_id = np.array([1, 2], dtype=np.int32)
        arr.chain_id = np.array(["A", "B"], dtype="U4")
        arr.hetero = np.array([False, True])

        p = ciffy.from_biotite(arr)
        assert p.size() == 1

    def test_backend_torch(self):
        from tests.utils import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import torch

        bt = ciffy.template("acgu").biotite()
        p = ciffy.from_biotite(bt, backend="torch")
        assert isinstance(p.coordinates, torch.Tensor)


class TestCLookups:
    """Direct tests for the C gperf lookup bindings."""

    def test_lookup_atom(self):
        from ciffy._c import _lookup_atom
        assert _lookup_atom("A_P") > 0
        assert _lookup_atom("ALA_CA") > 0
        assert _lookup_atom("INVALID_ATOM") == -1

    def test_lookup_element(self):
        from ciffy._c import _lookup_element
        assert _lookup_element("C") > 0
        assert _lookup_element("N") > 0
        assert _lookup_element("ZZ") == -1

    def test_lookup_residue(self):
        from ciffy._c import _lookup_residue
        assert _lookup_residue("A") > 0
        assert _lookup_residue("ALA") > 0
        assert _lookup_residue("ZZZ") == -1

    def test_batch_atoms(self):
        from ciffy._c import _lookup_atoms_batch, _lookup_atom
        res = np.array(["A", "ALA", "G"], dtype="U5")
        atoms = np.array(["C1'", "CA", "N9"], dtype="U6")
        result = _lookup_atoms_batch(res, atoms)
        assert result[0] == _lookup_atom("A_C1'")
        assert result[1] == _lookup_atom("ALA_CA")
        assert result[2] == _lookup_atom("G_N9")

    def test_batch_elements(self):
        from ciffy._c import _lookup_elements_batch, _lookup_element
        elems = np.array(["C", "N", "FE"], dtype="U2")
        result = _lookup_elements_batch(elems)
        assert result[0] == _lookup_element("C")
        assert result[1] == _lookup_element("N")
        assert result[2] == _lookup_element("FE")

    def test_batch_residues(self):
        from ciffy._c import _lookup_residues_batch, _lookup_residue
        res = np.array(["A", "ALA", "DA"], dtype="U5")
        result = _lookup_residues_batch(res)
        assert result[0] == _lookup_residue("A")
        assert result[1] == _lookup_residue("ALA")
        assert result[2] == _lookup_residue("DA")


# Module-level import for tests that use fixtures (fixtures return Polymer
# objects that already have ciffy imported)
import ciffy
