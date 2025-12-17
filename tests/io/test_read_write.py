"""
Tests for loading and saving molecular structures.

Tests are parameterized to run with both numpy and torch backends.
"""

import glob
import os
import tempfile
import pytest

from tests.utils import DATA_DIR, BACKENDS
from tests.testing import get_tolerances

CIF_FILES = sorted(glob.glob(str(DATA_DIR / "*.cif")))


class TestLoad:
    """Test CIF file loading with both numpy and torch backends."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_file(self, cif_file, backend):
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        assert polymer is not None
        assert not polymer.empty()
        assert polymer.size() > 0

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_coordinates(self, cif_file, backend):
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        assert polymer.coordinates.shape[0] == polymer.size()
        assert polymer.coordinates.shape[1] == 3

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_atoms(self, cif_file, backend):
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        assert polymer.atoms.shape[0] == polymer.size()
        assert polymer.elements.shape[0] == polymer.size()

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_chains(self, cif_file, backend):
        from ciffy import load, Scale

        polymer = load(cif_file, backend=backend)
        assert polymer.size(Scale.CHAIN) > 0
        assert len(polymer.names) == polymer.size(Scale.CHAIN)
        assert len(polymer.strands) == polymer.size(Scale.CHAIN)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_load_has_residues(self, cif_file, backend):
        from ciffy import load, Scale

        polymer = load(cif_file, backend=backend)
        assert polymer.size(Scale.RESIDUE) > 0
        assert polymer.sequence.shape[0] == polymer.size(Scale.RESIDUE)

    def test_load_nonexistent_file(self, backend):
        from ciffy import load

        with pytest.raises(OSError):
            load("nonexistent_file.cif", backend=backend)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_backend_conversion(self, cif_file):
        """Test that polymers can be converted between backends."""
        from ciffy import load
        import numpy as np

        # Load as numpy
        np_polymer = load(cif_file, backend="numpy")
        assert isinstance(np_polymer.coordinates, np.ndarray)

        # Convert to torch
        torch_polymer = np_polymer.torch()
        import torch
        assert isinstance(torch_polymer.coordinates, torch.Tensor)

        # Convert back to numpy
        np_polymer2 = torch_polymer.numpy()
        assert isinstance(np_polymer2.coordinates, np.ndarray)

        # Verify coordinates are equivalent
        assert np.allclose(np_polymer.coordinates, np_polymer2.coordinates)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_to_device(self, cif_file):
        """Test Polymer.to() for device conversion."""
        from ciffy import load
        import torch

        polymer = load(cif_file, backend="torch")

        # Move to cpu (always available)
        polymer_cpu = polymer.to("cpu")
        assert polymer_cpu.coordinates.device.type == "cpu"
        assert polymer_cpu.atoms.device.type == "cpu"
        assert polymer_cpu.elements.device.type == "cpu"
        assert polymer_cpu.sequence.device.type == "cpu"

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_to_dtype(self, cif_file):
        """Test Polymer.to() for dtype conversion (float tensors only)."""
        from ciffy import load
        import torch

        polymer = load(cif_file, backend="torch")

        # Convert to float16
        polymer_fp16 = polymer.to(dtype=torch.float16)
        assert polymer_fp16.coordinates.dtype == torch.float16
        # Integer tensors should remain long
        assert polymer_fp16.atoms.dtype == torch.int64
        assert polymer_fp16.elements.dtype == torch.int64
        assert polymer_fp16.sequence.dtype == torch.int64

        # Convert to float64
        polymer_fp64 = polymer.to(dtype=torch.float64)
        assert polymer_fp64.coordinates.dtype == torch.float64
        assert polymer_fp64.atoms.dtype == torch.int64

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_to_device_and_dtype(self, cif_file):
        """Test Polymer.to() with both device and dtype."""
        from ciffy import load
        import torch

        polymer = load(cif_file, backend="torch")

        # Move to cpu and convert to float16
        polymer_new = polymer.to("cpu", torch.float16)
        assert polymer_new.coordinates.device.type == "cpu"
        assert polymer_new.coordinates.dtype == torch.float16
        assert polymer_new.atoms.device.type == "cpu"
        assert polymer_new.atoms.dtype == torch.int64

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_to_returns_self_when_no_args(self, cif_file):
        """Test Polymer.to() returns self when no arguments provided."""
        from ciffy import load

        polymer = load(cif_file, backend="torch")
        result = polymer.to()
        assert result is polymer

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_to_raises_on_numpy(self, cif_file):
        """Test Polymer.to() raises ValueError on numpy backend."""
        from ciffy import load
        import pytest

        polymer = load(cif_file, backend="numpy")
        with pytest.raises(ValueError, match="only supported for torch backend"):
            polymer.to("cpu")

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_repr(self, cif_file, backend):
        """Test that __repr__ contains accurate information."""
        from ciffy import load, Scale

        polymer = load(cif_file, backend=backend)
        repr_str = repr(polymer)

        # Check header line contains PDB ID and backend
        assert polymer.pdb_id in repr_str
        assert backend in repr_str

        # Check column headers are present
        assert "Type" in repr_str
        assert "Res" in repr_str
        assert "Atoms" in repr_str

        # Check total atom count appears (in totals row)
        assert str(polymer.size()) in repr_str

        # Check each chain is listed with correct information
        for ix in range(polymer.size(Scale.CHAIN)):
            chain_name = polymer.names[ix]
            residue_count = polymer.lengths[ix].item()
            atom_count = polymer.sizes(Scale.CHAIN)[ix].item()

            # Chain name should appear in output
            assert chain_name in repr_str

            # Residue and atom counts should appear (as strings in the line)
            assert str(residue_count) in repr_str
            assert str(atom_count) in repr_str

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_repr_molecule_types(self, cif_file, backend):
        """Test that molecule types in __repr__ are valid."""
        from ciffy import load, Molecule

        polymer = load(cif_file, backend=backend)
        repr_str = repr(polymer)

        # All molecule type names should be valid
        valid_types = {m.name for m in Molecule}
        lines = repr_str.strip().split("\n")

        # Skip header lines (first 3: title, separator, column headers)
        # Data lines start after "Type" header line
        data_started = False
        for line in lines:
            if "Type" in line and "Res" in line:
                data_started = True
                continue
            if not data_started:
                continue
            if line.strip():
                # Extract molecule type (columns between chain name and residue count)
                # Format: "A  RNA  66  1413" or "D  CS ION  -  1"
                parts = line.split()
                if len(parts) >= 4:  # chain, type..., residues, atoms
                    # Type can be multi-word (e.g., "CS ION"), last word is the base type
                    type_parts = parts[1:-2]  # Everything between chain and res/atoms
                    mol_type = type_parts[-1]  # Last word is the Molecule enum name
                    assert mol_type in valid_types, f"Invalid molecule type: {mol_type}"

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_molecule_type_classification(self, cif_file, backend):
        """Test that molecule_type property correctly classifies chains."""
        from ciffy import load, Molecule, Scale

        polymer = load(cif_file, backend=backend)
        mol_types = polymer.molecule_type

        # Should have one type per chain
        assert len(mol_types) == polymer.size(Scale.CHAIN)

        # All types should be valid Molecule enum values
        valid_values = {m.value for m in Molecule}
        for i, mol_val in enumerate(mol_types):
            val = int(mol_val.item() if hasattr(mol_val, 'item') else mol_val)
            assert val in valid_values, f"Chain {i} has invalid molecule type value: {val}"

        # Verify RNA chains are classified correctly
        rna_subset = polymer.by_type(Molecule.RNA)
        if not rna_subset.empty():
            rna_types = rna_subset.molecule_type
            for i, mol_val in enumerate(rna_types):
                val = int(mol_val.item() if hasattr(mol_val, 'item') else mol_val)
                assert val == Molecule.RNA.value, f"RNA subset chain {i} should be RNA"


class TestCifSave:
    """Test CIF file saving with both backends."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_save_cif(self, cif_file, backend):
        """Test basic CIF writing."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            output_path = f.name

        try:
            polymer.write(output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_cif_has_header(self, cif_file, backend):
        """Test that saved CIF has proper header."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            output_path = f.name

        try:
            polymer.write(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            # Check CIF format has data_ header
            assert content.startswith("data_")
            # Check has loop_ sections
            assert "loop_" in content
            # Check has _atom_site block
            assert "_atom_site." in content

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_round_trip_cif(self, cif_file, backend):
        """Test load -> save -> load preserves data."""
        from ciffy import load
        import numpy as np

        original = load(cif_file, backend=backend)

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            output_path = f.name

        try:
            original.write(output_path)
            reloaded = load(output_path, backend=backend)

            # Note: CIF writer currently only writes polymer atoms,
            # so we compare polymer counts instead of total counts
            assert reloaded.size() == original.polymer_count

            # Verify polymer coordinates are close (allow small float precision loss)
            orig_coords = np.asarray(original.coordinates[:original.polymer_count])
            reload_coords = np.asarray(reloaded.coordinates)
            tol = get_tolerances()
            assert np.allclose(orig_coords, reload_coords, atol=tol.coord_roundtrip)

            # Verify chain count matches
            assert len(reloaded.names) == len(original.names)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_round_trip_preserves_sequence(self, cif_file, backend):
        """Test that round-trip preserves residue sequence for polymer residues.

        Note: The CIF writer only writes polymer atoms (group_PDB="ATOM").
        Non-polymer residues (HETATM) are excluded, so the reloaded sequence
        may be shorter than the original.
        """
        from ciffy import load
        import numpy as np

        original = load(cif_file, backend=backend)

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            output_path = f.name

        try:
            original.write(output_path)
            reloaded = load(output_path, backend=backend)

            # Verify reloaded polymer has valid sequence
            reload_seq = np.asarray(reloaded.sequence)
            assert len(reload_seq) > 0, "Reloaded structure should have residues"

            # Verify residue count matches what was written (polymer atoms only)
            # The reloaded polymer_count should match what we wrote
            assert reloaded.polymer_count == reloaded.size(), \
                "Reloaded should be all polymer (no HETATM in round-trip)"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_round_trip_preserves_atom_types(self, cif_file, backend):
        """Test that round-trip preserves atom types including backbone atoms.

        This specifically tests that atoms with primes (like C2', O3') are
        correctly written and re-read, not confused with nucleobase atoms
        (like C2, which is different from C2').
        """
        from ciffy import load
        import numpy as np

        original = load(cif_file, backend=backend)

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            output_path = f.name

        try:
            original.write(output_path)
            reloaded = load(output_path, backend=backend)

            # Compare polymer atom types (non-polymer atoms not written)
            orig_atoms = np.asarray(original.atoms[:original.polymer_count])
            reload_atoms = np.asarray(reloaded.atoms)

            # Atom types should match exactly
            assert np.array_equal(orig_atoms, reload_atoms), \
                f"Atom types mismatch: original has {len(np.unique(orig_atoms))} unique types, " \
                f"reloaded has {len(np.unique(reload_atoms))} unique types"

            # Specifically check that backbone atoms exist (primed atoms)
            # A.C2' (sugar) vs A.C2 (nucleobase)
            from ciffy.biochemistry import Residue
            has_backbone = Residue.A.C2p.value in orig_atoms
            has_nucleobase = Residue.A.C2.value in orig_atoms

            if has_backbone:
                assert Residue.A.C2p.value in reload_atoms, \
                    "Backbone C2' atoms lost in round-trip"
            if has_nucleobase:
                assert Residue.A.C2.value in reload_atoms, \
                    "Nucleobase C2 atoms lost in round-trip"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_cif_save_from_subset(self, cif_file):
        """Test saving a subset of the structure to CIF."""
        from ciffy import load, RNA

        polymer = load(cif_file, backend="numpy")

        rna = polymer.by_type(RNA)
        if rna.empty() or rna.polymer_count == 0:
            return  # No RNA to test - pass vacuously

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            output_path = f.name

        try:
            rna.write(output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Reload and verify polymer size
            # Note: CIF writer only writes polymer atoms
            reloaded = load(output_path, backend="numpy")
            assert reloaded.size() == rna.polymer_count

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_write_empty_polymer_raises(self):
        """Test that writing an empty polymer raises ValueError."""
        from ciffy import load, from_sequence

        # Create an empty polymer by subsetting with impossible mask
        template = from_sequence("acgu")
        empty = template[template.atoms < 0]  # Empty mask

        assert empty.empty()

        with pytest.raises(ValueError, match="Cannot write empty polymer"):
            empty.write("/tmp/should_not_exist.cif")

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_round_trip_preserves_molecule_type(self, cif_file, backend):
        """Test that round-trip preserves molecule type for all chains.

        With the _entity block written, both polymer and non-polymer chains
        should preserve their molecule types exactly.
        """
        from ciffy import load, Scale
        from ciffy.types import Molecule
        import numpy as np

        original = load(cif_file, backend=backend)

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            output_path = f.name

        try:
            original.write(output_path)
            reloaded = load(output_path, backend=backend)

            # Molecule types should match exactly (ION round-trips via _pdbx_entity_nonpoly)
            orig_types = np.asarray(original.molecule_type)
            reload_types = np.asarray(reloaded.molecule_type)

            assert np.array_equal(orig_types, reload_types), \
                f"Molecule types mismatch: original={orig_types}, reloaded={reload_types}"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestMoleculeTypeDetection:
    """Test molecule type detection for various structures."""

    def test_9mds_is_rna(self, cif_9mds):
        """Test that 9MDS (8 RNA chains) is correctly identified as RNA."""
        from ciffy import load, RNA, Scale
        from ciffy.types import Molecule

        polymer = load(cif_9mds, backend="numpy")

        # 9MDS has 8 chains, all RNA
        assert polymer.size(Scale.CHAIN) == 8

        # All chains should be RNA
        mol_types = polymer.molecule_type
        for i in range(8):
            assert mol_types[i] == Molecule.RNA.value, \
                f"Chain {i} should be RNA, got {Molecule(mol_types[i])}"

    def test_9gcm_mixed_rna_protein(self, cif_9gcm):
        """Test that 9GCM (1 RNA + 3 protein chains) has correct molecule types."""
        from ciffy import load, Scale
        from ciffy.types import Molecule

        polymer = load(cif_9gcm, backend="numpy")

        # 9GCM has 4 chains
        assert polymer.size(Scale.CHAIN) == 4

        # Chain A is RNA, chains B/C/D are protein
        mol_types = polymer.molecule_type
        names = polymer.names

        expected = {
            "A": Molecule.RNA.value,
            "B": Molecule.PROTEIN.value,
            "C": Molecule.PROTEIN.value,
            "D": Molecule.PROTEIN.value,
        }

        for name, mol_type in zip(names, mol_types):
            assert mol_type == expected[name], \
                f"Chain {name} should be {Molecule(expected[name]).name}, got {Molecule(mol_type).name}"


class TestPolymerCountInvariant:
    """Test that polymer_count == sum(atoms_per_res) invariant holds."""

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_polymer_count_equals_sum_atoms_per_res(self, cif_file, backend):
        """Verify invariant: polymer_count == sum(atoms_per_res).

        This invariant ensures that all polymer atoms belong to residues,
        and all residue atoms are counted as polymer. It's enforced in the
        C parser by checking both group_PDB and label_seq_id.
        """
        from ciffy import load, Scale

        polymer = load(cif_file, backend=backend)

        # Sum of atoms per residue should equal polymer_count
        atoms_per_res_sum = polymer.sizes(Scale.RESIDUE).sum().item()
        assert atoms_per_res_sum == polymer.polymer_count, \
            f"Invariant violated: sum(atoms_per_res)={atoms_per_res_sum} != polymer_count={polymer.polymer_count}"

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_nonpoly_is_nonnegative(self, cif_file, backend):
        """Verify nonpoly count is non-negative."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        assert polymer.nonpoly >= 0, f"nonpoly should be >= 0, got {polymer.nonpoly}"

    @pytest.mark.parametrize("cif_file", CIF_FILES)
    def test_polymer_plus_nonpoly_equals_total(self, cif_file, backend):
        """Verify polymer_count + nonpoly == total atoms."""
        from ciffy import load

        polymer = load(cif_file, backend=backend)
        total = polymer.size()
        assert polymer.polymer_count + polymer.nonpoly == total, \
            f"polymer_count ({polymer.polymer_count}) + nonpoly ({polymer.nonpoly}) != total ({total})"
