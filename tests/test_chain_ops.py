"""Tests for chain operations (join and extend)."""

import numpy as np
import pytest

import ciffy
from ciffy import Residue, Scale, join
from ciffy import from_sequence

# Identity rotation + Z-axis translation for ideal backbone spacing
LINEAR_EXTEND_TRANSFORM = np.array([0, 0, 0, 0, 0, 6.0], dtype=np.float32)


def template_with_coords(sequence: str, backend: str = "numpy") -> ciffy.Polymer:
    """Create a template with ideal coordinates for testing.

    This helper exists because from_sequence() now returns templates without
    coordinates. For tests that need coordinates, we build them manually.
    """
    template = from_sequence(sequence, backend=backend)
    if template.empty():
        return template

    # Build ideal coordinates by extending from empty
    poly = ciffy.Polymer()
    if backend == "torch":
        poly = poly.torch()
    sequences = [sequence] if isinstance(sequence, str) else sequence

    for seq in sequences:
        residue_indices = list(template.sequence[:len(seq)])
        for i, res_idx in enumerate(residue_indices):
            residue = Residue.from_index(int(res_idx))
            is_first = (i == 0)
            is_last = (i == len(residue_indices) - 1)

            atom_group = residue.terminal(start=is_first, end=is_last)
            atoms = atom_group.index()
            elements = atom_group.elements()
            coords = atom_group.ideal

            # Convert to torch if needed
            if backend == "torch":
                import torch
                coords = torch.from_numpy(coords)
                atoms = torch.from_numpy(atoms)
                elements = torch.from_numpy(elements)

            if poly.empty():
                poly = poly._append(residue, coords, atoms=atoms, elements=elements)
            else:
                transform = LINEAR_EXTEND_TRANSFORM
                if backend == "torch":
                    import torch
                    transform = torch.from_numpy(transform)
                poly = poly._append(residue, coords, transform, atoms=atoms, elements=elements)

    return poly


def extend_with_linear(poly, residue):
    """Helper to extend polymer with linear extension (for backward-compatible tests)."""
    # Get new residue data (internal, no terminals)
    atom_group = residue.terminal(start=False, end=False)
    atoms = atom_group.index()
    elements = atom_group.elements()
    coords = atom_group.ideal

    transform = LINEAR_EXTEND_TRANSFORM

    # Convert to match polymer backend
    if poly.backend == "torch":
        import torch
        coords = torch.from_numpy(coords)
        transform = torch.from_numpy(transform)
        atoms = torch.from_numpy(atoms)
        elements = torch.from_numpy(elements)

    return poly._append(residue, coords, transform, atoms=atoms, elements=elements)


class TestJoin:
    """Tests for ciffy.join() function."""

    def test_join_two_polymers(self):
        """Join two single-chain RNA polymers."""
        p1 = template_with_coords("ac")
        p2 = template_with_coords("gu")

        combined = join(p1, p2)

        # Should have 2 chains
        assert combined.size(Scale.CHAIN) == 2
        # Should have 4 residues total
        assert combined.size(Scale.RESIDUE) == 4
        # Should have all atoms from both
        assert combined.size() == p1.size() + p2.size()
        # Sequence should be concatenated
        assert combined.sequence_str() == "acgu"
        # Should have both chain names
        assert len(combined.names) == 2

    def test_join_three_polymers(self):
        """Join three polymers."""
        p1 = template_with_coords("a")
        p2 = template_with_coords("c")
        p3 = template_with_coords("g")

        combined = join(p1, p2, p3)

        assert combined.size(Scale.CHAIN) == 3
        assert combined.size(Scale.RESIDUE) == 3
        assert combined.sequence_str() == "acg"

    def test_join_multichain_polymers(self):
        """Join polymers that already have multiple chains."""
        # Build multi-chain by joining single chains
        p1 = join(template_with_coords("ac"), template_with_coords("gu"))  # 2 chains
        p2 = template_with_coords("aa")  # 1 chain

        combined = join(p1, p2)

        assert combined.size(Scale.CHAIN) == 3
        assert combined.size(Scale.RESIDUE) == 6

    def test_join_single_polymer(self):
        """Join with single polymer returns a copy."""
        p = template_with_coords("acgu")

        combined = join(p)

        assert combined.size(Scale.CHAIN) == 1
        assert combined.size(Scale.RESIDUE) == 4
        # Should be a copy, not the same object
        assert combined is not p
        # Modifying one shouldn't affect the other
        assert np.allclose(combined.coordinates, p.coordinates)

    def test_join_empty_polymers(self):
        """Join with empty polymers skips them."""
        p1 = template_with_coords("ac")
        p_empty = from_sequence("")

        combined = join(p1, p_empty)

        assert combined.size(Scale.CHAIN) == 1
        assert combined.size(Scale.RESIDUE) == 2

    def test_join_all_empty(self):
        """Join all empty polymers returns empty."""
        p1 = from_sequence("")
        p2 = from_sequence("")

        combined = join(p1, p2)

        assert combined.empty()
        assert combined.size(Scale.CHAIN) == 0

    def test_join_no_polymers_error(self):
        """Join with no arguments raises error."""
        with pytest.raises(ValueError, match="at least one polymer"):
            join()

    def test_join_with_loaded_structure(self):
        """Join works with structures loaded from files (HETATM is separate)."""
        p1 = template_with_coords("ac")
        p2 = ciffy.load("tests/data/9MDS.cif")

        # Should work - HETATM atoms are now in separate HeteroAtoms container
        result = join(p1, p2)
        assert result.size() == p1.size() + p2.size()

    def test_join_preserves_pdb_id_when_same(self):
        """Join preserves PDB ID when all are the same."""
        p1 = template_with_coords("ac")
        p2 = template_with_coords("gu")
        # Set same pdb_id on both
        p1._pdb_id = "TEST"
        p2._pdb_id = "TEST"

        combined = join(p1, p2)

        assert combined.pdb_id == "TEST"

    def test_join_different_pdb_ids_becomes_joined(self):
        """Join with different PDB IDs becomes 'joined'."""
        p1 = template_with_coords("ac")
        p2 = template_with_coords("gu")
        # Manually set different IDs
        p1._pdb_id = "TEST1"
        p2._pdb_id = "TEST2"

        combined = join(p1, p2)

        assert combined.pdb_id == "joined"

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_join_backend_preserved(self, backend):
        """Join preserves backend."""
        pytest.importorskip("torch")
        p1 = template_with_coords("ac", backend=backend)
        p2 = template_with_coords("gu", backend=backend)

        combined = join(p1, p2)

        assert combined.backend == backend


class TestAppend:
    """Tests for Polymer._append() method."""

    def test_extend_rna(self):
        """Extend RNA chain with a new residue."""
        p = template_with_coords("ac")
        initial_size = p.size()
        initial_res = p.size(Scale.RESIDUE)

        # Extend with guanine using helper
        extended = extend_with_linear(p, Residue.G)

        # Should have one more residue
        assert extended.size(Scale.RESIDUE) == initial_res + 1
        # Should have more atoms
        assert extended.size() > initial_size
        # Sequence should be updated
        assert extended.sequence_str() == "acg"
        # Still single chain
        assert extended.size(Scale.CHAIN) == 1

    def test_extend_preserves_original(self):
        """Extend returns new polymer, original unchanged."""
        p = template_with_coords("ac")
        original_size = p.size()
        original_seq = p.sequence_str()

        extended = extend_with_linear(p, Residue.G)

        # Original should be unchanged
        assert p.size() == original_size
        assert p.sequence_str() == original_seq

        # Extended should be different
        assert extended is not p
        assert extended.size() > original_size

    def test_extend_chain_multiple(self):
        """Extend chain multiple times."""
        p = template_with_coords("a")

        # Extend with c, g, u
        for residue in [Residue.C, Residue.G, Residue.U]:
            p = extend_with_linear(p, residue)

        assert p.size(Scale.RESIDUE) == 4
        assert p.sequence_str() == "acgu"

    def test_extend_multichain_extends_last(self):
        """Extend on multi-chain polymer extends the last chain."""
        # Build multi-chain by joining
        p = join(template_with_coords("ac"), template_with_coords("gu"))
        original_chains = p.size(Scale.CHAIN)
        original_residues = p.size(Scale.RESIDUE)

        atom_group = Residue.A.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
        transform = np.array([0, 0, 0, 0, 0, 6], dtype=np.float32)

        # Extend should add to the last chain
        p2 = p._append(Residue.A, coords, transform, atoms=atoms, elements=elements)

        assert p2.size(Scale.CHAIN) == original_chains  # Same number of chains
        assert p2.size(Scale.RESIDUE) == original_residues + 1  # One more residue

    def test_extend_template_error(self):
        """Extend fails on template (no coordinates)."""
        template = from_sequence("ac")
        atom_group = Residue.G.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal
        transform = np.array([0, 0, 0, 0, 0, 6], dtype=np.float32)

        with pytest.raises(AttributeError, match="coordinates"):
            template._append(Residue.G, coords, transform, atoms=atoms, elements=elements)

    def test_extend_with_custom_coords(self):
        """Extend with explicit coordinates."""
        p = template_with_coords("ac")

        # Get new residue data
        atom_group = Residue.G.terminal(start=False, end=False)
        atoms, elements, coords = atom_group.index(), atom_group.elements(), atom_group.ideal

        extended = p._append(Residue.G, coords, LINEAR_EXTEND_TRANSFORM, atoms=atoms, elements=elements)

        assert extended.size(Scale.RESIDUE) == 3
        assert extended.sequence_str() == "acg"

    def test_extend_residue_spacing(self):
        """Extended residues are properly spaced (non-overlapping)."""
        p = template_with_coords("a")
        extended = extend_with_linear(p, Residue.C)

        # Get centroids of each residue
        coords = extended.coordinates
        res_sizes = extended.counts(Scale.RESIDUE)
        first_res_atoms = res_sizes[0].item()

        first_centroid = coords[:first_res_atoms].mean(axis=0)
        second_centroid = coords[first_res_atoms:].mean(axis=0)

        # Residues should be spaced appropriately (non-overlapping)
        # Frame-based positioning extends along backbone direction, not necessarily global Z
        distance = np.linalg.norm(second_centroid - first_centroid)
        assert distance > 5.0, f"Centroid distance {distance:.2f}Å too small (residues may clash)"
        assert distance < 15.0, f"Centroid distance {distance:.2f}Å too large (unusual spacing)"

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_extend_backend_preserved(self, backend):
        """Extend preserves backend."""
        pytest.importorskip("torch")

        p = template_with_coords("ac", backend=backend)

        # Use explicit extend with numpy arrays (will be converted)
        extended = extend_with_linear(p, Residue.G)

        assert extended.backend == backend


class TestJoinAndExtendIntegration:
    """Integration tests combining join and extend."""

    def test_extend_then_join(self):
        """Extend chain then join with another."""
        p1 = template_with_coords("a")
        p1 = extend_with_linear(p1, Residue.C)

        p2 = template_with_coords("gu")

        combined = join(p1, p2)

        assert combined.size(Scale.CHAIN) == 2
        assert combined.size(Scale.RESIDUE) == 4
        assert combined.sequence_str() == "acgu"

    def test_join_then_iterate_chains(self):
        """Join polymers then iterate over chains."""
        p1 = template_with_coords("ac")
        p2 = template_with_coords("gu")

        combined = join(p1, p2)

        chains = list(combined.chains())
        assert len(chains) == 2
        assert chains[0].sequence_str() == "ac"
        assert chains[1].sequence_str() == "gu"


class TestFromSequenceTemplate:
    """Tests for from_sequence() returning templates."""

    def test_template_has_no_coordinates(self):
        """from_sequence() returns template without coordinates."""
        template = from_sequence("acgu")

        assert template.size(Scale.RESIDUE) == 4
        assert template.size() > 0  # Has atoms

        with pytest.raises(AttributeError, match="coordinates"):
            _ = template.coordinates

    def test_template_has_atoms_and_elements(self):
        """Template has atom and element data."""
        template = from_sequence("acgu")

        # Should have atoms and elements
        assert template.atoms is not None
        assert template.elements is not None
        assert len(template.atoms) == template.size()
        assert len(template.elements) == template.size()

    def test_template_copy_coordinates(self):
        """Can add coordinates to template with copy(coordinates=...)."""
        template = from_sequence("acgu")

        # Create dummy coordinates
        coords = np.zeros((template.size(), 3), dtype=np.float32)

        polymer = template.copy(coordinates=coords)

        assert polymer.coordinates is not None
        assert np.allclose(polymer.coordinates, coords)
