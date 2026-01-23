"""
Tests for CIF semicolon multi-line text block parsing.

CIF format uses semicolon-delimited blocks for multi-line text values:
    ;This is a multi-line
    text value
    ;

These can appear as field values within loop_ data rows.
"""

import pytest
from tests.utils import get_test_cif


class TestSemicolonBlocks:
    """Test parsing of CIF files containing semicolon multi-line values."""

    def test_load_file_with_semicolon_blocks(self):
        """3H5X.cif contains semicolon blocks in _pdbx_entity_nonpoly and loads correctly."""
        import ciffy

        # 3H5X has semicolon blocks in the _pdbx_entity_nonpoly block
        poly = ciffy.load(get_test_cif("3H5X"))

        # Basic sanity checks - file should load without error
        assert poly.pdb_id == "3H5X"
        assert poly.size() > 0
        assert poly.size(ciffy.Scale.CHAIN) == 3

    def test_load_preserves_atom_count(self):
        """Semicolon block parsing doesn't affect atom counts."""
        import ciffy

        poly = ciffy.load(get_test_cif("3H5X"))

        # Verify atom counts are consistent
        total_atoms = poly.size()
        atoms_per_chain = sum(poly.counts(ciffy.Scale.CHAIN).tolist())
        assert total_atoms == atoms_per_chain

    def test_load_preserves_residue_count(self):
        """Semicolon block parsing doesn't affect residue counts."""
        import ciffy

        poly = ciffy.load(get_test_cif("3H5X"))

        # Verify residue counts are consistent
        total_residues = poly.size(ciffy.Scale.RESIDUE)
        # Sum atoms per residue should equal total atoms
        atoms_per_residue = poly.counts(ciffy.Scale.RESIDUE)
        assert sum(atoms_per_residue.tolist()) == poly.size()
        assert total_residues > 0

    def test_round_trip_with_semicolon_file(self, tmp_path):
        """File with semicolon blocks can be loaded, saved, and reloaded."""
        import ciffy
        import numpy as np

        poly = ciffy.load(get_test_cif("3H5X"))
        original_coords = np.array(poly.coordinates)

        # Save and reload
        out_path = tmp_path / "3H5X_copy.cif"
        poly.write(str(out_path))

        reloaded = ciffy.load(str(out_path))

        # Verify structure preserved
        assert reloaded.size() == poly.size()
        assert np.allclose(np.array(reloaded.coordinates), original_coords, atol=0.001)


class TestSemicolonBlockCorrectness:
    """Test that semicolon block content is correctly extracted."""

    def test_multiline_description_content(self, tmp_path):
        """Verify multi-line description content is correctly extracted."""
        import ciffy

        # CIF with semicolon block in _entity.pdbx_description
        cif_content = """data_TEST
_entry.id TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer
;This is a multi-line
description that spans
three lines
;
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 polypeptide(L) MG
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
A 1 1 MET 1 A
A 1 2 GLY 2 A
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET A 1 1 0.0 0.0 0.0 A 1
ATOM 2 C CA . MET A 1 1 1.5 0.0 0.0 A 1
ATOM 3 N N . GLY A 1 2 3.3 1.5 0.0 A 1
ATOM 4 C CA . GLY A 1 2 4.0 2.8 0.0 A 1
#
"""
        cif_path = tmp_path / "test_desc.cif"
        cif_path.write_text(cif_content)

        # Load with descriptions (skip=[])
        poly = ciffy.load(str(cif_path), skip=[])

        # Verify description was extracted
        assert poly.descriptions is not None
        assert len(poly.descriptions) == 1  # One chain

        # Verify content - should be the text between semicolons (with newlines)
        desc = poly.descriptions[0]
        assert "This is a multi-line" in desc
        assert "description that spans" in desc
        assert "three lines" in desc

    def test_field_after_semicolon_block_in_same_row(self, tmp_path):
        """Verify fields after semicolon block in same logical row are parsed correctly."""
        import ciffy

        # CIF where semicolon block is NOT the last field in the row
        # _entity: id, pdbx_description, type (description is in middle)
        cif_content = """data_TEST
_entry.id TEST
#
loop_
_entity.id
_entity.pdbx_description
_entity.type
1
;Multi-line description
;
polymer
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 polypeptide(L) M
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
A 1 1 MET 1 A
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET A 1 1 0.0 0.0 0.0 A 1
ATOM 2 C CA . MET A 1 1 1.5 0.0 0.0 A 1
#
"""
        cif_path = tmp_path / "test_field_after.cif"
        cif_path.write_text(cif_content)

        # If parsing is wrong, the "polymer" type would be missed and entity
        # would be classified incorrectly (or parsing would fail entirely)
        poly = ciffy.load(str(cif_path), skip=[])

        # Verify structure loaded correctly
        assert poly.pdb_id == "TEST"
        assert poly.size() == 2  # 2 atoms
        # If "polymer" wasn't parsed correctly, molecule_types would be wrong
        assert poly.size(ciffy.Scale.CHAIN) == 1

        # Verify description extracted from semicolon block
        assert poly.descriptions is not None
        assert "Multi-line description" in poly.descriptions[0]

    def test_multiple_rows_with_semicolon_blocks(self, tmp_path):
        """Verify multiple rows with semicolon blocks are parsed correctly."""
        import ciffy

        # CIF with multiple entities, each having semicolon block descriptions
        cif_content = """data_TEST
_entry.id TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer
;First entity
description
;
2 polymer
;Second entity
description
;
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 polypeptide(L) M
2 polypeptide(L) G
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
A 1 1 MET 1 A
B 2 1 GLY 1 B
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET A 1 1 0.0 0.0 0.0 A 1
ATOM 2 C CA . MET A 1 1 1.5 0.0 0.0 A 1
ATOM 3 N N . GLY B 2 1 5.0 0.0 0.0 B 1
ATOM 4 C CA . GLY B 2 1 6.5 0.0 0.0 B 1
#
"""
        cif_path = tmp_path / "test_multiple.cif"
        cif_path.write_text(cif_content)

        poly = ciffy.load(str(cif_path), skip=[])

        # Verify both chains loaded
        assert poly.size(ciffy.Scale.CHAIN) == 2
        assert poly.size() == 4  # 4 atoms total

        # Verify descriptions for both chains
        assert poly.descriptions is not None
        assert len(poly.descriptions) == 2
        assert "First entity" in poly.descriptions[0]
        assert "Second entity" in poly.descriptions[1]

    def test_row_count_with_semicolon_blocks(self, tmp_path):
        """Verify correct row count when semicolon blocks span multiple physical lines."""
        import ciffy

        # CIF where entity block has 3 logical rows but many physical lines
        cif_content = """data_TEST
_entry.id TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer 'Normal description'
2 polymer
;This description
spans many
physical lines
but is one field
;
3 polymer 'Another normal one'
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 polypeptide(L) M
2 polypeptide(L) G
3 polypeptide(L) K
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
A 1 1 MET 1 A
B 2 1 GLY 1 B
C 3 1 LYS 1 C
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
C 3
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET A 1 1 0.0 0.0 0.0 A 1
ATOM 2 N N . GLY B 2 1 5.0 0.0 0.0 B 1
ATOM 3 N N . LYS C 3 1 10.0 0.0 0.0 C 1
#
"""
        cif_path = tmp_path / "test_row_count.cif"
        cif_path.write_text(cif_content)

        poly = ciffy.load(str(cif_path), skip=[])

        # If row counting is wrong, we'd get wrong chain count or parsing error
        assert poly.size(ciffy.Scale.CHAIN) == 3
        assert poly.size() == 3  # 3 atoms

        # Verify all three descriptions are correct
        assert poly.descriptions is not None
        assert len(poly.descriptions) == 3
        assert "Normal description" in poly.descriptions[0]
        assert "spans many" in poly.descriptions[1]
        assert "physical lines" in poly.descriptions[1]
        assert "Another normal one" in poly.descriptions[2]


class TestSemicolonBlockSynthetic:
    """Test semicolon parsing with synthetic CIF content."""

    def test_multiline_value_in_loop(self, tmp_path):
        """Loop with semicolon multi-line value parses correctly."""
        import ciffy

        # Create a minimal CIF with semicolon block in loop data
        cif_content = """data_TEST
_entry.id TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer 'Test protein'
2 non-polymer
;This is a multi-line
description for entity 2
;
3 water 'water molecules'
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 polypeptide(L) MGKLF
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
A 1 1 MET 1 A
A 1 2 GLY 2 A
A 1 3 LYS 3 A
A 1 4 LEU 4 A
A 1 5 PHE 5 A
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET A 1 1 0.0 0.0 0.0 A 1
ATOM 2 C CA . MET A 1 1 1.5 0.0 0.0 A 1
ATOM 3 C C . MET A 1 1 2.0 1.5 0.0 A 1
ATOM 4 O O . MET A 1 1 1.5 2.5 0.0 A 1
ATOM 5 N N . GLY A 1 2 3.3 1.5 0.0 A 1
ATOM 6 C CA . GLY A 1 2 4.0 2.8 0.0 A 1
ATOM 7 C C . GLY A 1 2 5.5 2.8 0.0 A 1
ATOM 8 O O . GLY A 1 2 6.2 1.8 0.0 A 1
ATOM 9 N N . LYS A 1 3 6.0 4.0 0.0 A 1
ATOM 10 C CA . LYS A 1 3 7.4 4.3 0.0 A 1
ATOM 11 C C . LYS A 1 3 8.0 5.7 0.0 A 1
ATOM 12 O O . LYS A 1 3 7.3 6.7 0.0 A 1
ATOM 13 N N . LEU A 1 4 9.3 5.8 0.0 A 1
ATOM 14 C CA . LEU A 1 4 10.0 7.1 0.0 A 1
ATOM 15 C C . LEU A 1 4 11.5 7.0 0.0 A 1
ATOM 16 O O . LEU A 1 4 12.1 5.9 0.0 A 1
ATOM 17 N N . PHE A 1 5 12.1 8.2 0.0 A 1
ATOM 18 C CA . PHE A 1 5 13.5 8.4 0.0 A 1
ATOM 19 C C . PHE A 1 5 14.0 9.8 0.0 A 1
ATOM 20 O O . PHE A 1 5 13.2 10.7 0.0 A 1
#
"""
        cif_path = tmp_path / "test_semicolon.cif"
        cif_path.write_text(cif_content)

        # Should load without error
        poly = ciffy.load(str(cif_path))

        assert poly.pdb_id == "TEST"
        assert poly.size() == 20  # 20 atoms
        assert poly.size(ciffy.Scale.RESIDUE) == 5  # 5 residues
