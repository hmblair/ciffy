"""
CCD (Chemical Component Dictionary) parsing.

Functions to parse the PDB Chemical Component Dictionary and load residue definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .config import IONS, RESIDUE_WHITELIST, Molecule
from .names import clean_atom_name, to_class_name
from .residue import ResidueDefinition


def _determine_molecule_type(comp_type: str, name: str, comp_id: str) -> int:
    """Determine Molecule type index from CCD type string."""
    t = comp_type.upper()

    # Polymer types
    if 'RNA' in t:
        return Molecule.RNA
    if 'DNA' in t:
        return Molecule.DNA
    if 'D-PEPTIDE' in t:
        return Molecule.PROTEIN_D
    if 'PEPTIDE' in t:
        return Molecule.PROTEIN

    # Non-polymer types
    if 'NON-POLYMER' in t:
        if comp_id == "HOH" or name.upper() == "WATER":
            return Molecule.WATER
        if comp_id in IONS:
            return Molecule.ION
        return Molecule.LIGAND

    return Molecule.OTHER


def _get_abbreviation(one_letter: str, comp_type: str) -> str:
    """Get single-letter abbreviation (lowercase for nucleotides)."""
    if one_letter and one_letter != '?':
        t = comp_type.upper()
        if 'RNA' in t or 'DNA' in t:
            return one_letter.lower()
        return one_letter.upper()
    return '~'


# =============================================================================
# CCD PARSER - Class-based parser with encapsulated state
# =============================================================================


@dataclass
class LoopColumns:
    """Column indices for a CIF loop."""
    names: list[str] = field(default_factory=list)

    def add(self, field_name: str) -> int:
        """Add a column and return its index."""
        self.names.append(field_name)
        return len(self.names) - 1

    def reset(self) -> None:
        """Reset column tracking."""
        self.names.clear()


@dataclass
class ComponentState:
    """State for the current component being parsed."""
    comp_id: str = ""
    name: str = ""
    comp_type: str = ""
    status: str = ""
    one_letter: str = ""
    atoms: list[str] = field(default_factory=list)
    ideal_coords: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    bonds: list[tuple[str, str]] = field(default_factory=list)
    torsions: dict[str, tuple[str, str, str, str]] = field(default_factory=dict)

    def reset(self, comp_id: str) -> None:
        """Reset state for a new component."""
        self.comp_id = comp_id
        self.name = ""
        self.comp_type = ""
        self.status = ""
        self.one_letter = ""
        self.atoms = []
        self.ideal_coords = {}
        self.bonds = []
        self.torsions = {}

    def to_residue(self) -> ResidueDefinition | None:
        """Create ResidueDefinition from current state if valid."""
        if not self.comp_id or self.status == "OBS":
            return None
        return ResidueDefinition(
            name=to_class_name(self.comp_id),
            cif_names=[self.comp_id],
            molecule_type=_determine_molecule_type(self.comp_type, self.name, self.comp_id),
            abbreviation=_get_abbreviation(self.one_letter, self.comp_type),
            atoms=self.atoms.copy(),
            ideal_coords=self.ideal_coords.copy(),
            bonds=self.bonds.copy(),
            torsions=self.torsions.copy() if self.torsions else None,
        )


class CCDParser:
    """
    Parser for PDB Chemical Component Dictionary files.

    Encapsulates parsing state and provides a clean iterator interface.
    Handles _chem_comp, _chem_comp_atom, _chem_comp_bond, and _chem_comp_tor categories.
    """

    def __init__(self, whitelist: set[str] | None = None):
        """
        Initialize parser.

        Args:
            whitelist: If provided, only yield components in this set.
        """
        self.whitelist = whitelist
        self._state = ComponentState()

        # Loop state
        self._in_atom_loop = False
        self._in_bond_loop = False
        self._in_torsion_loop = False

        # Column tracking for each loop type
        self._atom_cols = LoopColumns()
        self._bond_cols = LoopColumns()
        self._torsion_cols = LoopColumns()

        # Column indices (cached for performance)
        self._atom_id_col = -1
        self._x_ideal_col = -1
        self._y_ideal_col = -1
        self._z_ideal_col = -1
        self._bond_atom1_col = -1
        self._bond_atom2_col = -1
        self._tor_id_col = -1
        self._tor_atom1_col = -1
        self._tor_atom2_col = -1
        self._tor_atom3_col = -1
        self._tor_atom4_col = -1

    def _reset_loops(self) -> None:
        """Reset all loop states."""
        self._in_atom_loop = False
        self._in_bond_loop = False
        self._in_torsion_loop = False
        self._atom_cols.reset()
        self._bond_cols.reset()
        self._torsion_cols.reset()
        self._reset_column_indices()

    def _reset_column_indices(self) -> None:
        """Reset cached column indices."""
        self._atom_id_col = -1
        self._x_ideal_col = -1
        self._y_ideal_col = -1
        self._z_ideal_col = -1
        self._bond_atom1_col = -1
        self._bond_atom2_col = -1
        self._tor_id_col = -1
        self._tor_atom1_col = -1
        self._tor_atom2_col = -1
        self._tor_atom3_col = -1
        self._tor_atom4_col = -1

    def _try_yield(self) -> ResidueDefinition | None:
        """Try to create a residue from current state, respecting whitelist."""
        if self.whitelist is not None and self._state.comp_id not in self.whitelist:
            return None
        return self._state.to_residue()

    @staticmethod
    def _parse_float(s: str) -> float | None:
        """Parse a float, returning None for missing values."""
        if s == '?' or s == '.':
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _handle_chem_comp(self, line: str) -> None:
        """Handle _chem_comp.* fields."""
        if line.startswith('_chem_comp.id '):
            self._state.comp_id = line.split()[-1].strip()
        elif line.startswith('_chem_comp.name '):
            parts = line.split(None, 1)
            if len(parts) > 1:
                self._state.name = parts[1].strip().strip('"')
        elif line.startswith('_chem_comp.type '):
            parts = line.split(None, 1)
            if len(parts) > 1:
                self._state.comp_type = parts[1].strip().strip('"')
        elif line.startswith('_chem_comp.pdbx_release_status '):
            self._state.status = line.split()[-1].strip()
        elif line.startswith('_chem_comp.one_letter_code '):
            val = line.split()[-1].strip()
            if val != '?':
                self._state.one_letter = val

    def _handle_atom_header(self, line: str) -> None:
        """Handle _chem_comp_atom.* header lines."""
        col_name = line.strip().split()[0]
        field_name = col_name.split('.')[-1]
        parts = line.split()

        # Check for single-value format (e.g., "_chem_comp_atom.atom_id MG")
        if len(parts) >= 2:
            value = parts[-1]
            if field_name == 'atom_id':
                atom_id = clean_atom_name(value)
                if atom_id not in self._state.atoms:
                    self._state.atoms.append(atom_id)
            elif field_name == 'pdbx_model_Cartn_x_ideal' and self._state.atoms:
                try:
                    coord = list(self._state.ideal_coords.get(self._state.atoms[-1], [None, None, None]))
                    coord[0] = float(value)
                    self._state.ideal_coords[self._state.atoms[-1]] = coord
                except ValueError:
                    pass
            elif field_name == 'pdbx_model_Cartn_y_ideal' and self._state.atoms:
                try:
                    coord = list(self._state.ideal_coords.get(self._state.atoms[-1], [None, None, None]))
                    coord[1] = float(value)
                    self._state.ideal_coords[self._state.atoms[-1]] = coord
                except ValueError:
                    pass
            elif field_name == 'pdbx_model_Cartn_z_ideal' and self._state.atoms:
                try:
                    coord = list(self._state.ideal_coords.get(self._state.atoms[-1], [None, None, None]))
                    coord[2] = float(value)
                    if all(c is not None for c in coord):
                        self._state.ideal_coords[self._state.atoms[-1]] = tuple(coord)
                except ValueError:
                    pass
        else:
            # Loop header format - track column position
            col_idx = self._atom_cols.add(field_name)
            if field_name == 'atom_id':
                self._atom_id_col = col_idx
            elif field_name == 'pdbx_model_Cartn_x_ideal':
                self._x_ideal_col = col_idx
            elif field_name == 'pdbx_model_Cartn_y_ideal':
                self._y_ideal_col = col_idx
            elif field_name == 'pdbx_model_Cartn_z_ideal':
                self._z_ideal_col = col_idx
            self._in_atom_loop = True

    def _handle_atom_data(self, parts: list[str]) -> None:
        """Handle atom loop data line."""
        if len(parts) < 2 or parts[0] != self._state.comp_id:
            return

        if self._atom_id_col >= 0 and self._atom_id_col < len(parts):
            atom_id = clean_atom_name(parts[self._atom_id_col])
            if atom_id not in self._state.atoms:
                self._state.atoms.append(atom_id)

            # Parse ideal coordinates if available
            if (self._x_ideal_col >= 0 and self._y_ideal_col >= 0 and self._z_ideal_col >= 0 and
                self._x_ideal_col < len(parts) and self._y_ideal_col < len(parts) and
                self._z_ideal_col < len(parts)):
                x = self._parse_float(parts[self._x_ideal_col])
                y = self._parse_float(parts[self._y_ideal_col])
                z = self._parse_float(parts[self._z_ideal_col])
                if x is not None and y is not None and z is not None:
                    self._state.ideal_coords[atom_id] = (x, y, z)

    def _handle_bond_header(self, line: str) -> None:
        """Handle _chem_comp_bond.* header lines."""
        col_name = line.strip().split()[0]
        field_name = col_name.split('.')[-1]
        parts = line.split()

        if len(parts) == 1:
            # Loop header format
            col_idx = self._bond_cols.add(field_name)
            if field_name == 'atom_id_1':
                self._bond_atom1_col = col_idx
            elif field_name == 'atom_id_2':
                self._bond_atom2_col = col_idx
            self._in_bond_loop = True

    def _handle_bond_data(self, parts: list[str]) -> None:
        """Handle bond loop data line."""
        if len(parts) < 3 or parts[0] != self._state.comp_id:
            return

        if (self._bond_atom1_col >= 0 and self._bond_atom2_col >= 0 and
            self._bond_atom1_col < len(parts) and self._bond_atom2_col < len(parts)):
            atom1 = clean_atom_name(parts[self._bond_atom1_col])
            atom2 = clean_atom_name(parts[self._bond_atom2_col])
            self._state.bonds.append((atom1, atom2))

    def _handle_torsion_header(self, line: str) -> None:
        """Handle _chem_comp_tor.* header lines."""
        col_name = line.strip().split()[0]
        field_name = col_name.split('.')[-1]
        parts = line.split()

        if len(parts) == 1:
            # Loop header format
            col_idx = self._torsion_cols.add(field_name)
            if field_name == 'id':
                self._tor_id_col = col_idx
            elif field_name == 'atom_id_1':
                self._tor_atom1_col = col_idx
            elif field_name == 'atom_id_2':
                self._tor_atom2_col = col_idx
            elif field_name == 'atom_id_3':
                self._tor_atom3_col = col_idx
            elif field_name == 'atom_id_4':
                self._tor_atom4_col = col_idx
            self._in_torsion_loop = True

    def _handle_torsion_data(self, parts: list[str]) -> None:
        """Handle torsion loop data line."""
        if len(parts) < 5 or parts[0] != self._state.comp_id:
            return

        if (self._tor_id_col >= 0 and self._tor_atom1_col >= 0 and self._tor_atom2_col >= 0 and
            self._tor_atom3_col >= 0 and self._tor_atom4_col >= 0 and
            self._tor_id_col < len(parts) and self._tor_atom4_col < len(parts)):
            tor_id = parts[self._tor_id_col].lower()
            a1 = clean_atom_name(parts[self._tor_atom1_col])
            a2 = clean_atom_name(parts[self._tor_atom2_col])
            a3 = clean_atom_name(parts[self._tor_atom3_col])
            a4 = clean_atom_name(parts[self._tor_atom4_col])
            self._state.torsions[tor_id] = (a1, a2, a3, a4)

    def parse(self, filepath: str) -> Iterator[ResidueDefinition]:
        """
        Parse a CCD file and yield residue definitions.

        Args:
            filepath: Path to components.cif

        Yields:
            ResidueDefinition for each valid component.
        """
        with open(filepath, 'r') as f:
            for line in f:
                line = line.rstrip('\n')

                # New component
                if line.startswith('data_'):
                    if res := self._try_yield():
                        yield res
                    self._state.reset(line[5:])
                    self._reset_loops()
                    continue

                if not self._state.comp_id:
                    continue

                # Parse _chem_comp fields
                if line.startswith('_chem_comp.'):
                    self._handle_chem_comp(line)

                # Detect loop start
                elif line.startswith('loop_'):
                    self._reset_loops()

                # Atom loop
                elif line.startswith('_chem_comp_atom.'):
                    self._handle_atom_header(line)
                elif self._in_atom_loop and line.startswith('_'):
                    pass  # Other header in atom context
                elif self._in_atom_loop and line.strip() and not line.startswith('#'):
                    self._handle_atom_data(line.split())

                # Bond loop
                elif line.startswith('_chem_comp_bond.'):
                    self._handle_bond_header(line)
                elif self._in_bond_loop and line.startswith('_'):
                    pass
                elif self._in_bond_loop and line.strip() and not line.startswith('#'):
                    self._handle_bond_data(line.split())

                # Torsion loop
                elif line.startswith('_chem_comp_tor.'):
                    self._handle_torsion_header(line)
                elif self._in_torsion_loop and line.startswith('_'):
                    pass
                elif self._in_torsion_loop and line.strip() and not line.startswith('#'):
                    self._handle_torsion_data(line.split())

                # Comment ends loops
                elif line.startswith('#'):
                    self._in_atom_loop = False
                    self._in_bond_loop = False
                    self._in_torsion_loop = False

        # Yield last component
        if res := self._try_yield():
            yield res


# =============================================================================
# PUBLIC API
# =============================================================================


def parse_ccd(filepath: str, whitelist: set[str] | None = None) -> Iterator[ResidueDefinition]:
    """
    Parse the CCD file and yield residue definitions.

    Args:
        filepath: Path to components.cif
        whitelist: If provided, only yield components in this set.

    Yields:
        ResidueDefinition for each component (skips obsolete).
    """
    parser = CCDParser(whitelist)
    yield from parser.parse(filepath)


def load_residues_from_ccd(
    ccd_path: str,
    whitelist: set[str] | None = RESIDUE_WHITELIST
) -> list[ResidueDefinition]:
    """Load and sort residue definitions from CCD."""
    print(f"Parsing CCD: {ccd_path}")
    if whitelist:
        print(f"  Using whitelist with {len(whitelist)} entries")

    components = list(parse_ccd(ccd_path, whitelist))

    # Group by molecule type and sort each group
    groups: dict[int, list[ResidueDefinition]] = {}
    for comp in components:
        groups.setdefault(comp.molecule_type, []).append(comp)

    for mol_type in groups:
        groups[mol_type].sort(key=lambda c: c.name)

    # Combine in canonical order
    order = [Molecule.RNA, Molecule.DNA, Molecule.PROTEIN, Molecule.PROTEIN_D,
             Molecule.WATER, Molecule.ION, Molecule.LIGAND, Molecule.OTHER]
    all_residues = []
    for mol_type in order:
        all_residues.extend(groups.get(mol_type, []))

    # Print summary
    print(f"  RNA: {len(groups.get(Molecule.RNA, []))}")
    print(f"  DNA: {len(groups.get(Molecule.DNA, []))}")
    print(f"  L-peptides: {len(groups.get(Molecule.PROTEIN, []))}")
    print(f"  D-peptides: {len(groups.get(Molecule.PROTEIN_D, []))}")
    print(f"  Water: {len(groups.get(Molecule.WATER, []))}, "
          f"Ions: {len(groups.get(Molecule.ION, []))}, "
          f"Ligands: {len(groups.get(Molecule.LIGAND, []))}")
    print(f"  Total: {len(all_residues)} residues")

    return all_residues
