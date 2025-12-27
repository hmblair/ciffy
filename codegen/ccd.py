"""
CCD (Chemical Component Dictionary) parsing.

Functions to parse the PDB Chemical Component Dictionary and load residue definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import (
    IONS,
    RESIDUE_WHITELIST,
    Molecule,
    MOLECULE_CLASSIFICATION_RULES,
    MOLECULE_CLASSIFICATION_DEFAULT,
)
from .names import clean_atom_name, to_class_name
from .residue import ResidueDefinition


class CCDParseError(Exception):
    """Error parsing the Chemical Component Dictionary."""
    pass


def _determine_molecule_type(comp_type: str, name: str, comp_id: str) -> int:
    """
    Determine Molecule type index from CCD type string.

    Uses classification rules loaded from residues.yaml. Rules are evaluated
    in order; first pattern match wins. Special handling for water and ions
    within the LIGAND category.
    """
    t = comp_type.upper()

    # Apply classification rules from YAML (order matters)
    for rule in MOLECULE_CLASSIFICATION_RULES:
        if rule.pattern.upper() in t:
            molecule_type = getattr(Molecule, rule.molecule_type)

            # Special handling for non-polymers: distinguish water, ions, ligands
            if molecule_type == Molecule.LIGAND:
                if comp_id == "HOH" or name.upper() == "WATER":
                    return Molecule.WATER
                if comp_id in IONS:
                    return Molecule.ION

            return molecule_type

    # Default if no pattern matches
    return getattr(Molecule, MOLECULE_CLASSIFICATION_DEFAULT)


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
class AtomLoopState:
    """
    State for parsing _chem_comp_atom loops.

    Tracks which columns contain which fields, allowing data rows
    to be parsed correctly regardless of column order.
    """
    active: bool = False
    column_count: int = 0
    # Column indices (-1 = not present)
    atom_id_col: int = -1
    x_ideal_col: int = -1
    y_ideal_col: int = -1
    z_ideal_col: int = -1

    def add_column(self, field_name: str) -> None:
        """Record column index for a field."""
        if field_name == 'atom_id':
            self.atom_id_col = self.column_count
        elif field_name == 'pdbx_model_Cartn_x_ideal':
            self.x_ideal_col = self.column_count
        elif field_name == 'pdbx_model_Cartn_y_ideal':
            self.y_ideal_col = self.column_count
        elif field_name == 'pdbx_model_Cartn_z_ideal':
            self.z_ideal_col = self.column_count
        self.column_count += 1
        self.active = True

    def reset(self) -> None:
        """Reset to initial state."""
        self.active = False
        self.column_count = 0
        self.atom_id_col = -1
        self.x_ideal_col = -1
        self.y_ideal_col = -1
        self.z_ideal_col = -1


@dataclass
class BondLoopState:
    """State for parsing _chem_comp_bond loops."""
    active: bool = False
    column_count: int = 0
    atom1_col: int = -1
    atom2_col: int = -1

    def add_column(self, field_name: str) -> None:
        """Record column index for a field."""
        if field_name == 'atom_id_1':
            self.atom1_col = self.column_count
        elif field_name == 'atom_id_2':
            self.atom2_col = self.column_count
        self.column_count += 1
        self.active = True

    def reset(self) -> None:
        """Reset to initial state."""
        self.active = False
        self.column_count = 0
        self.atom1_col = -1
        self.atom2_col = -1


@dataclass
class TorsionLoopState:
    """State for parsing _chem_comp_tor loops."""
    active: bool = False
    column_count: int = 0
    id_col: int = -1
    atom1_col: int = -1
    atom2_col: int = -1
    atom3_col: int = -1
    atom4_col: int = -1

    def add_column(self, field_name: str) -> None:
        """Record column index for a field."""
        if field_name == 'id':
            self.id_col = self.column_count
        elif field_name == 'atom_id_1':
            self.atom1_col = self.column_count
        elif field_name == 'atom_id_2':
            self.atom2_col = self.column_count
        elif field_name == 'atom_id_3':
            self.atom3_col = self.column_count
        elif field_name == 'atom_id_4':
            self.atom4_col = self.column_count
        self.column_count += 1
        self.active = True

    def reset(self) -> None:
        """Reset to initial state."""
        self.active = False
        self.column_count = 0
        self.id_col = -1
        self.atom1_col = -1
        self.atom2_col = -1
        self.atom3_col = -1
        self.atom4_col = -1


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

        # Loop state for each CIF category
        self._atom_loop = AtomLoopState()
        self._bond_loop = BondLoopState()
        self._torsion_loop = TorsionLoopState()

    def _reset_loops(self) -> None:
        """Reset all loop states."""
        self._atom_loop.reset()
        self._bond_loop.reset()
        self._torsion_loop.reset()

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

    def _update_atom_coordinate(self, atom_name: str, coord_index: int, value: str) -> None:
        """
        Update a single coordinate component for an atom.

        Args:
            atom_name: Name of the atom to update.
            coord_index: Which coordinate (0=x, 1=y, 2=z).
            value: String value to parse as float.
        """
        try:
            coord = list(self._state.ideal_coords.get(atom_name, [None, None, None]))
            coord[coord_index] = float(value)
            # Store as tuple only when all coordinates are present
            if all(c is not None for c in coord):
                self._state.ideal_coords[atom_name] = tuple(coord)
            else:
                self._state.ideal_coords[atom_name] = coord
        except ValueError:
            pass  # Ignore unparseable coordinate values

    def _handle_atom_single_value(self, field_name: str, value: str) -> None:
        """
        Handle single-value atom field (non-loop format).

        Some simple components use "_chem_comp_atom.field value" format
        instead of loop format.
        """
        if field_name == 'atom_id':
            atom_id = clean_atom_name(value)
            if atom_id not in self._state.atoms:
                self._state.atoms.append(atom_id)
        elif self._state.atoms:
            # Coordinate fields - apply to most recently added atom
            atom_name = self._state.atoms[-1]
            if field_name == 'pdbx_model_Cartn_x_ideal':
                self._update_atom_coordinate(atom_name, 0, value)
            elif field_name == 'pdbx_model_Cartn_y_ideal':
                self._update_atom_coordinate(atom_name, 1, value)
            elif field_name == 'pdbx_model_Cartn_z_ideal':
                self._update_atom_coordinate(atom_name, 2, value)

    def _handle_atom_loop_header(self, field_name: str) -> None:
        """
        Track column position for atom loop format.

        Records which column index corresponds to each field name.
        """
        self._atom_loop.add_column(field_name)

    def _handle_atom_header(self, line: str) -> None:
        """
        Handle _chem_comp_atom.* header lines.

        CIF files can have atoms in two formats:
        1. Single-value: "_chem_comp_atom.atom_id MG" (simple components)
        2. Loop format: column headers followed by data rows (most components)

        This method detects which format is used and delegates accordingly.
        """
        parts = line.split()
        field_name = parts[0].split('.')[-1]

        if len(parts) >= 2:
            # Single-value format: "_chem_comp_atom.field value"
            self._handle_atom_single_value(field_name, parts[-1])
        else:
            # Loop header format: just the field name
            self._handle_atom_loop_header(field_name)

    def _handle_atom_data(self, parts: list[str]) -> None:
        """Handle atom loop data line."""
        if len(parts) < 2 or parts[0] != self._state.comp_id:
            return

        loop = self._atom_loop
        if loop.atom_id_col >= 0 and loop.atom_id_col < len(parts):
            atom_id = clean_atom_name(parts[loop.atom_id_col])
            if atom_id not in self._state.atoms:
                self._state.atoms.append(atom_id)

            # Parse ideal coordinates if available
            if (loop.x_ideal_col >= 0 and loop.y_ideal_col >= 0 and loop.z_ideal_col >= 0 and
                loop.x_ideal_col < len(parts) and loop.y_ideal_col < len(parts) and
                loop.z_ideal_col < len(parts)):
                x = self._parse_float(parts[loop.x_ideal_col])
                y = self._parse_float(parts[loop.y_ideal_col])
                z = self._parse_float(parts[loop.z_ideal_col])
                if x is not None and y is not None and z is not None:
                    self._state.ideal_coords[atom_id] = (x, y, z)

    def _handle_bond_header(self, line: str) -> None:
        """Handle _chem_comp_bond.* header lines."""
        parts = line.split()
        if len(parts) == 1:
            # Loop header format
            field_name = parts[0].split('.')[-1]
            self._bond_loop.add_column(field_name)

    def _handle_bond_data(self, parts: list[str]) -> None:
        """Handle bond loop data line."""
        if len(parts) < 3 or parts[0] != self._state.comp_id:
            return

        loop = self._bond_loop
        if (loop.atom1_col >= 0 and loop.atom2_col >= 0 and
            loop.atom1_col < len(parts) and loop.atom2_col < len(parts)):
            atom1 = clean_atom_name(parts[loop.atom1_col])
            atom2 = clean_atom_name(parts[loop.atom2_col])
            self._state.bonds.append((atom1, atom2))

    def _handle_torsion_header(self, line: str) -> None:
        """Handle _chem_comp_tor.* header lines."""
        parts = line.split()
        if len(parts) == 1:
            # Loop header format
            field_name = parts[0].split('.')[-1]
            self._torsion_loop.add_column(field_name)

    def _handle_torsion_data(self, parts: list[str]) -> None:
        """Handle torsion loop data line."""
        if len(parts) < 5 or parts[0] != self._state.comp_id:
            return

        loop = self._torsion_loop
        if (loop.id_col >= 0 and loop.atom1_col >= 0 and loop.atom2_col >= 0 and
            loop.atom3_col >= 0 and loop.atom4_col >= 0 and
            loop.id_col < len(parts) and loop.atom4_col < len(parts)):
            tor_id = parts[loop.id_col].lower()
            a1 = clean_atom_name(parts[loop.atom1_col])
            a2 = clean_atom_name(parts[loop.atom2_col])
            a3 = clean_atom_name(parts[loop.atom3_col])
            a4 = clean_atom_name(parts[loop.atom4_col])
            self._state.torsions[tor_id] = (a1, a2, a3, a4)

    def parse(self, filepath: str) -> Iterator[ResidueDefinition]:
        """
        Parse a CCD file and yield residue definitions.

        Args:
            filepath: Path to components.cif

        Yields:
            ResidueDefinition for each valid component.

        Raises:
            CCDParseError: If the file cannot be opened or parsed.
        """
        try:
            f = open(filepath, 'r')
        except FileNotFoundError:
            raise CCDParseError(
                f"CCD file not found: {filepath}\n"
                f"Run 'python -m codegen.cli --download' to download it, or download manually from:\n"
                f"https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
            ) from None
        except PermissionError:
            raise CCDParseError(f"Permission denied reading CCD file: {filepath}") from None
        except OSError as e:
            raise CCDParseError(f"Error opening CCD file {filepath}: {e}") from None

        with f:
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
                elif self._atom_loop.active and line.startswith('_'):
                    pass  # Other header in atom context
                elif self._atom_loop.active and line.strip() and not line.startswith('#'):
                    self._handle_atom_data(line.split())

                # Bond loop
                elif line.startswith('_chem_comp_bond.'):
                    self._handle_bond_header(line)
                elif self._bond_loop.active and line.startswith('_'):
                    pass
                elif self._bond_loop.active and line.strip() and not line.startswith('#'):
                    self._handle_bond_data(line.split())

                # Torsion loop
                elif line.startswith('_chem_comp_tor.'):
                    self._handle_torsion_header(line)
                elif self._torsion_loop.active and line.startswith('_'):
                    pass
                elif self._torsion_loop.active and line.strip() and not line.startswith('#'):
                    self._handle_torsion_data(line.split())

                # Comment ends loops
                elif line.startswith('#'):
                    self._atom_loop.active = False
                    self._bond_loop.active = False
                    self._torsion_loop.active = False

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
