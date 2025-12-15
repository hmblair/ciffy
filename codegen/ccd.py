"""
CCD (Chemical Component Dictionary) parsing.

Functions to parse the PDB Chemical Component Dictionary and load residue definitions.
"""

from __future__ import annotations

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


def parse_ccd(filepath: str, whitelist: set[str] | None = None) -> Iterator[ResidueDefinition]:
    """Parse the CCD file and yield residue definitions.

    Args:
        filepath: Path to components.cif
        whitelist: If provided, only yield components in this set.

    Yields:
        ResidueDefinition for each component (skips obsolete).
    """
    # State for current component
    comp_id = ""
    name = ""
    comp_type = ""
    status = ""
    one_letter = ""
    atoms: list[str] = []
    ideal_coords: dict[str, tuple[float, float, float]] = {}
    bonds: list[tuple[str, str]] = []
    in_atom_loop = False
    in_bond_loop = False
    # Column indices for atom loop parsing
    atom_columns: list[str] = []
    atom_id_col = -1
    x_ideal_col = -1
    y_ideal_col = -1
    z_ideal_col = -1
    # Column indices for bond loop parsing
    bond_columns: list[str] = []
    bond_atom1_col = -1
    bond_atom2_col = -1

    def make_residue() -> ResidueDefinition | None:
        """Create ResidueDefinition from current state if valid."""
        if not comp_id or status == "OBS":
            return None
        if whitelist is not None and comp_id not in whitelist:
            return None
        return ResidueDefinition(
            name=to_class_name(comp_id),
            cif_names=[comp_id],
            molecule_type=_determine_molecule_type(comp_type, name, comp_id),
            abbreviation=_get_abbreviation(one_letter, comp_type),
            atoms=atoms.copy(),
            ideal_coords=ideal_coords.copy(),
            bonds=bonds.copy(),
        )

    def _parse_float(s: str) -> float | None:
        """Parse a float, returning None for missing values."""
        if s == '?' or s == '.':
            return None
        try:
            return float(s)
        except ValueError:
            return None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # New component
            if line.startswith('data_'):
                if res := make_residue():
                    yield res
                # Reset state
                comp_id = line[5:]
                name = ""
                comp_type = ""
                status = ""
                one_letter = ""
                atoms = []
                ideal_coords = {}
                bonds = []
                in_atom_loop = False
                in_bond_loop = False
                atom_columns = []
                atom_id_col = -1
                x_ideal_col = -1
                y_ideal_col = -1
                z_ideal_col = -1
                bond_columns = []
                bond_atom1_col = -1
                bond_atom2_col = -1
                continue

            if not comp_id:
                continue

            # Parse _chem_comp fields
            if line.startswith('_chem_comp.id '):
                comp_id = line.split()[-1].strip()
            elif line.startswith('_chem_comp.name '):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    name = parts[1].strip().strip('"')
            elif line.startswith('_chem_comp.type '):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    comp_type = parts[1].strip().strip('"')
            elif line.startswith('_chem_comp.pdbx_release_status '):
                status = line.split()[-1].strip()
            elif line.startswith('_chem_comp.one_letter_code '):
                val = line.split()[-1].strip()
                if val != '?':
                    one_letter = val

            # Detect loop start - reset loop states
            elif line.startswith('loop_'):
                in_atom_loop = False
                in_bond_loop = False
                atom_columns = []
                bond_columns = []
            elif line.startswith('_chem_comp_atom.'):
                col_name = line.strip().split()[0]  # e.g., "_chem_comp_atom.atom_id"
                field = col_name.split('.')[-1]  # e.g., "atom_id"
                parts = line.split()

                # Check for single-value format (e.g., "_chem_comp_atom.atom_id MG")
                if len(parts) >= 2:
                    value = parts[-1]
                    if field == 'atom_id':
                        atom_id = clean_atom_name(value)
                        if atom_id not in atoms:
                            atoms.append(atom_id)
                            # Single-value format: store coords later when we see them
                    elif field == 'pdbx_model_Cartn_x_ideal' and atoms:
                        try:
                            _single_x = float(value)
                            ideal_coords.setdefault(atoms[-1], [None, None, None])[0] = _single_x
                        except ValueError:
                            pass
                    elif field == 'pdbx_model_Cartn_y_ideal' and atoms:
                        try:
                            _single_y = float(value)
                            ideal_coords.setdefault(atoms[-1], [None, None, None])[1] = _single_y
                        except ValueError:
                            pass
                    elif field == 'pdbx_model_Cartn_z_ideal' and atoms:
                        try:
                            _single_z = float(value)
                            coord = ideal_coords.get(atoms[-1], [None, None, None])
                            coord[2] = float(value)
                            if all(c is not None for c in coord):
                                ideal_coords[atoms[-1]] = tuple(coord)
                        except ValueError:
                            pass
                else:
                    # Loop header format - track column position
                    atom_columns.append(field)
                    col_idx = len(atom_columns) - 1
                    if field == 'atom_id':
                        atom_id_col = col_idx
                    elif field == 'pdbx_model_Cartn_x_ideal':
                        x_ideal_col = col_idx
                    elif field == 'pdbx_model_Cartn_y_ideal':
                        y_ideal_col = col_idx
                    elif field == 'pdbx_model_Cartn_z_ideal':
                        z_ideal_col = col_idx
                    in_atom_loop = True
            elif in_atom_loop and line.startswith('_'):
                pass
            elif in_atom_loop and line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2 and parts[0] == comp_id:
                    # Parse atom_id
                    if atom_id_col >= 0 and atom_id_col < len(parts):
                        atom_id = clean_atom_name(parts[atom_id_col])
                        if atom_id not in atoms:
                            atoms.append(atom_id)
                        # Parse ideal coordinates if available
                        if (x_ideal_col >= 0 and y_ideal_col >= 0 and z_ideal_col >= 0 and
                            x_ideal_col < len(parts) and y_ideal_col < len(parts) and z_ideal_col < len(parts)):
                            x = _parse_float(parts[x_ideal_col])
                            y = _parse_float(parts[y_ideal_col])
                            z = _parse_float(parts[z_ideal_col])
                            if x is not None and y is not None and z is not None:
                                ideal_coords[atom_id] = (x, y, z)

            # Detect bond definitions
            elif line.startswith('_chem_comp_bond.'):
                col_name = line.strip().split()[0]
                field = col_name.split('.')[-1]
                parts = line.split()

                if len(parts) == 1:
                    # Loop header format - track column position
                    bond_columns.append(field)
                    col_idx = len(bond_columns) - 1
                    if field == 'atom_id_1':
                        bond_atom1_col = col_idx
                    elif field == 'atom_id_2':
                        bond_atom2_col = col_idx
                    in_bond_loop = True
            elif in_bond_loop and line.startswith('_'):
                pass
            elif in_bond_loop and line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3 and parts[0] == comp_id:
                    # Parse bond atom pair
                    if (bond_atom1_col >= 0 and bond_atom2_col >= 0 and
                        bond_atom1_col < len(parts) and bond_atom2_col < len(parts)):
                        atom1 = clean_atom_name(parts[bond_atom1_col])
                        atom2 = clean_atom_name(parts[bond_atom2_col])
                        bonds.append((atom1, atom2))

            elif line.startswith('#'):
                in_atom_loop = False
                in_bond_loop = False

    # Yield last component
    if res := make_residue():
        yield res


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
