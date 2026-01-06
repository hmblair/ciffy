"""
Parse backbone dihedral definitions from MonomerLibrary.

This module parses the MonomerLibrary links_and_mods.cif file to extract
inter-residue dihedral angle definitions (phi, psi, omega for peptides,
alpha-zeta for nucleic acids).

Data source: https://github.com/MonomerLibrary/monomers
License: LGPL-3.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .names import clean_atom_name


class MonLibParseError(Exception):
    """Error parsing the MonomerLibrary file."""
    pass


@dataclass
class LinkBond:
    """A bond definition from a chemical link (inter-residue)."""
    link_id: str  # e.g., "TRANS", "p"
    atom_1: str  # e.g., "C", "O3'"
    comp_1: int  # 1 = previous residue, 2 = next residue
    atom_2: str  # e.g., "N", "P"
    comp_2: int
    bond_type: str  # "SINGLE", "DOUBLE", etc.
    value_dist: float  # Bond length in Angstroms
    value_dist_esd: float  # Standard deviation


@dataclass
class LinkAngle:
    """A bond angle definition from a chemical link (inter-residue)."""
    link_id: str  # e.g., "TRANS", "p"
    atom_1: str  # First atom
    comp_1: int
    atom_2: str  # Central atom (vertex)
    comp_2: int
    atom_3: str  # Third atom
    comp_3: int
    value_angle: float  # Angle in degrees
    value_angle_esd: float  # Standard deviation


@dataclass
class LinkTorsion:
    """A torsion angle definition from a chemical link."""
    link_id: str  # e.g., "TRANS", "p"
    torsion_id: str  # e.g., "phi", "psi", "alpha"
    atoms: tuple[tuple[str, int], tuple[str, int], tuple[str, int], tuple[str, int]]
    # Each atom is (atom_name, comp_id) where comp_id is 1 or 2


@dataclass
class BackboneDihedrals:
    """Parsed backbone dihedral definitions."""
    # Protein backbone (from TRANS/CIS links)
    phi: tuple[tuple[str, int], ...] | None = None  # C(1)-N(2)-CA(2)-C(2)
    psi: tuple[tuple[str, int], ...] | None = None  # N(1)-CA(1)-C(1)-N(2)
    omega: tuple[tuple[str, int], ...] | None = None  # CA(1)-C(1)-N(2)-CA(2)

    # Nucleic acid backbone (from p link)
    alpha: tuple[tuple[str, int], ...] | None = None  # O3'(1)-P(2)-O5'(2)-C5'(2)
    beta: tuple[tuple[str, int], ...] | None = None
    gamma: tuple[tuple[str, int], ...] | None = None
    delta: tuple[tuple[str, int], ...] | None = None
    epsilon: tuple[tuple[str, int], ...] | None = None  # C4'(1)-C3'(1)-O3'(1)-P(2)
    zeta: tuple[tuple[str, int], ...] | None = None  # C3'(1)-O3'(1)-P(2)-O5'(2)


def parse_link_torsions(filepath: Path) -> Iterator[LinkTorsion]:
    """
    Parse _chem_link_tor from links_and_mods.cif.

    Yields LinkTorsion objects for each torsion definition found.

    Raises:
        MonLibParseError: If the file cannot be opened.
    """
    current_link = ""
    in_link_tor_loop = False
    columns: list[str] = []

    # Column indices
    link_id_col = -1
    tor_id_col = -1
    atom1_comp_col = -1
    atom1_id_col = -1
    atom2_comp_col = -1
    atom2_id_col = -1
    atom3_comp_col = -1
    atom3_id_col = -1
    atom4_comp_col = -1
    atom4_id_col = -1

    try:
        f = open(filepath, 'r', encoding='utf-8')
    except FileNotFoundError:
        raise MonLibParseError(
            f"MonomerLibrary file not found: {filepath}\n"
            f"Run 'python -m codegen.cli --download' to download it, or download manually from:\n"
            f"https://github.com/MonomerLibrary/monomers"
        ) from None
    except OSError as e:
        raise MonLibParseError(f"Error opening MonomerLibrary file {filepath}: {e}") from None

    with f:
        for line in f:
            line = line.rstrip('\n')

            # Track current link block
            if line.startswith('data_link_'):
                current_link = line[10:]  # e.g., "TRANS", "p"
                in_link_tor_loop = False
                columns = []
                continue

            # Detect loop start
            if line.startswith('loop_'):
                in_link_tor_loop = False
                columns = []
                continue

            # Parse _chem_link_tor columns
            if line.startswith('_chem_link_tor.'):
                field = line.strip().split('.')[1].split()[0]
                columns.append(field)
                col_idx = len(columns) - 1

                if field == 'link_id':
                    link_id_col = col_idx
                elif field == 'id':
                    tor_id_col = col_idx
                elif field == 'atom_1_comp_id':
                    atom1_comp_col = col_idx
                elif field == 'atom_id_1':
                    atom1_id_col = col_idx
                elif field == 'atom_2_comp_id':
                    atom2_comp_col = col_idx
                elif field == 'atom_id_2':
                    atom2_id_col = col_idx
                elif field == 'atom_3_comp_id':
                    atom3_comp_col = col_idx
                elif field == 'atom_id_3':
                    atom3_id_col = col_idx
                elif field == 'atom_4_comp_id':
                    atom4_comp_col = col_idx
                elif field == 'atom_id_4':
                    atom4_id_col = col_idx

                in_link_tor_loop = True
                continue

            # Parse data lines
            if in_link_tor_loop and line.strip() and not line.startswith('#') and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        link_id = parts[link_id_col] if link_id_col >= 0 else current_link
                        tor_id = parts[tor_id_col].lower() if tor_id_col >= 0 else ""

                        atoms = (
                            (clean_atom_name(parts[atom1_id_col]), int(parts[atom1_comp_col])),
                            (clean_atom_name(parts[atom2_id_col]), int(parts[atom2_comp_col])),
                            (clean_atom_name(parts[atom3_id_col]), int(parts[atom3_comp_col])),
                            (clean_atom_name(parts[atom4_id_col]), int(parts[atom4_comp_col])),
                        )

                        yield LinkTorsion(
                            link_id=link_id,
                            torsion_id=tor_id,
                            atoms=atoms,
                        )
                    except (ValueError, IndexError):
                        continue

            # End of loop
            if line.startswith('#') or (line.startswith('_') and not line.startswith('_chem_link_tor')):
                in_link_tor_loop = False


def parse_link_bonds(filepath: Path) -> Iterator[LinkBond]:
    """
    Parse _chem_link_bond from links_and_mods.cif.

    Yields LinkBond objects for each inter-residue bond definition found.

    Raises:
        MonLibParseError: If the file cannot be opened.
    """
    current_link = ""
    in_bond_loop = False
    columns: list[str] = []

    # Column indices (set when reading loop header)
    link_id_col = -1
    atom1_comp_col = -1
    atom1_id_col = -1
    atom2_comp_col = -1
    atom2_id_col = -1
    type_col = -1
    dist_col = -1
    dist_esd_col = -1

    try:
        f = open(filepath, 'r', encoding='utf-8')
    except FileNotFoundError:
        raise MonLibParseError(
            f"MonomerLibrary file not found: {filepath}\n"
            f"Run 'python -m codegen.cli --download' to download it."
        ) from None
    except OSError as e:
        raise MonLibParseError(f"Error opening MonomerLibrary file {filepath}: {e}") from None

    with f:
        for line in f:
            line = line.rstrip('\n')

            # Track current link block
            if line.startswith('data_link_'):
                current_link = line[10:]
                in_bond_loop = False
                columns = []
                continue

            # Detect loop start
            if line.startswith('loop_'):
                in_bond_loop = False
                columns = []
                continue

            # Parse _chem_link_bond columns
            if line.startswith('_chem_link_bond.'):
                field = line.strip().split('.')[1].split()[0]
                columns.append(field)
                col_idx = len(columns) - 1

                if field == 'link_id':
                    link_id_col = col_idx
                elif field == 'atom_1_comp_id':
                    atom1_comp_col = col_idx
                elif field == 'atom_id_1':
                    atom1_id_col = col_idx
                elif field == 'atom_2_comp_id':
                    atom2_comp_col = col_idx
                elif field == 'atom_id_2':
                    atom2_id_col = col_idx
                elif field == 'type':
                    type_col = col_idx
                elif field == 'value_dist':
                    dist_col = col_idx
                elif field == 'value_dist_esd':
                    dist_esd_col = col_idx

                in_bond_loop = True
                continue

            # Parse data lines (8 columns for bonds)
            if in_bond_loop and line.strip() and not line.startswith('#') and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        link_id = parts[link_id_col] if link_id_col >= 0 else current_link

                        yield LinkBond(
                            link_id=link_id,
                            atom_1=clean_atom_name(parts[atom1_id_col]),
                            comp_1=int(parts[atom1_comp_col]),
                            atom_2=clean_atom_name(parts[atom2_id_col]),
                            comp_2=int(parts[atom2_comp_col]),
                            bond_type=parts[type_col],
                            value_dist=float(parts[dist_col]),
                            value_dist_esd=float(parts[dist_esd_col]),
                        )
                    except (ValueError, IndexError):
                        continue

            # End of loop
            if line.startswith('#') or (line.startswith('_') and not line.startswith('_chem_link_bond')):
                in_bond_loop = False


def parse_link_angles(filepath: Path) -> Iterator[LinkAngle]:
    """
    Parse _chem_link_angle from links_and_mods.cif.

    Yields LinkAngle objects for each inter-residue angle definition found.

    Raises:
        MonLibParseError: If the file cannot be opened.
    """
    current_link = ""
    in_angle_loop = False
    columns: list[str] = []

    # Column indices
    link_id_col = -1
    atom1_comp_col = -1
    atom1_id_col = -1
    atom2_comp_col = -1
    atom2_id_col = -1
    atom3_comp_col = -1
    atom3_id_col = -1
    angle_col = -1
    angle_esd_col = -1

    try:
        f = open(filepath, 'r', encoding='utf-8')
    except FileNotFoundError:
        raise MonLibParseError(
            f"MonomerLibrary file not found: {filepath}\n"
            f"Run 'python -m codegen.cli --download' to download it."
        ) from None
    except OSError as e:
        raise MonLibParseError(f"Error opening MonomerLibrary file {filepath}: {e}") from None

    with f:
        for line in f:
            line = line.rstrip('\n')

            # Track current link block
            if line.startswith('data_link_'):
                current_link = line[10:]
                in_angle_loop = False
                columns = []
                continue

            # Detect loop start
            if line.startswith('loop_'):
                in_angle_loop = False
                columns = []
                continue

            # Parse _chem_link_angle columns
            if line.startswith('_chem_link_angle.'):
                field = line.strip().split('.')[1].split()[0]
                columns.append(field)
                col_idx = len(columns) - 1

                if field == 'link_id':
                    link_id_col = col_idx
                elif field == 'atom_1_comp_id':
                    atom1_comp_col = col_idx
                elif field == 'atom_id_1':
                    atom1_id_col = col_idx
                elif field == 'atom_2_comp_id':
                    atom2_comp_col = col_idx
                elif field == 'atom_id_2':
                    atom2_id_col = col_idx
                elif field == 'atom_3_comp_id':
                    atom3_comp_col = col_idx
                elif field == 'atom_id_3':
                    atom3_id_col = col_idx
                elif field == 'value_angle':
                    angle_col = col_idx
                elif field == 'value_angle_esd':
                    angle_esd_col = col_idx

                in_angle_loop = True
                continue

            # Parse data lines (9 columns for angles)
            if in_angle_loop and line.strip() and not line.startswith('#') and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        link_id = parts[link_id_col] if link_id_col >= 0 else current_link

                        yield LinkAngle(
                            link_id=link_id,
                            atom_1=clean_atom_name(parts[atom1_id_col]),
                            comp_1=int(parts[atom1_comp_col]),
                            atom_2=clean_atom_name(parts[atom2_id_col]),
                            comp_2=int(parts[atom2_comp_col]),
                            atom_3=clean_atom_name(parts[atom3_id_col]),
                            comp_3=int(parts[atom3_comp_col]),
                            value_angle=float(parts[angle_col]),
                            value_angle_esd=float(parts[angle_esd_col]),
                        )
                    except (ValueError, IndexError):
                        continue

            # End of loop
            if line.startswith('#') or (line.startswith('_') and not line.startswith('_chem_link_angle')):
                in_angle_loop = False


def load_backbone_dihedrals(filepath: Path | None = None) -> BackboneDihedrals:
    """
    Load backbone dihedral definitions from MonomerLibrary.

    Args:
        filepath: Path to links_and_mods.cif. If None, downloads if necessary.

    Returns:
        BackboneDihedrals with peptide and nucleic acid backbone angle definitions.
    """
    if filepath is None:
        from .cli import get_monlib_path
        filepath = get_monlib_path()

    dihedrals = BackboneDihedrals()

    for torsion in parse_link_torsions(filepath):
        # Protein backbone (TRANS link)
        if torsion.link_id.upper() == "TRANS":
            if torsion.torsion_id == "phi":
                dihedrals.phi = torsion.atoms
            elif torsion.torsion_id == "psi":
                dihedrals.psi = torsion.atoms
            elif torsion.torsion_id == "omega":
                dihedrals.omega = torsion.atoms

        # Nucleic acid backbone (p link = phosphodiester)
        elif torsion.link_id.lower() == "p":
            if torsion.torsion_id == "alpha":
                dihedrals.alpha = torsion.atoms
            elif torsion.torsion_id == "beta":
                dihedrals.beta = torsion.atoms
            elif torsion.torsion_id == "gamma":
                dihedrals.gamma = torsion.atoms
            elif torsion.torsion_id == "delta":
                dihedrals.delta = torsion.atoms
            elif torsion.torsion_id == "epsilon":
                dihedrals.epsilon = torsion.atoms
            elif torsion.torsion_id == "zeta":
                dihedrals.zeta = torsion.atoms

    return dihedrals


def convert_to_residue_offset(
    atoms: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    """
    Convert MonomerLibrary comp_id (1 or 2) to residue offset (-1 or 0).

    MonomerLibrary uses:
    - comp_id=1: first residue in link (previous residue)
    - comp_id=2: second residue in link (current residue)

    We convert to:
    - offset=-1: previous residue
    - offset=0: current residue

    This matches our convention where the 4th atom "owns" the dihedral.
    """
    return tuple((atom_name, comp_id - 2) for atom_name, comp_id in atoms)
