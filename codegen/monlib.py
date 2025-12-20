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

    with open(filepath, 'r', encoding='utf-8') as f:
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

                        # Clean atom names (remove quotes)
                        def clean_atom(s: str) -> str:
                            return s.strip('"').strip("'")

                        atoms = (
                            (clean_atom(parts[atom1_id_col]), int(parts[atom1_comp_col])),
                            (clean_atom(parts[atom2_id_col]), int(parts[atom2_comp_col])),
                            (clean_atom(parts[atom3_id_col]), int(parts[atom3_comp_col])),
                            (clean_atom(parts[atom4_id_col]), int(parts[atom4_comp_col])),
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
