"""
ChimeraX defattr file generation.

Functions for creating attribute files that can be loaded into ChimeraX
to color molecular structures by arbitrary per-residue or per-atom values.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union

import numpy as np

from ..types import Scale

if TYPE_CHECKING:
    from ..polymer import Polymer


def to_defattr(
    polymer: "Polymer",
    values: np.ndarray,
    path: str,
    *,
    scale: Scale = Scale.RESIDUE,
    attr_name: str = "value",
    chain: Union[str, int, None] = None,
) -> None:
    """
    Write values to a ChimeraX defattr file.

    Creates an attribute definition file that can be loaded into ChimeraX
    to assign custom values to residues or atoms for coloring.

    Args:
        polymer: Polymer structure providing residue/atom information.
        values: Array of values to write. Shape must match the scale:
            - Scale.RESIDUE: (num_residues,)
            - Scale.ATOM: (num_atoms,)
        path: Output file path.
        scale: Scale of values (RESIDUE or ATOM).
        attr_name: Attribute name in ChimeraX.
        chain: Chain filter (name string or index). None for all chains.

    Raises:
        ValueError: If values shape doesn't match polymer at given scale.
        ValueError: If scale is not RESIDUE or ATOM.

    Example:
        >>> import ciffy
        >>> polymer = ciffy.load("structure.cif")
        >>> values = np.random.rand(polymer.size(ciffy.RESIDUE))
        >>> ciffy.visualize.to_defattr(polymer, values, "output.defattr")
    """
    values = np.asarray(values)

    if scale == Scale.RESIDUE:
        _write_residue_defattr(polymer, values, path, attr_name, chain)
    elif scale == Scale.ATOM:
        _write_atom_defattr(polymer, values, path, attr_name, chain)
    else:
        raise ValueError(f"Scale must be RESIDUE or ATOM, got {scale}")


def _write_residue_defattr(
    polymer: "Polymer",
    values: np.ndarray,
    path: str,
    attr_name: str,
    chain: Union[str, int, None],
) -> None:
    """Write residue-level defattr file."""
    n_residues = polymer.size(Scale.RESIDUE)
    if values.shape[0] != n_residues:
        raise ValueError(
            f"Values length ({values.shape[0]}) must match number of "
            f"residues ({n_residues})"
        )

    # Get residue-to-chain mapping
    chain_sizes = polymer.sizes(Scale.CHAIN)

    # Build chain name lookup
    chain_names = polymer.names
    if chain is not None:
        if isinstance(chain, int):
            chain_names = [chain_names[chain]]
        else:
            chain_names = [chain]

    with open(path, 'w') as f:
        f.write(f"attribute: {attr_name}\n")
        f.write("recipient: residues\n\n")

        res_idx = 0
        for chain_idx, chain_name in enumerate(polymer.names):
            # Get residues in this chain
            chain_atoms = chain_sizes[chain_idx].item()
            res_sizes = polymer.sizes(Scale.RESIDUE)

            # Count residues in this chain
            atom_count = 0
            chain_res_start = res_idx
            while res_idx < len(res_sizes) and atom_count < chain_atoms:
                atom_count += res_sizes[res_idx].item()
                res_idx += 1
            chain_res_end = res_idx

            # Skip if filtering and this chain doesn't match
            if chain is not None:
                if isinstance(chain, int) and chain_idx != chain:
                    continue
                if isinstance(chain, str) and chain_name != chain:
                    continue

            # Write residues for this chain
            for local_res, global_res in enumerate(range(chain_res_start, chain_res_end)):
                resnum = local_res + 1  # 1-indexed
                value = values[global_res]
                if np.isnan(value):
                    value = 0.0
                f.write(f"\t/{chain_name}:{resnum}\t{value}\n")


def _write_atom_defattr(
    polymer: "Polymer",
    values: np.ndarray,
    path: str,
    attr_name: str,
    chain: Union[str, int, None],
) -> None:
    """Write atom-level defattr file."""
    n_atoms = polymer.size()
    if values.shape[0] != n_atoms:
        raise ValueError(
            f"Values length ({values.shape[0]}) must match number of "
            f"atoms ({n_atoms})"
        )

    atom_names = polymer.atom_names()
    res_sizes = polymer.sizes(Scale.RESIDUE)
    chain_sizes = polymer.sizes(Scale.CHAIN)

    with open(path, 'w') as f:
        f.write(f"attribute: {attr_name}\n")
        f.write("recipient: atoms\n\n")

        atom_idx = 0
        res_idx = 0
        for chain_idx, chain_name in enumerate(polymer.names):
            chain_atoms = chain_sizes[chain_idx].item()

            # Skip if filtering and this chain doesn't match
            if chain is not None:
                if isinstance(chain, int) and chain_idx != chain:
                    atom_idx += chain_atoms
                    # Advance res_idx too
                    temp_atoms = 0
                    while res_idx < len(res_sizes) and temp_atoms < chain_atoms:
                        temp_atoms += res_sizes[res_idx].item()
                        res_idx += 1
                    continue
                if isinstance(chain, str) and chain_name != chain:
                    atom_idx += chain_atoms
                    temp_atoms = 0
                    while res_idx < len(res_sizes) and temp_atoms < chain_atoms:
                        temp_atoms += res_sizes[res_idx].item()
                        res_idx += 1
                    continue

            # Process atoms in this chain
            chain_atom_count = 0
            local_res = 0
            while chain_atom_count < chain_atoms and res_idx < len(res_sizes):
                res_atom_count = res_sizes[res_idx].item()
                resnum = local_res + 1  # 1-indexed

                for _ in range(res_atom_count):
                    atom_name = atom_names[atom_idx]
                    # ChimeraX uses ' for prime in atom names
                    atom_name = atom_name.replace('p', "'")
                    value = values[atom_idx]
                    if np.isnan(value):
                        value = 0.0
                    f.write(f"\t/{chain_name}:{resnum}@{atom_name}\t{value}\n")
                    atom_idx += 1

                chain_atom_count += res_atom_count
                res_idx += 1
                local_res += 1
