"""
CIF file loading functionality.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING, Union, List

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..biochemistry import Molecule

from ..polymer import Field

def load(
    file: str | Path,
    backend: str | None = None,
    molecule_types: Union["Molecule", List["Molecule"], None] = None,
    chains: Union[str, List[str], None] = None,
    model: int = 1,
    skip: Union[str, List[str], None] = ("descriptions", "connections"),
    alt_loc: str | None = "A",
) -> "Polymer":
    """
    Load a molecular structure from a CIF file.

    Parses the CIF file using the C extension and constructs a Polymer
    object with coordinates, atoms, elements, and structural information.

    Args:
        file: Path to the CIF file.
        backend: Array backend, either "numpy" or "torch". Default is "numpy".
        molecule_types: Filter to load only specific molecule types.
            Can be a single Molecule enum (e.g., Molecule.RNA) or a list
            of Molecule enums. If None, all molecules are loaded.
            This enables partial loading for improved performance.
        chains: Filter to load only specific chains by name.
            Can be a single chain name (e.g., "A") or a list of chain names.
            If None, all chains are loaded. Can be combined with molecule_types.
        model: Model number to load for multi-model structures (e.g., NMR
            ensembles). Default is 1. For structures with multiple models,
            each model may have a different number of atoms.
        skip: Fields to skip loading. Can be:
            - A tuple/list of field names (default: ["descriptions", "connections"])
            - "metadata": Skip heavy atom-level fields (coordinates, bfactors,
              atoms, elements, atoms_per_res). Useful for fast indexing.
            - A single field name: Skip that field (e.g., "bfactors")
            - None or []: Load all fields including descriptions and connections
            Skippable fields: coordinates, bfactors, atoms (types), elements,
            sequence (residues), res_per_chain, atoms_per_res, resolution,
            descriptions, connections. Core fields (chains, names, etc.)
            cannot be skipped.
        alt_loc: Which alternate conformation to keep for atoms with multiple
            positions. Default is "A". Set to None to keep all conformations
            (may result in duplicate atoms per residue).

    Returns:
        Polymer object containing the parsed structure.

    Raises:
        OSError: If the file does not exist.
        RuntimeError: If parsing fails.
        ValueError: If backend is not "numpy" or "torch", if skip contains
            invalid or core field names, if model < 1, or if the requested
            model does not exist in the structure.

    Example:
        >>> polymer = load("1abc.cif", backend="numpy")
        >>> print(polymer)
        PDB 1ABC with 1234 atoms (numpy).

        >>> # Load with connections (H-bonds, metal coordination, etc.)
        >>> polymer = load("1abc.cif", skip=["descriptions"])
        >>> print(polymer.connections.shape)
        (4404, 2)

        >>> # Load everything including descriptions and connections
        >>> polymer = load("1abc.cif", skip=[])
        >>> print(polymer.descriptions)
        ['RNA (66-MER)', 'CESIUM ION', ...]

        >>> # Load only RNA chains (partial loading)
        >>> from ciffy import Molecule
        >>> rna = load("1abc.cif", molecule_types=Molecule.RNA)

        >>> # Load RNA and DNA chains
        >>> rna_dna = load("1abc.cif", molecule_types=[Molecule.RNA, Molecule.DNA])

        >>> # Load specific chains by name
        >>> chain_a = load("1abc.cif", chains="A")
        >>> chains_ab = load("1abc.cif", chains=["A", "B"])

        >>> # Combine filters: only RNA chains named A or B
        >>> rna_ab = load("1abc.cif", molecule_types=Molecule.RNA, chains=["A", "B"])

        >>> # Skip loading B-factors for faster loading
        >>> polymer = load("1abc.cif", skip=["descriptions", "connections", "bfactors"])
    """
    # Import here to avoid circular imports
    from ..polymer import Polymer
    from ..biochemistry import Scale, Molecule
    from .._c import _load

    # Convert Path to str for C extension compatibility
    if isinstance(file, Path):
        file = str(file)

    # Handle backend parameter
    if backend is None:
        backend = "numpy"

    if backend not in ("numpy", "torch"):
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")

    if model < 1:
        raise ValueError(f"model must be >= 1, got {model}")

    if not os.path.isfile(file):
        raise OSError(f'The file "{file}" does not exist.')

    # Convert molecule_types to list of ints for C extension
    mol_type_filter = None
    if molecule_types is not None:
        if isinstance(molecule_types, Molecule):
            mol_type_filter = [int(molecule_types)]
        else:
            mol_type_filter = [int(mt) for mt in molecule_types]

    # Convert chains to list of strings for C extension
    chain_filter = None
    if chains is not None:
        if isinstance(chains, str):
            chain_filter = [chains]
        else:
            chain_filter = list(chains)

    # Validate and parse skip parameter
    skip_set = set()
    if skip is not None:
        if isinstance(skip, str):
            skip_set.add(skip)
        elif hasattr(skip, '__iter__'):
            skip_set.update(skip)
        else:
            raise TypeError(
                f"skip must be None, a string, or an iterable of strings, got {type(skip).__name__}"
            )
    load_connections = "connections" not in skip_set

    # Normalize skip to list for C extension (exclude "connections" - handled separately)
    if skip is None:
        skip_for_c = None
    elif isinstance(skip, str):
        skip_for_c = skip if skip != "connections" else None
    else:
        skip_for_c = [s for s in skip if s != "connections"]
        if not skip_for_c:
            skip_for_c = None

    # Load returns a dict with all parsed data
    data = _load(file, skip=skip_for_c, molecule_types=mol_type_filter, chains=chain_filter,
                 connections=load_connections, alt_loc=alt_loc, model=model)

    # Extract fields from dict
    id = data["id"]
    coordinates = data["coordinates"]
    atoms = data["atoms"]
    elements = data["elements"]
    residues = data["residues"]
    atoms_per_res = data["atoms_per_res"]
    atoms_per_chain = data["atoms_per_chain"]
    res_per_chain = data["res_per_chain"]
    chain_names = data["chain_names"]
    strand_names = data["strand_names"]
    molecule_types = data["molecule_types"]

    # Filter out chains with 0 residues (ION/WATER/LIGAND-only chains)
    # These chains only contain HETATM atoms and shouldn't be part of Polymer
    chain_mask = res_per_chain > 0
    if not np.all(chain_mask):
        atoms_per_chain = atoms_per_chain[chain_mask]
        res_per_chain = res_per_chain[chain_mask]
        chain_names = [n for n, m in zip(chain_names, chain_mask) if m]
        strand_names = [n for n, m in zip(strand_names, chain_mask) if m]
        molecule_types = molecule_types[chain_mask]
        # descriptions is per-chain if present
        descriptions = data.get("descriptions", None)
        if descriptions is not None:
            descriptions = [d for d, m in zip(descriptions, chain_mask) if m]
            data["descriptions"] = descriptions

    # Compute total atoms - use sum of atoms_per_chain if coordinates is None (skip='metadata')
    total_atoms = len(coordinates) if coordinates is not None else int(np.sum(atoms_per_chain))
    mol_sizes = np.array([total_atoms], dtype=np.int64)

    sizes = {
        Scale.RESIDUE: atoms_per_res,
        Scale.CHAIN: atoms_per_chain,
        Scale.MOLECULE: mol_sizes,
    }

    # Get descriptions if loaded
    descriptions = data.get("descriptions", None)

    # Get B-factors, resolution, and deposit date
    bfactors = data.get("bfactors", None)
    resolution = data.get("resolution", None)
    # C extension uses -1.0 as sentinel for unavailable; convert to None
    if resolution is not None and resolution < 0:
        resolution = None

    # Convert deposit date string to datetime.date
    deposition_date = None
    date_str = data.get("date")
    if date_str is not None:
        from datetime import date
        try:
            year, month, day = date_str.split("-")
            deposition_date = date(int(year), int(month), int(day))
        except (ValueError, AttributeError):
            pass  # Invalid format, leave as None

    # Get connections if loaded
    connections = data.get("connections", None)
    connection_types = data.get("connection_types", None)

    # Extract HETATM data if present
    hetatm_coords = data.get("hetatm_coordinates")
    hetatm_elements = data.get("hetatm_elements")
    hetatm_chains = data.get("hetatm_chains")
    hetatm_bfactors = data.get("hetatm_bfactors")

    hetero = None
    if hetatm_coords is not None and len(hetatm_coords) > 0:
        from ..hetero import HeteroAtoms
        hetero = HeteroAtoms(
            coordinates=hetatm_coords,
            elements=hetatm_elements,
            chains=hetatm_chains,
            bfactors=hetatm_bfactors,
            pdb_id=id,
        )

    # Create hierarchy from sizes and lengths
    from ..polymer.hierarchy import _Hierarchy
    hierarchy = _Hierarchy.from_sizes_and_lengths(
        sizes=sizes,
        lengths=res_per_chain,
        ref=coordinates,
    )

    # Create Polymer with Field objects
    polymer = Polymer(
        hierarchy,
        # Field objects (arrays with scale)
        coordinates=Field(coordinates, Scale.ATOM),
        atoms=Field(atoms, Scale.ATOM),
        elements=Field(elements, Scale.ATOM),
        sequence=Field(residues, Scale.RESIDUE),
        molecule_types=Field(molecule_types, Scale.CHAIN),
        bfactors=Field(bfactors, Scale.ATOM),
        # Metadata (non-array values)
        pdb_id=id,
        names=chain_names,
        strands=strand_names,
        descriptions=descriptions,
        resolution=resolution,
        date=deposition_date,
        # Internal state
        connections=connections,
        connection_types=connection_types,
        hetero=hetero,
    )

    # Convert to torch if requested
    if backend == "torch":
        return polymer.torch()

    return polymer


def load_metadata(file: str | Path) -> dict:
    """
    Load only metadata from a CIF file (fast path for indexing).

    Skips parsing of coordinates, atom types, and elements, returning
    only the information needed for dataset indexing: atom counts,
    chain structure, and molecule types.

    This is ~3x faster than full load() for large structures.

    Equivalent to: load(file, skip='metadata')

    Args:
        file: Path to the CIF file.

    Returns:
        Dict with keys:
            - id: PDB identifier (str)
            - atoms: Total atom count (int)
            - residues: Total residue count (int)
            - chains: Number of chains (int)
            - atoms_per_chain: Array of atom counts per chain (np.ndarray)
            - residues_per_chain: Array of residue counts per chain (np.ndarray)
            - molecule_types: Array of molecule type per chain (np.ndarray)
              Values correspond to Molecule enum (0=PROTEIN, 1=RNA, 2=DNA, etc.)
            - date: Initial deposition date (datetime.date) or None

    Raises:
        OSError: If the file does not exist.
        RuntimeError: If parsing fails.

    Example:
        >>> meta = load_metadata("8cam.cif")
        >>> print(f"{meta['chains']} chains, {meta['atoms']} total atoms")
        377 chains, 86648 total atoms
        >>> print(f"Chain 0 has {meta['atoms_per_chain'][0]} atoms")
        Chain 0 has 190 atoms
        >>> print(f"Molecule types: {meta['molecule_types'][:5]}")
        Molecule types: [0 0 0 0 0]  # All protein
    """
    from .._c import _load

    # Convert Path to str for C extension compatibility
    if isinstance(file, Path):
        file = str(file)

    if not os.path.isfile(file):
        raise OSError(f'The file "{file}" does not exist.')

    data = _load(file, skip='metadata')

    atoms_per_chain = data["atoms_per_chain"]
    res_per_chain = data["res_per_chain"]
    molecule_types = data["molecule_types"]

    # Filter out chains with 0 residues (ION/WATER/LIGAND-only chains)
    chain_mask = res_per_chain > 0
    if not np.all(chain_mask):
        atoms_per_chain = atoms_per_chain[chain_mask]
        res_per_chain = res_per_chain[chain_mask]
        molecule_types = molecule_types[chain_mask]

    # Convert date string to datetime.date
    date_value = None
    date_str = data.get("date")
    if date_str is not None:
        from datetime import date
        try:
            year, month, day = date_str.split("-")
            date_value = date(int(year), int(month), int(day))
        except (ValueError, AttributeError):
            pass  # Invalid format, leave as None

    return {
        "id": data["id"],
        "atoms": int(atoms_per_chain.sum()),
        "residues": int(res_per_chain.sum()),
        "chains": len(atoms_per_chain),
        "atoms_per_chain": atoms_per_chain,
        "residues_per_chain": res_per_chain,
        "molecule_types": molecule_types,
        "date": date_value,
    }
