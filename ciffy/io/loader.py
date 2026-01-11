"""
CIF file loading functionality.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING, Union, List, Sequence, overload

import numpy as np

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..biochemistry import Molecule

from ..polymer import Field


@overload
def load(
    file: str | Path,
    backend: str | None = None,
    molecule_types: Union["Molecule", List["Molecule"], None] = None,
    chains: Union[str, List[str], None] = None,
    model: int = 1,
    skip: Union[str, List[str], None] = ("descriptions", "connections"),
    alt_loc: str | None = "A",
) -> "Polymer": ...


@overload
def load(
    file: Sequence[str | Path],
    backend: str | None = None,
    molecule_types: Union["Molecule", List["Molecule"], None] = None,
    chains: Union[str, List[str], None] = None,
    model: int = 1,
    skip: Union[str, List[str], None] = ("descriptions", "connections"),
    alt_loc: str | None = "A",
) -> List["Polymer | None"]: ...


def load(
    file: str | Path | Sequence[str | Path],
    backend: str | None = None,
    molecule_types: Union["Molecule", List["Molecule"], None] = None,
    chains: Union[str, List[str], None] = None,
    model: int = 1,
    skip: Union[str, List[str], None] = ("descriptions", "connections"),
    alt_loc: str | None = "A",
) -> "Polymer | List[Polymer | None]":
    """
    Load molecular structure(s) from CIF file(s).

    Parses CIF file(s) using the C extension and constructs Polymer object(s)
    with coordinates, atoms, elements, and structural information.

    When given multiple files, uses parallel loading with OpenMP threads
    for significantly faster loading (typically 4-6x speedup).

    Args:
        file: Path to a CIF file, or a sequence of paths for parallel loading.
            When a sequence is provided, files are loaded in parallel using
            OpenMP threads, providing significant speedup on multi-core systems.
        backend: Array backend, either "numpy" or "torch". Default is "numpy".
        molecule_types: Filter to load only specific molecule types.
            Can be a single Molecule enum (e.g., Molecule.RNA) or a list
            of Molecule enums. If None, all molecules are loaded.
            This enables partial loading for improved performance.
            Note: Only applies to single-file loading.
        chains: Filter to load only specific chains by name.
            Can be a single chain name (e.g., "A") or a list of chain names.
            If None, all chains are loaded. Can be combined with molecule_types.
            Note: Only applies to single-file loading.
        model: Model number to load for multi-model structures (e.g., NMR
            ensembles). Default is 1. For structures with multiple models,
            each model may have a different number of atoms.
            Note: Only applies to single-file loading.
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
            Note: Only applies to single-file loading.
        alt_loc: Which alternate conformation to keep for atoms with multiple
            positions. Default is "A". Set to None to keep all conformations
            (may result in duplicate atoms per residue).
            Note: Only applies to single-file loading.

    Returns:
        For single file: Polymer object containing the parsed structure.
        For multiple files: List of Polymer objects (None for failed loads).

    Raises:
        OSError: If a single file does not exist.
        RuntimeError: If parsing fails.
        ValueError: If backend is not "numpy" or "torch", if skip contains
            invalid or core field names, if model < 1, or if the requested
            model does not exist in the structure.

    Example:
        >>> # Single file
        >>> polymer = load("1abc.cif", backend="numpy")
        >>> print(polymer)
        PDB 1ABC with 1234 atoms (numpy).

        >>> # Multiple files (parallel loading, ~5x faster)
        >>> from pathlib import Path
        >>> files = list(Path("./structures/").glob("*.cif"))[:100]
        >>> polymers = load(files)
        >>> valid = [p for p in polymers if p is not None]

        >>> # Load with connections (H-bonds, metal coordination, etc.)
        >>> polymer = load("1abc.cif", skip=["descriptions"])
        >>> print(polymer.connections.shape)
        (4404, 2)

        >>> # Load only RNA chains (partial loading)
        >>> from ciffy import Molecule
        >>> rna = load("1abc.cif", molecule_types=Molecule.RNA)

        >>> # Load specific chains by name
        >>> chain_a = load("1abc.cif", chains="A")
    """
    # Check if file is a sequence of paths (but not a single string/Path)
    if not isinstance(file, (str, Path)) and hasattr(file, '__iter__'):
        return _load_multiple(file, backend=backend)

    # Single file path - use existing logic
    return _load_single(
        file,
        backend=backend,
        molecule_types=molecule_types,
        chains=chains,
        model=model,
        skip=skip,
        alt_loc=alt_loc,
    )


def _load_multiple(
    files: Sequence[str | Path],
    backend: str | None = None,
) -> List["Polymer | None"]:
    """
    Load multiple CIF files in parallel using OpenMP.

    Internal implementation for parallel loading.
    """
    from .._c import _load_batch

    if backend is None:
        backend = "numpy"

    if backend not in ("numpy", "torch"):
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")

    # Convert all paths to strings
    file_strs = [str(f) for f in files]

    # Load in parallel at C level (releases GIL)
    results = _load_batch(file_strs)

    # Convert dicts to Polymers
    polymers = []
    for data in results:
        if data is None:
            polymers.append(None)
        else:
            try:
                polymer = _dict_to_polymer(data, backend=backend)
                polymers.append(polymer)
            except Exception:
                polymers.append(None)

    return polymers


def _load_single(
    file: str | Path,
    backend: str | None = None,
    molecule_types: Union["Molecule", List["Molecule"], None] = None,
    chains: Union[str, List[str], None] = None,
    model: int = 1,
    skip: Union[str, List[str], None] = ("descriptions", "connections"),
    alt_loc: str | None = "A",
) -> "Polymer":
    """
    Load a single CIF file.

    Internal implementation for single-file loading with full options.
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
    molecule_types = data["molecule_types"]

    # Filter out chains with 0 residues (ION/WATER/LIGAND-only chains)
    # These chains only contain HETATM atoms and shouldn't be part of Polymer
    chain_mask = res_per_chain > 0
    if not np.all(chain_mask):
        atoms_per_chain = atoms_per_chain[chain_mask]
        res_per_chain = res_per_chain[chain_mask]
        chain_names = [n for n, m in zip(chain_names, chain_mask) if m]
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
        from ..polymer.hetero import HeteroAtoms
        hetero = HeteroAtoms.from_arrays(
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


@overload
def load_metadata(file: str | Path) -> dict: ...


@overload
def load_metadata(file: Sequence[str | Path]) -> List[dict | None]: ...


def load_metadata(
    file: str | Path | Sequence[str | Path],
) -> dict | List[dict | None]:
    """
    Load only metadata from CIF file(s) (fast path for indexing).

    Skips parsing of coordinates, atom types, and elements, returning
    only the information needed for dataset indexing: atom counts,
    chain structure, and molecule types.

    This is ~3x faster than full load() for large structures.
    When given multiple files, uses parallel loading (~5x additional speedup).

    Equivalent to: load(file, skip='metadata')

    Args:
        file: Path to a CIF file, or a sequence of paths for parallel loading.

    Returns:
        For single file: Dict with keys:
            - id: PDB identifier (str)
            - atoms: Total atom count (int)
            - residues: Total residue count (int)
            - chains: Number of chains (int)
            - atoms_per_chain: Array of atom counts per chain (np.ndarray)
            - residues_per_chain: Array of residue counts per chain (np.ndarray)
            - molecule_types: Array of molecule type per chain (np.ndarray)
              Values correspond to Molecule enum (0=PROTEIN, 1=RNA, 2=DNA, etc.)
            - date: Initial deposition date (datetime.date) or None
        For multiple files: List of dicts (None for failed loads).

    Raises:
        OSError: If a single file does not exist.
        RuntimeError: If parsing fails.

    Example:
        >>> # Single file
        >>> meta = load_metadata("8cam.cif")
        >>> print(f"{meta['chains']} chains, {meta['atoms']} total atoms")
        377 chains, 86648 total atoms

        >>> # Multiple files (parallel, ~5x faster)
        >>> from pathlib import Path
        >>> files = list(Path("./structures/").glob("*.cif"))
        >>> metas = load_metadata(files)
        >>> valid = [m for m in metas if m is not None]
    """
    # Check if file is a sequence of paths (but not a single string/Path)
    if not isinstance(file, (str, Path)) and hasattr(file, '__iter__'):
        return _load_metadata_multiple(file)

    return _load_metadata_single(file)


def _load_metadata_single(file: str | Path) -> dict:
    """Load metadata from a single CIF file."""
    from .._c import _load

    # Convert Path to str for C extension compatibility
    if isinstance(file, Path):
        file = str(file)

    if not os.path.isfile(file):
        raise OSError(f'The file "{file}" does not exist.')

    data = _load(file, skip='metadata')
    return _process_metadata(data)


def _load_metadata_multiple(files: Sequence[str | Path]) -> List[dict | None]:
    """Load metadata from multiple CIF files in parallel."""
    from .._c import _load_batch

    # Convert all paths to strings
    file_strs = [str(f) for f in files]

    # Load in parallel at C level with metadata skip (releases GIL)
    results = _load_batch(file_strs, skip='metadata')

    # Process each result
    metas = []
    for data in results:
        if data is None:
            metas.append(None)
        else:
            try:
                metas.append(_process_metadata(data))
            except Exception:
                metas.append(None)

    return metas


def _process_metadata(data: dict) -> dict:
    """Process raw metadata dict into final format."""
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


def _dict_to_polymer(data: dict, backend: str = "numpy") -> "Polymer":
    """
    Convert a dict from _load/_load_batch to a Polymer object.

    Internal helper - shared between load() and load_batch().
    """
    from ..polymer import Polymer
    from ..biochemistry import Scale

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
    molecule_types = data["molecule_types"]

    # Filter out chains with 0 residues (ION/WATER/LIGAND-only chains)
    chain_mask = res_per_chain > 0
    if not np.all(chain_mask):
        atoms_per_chain = atoms_per_chain[chain_mask]
        res_per_chain = res_per_chain[chain_mask]
        chain_names = [n for n, m in zip(chain_names, chain_mask) if m]
        molecule_types = molecule_types[chain_mask]
        descriptions = data.get("descriptions", None)
        if descriptions is not None:
            descriptions = [d for d, m in zip(descriptions, chain_mask) if m]
            data["descriptions"] = descriptions

    # Compute total atoms
    total_atoms = len(coordinates) if coordinates is not None else int(np.sum(atoms_per_chain))
    mol_sizes = np.array([total_atoms], dtype=np.int64)

    sizes = {
        Scale.RESIDUE: atoms_per_res,
        Scale.CHAIN: atoms_per_chain,
        Scale.MOLECULE: mol_sizes,
    }

    # Get optional fields
    descriptions = data.get("descriptions", None)
    bfactors = data.get("bfactors", None)
    resolution = data.get("resolution", None)
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
            pass

    connections = data.get("connections", None)
    connection_types = data.get("connection_types", None)

    # Extract HETATM data if present
    hetatm_coords = data.get("hetatm_coordinates")
    hetatm_elements = data.get("hetatm_elements")
    hetatm_chains = data.get("hetatm_chains")
    hetatm_bfactors = data.get("hetatm_bfactors")

    hetero = None
    if hetatm_coords is not None and len(hetatm_coords) > 0:
        from ..polymer.hetero import HeteroAtoms
        hetero = HeteroAtoms.from_arrays(
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
        coordinates=Field(coordinates, Scale.ATOM),
        atoms=Field(atoms, Scale.ATOM),
        elements=Field(elements, Scale.ATOM),
        sequence=Field(residues, Scale.RESIDUE),
        molecule_types=Field(molecule_types, Scale.CHAIN),
        bfactors=Field(bfactors, Scale.ATOM),
        pdb_id=id,
        names=chain_names,
        descriptions=descriptions,
        resolution=resolution,
        date=deposition_date,
        connections=connections,
        connection_types=connection_types,
        hetero=hetero,
    )

    if backend == "torch":
        return polymer.torch()

    return polymer
