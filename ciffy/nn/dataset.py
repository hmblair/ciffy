"""
PyTorch Dataset for loading CIF files.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..polymer import Polymer

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Placeholder for type hints

from ..types import Scale, Molecule


class PolymerDataset(Dataset):
    """
    PyTorch Dataset for loading CIF files from a directory.

    Supports iteration at molecule or chain scale, with optional
    filtering by maximum atom count and molecule type.

    Example:
        >>> from ciffy.nn import PolymerDataset
        >>> from ciffy import Scale, Molecule
        >>> # Basic usage
        >>> dataset = PolymerDataset("./structures/", scale=Scale.CHAIN, max_atoms=5000)
        >>> print(f"Found {len(dataset)} chains")
        >>> chain = dataset[0]  # Load first chain
        >>>
        >>> # Only RNA chains
        >>> dataset = PolymerDataset("./structures/", molecule_types=Molecule.RNA)
        >>>
        >>> # Only protein and RNA chains
        >>> dataset = PolymerDataset("./structures/", molecule_types=(Molecule.PROTEIN, Molecule.RNA))
    """

    def __init__(
        self,
        directory: str | Path,
        scale: Scale = Scale.MOLECULE,
        max_atoms: int | None = None,
        backend: str = "torch",
        molecule_types: Molecule | tuple[Molecule, ...] | None = None,
    ):
        """
        Initialize dataset by scanning directory for CIF files.

        Args:
            directory: Path to directory containing .cif files.
            scale: Iteration scale (MOLECULE or CHAIN only).
                - MOLECULE: iterate over full structures
                - CHAIN: iterate over individual chains
            max_atoms: Maximum atoms per item. Items exceeding this
                are filtered out. None = no limit.
            backend: Backend for loaded polymers ("torch" or "numpy").
            molecule_types: Filter to specific molecule type(s). Can be
                a single Molecule or tuple of Molecules. Chains not matching
                any specified type are excluded. None = no filtering.
                Common types: Molecule.PROTEIN, Molecule.RNA, Molecule.DNA

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If scale is not MOLECULE or CHAIN.
            FileNotFoundError: If directory does not exist.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PolymerDataset. "
                "Install with: pip install torch"
            )

        if scale not in (Scale.MOLECULE, Scale.CHAIN):
            raise ValueError(
                f"scale must be MOLECULE or CHAIN, got {scale.name}"
            )

        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        self.scale = scale
        self.max_atoms = max_atoms
        self.backend = backend

        # Normalize molecule_types to tuple or None
        if molecule_types is None:
            self.molecule_types = None
        elif isinstance(molecule_types, Molecule):
            self.molecule_types = (molecule_types,)
        else:
            self.molecule_types = tuple(molecule_types)

        # Build index: list of (file_path, chain_idx or None)
        self._index: list[tuple[Path, int | None]] = []
        self._build_index(directory)

    def _build_index(self, directory: Path) -> None:
        """Scan directory and build index of valid items."""
        from .. import load_metadata

        cif_files = sorted(directory.glob("*.cif"))

        # Pre-compute type filter values if needed
        type_filter = None
        if self.molecule_types is not None:
            type_filter = {m.value for m in self.molecule_types}

        for path in cif_files:
            try:
                # Use fast metadata-only loading (~3x faster than full load)
                meta = load_metadata(str(path))
            except Exception:
                # Skip files that fail to load
                continue

            atoms_per_chain = meta["atoms_per_chain"]
            mol_types = meta["molecule_types"]

            if self.scale == Scale.MOLECULE:
                # For molecule scale, check if structure has any matching chains
                if type_filter is not None:
                    has_matching = any(int(t) in type_filter for t in mol_types)
                    if not has_matching:
                        continue

                # Check total atom count
                if self.max_atoms is None or meta["atoms"] <= self.max_atoms:
                    self._index.append((path, None))

            else:  # Scale.CHAIN
                # Check each chain against filters
                for chain_idx in range(meta["chains"]):
                    chain_mol_type = int(mol_types[chain_idx])

                    # Skip chains not matching molecule_types filter
                    if type_filter is not None and chain_mol_type not in type_filter:
                        continue

                    # Check atom count
                    chain_atoms = int(atoms_per_chain[chain_idx])
                    if self.max_atoms is not None and chain_atoms > self.max_atoms:
                        continue

                    self._index.append((path, chain_idx))

    def __len__(self) -> int:
        """Return number of valid items (structures or chains)."""
        return len(self._index)

    def __getitem__(self, idx: int) -> Polymer:
        """
        Load and return polymer/chain at index.

        Args:
            idx: Index into dataset.

        Returns:
            Polymer object (full structure or single chain),
            with any configured filtering applied.

        Note:
            At CHAIN scale, molecule_types filtering is done during
            index building, so no filtering is needed here.
            At MOLECULE scale, chain filtering is applied after loading
            to remove non-matching chains from mixed structures.
        """
        from .. import load

        path, chain_idx = self._index[idx]
        polymer = load(str(path), backend=self.backend)

        if chain_idx is not None:
            # Chain scale: filtering already done during indexing
            polymer = polymer.by_index(chain_idx)
        elif self.molecule_types is not None:
            # Molecule scale: filter out non-matching chains
            polymer = self._filter_by_molecule_type(polymer)

        return polymer

    def _filter_by_molecule_type(self, polymer: Polymer) -> Polymer:
        """Filter polymer to only include chains of specified molecule types."""
        import numpy as np

        # Get molecule types as numpy for comparison
        mol_types = polymer.molecule_type
        if hasattr(mol_types, 'numpy'):
            mol_types = mol_types.numpy()
        mol_types = np.asarray(mol_types)

        # Build mask for matching types
        type_values = [m.value for m in self.molecule_types]
        mask = np.isin(mol_types, type_values)

        # Get matching chain indices
        matching_indices = np.nonzero(mask)[0]

        if len(matching_indices) == 0:
            # Return empty polymer (first 0 atoms)
            return polymer[:0]

        return polymer.by_index(matching_indices)
