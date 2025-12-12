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

from ..types import Scale


class PolymerDataset(Dataset):
    """
    PyTorch Dataset for loading CIF files from a directory.

    Supports iteration at molecule or chain scale, with optional
    filtering by maximum atom count.

    Example:
        >>> from ciffy.nn import PolymerDataset
        >>> from ciffy import Scale
        >>> dataset = PolymerDataset("./structures/", scale=Scale.CHAIN, max_atoms=5000)
        >>> print(f"Found {len(dataset)} chains")
        >>> chain = dataset[0]  # Load first chain
    """

    def __init__(
        self,
        directory: str | Path,
        scale: Scale = Scale.MOLECULE,
        max_atoms: int | None = None,
        backend: str = "torch",
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

        # Build index: list of (file_path, chain_idx or None)
        self._index: list[tuple[Path, int | None]] = []
        self._build_index(directory)

    def _build_index(self, directory: Path) -> None:
        """Scan directory and build index of valid items."""
        from .. import load

        cif_files = sorted(directory.glob("*.cif"))

        for path in cif_files:
            try:
                polymer = load(str(path), backend=self.backend)
            except Exception:
                # Skip files that fail to load
                continue

            if self.scale == Scale.MOLECULE:
                # Check total atom count
                if self.max_atoms is None or polymer.size() <= self.max_atoms:
                    self._index.append((path, None))
            else:  # Scale.CHAIN
                # Check each chain's atom count
                for chain_idx in range(polymer.size(Scale.CHAIN)):
                    chain = polymer.by_index(chain_idx)
                    if self.max_atoms is None or chain.size() <= self.max_atoms:
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
            Polymer object (full structure or single chain).
        """
        from .. import load

        path, chain_idx = self._index[idx]
        polymer = load(str(path), backend=self.backend)

        if chain_idx is not None:
            polymer = polymer.by_index(chain_idx)

        return polymer
