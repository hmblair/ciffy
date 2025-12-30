"""DataModule for residue flow model training.

Handles data extraction and exposes training data for PCA computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


class FlowDataModule(LightningDataModule):
    """DataModule for residue flow training.

    This module:
    - Extracts residue conformations from CIF files
    - Splits by structure to prevent data leakage
    - Exposes train_data for PCA computation in ResidueFlowModule.setup()

    The train_data attribute is a numpy array of shape (n_instances, n_features)
    containing flattened coordinates + SE(3) transforms.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> dm = FlowDataModule(
        ...     cif_paths=list(Path("./data").glob("*.cif")),
        ...     residue=Residue.A,
        ... )
        >>> # After setup(), dm.train_data is available for PCA
    """

    def __init__(
        self,
        cif_paths: list[Path],
        residue: "Residue",
        train_split: float = 0.8,
        min_coverage: float = 0.9,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        """Initialize the flow data module.

        Args:
            cif_paths: List of paths to CIF files.
            residue: Residue type to extract (e.g., Residue.A).
            train_split: Fraction of structures for training.
            min_coverage: Minimum atom coverage for extraction.
            batch_size: Training batch size.
            seed: Random seed for splitting.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["cif_paths", "residue"])

        self.cif_paths = cif_paths
        self.residue = residue
        self.train_split = train_split
        self.min_coverage = min_coverage
        self.batch_size = batch_size
        self.seed = seed

        # Set in setup()
        self.train_data: np.ndarray | None = None
        self.test_data: np.ndarray | None = None
        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None
        self.atoms: np.ndarray | None = None

    def setup(self, stage: str) -> None:
        """Extract residue data and split by structure.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.
        """
        if stage not in ("fit", "validate") or self.train_data is not None:
            return

        from ciffy.nn.flow.residue.data import extract_residues_with_links
        from ciffy.nn.split import split_by_structure

        # Split paths by structure (not by instance) to prevent leakage
        np.random.seed(self.seed)
        n_train = int(len(self.cif_paths) * self.train_split)
        indices = np.random.permutation(len(self.cif_paths))
        train_paths = [self.cif_paths[i] for i in indices[:n_train]]
        test_paths = [self.cif_paths[i] for i in indices[n_train:]]

        # Extract residues with link transforms
        train_coords, train_transforms, self.atoms = extract_residues_with_links(
            train_paths,
            self.residue,
            min_coverage=self.min_coverage,
            verbose=False,
        )

        # Create extended representation: [coords_flat, transforms]
        train_coords_flat = train_coords.reshape(len(train_coords), -1)
        self.train_data = np.concatenate([train_coords_flat, train_transforms], axis=1)

        # Extract test data
        if len(test_paths) > 0:
            test_coords, test_transforms, _ = extract_residues_with_links(
                test_paths,
                self.residue,
                min_coverage=self.min_coverage,
                verbose=False,
            )
            if len(test_coords) > 0:
                test_coords_flat = test_coords.reshape(len(test_coords), -1)
                self.test_data = np.concatenate([test_coords_flat, test_transforms], axis=1)
            else:
                self.test_data = self.train_data  # Use train as val if no test samples
        else:
            self.test_data = self.train_data  # Use train as val if no test

        # Create PyTorch datasets
        self.train_dataset = TensorDataset(
            torch.from_numpy(self.train_data).float()
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(self.test_data).float()
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    @property
    def n_atoms(self) -> int:
        """Number of atoms per residue."""
        return len(self.atoms) if self.atoms is not None else 0

    @property
    def n_features(self) -> int:
        """Number of features (coords + transforms)."""
        return self.train_data.shape[1] if self.train_data is not None else 0


__all__ = ["FlowDataModule"]
