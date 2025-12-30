"""DataModules for diffusion model training.

Provides LightningDataModules that handle:
- Loading polymer datasets
- Filtering for flow model compatibility
- On-the-fly latent encoding (for latent diffusion)
- Variable-length batching with padding
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from ciffy import Molecule, Scale
from ciffy.nn.dataset import PolymerDataset
from ciffy.nn.filtered_dataset import FilterConfig, FilteredPolymerDataset

if TYPE_CHECKING:
    from ciffy.nn.flow import PolymerFlowModel


class LatentEncodingDataset(Dataset):
    """Dataset that encodes polymers to latent space on-the-fly.

    Wraps a FilteredPolymerDataset and uses a flow model to encode
    coordinates to latents. This avoids caching all latents in memory.

    Note: This dataset performs GPU encoding in __getitem__, so it must
    be used with num_workers=0 in the DataLoader.
    """

    def __init__(
        self,
        filtered_dataset: FilteredPolymerDataset,
        flow_model: "PolymerFlowModel",
        device: str = "cpu",
    ) -> None:
        self.filtered_dataset = filtered_dataset
        self.flow_model = flow_model
        self.device = device

    def __len__(self) -> int:
        return len(self.filtered_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get encoded latents and sequence for a sample.

        Returns:
            (latents, sequence) where:
            - latents: (n_residues, latent_dim)
            - sequence: (n_residues,) long tensor
        """
        polymer = self.filtered_dataset[idx]

        coords = polymer.coordinates
        sequence = polymer.sequence

        # Convert to tensors
        if not isinstance(coords, torch.Tensor):
            coords = torch.from_numpy(coords).float()
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.long)

        # Encode to latent space
        with torch.no_grad():
            coords = coords.to(self.device)
            latents = self.flow_model.encode(coords, sequence.numpy())

        return latents.cpu(), sequence


def latent_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length latent sequences with padding.

    Args:
        batch: List of (latents, sequence) tuples.

    Returns:
        (latents, sequences, mask) where:
        - latents: (batch, max_len, latent_dim)
        - sequences: (batch, max_len)
        - mask: (batch, max_len) bool, True = padding
    """
    latents_list = [item[0] for item in batch]
    seq_list = [item[1] for item in batch]

    max_len = max(z.shape[0] for z in latents_list)
    latent_dim = latents_list[0].shape[1]
    batch_size = len(batch)

    # Initialize padded tensors
    latents = torch.zeros(batch_size, max_len, latent_dim)
    sequences = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (z, s) in enumerate(zip(latents_list, seq_list)):
        length = z.shape[0]
        latents[i, :length] = z
        sequences[i, :length] = s
        mask[i, :length] = False

    return latents, sequences, mask


class LatentDiffusionDataModule(LightningDataModule):
    """DataModule for latent diffusion training.

    Handles:
    - Loading polymer dataset from a directory of CIF files
    - Filtering for flow model compatibility
    - On-the-fly latent encoding
    - Train/val splitting
    - Variable-length batching with padding

    Example:
        >>> flow_model = load_pretrained("rna")
        >>> dm = LatentDiffusionDataModule(
        ...     data_dir="./structures",
        ...     flow_model=flow_model,
        ...     batch_size=32,
        ... )
        >>> trainer.fit(module, dm)
    """

    def __init__(
        self,
        data_dir: str | Path,
        flow_model: "PolymerFlowModel",
        batch_size: int = 32,
        molecule_types: tuple[str, ...] = ("RNA",),
        min_residues: int = 10,
        max_residues: int = 500,
        val_fraction: float = 0.1,
        num_workers: int = 0,
    ) -> None:
        """Initialize the data module.

        Args:
            data_dir: Directory containing CIF files.
            flow_model: Pre-trained flow model for encoding and validation.
            batch_size: Training batch size.
            molecule_types: Filter to specific molecule types.
            min_residues: Minimum residues per chain.
            max_residues: Maximum residues per chain.
            val_fraction: Fraction of data for validation.
            num_workers: Number of DataLoader workers. Must be 0 for latent
                encoding (GPU operations in __getitem__).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["flow_model"])

        self.data_dir = Path(data_dir)
        self.flow_model = flow_model
        self.batch_size = batch_size
        self.molecule_types = molecule_types
        self.min_residues = min_residues
        self.max_residues = max_residues
        self.val_fraction = val_fraction
        self.num_workers = num_workers

        # Set in setup()
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str) -> None:
        """Set up datasets for training/validation.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.
        """
        if stage not in ("fit", "validate"):
            return

        # Parse molecule types
        mol_types = tuple(
            Molecule[m] if isinstance(m, str) else m for m in self.molecule_types
        )

        # Create base dataset
        base_dataset = PolymerDataset(
            self.data_dir,
            scale=Scale.CHAIN,
            molecule_types=mol_types,
            backend="torch",
        )

        # Create filter config
        filter_config = FilterConfig(
            min_residues=self.min_residues,
            max_residues=self.max_residues,
            poly_only=True,
            reject_unknown_residues=True,
            flow_model=self.flow_model,
        )

        # Filter dataset
        filtered_dataset = FilteredPolymerDataset(base_dataset, filter_config)

        if len(filtered_dataset) == 0:
            raise ValueError(
                f"No valid samples found in {self.data_dir} after filtering. "
                f"Check molecule_types={self.molecule_types}, "
                f"min_residues={self.min_residues}, max_residues={self.max_residues}"
            )

        # Create encoding dataset
        # Note: device is set later when we know the accelerator
        encoding_dataset = LatentEncodingDataset(
            filtered_dataset,
            self.flow_model,
            device="cpu",  # Will be moved to correct device in on_after_batch_transfer
        )

        # Split into train/val
        n_val = int(len(encoding_dataset) * self.val_fraction)
        n_train = len(encoding_dataset) - n_val

        self.train_dataset, self.val_dataset = random_split(
            encoding_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=latent_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=latent_collate_fn,
            pin_memory=True,
        )


class CoordinateDiffusionDataModule(LightningDataModule):
    """DataModule for coordinate diffusion training.

    Similar to LatentDiffusionDataModule but works directly with
    coordinates instead of latents. Simpler but more memory-intensive.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 8,
        molecule_types: tuple[str, ...] = ("RNA",),
        min_atoms: int = 50,
        max_atoms: int = 2000,
        val_fraction: float = 0.1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.molecule_types = molecule_types
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.val_fraction = val_fraction
        self.num_workers = num_workers

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str) -> None:
        if stage not in ("fit", "validate"):
            return

        mol_types = tuple(
            Molecule[m] if isinstance(m, str) else m for m in self.molecule_types
        )

        base_dataset = PolymerDataset(
            self.data_dir,
            scale=Scale.CHAIN,
            min_atoms=self.min_atoms,
            max_atoms=self.max_atoms,
            molecule_types=mol_types,
            backend="torch",
        )

        # Split
        n_val = int(len(base_dataset) * self.val_fraction)
        n_train = len(base_dataset) - n_val

        self.train_dataset, self.val_dataset = random_split(
            base_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,  # Return list of polymers
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,
            pin_memory=True,
        )


__all__ = [
    "LatentEncodingDataset",
    "latent_collate_fn",
    "LatentDiffusionDataModule",
    "CoordinateDiffusionDataModule",
]
