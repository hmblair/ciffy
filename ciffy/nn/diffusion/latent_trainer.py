"""Trainer for latent diffusion models on polymer structures.

This module provides the LatentDiffusionTrainer, which handles:
- On-the-fly encoding of polymer coordinates to latent space
- Variable-length sequence batching with padding
- EMA weight tracking for the denoiser
- RMSD-based validation via sample generation

Example:
    >>> from ciffy.nn.diffusion import LatentDiffusionTrainer, LatentDiffusionTrainingConfig
    >>>
    >>> config = LatentDiffusionTrainingConfig.from_yaml("config.yaml")
    >>> trainer = LatentDiffusionTrainer(config)
    >>> trainer.train()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

from ciffy import Scale

from ..base_trainer import (
    BaseConfig,
    BaseTrainer,
    MetricsLogger,
    OutputConfig,
    TrainingConfig,
    WandbConfig,
)
from ..dataset import PolymerDataset
from ..trainer_registry import register_trainer
from .ema import EMA
from .latent_diffusion import LatentDiffusionConfig, LatentDiffusionModel

logger = logging.getLogger(__name__)


@dataclass
class LatentDiffusionDataConfig:
    """Dataset configuration for latent diffusion training.

    Attributes:
        data_dir: Directory containing CIF files.
        batch_size: Training batch size.
        molecule_types: Filter to specific molecule types (e.g., ["RNA"]).
        min_residues: Minimum residues per chain.
        max_residues: Maximum residues per chain.
    """

    data_dir: str = ""
    batch_size: int = 32
    molecule_types: tuple[str, ...] = ("RNA",)
    min_residues: int = 10
    max_residues: int = 500


@dataclass
class LatentDiffusionTrainingConfig(BaseConfig):
    """Full configuration for latent diffusion training.

    Combines model, data, training, output, and logging configurations.

    Example:
        >>> config = LatentDiffusionTrainingConfig(
        ...     model=LatentDiffusionConfig(),
        ...     data=LatentDiffusionDataConfig(data_dir="./data"),
        ...     training=TrainingConfig(epochs=100),
        ... )
    """

    model: LatentDiffusionConfig = field(default_factory=LatentDiffusionConfig)
    data: LatentDiffusionDataConfig = field(default_factory=LatentDiffusionDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # EMA settings
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000

    # Validation settings
    val_every: int = 10
    val_samples: int = 5
    val_steps: int = 50  # DDIM steps for validation sampling


class LatentEncodingDataset(Dataset):
    """Dataset that encodes polymers to latent space on-the-fly.

    Filters polymers by residue count and encodes coordinates using
    the flow model during __getitem__. This avoids memory overhead
    of caching all latents.
    """

    def __init__(
        self,
        polymer_dataset: "PolymerDataset",
        flow_model: "nn.Module",
        min_residues: int = 10,
        max_residues: int = 500,
        device: str = "cpu",
    ) -> None:
        """Initialize the encoding dataset.

        Args:
            polymer_dataset: Source dataset of Polymer objects.
            flow_model: Flow model for encoding coordinates to latents.
            min_residues: Minimum residues per chain.
            max_residues: Maximum residues per chain.
            device: Device for encoding.
        """
        self.polymer_dataset = polymer_dataset
        self.flow_model = flow_model
        self.min_residues = min_residues
        self.max_residues = max_residues
        self.device = device

        # Build index of valid samples (filter by residue count)
        self.valid_indices: list[int] = []
        n_none = 0
        n_too_small = 0
        n_too_large = 0
        n_errors = 0

        for idx in range(len(polymer_dataset)):
            try:
                polymer = polymer_dataset[idx]
                if polymer is None:
                    n_none += 1
                    continue
                n_res = polymer.size(Scale.RESIDUE)
                if n_res < min_residues:
                    n_too_small += 1
                elif n_res > max_residues:
                    n_too_large += 1
                else:
                    self.valid_indices.append(idx)
            except Exception as e:
                n_errors += 1
                logger.debug(f"Error loading sample {idx}: {e}")
                continue

        # Log filtering statistics
        total = len(polymer_dataset)
        valid = len(self.valid_indices)
        logger.info(
            f"LatentEncodingDataset: {valid}/{total} samples valid "
            f"(filtered: {n_too_small} too small, {n_too_large} too large, "
            f"{n_none} None, {n_errors} errors)"
        )

        if valid == 0:
            raise ValueError(
                f"No valid samples in dataset!\n"
                f"  Total samples: {total}\n"
                f"  Too small (<{min_residues} residues): {n_too_small}\n"
                f"  Too large (>{max_residues} residues): {n_too_large}\n"
                f"  None/empty: {n_none}\n"
                f"  Load errors: {n_errors}\n"
                f"Consider adjusting min_residues/max_residues in config."
            )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Get encoded latents and sequence for a sample.

        Returns:
            Tuple of (latents, sequence) where:
                - latents: (n_residues, latent_dim) tensor
                - sequence: (n_residues,) long tensor of residue types
        """
        polymer_idx = self.valid_indices[idx]
        polymer = self.polymer_dataset[polymer_idx]

        # Get only polymer atoms (exclude HETATM like water/ions)
        polymer = polymer.poly()

        coords = polymer.coordinates
        sequence = polymer.sequence

        # Convert to tensors
        if not isinstance(coords, torch.Tensor):
            coords = torch.from_numpy(coords).float()
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.long)

        # Encode to latent space (no gradient needed)
        with torch.no_grad():
            coords = coords.to(self.device)
            latents = self.flow_model.encode(coords, sequence.numpy())

        return latents.cpu(), sequence


def latent_collate_fn(
    batch: list[tuple["torch.Tensor", "torch.Tensor"]],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Collate variable-length latent sequences with padding.

    Args:
        batch: List of (latents, sequence) tuples.

    Returns:
        Tuple of (latents, sequences, mask) where:
            - latents: (batch, max_len, latent_dim)
            - sequences: (batch, max_len)
            - mask: (batch, max_len) bool mask, True = padding
    """
    latents_list = [item[0] for item in batch]
    seq_list = [item[1] for item in batch]

    # Find max length
    max_len = max(z.shape[0] for z in latents_list)
    latent_dim = latents_list[0].shape[1]

    batch_size = len(batch)

    # Initialize padded tensors
    latents = torch.zeros(batch_size, max_len, latent_dim)
    sequences = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padding

    for i, (z, s) in enumerate(zip(latents_list, seq_list)):
        length = z.shape[0]
        latents[i, :length] = z
        sequences[i, :length] = s
        mask[i, :length] = False

    return latents, sequences, mask


@register_trainer("latent_diffusion", LatentDiffusionTrainingConfig)
class LatentDiffusionTrainer(BaseTrainer):
    """Trainer for latent diffusion models on polymer structures.

    Extends BaseTrainer with:
        - On-the-fly encoding of coordinates to latent space
        - Custom loss function for diffusion
        - EMA weight tracking for denoiser
        - RMSD-based validation via sample generation

    Example:
        >>> config = LatentDiffusionTrainingConfig.from_yaml("config.yaml")
        >>> trainer = LatentDiffusionTrainer(config)
        >>> trainer.train()
    """

    config: LatentDiffusionTrainingConfig

    def __init__(
        self,
        config: LatentDiffusionTrainingConfig,
        model: LatentDiffusionModel | None = None,
        dataset: PolymerDataset | None = None,
        device: Optional["torch.device"] = None,
        logger: MetricsLogger | None = None,
        quiet: bool = False,
    ) -> None:
        """Initialize the latent diffusion trainer.

        Args:
            config: Training configuration.
            model: Optional pre-initialized model.
            dataset: Optional pre-created dataset.
            device: Device to train on. If None, uses config.training.device.
            logger: Optional metrics logger (e.g., WandbLogger).
            quiet: If True, suppress progress bars and reduce logging.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LatentDiffusionTrainer")

        # Determine device early (needed for encoding dataset)
        if device is None:
            device = torch.device(config.training.device)
            if device.type == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model if not provided
        if model is None:
            model = LatentDiffusionModel(config.model)
        model = model.to(device)

        # Create dataset if not provided
        if dataset is None:
            dataset = self._create_polymer_dataset(config)

        # Create encoding dataset BEFORE super().__init__ (which calls create_dataloader)
        self._encoding_dataset = LatentEncodingDataset(
            polymer_dataset=dataset,
            flow_model=model.flow_model,
            min_residues=config.data.min_residues,
            max_residues=config.data.max_residues,
            device=str(device),
        )

        if not quiet:
            logger.info(f"Found {len(self._encoding_dataset)} valid samples")

        # Initialize base trainer
        super().__init__(
            config=config,
            model=model,
            dataset=dataset,
            device=device,
            logger=logger,
            quiet=quiet,
        )

        # Setup EMA for denoiser only (flow model is frozen)
        self.ema = EMA(
            self.model.denoiser,
            decay=config.ema_decay,
            warmup_steps=config.ema_warmup_steps,
        )

    @property
    def train_dataset_size(self) -> int:
        """Return total training dataset size for progress reporting."""
        return len(self._encoding_dataset)

    def _create_polymer_dataset(
        self,
        config: LatentDiffusionTrainingConfig,
    ) -> PolymerDataset:
        """Create the polymer dataset from config."""
        from ciffy import Molecule
        from pathlib import Path

        data_dir = Path(config.data.data_dir)
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        mol_types = tuple(
            getattr(Molecule, m) for m in config.data.molecule_types
        )

        dataset = PolymerDataset(
            directory=config.data.data_dir,
            scale=Scale.CHAIN,
            molecule_types=mol_types,
            backend="torch",
        )

        if len(dataset) == 0:
            # List files in directory for debugging
            cif_files = list(data_dir.glob("*.cif"))[:10]
            raise ValueError(
                f"No samples found in dataset!\n"
                f"  Directory: {data_dir}\n"
                f"  Molecule types: {config.data.molecule_types}\n"
                f"  CIF files found: {len(list(data_dir.glob('*.cif')))}\n"
                f"  First few: {[f.name for f in cif_files]}"
            )

        logger.info(f"PolymerDataset: {len(dataset)} chains from {data_dir}")
        return dataset

    def create_optimizer(self) -> "optim.Optimizer":
        """Create optimizer for denoiser only (flow model is frozen)."""
        return optim.AdamW(
            self.model.denoiser.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

    def create_dataloader(self) -> "DataLoader":
        """Create DataLoader for training."""
        return DataLoader(
            self._encoding_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=0,  # Encoding uses GPU, can't use multiple workers
            collate_fn=latent_collate_fn,
        )

    def create_loss_fn(
        self,
    ) -> Callable[["nn.Module", Any], dict[str, "torch.Tensor"]]:
        """Create the diffusion loss function."""
        device = self.device

        def diffusion_loss_fn(
            model: LatentDiffusionModel,
            batch: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"],
        ) -> dict[str, "torch.Tensor"]:
            latents, sequences, mask = batch
            latents = latents.to(device)
            sequences = sequences.to(device)
            mask = mask.to(device)

            loss, metrics = model.training_step_batch(latents, sequences, mask)

            return {"loss": loss, **{k: torch.tensor(v) for k, v in metrics.items()}}

        return diffusion_loss_fn

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Hook for EMA update and periodic validation."""
        # Update EMA after each epoch
        self.ema.update(self.model.denoiser)

        # Periodic validation
        if (epoch + 1) % self.config.val_every == 0:
            val_metrics = self._validate()
            metrics.update(val_metrics)

            # Generate and save samples
            self._generate_samples(epoch)

    def _validate(self) -> dict[str, float]:
        """Compute validation metrics via sample generation."""
        if len(self._encoding_dataset) == 0:
            return {}

        self.model.eval()
        rmsds = []

        try:
            with self.ema.apply(self.model.denoiser):
                # Sample a few structures and compute RMSD
                n_samples = min(self.config.val_samples, len(self._encoding_dataset))

                for idx in range(n_samples):
                    latents, sequence = self._encoding_dataset[idx]
                    latents = latents.to(self.device)
                    sequence_np = sequence.numpy()

                    # Decode original latents for reference
                    original_coords = self.model.decode(latents, sequence_np)

                    # Generate sample via reverse diffusion
                    try:
                        sampled_coords = self.model.sample(
                            sequence,
                            n_samples=1,
                            num_steps=self.config.val_steps,
                        )

                        # Compute RMSD
                        from ciffy import rmsd

                        rmsd_val = rmsd(
                            sampled_coords.cpu().numpy(),
                            original_coords.cpu().numpy(),
                        )
                        rmsds.append(float(rmsd_val))
                    except Exception as e:
                        if not self.quiet:
                            logger.debug(f"Validation sample failed: {e}")
                        continue
        finally:
            self.model.train()

        if len(rmsds) == 0:
            return {}

        return {
            "val_rmsd_mean": float(np.mean(rmsds)),
            "val_rmsd_std": float(np.std(rmsds)),
            "val_n_samples": len(rmsds),
        }

    def _generate_samples(self, epoch: int) -> None:
        """Generate and save sample structures as CIF files.

        For each validation sequence, saves:
        - ground_truth.cif: The original structure
        - sample_0.cif, sample_1.cif, ...: Generated samples

        Directory structure:
            samples/epoch_0010/
            ├── seq_0/
            │   ├── ground_truth.cif
            │   ├── sample_0.cif
            │   └── sample_1.cif
            └── seq_1/
                ├── ground_truth.cif
                └── ...
        """
        if len(self._encoding_dataset) == 0:
            return

        sample_dir = self.sample_dir / f"epoch_{epoch + 1:04d}"

        # Use multiple validation sequences (up to val_samples)
        n_samples = min(self.config.val_samples, len(self._encoding_dataset))

        self.model.eval()
        try:
            with torch.no_grad(), self.ema.apply(self.model.denoiser):
                for seq_idx in range(n_samples):
                    seq_dir = sample_dir / f"seq_{seq_idx}"
                    seq_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        # Get original polymer for ground truth
                        polymer_idx = self._encoding_dataset.valid_indices[seq_idx]
                        polymer = self._encoding_dataset.polymer_dataset[polymer_idx]

                        # Save ground truth
                        polymer.write(str(seq_dir / "ground_truth.cif"))

                        # Generate samples
                        samples = self.model.sample_to_polymer(
                            polymer,
                            n_samples=3,  # 3 samples per sequence
                            num_steps=self.config.val_steps,
                        )

                        if not isinstance(samples, list):
                            samples = [samples]

                        # Save samples as CIF
                        for i, sample in enumerate(samples):
                            sample.write(str(seq_dir / f"sample_{i}.cif"))

                    except Exception as e:
                        if not self.quiet:
                            logger.debug(f"Sample generation for seq {seq_idx} failed: {e}")
                        continue

                if not self.quiet:
                    logger.info(f"Saved validation samples to: {sample_dir}")

        except Exception as e:
            if not self.quiet:
                logger.warning(f"Sample generation failed: {e}")
        finally:
            self.model.train()


__all__ = [
    "LatentDiffusionDataConfig",
    "LatentDiffusionTrainingConfig",
    "LatentEncodingDataset",
    "latent_collate_fn",
    "LatentDiffusionTrainer",
]
