"""Trainer for coordinate diffusion models on polymer structures.

This module provides the CoordinateDiffusionTrainer, which handles:
- Variable-length batching with padding
- EMA weight tracking for the denoiser
- RMSD-based validation via sample generation

Uses PyTorch Lightning Fabric for device handling and mixed precision.

Example:
    >>> from ciffy.nn.diffusion import CoordinateDiffusionTrainer, CoordinateDiffusionTrainingConfig
    >>>
    >>> config = CoordinateDiffusionTrainingConfig.from_yaml("config.yaml")
    >>> trainer = CoordinateDiffusionTrainer(config)
    >>> trainer.train()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

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
    from lightning.fabric import Fabric

from ciffy import Scale

from ..base_trainer import (
    BaseConfig,
    MetricsLogger,
    OutputConfig,
    TrainingConfig,
    WandbConfig,
)
from ..dataset import PolymerDataset
from ..trainer_registry import register_trainer
from ..fabric_utils import create_fabric
from .ema import EMA
from .coordinate_diffusion import CoordinateDiffusionConfig, CoordinateDiffusionModel

logger = logging.getLogger(__name__)


@dataclass
class CoordinateDiffusionDataConfig:
    """Dataset configuration for coordinate diffusion training.

    Attributes:
        data_dir: Directory containing CIF files.
        batch_size: Training batch size.
        molecule_types: Filter to specific molecule types (e.g., ["RNA"]).
        min_atoms: Minimum atoms per chain.
        max_atoms: Maximum atoms per chain.
        limit: Optional limit on number of samples (for testing).
    """

    data_dir: str = ""
    batch_size: int = 8  # Smaller due to full coordinate tensors
    molecule_types: tuple[str, ...] = ("RNA",)
    min_atoms: int = 50
    max_atoms: int = 2000
    limit: Optional[int] = None  # For quick testing


@dataclass
class CoordinateDiffusionTrainingConfig(BaseConfig):
    """Full configuration for coordinate diffusion training.

    Combines model, data, training, output, and logging configurations.

    Example:
        >>> config = CoordinateDiffusionTrainingConfig(
        ...     model=CoordinateDiffusionConfig(),
        ...     data=CoordinateDiffusionDataConfig(data_dir="./data"),
        ...     training=TrainingConfig(epochs=100),
        ... )
    """

    model: CoordinateDiffusionConfig = field(default_factory=CoordinateDiffusionConfig)
    data: CoordinateDiffusionDataConfig = field(default_factory=CoordinateDiffusionDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # EMA settings
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000

    # Validation settings
    val_every: int = 10
    val_samples: int = 3
    val_steps: int = 50  # DDIM steps for validation sampling


class CoordinateDataset(Dataset):
    """Dataset wrapper that returns polymers with torch tensors."""

    def __init__(
        self,
        polymer_dataset: "PolymerDataset",
        device: str = "cpu",
    ) -> None:
        self.polymer_dataset = polymer_dataset
        self.device = device
        # Pre-filter valid indices (some polymers may fail to load)
        self._valid_indices = []
        for i in range(len(polymer_dataset)):
            try:
                p = polymer_dataset[i]
                if p is not None and p.size() > 0:
                    self._valid_indices.append(i)
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> "Polymer":
        """Get polymer with torch backend."""
        real_idx = self._valid_indices[idx]
        polymer = self.polymer_dataset[real_idx]
        return polymer.torch()


def coordinate_collate_fn(
    batch: list["Polymer"],
) -> list["Polymer"]:
    """Collate polymers - just return as list.

    For coordinate diffusion, we process each polymer individually
    in the training loop since PolymerEmbedding needs the full Polymer.
    """
    return batch


@register_trainer("coordinate_diffusion", CoordinateDiffusionTrainingConfig)
class CoordinateDiffusionTrainer:
    """Trainer for coordinate diffusion models.

    Handles training loop, EMA, validation, and checkpointing.

    Example:
        >>> config = CoordinateDiffusionTrainingConfig(
        ...     data=CoordinateDiffusionDataConfig(data_dir="./rna"),
        ...     training=TrainingConfig(epochs=100, device="cuda"),
        ... )
        >>> trainer = CoordinateDiffusionTrainer(config)
        >>> result = trainer.train()
    """

    def __init__(
        self,
        config: CoordinateDiffusionTrainingConfig,
        metrics_logger: Optional[MetricsLogger] = None,
        quiet: bool = False,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CoordinateDiffusionTrainer")

        self.config = config
        self.metrics_logger = metrics_logger
        self.quiet = quiet

        # Create model
        self.model = CoordinateDiffusionModel(config.model)

        # Setup Fabric, optimizer, model
        self._fabric: Optional["Fabric"] = None
        self._setup_training()

        # Setup EMA
        self.ema = EMA(
            self.model.denoiser,
            decay=config.ema_decay,
            warmup_steps=config.ema_warmup_steps,
        )

        # Setup data
        self._dataset: Optional[FilteredPolymerDataset] = None
        self._coord_dataset: Optional[CoordinateDataset] = None
        self.dataloader: Optional[DataLoader] = None

        # Setup output directories
        self.checkpoint_dir = Path(config.output.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = Path(config.output.sample_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.best_checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
        self.best_loss = float("inf")
        self.current_epoch = 0

    def _get_fabric(self) -> "Fabric":
        """Get or create Fabric instance."""
        if self._fabric is None:
            self._fabric = create_fabric(
                device=self.config.training.device,
                precision=self.config.training.precision,
                num_devices=self.config.training.num_devices,
            )
        return self._fabric

    def _setup_training(self) -> None:
        """Setup optimizer and wrap model with Fabric."""
        fabric = self._get_fabric()

        # Create optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        # Setup with Fabric
        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

    @property
    def device(self) -> "torch.device":
        """Get device from Fabric."""
        return self._get_fabric().device

    def setup_data(self, cif_files: Optional[list[Path]] = None) -> None:
        """Setup dataset and dataloader.

        Args:
            cif_files: Optional list of CIF files. If None, loads from config.data.data_dir.
        """
        from ciffy import Molecule

        config = self.config

        if cif_files is None:
            data_dir = Path(config.data.data_dir)
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            cif_files = sorted(data_dir.glob("*.cif"))

        # Apply limit for testing
        if config.data.limit is not None:
            cif_files = cif_files[:config.data.limit]

        # Convert molecule types
        mol_types = tuple(
            getattr(Molecule, m) for m in config.data.molecule_types
        )

        # Create PolymerDataset from files
        # PolymerDataset needs a directory, so we use the parent of first file
        # and restrict via PDB IDs
        if len(cif_files) == 0:
            raise ValueError("No CIF files provided")

        # Get directory and allowed PDB IDs
        data_dir = cif_files[0].parent
        allowed_ids = {f.stem.upper() for f in cif_files}

        # Create dataset from directory
        self._dataset = PolymerDataset(
            data_dir,
            scale=Scale.CHAIN,
            molecule_types=mol_types,
            min_atoms=config.data.min_atoms,
            max_atoms=config.data.max_atoms,
            backend="numpy",  # Will convert to torch in CoordinateDataset
        )

        # Filter to only the specified files
        if len(allowed_ids) < len(list(data_dir.glob("*.cif"))):
            # Need to filter the dataset index to only include allowed files
            self._dataset._index = [
                (path, chain_idx)
                for path, chain_idx in self._dataset._index
                if path.stem.upper() in allowed_ids
            ]

        if len(self._dataset) == 0:
            raise ValueError(
                f"No samples found in dataset!\n"
                f"  Files: {len(cif_files)} CIF files\n"
                f"  Molecule types: {config.data.molecule_types}\n"
                f"  Atom range: [{config.data.min_atoms}, {config.data.max_atoms}]"
            )

        # Wrap for torch output
        self._coord_dataset = CoordinateDataset(
            self._dataset,
            device=str(self.device),
        )

        # Create dataloader
        self.dataloader = DataLoader(
            self._coord_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            collate_fn=coordinate_collate_fn,
            num_workers=config.training.num_workers,
        )

        logger.info(f"Dataset: {len(self._dataset)} polymers")

    def train(
        self,
        cif_files: Optional[list[Path]] = None,
        resume_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            cif_files: Optional list of CIF files. If None, uses config.data.data_dir.
            resume_path: Optional checkpoint path to resume from.
            progress_callback: Optional callback called after each epoch.

        Returns:
            Dictionary containing training results.
        """
        from ..training import load_checkpoint, save_checkpoint

        # Setup data if not already done
        if self.dataloader is None:
            self.setup_data(cif_files)

        fabric = self._get_fabric()
        total_epochs = self.config.training.epochs
        start_epoch = 0

        # Resume from checkpoint if specified
        if resume_path is not None:
            ckpt = load_checkpoint(Path(resume_path), self.model, self.optimizer)
            start_epoch = ckpt.get("epoch", 0) + 1
            self.best_loss = ckpt.get("metrics", {}).get("loss", float("inf"))
            if not self.quiet:
                logger.info(f"Resumed from epoch {start_epoch}")

        metrics: dict[str, float] = {}
        total_samples = 0

        if not self.quiet:
            precision_plugin = getattr(fabric, "_precision", None)
            precision_str = getattr(precision_plugin, "precision", "32-true") if precision_plugin else "32-true"
            logger.info(f"Training with Fabric ({fabric.device}, precision={precision_str})")

        try:
            for epoch in range(start_epoch, total_epochs):
                self.current_epoch = epoch
                self.model.train()

                # Train one epoch
                epoch_loss = 0.0
                n_samples = 0

                for polymers in self.dataloader:
                    # Process each polymer in the batch individually
                    batch_loss = 0.0
                    for polymer in polymers:
                        polymer = polymer.to(self.device)

                        self.optimizer.zero_grad()

                        # Compute loss
                        loss, batch_metrics = self.model.training_step(polymer)

                        # Backward with Fabric
                        fabric.backward(loss)

                        # Gradient clipping
                        if self.config.training.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.training.grad_clip,
                            )

                        self.optimizer.step()

                        batch_loss += loss.item()
                        n_samples += 1

                    epoch_loss += batch_loss

                # Average metrics
                avg_loss = epoch_loss / max(n_samples, 1)
                metrics = {
                    "loss": avg_loss,
                    "n_samples": n_samples,
                }
                total_samples += n_samples

                # Log metrics
                if not self.quiet:
                    self._log_epoch(epoch, total_epochs, metrics)

                # Log to external logger
                if self.metrics_logger is not None:
                    self.metrics_logger.log(metrics, step=epoch)

                # Progress callback
                if progress_callback is not None:
                    progress_callback(epoch + 1, total_epochs, metrics)

                # Update EMA
                self.ema.update(self.model.denoiser)

                # Periodic validation
                if self.config.val_every > 0 and (epoch + 1) % self.config.val_every == 0:
                    val_metrics = self._validate()
                    metrics.update(val_metrics)

                    if self.metrics_logger is not None and val_metrics:
                        self.metrics_logger.log(val_metrics, step=epoch)

                # Save periodic checkpoint
                if (epoch + 1) % self.config.output.save_every == 0:
                    self._save_checkpoint(epoch, metrics, is_best=False)

                # Save best checkpoint
                current_loss = metrics.get("loss", float("inf"))
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self._save_checkpoint(epoch, metrics, is_best=True)

            # Save final checkpoint
            self._save_checkpoint(total_epochs - 1, metrics, is_best=False, is_final=True)

            if not self.quiet:
                logger.info("Training complete!")

        finally:
            if self.metrics_logger is not None:
                self.metrics_logger.finish()

        return {
            "status": "success",
            "final_loss": metrics.get("loss"),
            "best_loss": self.best_loss,
            "epochs_trained": total_epochs - start_epoch,
            "total_epochs": total_epochs,
            "n_samples": total_samples,
            "checkpoint_path": str(self.best_checkpoint_path),
        }

    def _log_epoch(self, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics to console."""
        parts = [f"Epoch {epoch + 1}/{total_epochs}"]

        if "loss" in metrics:
            parts.append(f"Loss: {metrics['loss']:.4f}")
        if "val_rmsd_mean" in metrics:
            parts.append(f"Val RMSD: {metrics['val_rmsd_mean']:.2f}Å")

        logger.info(" | ".join(parts))

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save a training checkpoint."""
        from ..training import save_checkpoint

        if is_best:
            path = self.best_checkpoint_path
        elif is_final:
            path = self.checkpoint_dir / "checkpoint_final.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch{epoch + 1:04d}.pt"

        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch + 1,
            metrics=metrics,
            config=self.config,
        )

    def _validate(self) -> dict[str, float]:
        """Compute validation metrics via sample generation."""
        if self._coord_dataset is None or len(self._coord_dataset) == 0:
            return {}

        self.model.eval()
        rmsds = []

        try:
            with self.ema.apply(self.model.denoiser):
                n_samples = min(self.config.val_samples, len(self._coord_dataset))

                for idx in range(n_samples):
                    polymer = self._coord_dataset[idx].to(self.device)

                    # Get original coordinates
                    original_coords = polymer.coordinates.cpu().numpy()

                    # Generate sample
                    try:
                        samples = self.model.sample(
                            polymer.numpy(),
                            n_samples=1,
                            num_steps=self.config.val_steps,
                        )

                        if samples:
                            from ciffy import rmsd
                            rmsd_val = rmsd(samples[0].coordinates, original_coords)
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


__all__ = [
    "CoordinateDiffusionDataConfig",
    "CoordinateDiffusionTrainingConfig",
    "CoordinateDataset",
    "CoordinateDiffusionTrainer",
]
