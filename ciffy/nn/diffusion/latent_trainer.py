"""Trainer for latent diffusion models on polymer structures.

This module provides the LatentDiffusionTrainer, which handles:
- On-the-fly encoding of polymer coordinates to latent space
- Variable-length sequence batching with padding
- EMA weight tracking for the denoiser
- RMSD-based validation via sample generation

Uses PyTorch Lightning Fabric for device handling and mixed precision.

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
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Union, runtime_checkable

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
    OutputConfig,
    TrainingConfig,
    WandbConfig,
)
from ..data_validation import validate_flow_model_compatibility
from ..dataset import PolymerDataset
from ..filtered_dataset import FilterConfig, FilteredPolymerDataset
from ..trainer_registry import register_trainer
from ..fabric_utils import create_fabric
from .ema import EMA
from .latent_diffusion import LatentDiffusionConfig, LatentDiffusionModel

logger = logging.getLogger(__name__)


@runtime_checkable
class MetricsLogger(Protocol):
    """Protocol for metrics logging (wandb, tensorboard, etc.)."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics for a given step."""
        ...

    def info(self, message: str) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str) -> None:
        """Log a warning message."""
        ...

    def finish(self) -> None:
        """Finalize logging."""
        ...


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
    """Dataset that encodes pre-filtered polymers to latent space on-the-fly.

    This is a thin wrapper that handles encoding only - all filtering
    is done by FilteredPolymerDataset. This avoids memory overhead
    of caching all latents.

    Attributes:
        filtered_dataset: The underlying FilteredPolymerDataset.
        flow_model: Flow model for encoding coordinates to latents.
        device: Device for encoding.
    """

    def __init__(
        self,
        filtered_dataset: "FilteredPolymerDataset",
        flow_model: "nn.Module",
        device: str = "cpu",
    ) -> None:
        """Initialize the encoding dataset.

        Args:
            filtered_dataset: Pre-filtered dataset of Polymer objects.
            flow_model: Flow model for encoding coordinates to latents.
            device: Device for encoding.
        """
        self.filtered_dataset = filtered_dataset
        self.flow_model = flow_model
        self.device = device

    def __len__(self) -> int:
        return len(self.filtered_dataset)

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Get encoded latents and sequence for a sample.

        Returns:
            Tuple of (latents, sequence) where:
                - latents: (n_residues, latent_dim) tensor
                - sequence: (n_residues,) long tensor of residue types
        """
        polymer = self.filtered_dataset[idx]

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
class LatentDiffusionTrainer:
    """Trainer for latent diffusion models on polymer structures.

    Uses PyTorch Lightning Fabric for device handling and mixed precision.
    Provides:
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
        metrics_logger: MetricsLogger | None = None,
        quiet: bool = False,
    ) -> None:
        """Initialize the latent diffusion trainer.

        Args:
            config: Training configuration.
            model: Optional pre-initialized model.
            dataset: Optional pre-created dataset.
            metrics_logger: Optional metrics logger (e.g., WandbLogger).
            quiet: If True, suppress progress bars and reduce logging.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LatentDiffusionTrainer")

        self.config = config
        self.quiet = quiet
        self.metrics_logger = metrics_logger
        self._fabric: "Fabric | None" = None

        # Create Fabric for device handling
        fabric = self._get_fabric()

        # Create model if not provided
        if model is None:
            model = LatentDiffusionModel(config.model)

        # Create base dataset if not provided
        if dataset is None:
            dataset = self._create_polymer_dataset(config)

        # Upfront validation: Quick compatibility check before building full dataset
        if not quiet:
            logger.info("Checking flow model / data compatibility...")

        compat_report = validate_flow_model_compatibility(
            flow_model=model.flow_model,
            polymer_dataset=dataset,
            sample_count=min(100, len(dataset)),
            min_residues=config.data.min_residues,
            max_residues=config.data.max_residues,
        )

        if not compat_report.is_compatible:
            raise ValueError(
                f"Flow model incompatible with dataset!\n\n"
                f"{compat_report.format_summary()}\n\n"
                f"The flow model expects specific atom counts per residue type "
                f"that don't match the structures in your dataset."
            )

        if compat_report.valid_fraction < 0.5 and not quiet:
            logger.warning(
                f"Low data compatibility ({compat_report.valid_fraction * 100:.1f}%):\n"
                f"{compat_report.format_summary()}"
            )

        # Create filtered dataset with full diagnostics
        filter_config = FilterConfig(
            min_residues=config.data.min_residues,
            max_residues=config.data.max_residues,
            poly_only=True,
            reject_unknown_residues=True,
            flow_model=model.flow_model,
        )

        self._filtered_dataset = FilteredPolymerDataset(dataset, filter_config)

        # Create encoding dataset
        self._encoding_dataset = LatentEncodingDataset(
            filtered_dataset=self._filtered_dataset,
            flow_model=model.flow_model,
            device=str(fabric.device),
        )

        if not quiet:
            logger.info(f"Found {len(self._encoding_dataset)} valid samples")

        # Setup model and optimizer with Fabric
        self.model = model
        self.optimizer = optim.AdamW(
            model.denoiser.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Wrap with Fabric for device placement
        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

        # Create dataloader
        self.dataloader = DataLoader(
            self._encoding_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=0,  # Encoding uses GPU, can't use multiple workers
            collate_fn=latent_collate_fn,
        )

        # Setup output directories
        self.checkpoint_dir = Path(config.output.checkpoint_dir)
        self.sample_dir = Path(config.output.sample_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"

        # Setup EMA for denoiser only (flow model is frozen)
        self.ema = EMA(
            self.model.denoiser,
            decay=config.ema_decay,
            warmup_steps=config.ema_warmup_steps,
        )

    def _get_fabric(self) -> "Fabric":
        """Get or create Fabric instance for training."""
        if self._fabric is None:
            self._fabric = create_fabric(
                device=self.config.training.device,
                precision=self.config.training.precision,
            )
            self._fabric.launch()
        return self._fabric

    @property
    def train_dataset_size(self) -> int:
        """Return total training dataset size for progress reporting."""
        return len(self._encoding_dataset)

    @property
    def device(self) -> "torch.device":
        """Return the device used for training."""
        return self._get_fabric().device

    def _create_polymer_dataset(
        self,
        config: LatentDiffusionTrainingConfig,
    ) -> PolymerDataset:
        """Create the polymer dataset from config."""
        from ciffy import Molecule

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

    def train(
        self,
        resume_path: str | Path | None = None,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            resume_path: Optional checkpoint path to resume from.
            progress_callback: Optional callback called after each epoch with
                signature: callback(epoch, total_epochs, metrics).

        Returns:
            Dictionary containing:
                - final_loss: Loss from the final epoch
                - best_loss: Best loss achieved during training
                - epochs_trained: Number of epochs completed
                - checkpoint_path: Path to best checkpoint
        """
        from ..training import load_checkpoint, save_checkpoint

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
                epoch_metrics: dict[str, float] = {}
                n_batches = 0

                for batch in self.dataloader:
                    latents, sequences, mask = batch
                    latents = fabric.to_device(latents)
                    sequences = fabric.to_device(sequences)
                    mask = fabric.to_device(mask)

                    self.optimizer.zero_grad()

                    # Compute loss
                    loss, batch_metrics = self.model.training_step_batch(
                        latents, sequences, mask
                    )

                    # Backward with Fabric (handles mixed precision)
                    fabric.backward(loss)

                    # Gradient clipping
                    if self.config.training.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.grad_clip,
                        )

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    for k, v in batch_metrics.items():
                        epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                    n_batches += 1

                # Average metrics
                metrics = {"loss": epoch_loss / n_batches}
                for k, v in epoch_metrics.items():
                    metrics[k] = v / n_batches
                metrics["n_samples"] = len(self._encoding_dataset)
                total_samples += int(metrics["n_samples"])

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
                if (epoch + 1) % self.config.val_every == 0:
                    val_metrics = self._validate()
                    metrics.update(val_metrics)

                    if self.metrics_logger is not None and val_metrics:
                        self.metrics_logger.log(val_metrics, step=epoch)

                    # Generate samples
                    self._generate_samples(epoch)

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
        if "mse_loss" in metrics:
            parts.append(f"MSE: {metrics['mse_loss']:.4f}")
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
                        polymer = self._filtered_dataset[seq_idx]

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
