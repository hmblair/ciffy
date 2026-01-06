"""
PyTorch Lightning module for training autoregressive residue latent models.

This module provides:
- ResidueLatentARModelModule: Lightning module for training ResidueLatentARModel
- Data handling for polymer -> latent conversion

Example:
    >>> from ciffy.nn.lightning import ResidueLatentARModelModule, ResidueLatentARModelDataModule
    >>> from lightning import Trainer
    >>>
    >>> module = ResidueLatentARModelModule(config)
    >>> datamodule = ResidueLatentARModelDataModule(cif_paths, encoder_model)
    >>>
    >>> trainer = Trainer(max_epochs=100, accelerator="gpu")
    >>> trainer.fit(module, datamodule)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from lightning import LightningModule, LightningDataModule

    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        from pytorch_lightning import LightningModule, LightningDataModule

        LIGHTNING_AVAILABLE = True
    except ImportError:
        LIGHTNING_AVAILABLE = False
        LightningModule = object
        LightningDataModule = object

from ...autoregressive import ResidueLatentARModel, ResidueLatentARModelConfig

if TYPE_CHECKING:
    from ....polymer import Polymer
    from ....biochemistry import Residue


@dataclass
class ResidueLatentARModelTrainingConfig:
    """Training configuration for ResidueLatentARModel.

    Args:
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Number of warmup steps for learning rate.
        batch_size: Batch size for training.
        grad_clip: Gradient clipping value (0 = disabled).
        use_scheduler: Whether to use cosine annealing scheduler.
    """
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 32
    grad_clip: float = 1.0
    use_scheduler: bool = True


@dataclass
class ResidueLatentARModelFullConfig:
    """Full configuration for AR training.

    Args:
        model: Model configuration.
        training: Training configuration.
    """
    model: ResidueLatentARModelConfig = field(default_factory=ResidueLatentARModelConfig)
    training: ResidueLatentARModelTrainingConfig = field(default_factory=ResidueLatentARModelTrainingConfig)


class ResidueLatentARModelModule(LightningModule):
    """
    Lightning module for training ResidueLatentARModel.

    Handles:
    - Model creation and initialization
    - Training step with loss computation
    - Validation with metrics logging
    - Optimizer and scheduler configuration

    Args:
        config: Full configuration object.
    """

    def __init__(self, config: Optional[ResidueLatentARModelFullConfig] = None, **kwargs):
        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning is required")
        super().__init__()

        if config is None:
            config = ResidueLatentARModelFullConfig()

        self.config = config
        self.save_hyperparameters({"config": asdict(config)})

        # Build model
        self.model = ResidueLatentARModel(config.model)

        # Track training steps for warmup
        self._train_steps = 0

    def forward(
        self,
        sequence: "torch.Tensor",
        latents: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> dict:
        return self.model(sequence, latents, padding_mask)

    def training_step(self, batch: dict, batch_idx: int) -> "torch.Tensor":
        sequence = batch["sequence"]
        latents = batch["latents"]
        padding_mask = batch.get("padding_mask", None)

        loss = self.model.compute_loss(sequence, latents, padding_mask)

        # Logging
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Log prediction metrics
        if batch_idx % 100 == 0:
            with torch.no_grad():
                outputs = self.model(sequence, latents, padding_mask)
                pred_mean = outputs["pred_mean"]
                mse = ((pred_mean - latents) ** 2).mean()
                self.log("train/mse", mse, on_step=True, on_epoch=False)

                if self.config.model.predict_std:
                    pred_std = outputs["pred_std"]
                    self.log("train/mean_std", pred_std.mean(), on_step=True, on_epoch=False)

        self._train_steps += 1
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> "torch.Tensor":
        sequence = batch["sequence"]
        latents = batch["latents"]
        padding_mask = batch.get("padding_mask", None)

        loss = self.model.compute_loss(sequence, latents, padding_mask)

        # Compute additional metrics
        outputs = self.model(sequence, latents, padding_mask)
        pred_mean = outputs["pred_mean"]
        mse = ((pred_mean - latents) ** 2).mean()

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/mse", mse, on_epoch=True)

        # Per-dimension correlation (how well does the model capture each latent dim)
        if latents.shape[0] >= 4:  # Need enough samples for correlation
            flat_pred = pred_mean.reshape(-1, pred_mean.shape[-1])
            flat_true = latents.reshape(-1, latents.shape[-1])
            correlations = []
            for d in range(flat_pred.shape[-1]):
                corr = torch.corrcoef(torch.stack([flat_pred[:, d], flat_true[:, d]]))[0, 1]
                correlations.append(corr)
            mean_corr = torch.stack(correlations).mean()
            self.log("val/mean_correlation", mean_corr, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        if not self.config.training.use_scheduler:
            return optimizer

        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.config.training.warmup_steps:
                return step / max(1, self.config.training.warmup_steps)
            return 1.0  # Constant after warmup (trainer handles total decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def get_model(self) -> ResidueLatentARModel:
        """Return the underlying model for inference."""
        return self.model


class PolymerLatentDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Dataset that encodes polymers to latent vectors on-the-fly.

    Args:
        polymers: List of Polymer objects.
        encoder: PolymerModel or dict of residue encoders.
        max_len: Maximum sequence length (longer sequences are cropped).
    """

    def __init__(
        self,
        polymers: List["Polymer"],
        encoder: Any,  # PolymerModel or dict
        max_len: int = 512,
    ):
        self.polymers = polymers
        self.encoder = encoder
        self.max_len = max_len

        # Precompute sequences for fast access
        self._sequences = []
        self._valid_indices = []
        for i, polymer in enumerate(polymers):
            seq = polymer.sequence
            if len(seq) > 0:
                self._sequences.append(seq)
                self._valid_indices.append(i)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        polymer_idx = self._valid_indices[idx]
        polymer = self.polymers[polymer_idx]
        sequence = self._sequences[idx]

        # Crop if too long
        if len(sequence) > self.max_len:
            start = torch.randint(0, len(sequence) - self.max_len + 1, (1,)).item()
            polymer = polymer.residue(list(range(start, start + self.max_len)))
            sequence = sequence[start:start + self.max_len]

        # Encode to latents
        if hasattr(self.encoder, 'encode_polymer'):
            # PolymerModel interface
            latents = self.encoder.encode_polymer(polymer)
        else:
            # Dict of residue encoders
            latents = self._encode_with_dict(polymer)

        return {
            "sequence": torch.tensor(sequence, dtype=torch.long),
            "latents": latents,
        }

    def _encode_with_dict(self, polymer: "Polymer") -> "torch.Tensor":
        """Encode polymer using dict of residue encoders."""
        from ....biochemistry import Scale

        n_residues = polymer.size(Scale.RESIDUE)
        latent_dim = None

        latents_list = []
        for i in range(n_residues):
            res_polymer = polymer.residue(i)
            res_type = res_polymer.sequence[0]

            encoder = self.encoder.get(res_type)
            if encoder is None:
                raise ValueError(f"No encoder for residue type {res_type}")

            coords = torch.tensor(res_polymer.coordinates, dtype=torch.float32)
            z = encoder.encode(coords.unsqueeze(0))  # (1, latent_dim)

            if latent_dim is None:
                latent_dim = z.shape[-1]

            latents_list.append(z.squeeze(0))

        return torch.stack(latents_list)  # (n_residues, latent_dim)


def collate_polymer_latents(batch: List[dict]) -> dict:
    """
    Collate function for variable-length polymer sequences.

    Pads sequences to the maximum length in the batch.
    """
    sequences = [item["sequence"] for item in batch]
    latents = [item["latents"] for item in batch]

    # Find max length
    max_len = max(len(s) for s in sequences)
    latent_dim = latents[0].shape[-1]

    # Pad sequences and latents
    padded_sequences = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_latents = torch.zeros(len(batch), max_len, latent_dim)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)

    for i, (seq, lat) in enumerate(zip(sequences, latents)):
        length = len(seq)
        padded_sequences[i, :length] = seq
        padded_latents[i, :length] = lat
        padding_mask[i, :length] = False

    return {
        "sequence": padded_sequences,
        "latents": padded_latents,
        "padding_mask": padding_mask,
    }


class ResidueLatentARModelDataModule(LightningDataModule):
    """
    Lightning DataModule for ResidueLatentARModel training.

    Handles:
    - Loading polymers from CIF files
    - Encoding to latent vectors
    - Train/val splitting
    - DataLoader creation

    Args:
        cif_paths: List of CIF file paths or glob pattern.
        encoder: PolymerModel or dict of residue encoders for latent encoding.
        val_fraction: Fraction of data for validation.
        batch_size: Batch size.
        max_len: Maximum sequence length.
        num_workers: DataLoader workers.
    """

    def __init__(
        self,
        cif_paths: List[str] | str,
        encoder: Any,
        val_fraction: float = 0.1,
        batch_size: int = 32,
        max_len: int = 512,
        num_workers: int = 4,
    ):
        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning is required")
        super().__init__()

        self.cif_paths = cif_paths
        self.encoder = encoder
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers

        self._train_dataset = None
        self._val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self._train_dataset is not None:
            return

        import ciffy
        from glob import glob

        # Load CIF files
        if isinstance(self.cif_paths, str):
            paths = glob(self.cif_paths)
        else:
            paths = self.cif_paths

        polymers = []
        for path in paths:
            try:
                polymer = ciffy.load(path)
                # Filter to polymer atoms and strip empty residues
                polymer = polymer.strip()
                if polymer.size() > 0:
                    polymers.append(polymer)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        if len(polymers) == 0:
            raise ValueError("No valid polymers loaded")

        # Split train/val
        n_val = max(1, int(len(polymers) * self.val_fraction))
        n_train = len(polymers) - n_val

        # Shuffle and split
        indices = torch.randperm(len(polymers)).tolist()
        train_polymers = [polymers[i] for i in indices[:n_train]]
        val_polymers = [polymers[i] for i in indices[n_train:]]

        self._train_dataset = PolymerLatentDataset(train_polymers, self.encoder, self.max_len)
        self._val_dataset = PolymerLatentDataset(val_polymers, self.encoder, self.max_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_polymer_latents,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_polymer_latents,
            pin_memory=True,
        )
