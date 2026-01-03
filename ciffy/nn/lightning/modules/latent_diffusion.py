"""LightningModule for latent diffusion training.

Wraps LatentDiffusionModel with Lightning training logic.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from .base import BaseCiffyModule
from ciffy.nn.config import OutputConfig, TrainingConfig, WandbConfig
from ciffy.nn.diffusion.latent_diffusion import (
    LatentDiffusionConfig,
    LatentDiffusionModel,
)
from ciffy.nn.diffusion.ema import EMA

if TYPE_CHECKING:
    from ciffy.nn.polymer import PolymerModel


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
class LatentDiffusionFullConfig:
    """Full configuration for latent diffusion training.

    Combines model, data, training, output, and logging configurations.
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
    val_steps: int = 50


class LatentDiffusionModule(BaseCiffyModule):
    """LightningModule for training latent diffusion models.

    This module wraps a LatentDiffusionModel and provides:
    - Training step with pre-encoded latent batches
    - Validation step with same loss computation
    - Configurable optimizer and scheduler via BaseCiffyModule

    The module expects batches from LatentDiffusionDataModule:
    - (latents, sequences, mask) tuple
    - latents: (batch, n_residues, latent_dim)
    - sequences: (batch, n_residues)
    - mask: (batch, n_residues) bool, True = padding

    Example:
        >>> config = LatentDiffusionFullConfig.from_yaml("config.yaml")
        >>> module = LatentDiffusionModule(config)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(module, datamodule)
    """

    def __init__(
        self,
        config: LatentDiffusionFullConfig,
        encoder_model: Optional["PolymerModel"] = None,
    ) -> None:
        """Initialize the latent diffusion module.

        Args:
            config: Full training configuration.
            encoder_model: Optional pre-loaded PolymerModel. If None, loads from
                config.model.encoder_path.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_model"])

        self.config = config
        self.training_config = config.training

        # Create the model
        self.model = LatentDiffusionModel(config.model, encoder_model=encoder_model)

        # EMA will be initialized on first training step (needs model on correct device)
        self._ema = None
        self._ema_decay = config.ema_decay
        self._ema_warmup_steps = config.ema_warmup_steps

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        batch_idx: int,
    ) -> torch.Tensor | None:
        """Compute training loss for a batch.

        Args:
            batch: Tuple of (latents, sequences, mask) from dataloader, or None.
            batch_idx: Batch index (unused).

        Returns:
            Loss tensor, or None if batch was empty.
        """
        if batch is None:
            return None

        latents, sequences, mask = batch

        loss, metrics = self.model.training_step_batch(latents, sequences, mask)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/noise_mse", metrics["noise_mse"], on_step=False, on_epoch=True)

        return loss

    def on_train_start(self) -> None:
        """Initialize EMA on training start (model is on correct device)."""
        if self._ema is None:
            self._ema = EMA(
                self.model.denoiser,
                decay=self._ema_decay,
                warmup_steps=self._ema_warmup_steps,
            )

    def on_before_zero_grad(self, optimizer) -> None:
        """Update EMA after each optimizer step."""
        if self._ema is not None:
            self._ema.update(self.model.denoiser)

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        batch_idx: int,
    ) -> torch.Tensor | None:
        """Compute validation loss for a batch.

        Args:
            batch: Tuple of (latents, sequences, mask) from dataloader, or None.
            batch_idx: Batch index (unused).

        Returns:
            Loss tensor, or None if batch was empty.
        """
        if batch is None:
            return None

        latents, sequences, mask = batch

        loss, metrics = self.model.training_step_batch(latents, sequences, mask)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/noise_mse", metrics["noise_mse"], sync_dist=True)

        # Store batch for coordinate-space validation
        if batch_idx == 0:
            self._val_batch = (latents, sequences, mask)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute coordinate-space metrics at end of validation."""
        # Skip during sanity check
        if self.trainer.sanity_checking:
            return

        # Only run every val_every epochs
        val_every = getattr(self.config, "val_every", 10)
        if self.current_epoch % val_every != 0:
            return

        # Get stored validation batch (consumed on first call)
        if not hasattr(self, "_val_batch") or self._val_batch is None:
            return

        latents, sequences, mask = self._val_batch
        self._val_batch = None  # Prevent double computation

        n_samples = min(getattr(self.config, "val_samples", 3), latents.shape[0])
        val_steps = getattr(self.config, "val_steps", 50)

        # Compute coordinate-space metrics
        coord_metrics = self._compute_coord_metrics(
            latents[:n_samples],
            sequences[:n_samples],
            mask[:n_samples] if mask is not None else None,
            num_steps=val_steps,
        )

        # Log and print metrics
        for key, value in coord_metrics.items():
            self.log(f"val/{key}", value, sync_dist=True)

        # Print summary
        if coord_metrics:
            rmsd = coord_metrics.get("coord_rmsd", 0)
            latent_rmsd = coord_metrics.get("latent_rmsd", 0)
            bond_frac = coord_metrics.get("bond_length_valid_frac", 0)
            print(f"\n  [Epoch {self.current_epoch}] coord_rmsd={rmsd:.2f}Å, "
                  f"latent_rmsd={latent_rmsd:.2f}, bond_valid={bond_frac:.1%}")

    def _compute_coord_metrics(
        self,
        latents: torch.Tensor,
        sequences: torch.Tensor,
        mask: torch.Tensor | None,
        num_steps: int = 50,
    ) -> dict[str, float]:
        """Compute coordinate-space validation metrics.

        Args:
            latents: (batch, n_res, latent_dim) ground truth latents.
            sequences: (batch, n_res) residue sequences.
            mask: (batch, n_res) padding mask (True = padding).
            num_steps: DDIM sampling steps.

        Returns:
            Dictionary of coordinate-space metrics.
        """
        from ciffy.biochemistry import Residue

        metrics = {}
        rmsds = []
        latent_rmsds = []
        bond_lengths = []

        batch_size = latents.shape[0]

        self.model.eval()

        # Use EMA weights for sampling if available
        ema_context = self._ema.apply(self.model.denoiser) if self._ema else nullcontext()

        with torch.no_grad(), ema_context:
            for i in range(batch_size):
                # Get unmasked sequence length
                if mask is not None:
                    seq_len = (~mask[i]).sum().item()
                else:
                    seq_len = sequences.shape[1]

                seq = sequences[i, :seq_len]
                gt_latent = latents[i, :seq_len]

                # 1. Sample from diffusion model (uses EMA weights)
                sampled_latent = self._sample_latent(seq, num_steps)

                # 2. Latent-space RMSD
                latent_rmsd = torch.sqrt(
                    ((sampled_latent - gt_latent) ** 2).mean()
                ).item()
                latent_rmsds.append(latent_rmsd)

                # 3. Decode both to coordinates
                seq_np = seq.cpu().numpy()
                gt_coords = self.model.decode(gt_latent, seq_np)
                sampled_coords = self.model.decode(sampled_latent, seq_np)

                # 4. Coordinate RMSD (after optimal alignment)
                rmsd = self._compute_rmsd(gt_coords, sampled_coords)
                rmsds.append(rmsd)

                # 5. Bond length statistics (O3'-P for RNA)
                bonds = self._compute_bond_lengths(sampled_coords, seq_np)
                bond_lengths.extend(bonds)

        # Aggregate metrics
        metrics["coord_rmsd"] = float(np.mean(rmsds))
        metrics["coord_rmsd_std"] = float(np.std(rmsds))
        metrics["latent_rmsd"] = float(np.mean(latent_rmsds))

        if bond_lengths:
            bond_lengths = np.array(bond_lengths)
            # O3'-P ideal distance is ~1.6 Å
            metrics["bond_length_mean"] = float(np.mean(bond_lengths))
            metrics["bond_length_std"] = float(np.std(bond_lengths))
            # Fraction of bonds within reasonable range (1.4-1.8 Å)
            reasonable = np.mean((bond_lengths > 1.4) & (bond_lengths < 1.8))
            metrics["bond_length_valid_frac"] = float(reasonable)

        return metrics

    def _sample_latent(
        self,
        sequence: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Sample latent from diffusion model for a single sequence.

        Args:
            sequence: (n_res,) residue sequence.
            num_steps: DDIM sampling steps.

        Returns:
            (n_res, latent_dim) sampled latent.
        """
        n_res = len(sequence)
        device = self.model.device

        # Start from noise
        x = torch.randn(1, n_res, self.model.latent_dim, device=device)
        seq_batch = sequence.unsqueeze(0).to(device)

        # Get timesteps
        timesteps = self.model.diffusion.get_sampling_timesteps(num_steps)

        # Reverse diffusion
        for i, t in enumerate(timesteps.tolist()):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            pred_noise = self.model.denoiser(x, t_batch, seq_batch)

            t_prev = timesteps[i + 1].item() if i < len(timesteps) - 1 else 0
            x = self.model.diffusion.ddim_step(x, pred_noise, t, t_prev, eta=0.0)

        return x.squeeze(0)

    def _compute_rmsd(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
    ) -> float:
        """Compute RMSD between two coordinate sets after optimal alignment.

        Uses Kabsch algorithm for optimal rotation.
        """
        # Move to numpy for Kabsch
        c1 = coords1.cpu().numpy()
        c2 = coords2.cpu().numpy()

        # Center both
        c1_centered = c1 - c1.mean(axis=0)
        c2_centered = c2 - c2.mean(axis=0)

        # Kabsch rotation
        H = c1_centered.T @ c2_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation and compute RMSD
        c2_aligned = c2_centered @ R
        rmsd = np.sqrt(((c1_centered - c2_aligned) ** 2).sum(axis=1).mean())

        return float(rmsd)

    def _compute_bond_lengths(
        self,
        coords: torch.Tensor,
        sequence: np.ndarray,
    ) -> list[float]:
        """Compute inter-residue bond lengths (O3'-P for nucleotides).

        Args:
            coords: (N, 3) coordinates.
            sequence: (n_res,) residue sequence as numpy array.

        Returns:
            List of O3'-P bond lengths.
        """
        from ciffy.biochemistry import Backbone, Residue

        bond_lengths = []

        # Get atoms per residue
        n_residues = len(sequence)
        atoms_per_residue = coords.shape[0] // n_residues

        # O3' and P indices within a residue
        # For RNA: O3' is typically at index 8, P at index 0
        try:
            o3p_idx = Backbone.O3p.value
            p_idx = Backbone.P.value
        except (AttributeError, ValueError):
            return bond_lengths

        coords_np = coords.cpu().numpy()

        for i in range(n_residues - 1):
            # O3' of residue i
            o3p_global = i * atoms_per_residue + o3p_idx
            # P of residue i+1
            p_global = (i + 1) * atoms_per_residue + p_idx

            if o3p_global < coords_np.shape[0] and p_global < coords_np.shape[0]:
                dist = np.linalg.norm(coords_np[o3p_global] - coords_np[p_global])
                bond_lengths.append(dist)

        return bond_lengths

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer for denoiser only (flow model is frozen)."""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        config = self.training_config

        # Only optimize denoiser parameters (flow model is frozen)
        optimizer = AdamW(
            self.model.denoiser.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        scheduler_config = config.scheduler
        if scheduler_config.scheduler_type == "none":
            return {"optimizer": optimizer}

        # Calculate epochs
        total_epochs = self.trainer.max_epochs
        warmup_epochs = scheduler_config.warmup_epochs
        main_epochs = max(1, total_epochs - warmup_epochs)

        # Create main scheduler
        if scheduler_config.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=scheduler_config.min_lr,
            )
        else:
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=main_epochs, eta_min=scheduler_config.min_lr
            )

        # Add warmup
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


__all__ = [
    "LatentDiffusionDataConfig",
    "LatentDiffusionFullConfig",
    "LatentDiffusionModule",
]
