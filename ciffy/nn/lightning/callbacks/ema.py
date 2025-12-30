"""EMA callback for maintaining exponential moving average of model weights.

Wraps the existing EMA class from ciffy.nn.diffusion.ema to integrate with
PyTorch Lightning's callback system.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer

from ciffy.nn.diffusion.ema import EMA


class EMACallback(Callback):
    """Callback that maintains EMA of model weights during training.

    This callback:
    - Initializes EMA from the model (or a submodule like 'denoiser')
    - Updates EMA after each training batch
    - Swaps to EMA weights for validation
    - Saves/loads EMA state with checkpoints

    Args:
        decay: EMA decay rate. Higher = slower updates. Default: 0.9999.
        warmup_steps: Number of steps to linearly ramp up decay. Default: 2000.
        ema_submodule: Name of submodule to track (e.g., 'denoiser').
            If None, tracks the entire model. Default: None.

    Example:
        >>> trainer = Trainer(
        ...     callbacks=[EMACallback(decay=0.9999, ema_submodule="denoiser")]
        ... )
        >>> trainer.fit(module, datamodule)
    """

    def __init__(
        self,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        ema_submodule: str | None = None,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.ema_submodule = ema_submodule
        self.ema: EMA | None = None
        self._original_weights_backed_up = False

    def _get_ema_model(self, pl_module: LightningModule) -> torch.nn.Module:
        """Get the model or submodule to track with EMA."""
        if self.ema_submodule is not None:
            # Track a specific submodule (e.g., denoiser)
            model = pl_module.model
            for attr in self.ema_submodule.split("."):
                model = getattr(model, attr)
            return model
        elif hasattr(pl_module, "model"):
            return pl_module.model
        else:
            return pl_module

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize EMA from model weights at training start."""
        model = self._get_ema_model(pl_module)
        self.ema = EMA(
            model,
            decay=self.decay,
            warmup_steps=self.warmup_steps,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA after each training step."""
        if self.ema is not None:
            model = self._get_ema_model(pl_module)
            self.ema.update(model)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Swap to EMA weights for validation."""
        if self.ema is not None and not self._original_weights_backed_up:
            model = self._get_ema_model(pl_module)
            # Store original weights and apply EMA
            self._backup_params = {}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self.ema.shadow_params:
                        self._backup_params[name] = param.data.clone()
                        param.data.copy_(self.ema.shadow_params[name])
            self._original_weights_backed_up = True

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Restore original weights after validation."""
        if self._original_weights_backed_up:
            model = self._get_ema_model(pl_module)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self._backup_params:
                        param.data.copy_(self._backup_params[name])
            self._backup_params = {}
            self._original_weights_backed_up = False

    def state_dict(self) -> dict[str, Any]:
        """Return callback state for checkpointing."""
        if self.ema is None:
            return {}
        return {
            "ema_state": self.ema.state_dict(),
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "ema_submodule": self.ema_submodule,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        if "ema_state" in state_dict and self.ema is not None:
            self.ema.load_state_dict(state_dict["ema_state"])

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Save EMA state to checkpoint."""
        checkpoint["ema_callback"] = self.state_dict()

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Load EMA state from checkpoint."""
        if "ema_callback" in checkpoint:
            # EMA will be initialized in on_fit_start, then we load state
            self._pending_state = checkpoint["ema_callback"]

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Load pending EMA state after EMA is initialized."""
        if hasattr(self, "_pending_state") and self._pending_state:
            self.load_state_dict(self._pending_state)
            delattr(self, "_pending_state")

    def get_ema_model(self, pl_module: LightningModule) -> torch.nn.Module | None:
        """Get a model with EMA weights applied.

        Useful for inference or final model saving.

        Returns:
            Model with EMA weights, or None if EMA not initialized.
        """
        if self.ema is None:
            return None

        import copy

        model = self._get_ema_model(pl_module)
        ema_model = copy.deepcopy(model)
        self.ema.copy_to(ema_model)
        return ema_model


__all__ = ["EMACallback"]
