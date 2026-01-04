"""Exponential Moving Average (EMA) for model weights.

EMA maintains a smoothed version of model weights that typically produces
better results at inference time, especially for generative models.

Example (Training):
    >>> model = MyModel()
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ema = EMA(model, decay=0.9999)
    >>>
    >>> for epoch in range(100):
    ...     for batch in dataloader:
    ...         loss = model(batch).mean()
    ...         loss.backward()
    ...         optimizer.step()
    ...         ema.update(model)  # Update EMA after each step
    ...
    ...     # Save checkpoint
    ...     ema.save(f'checkpoint_{epoch}.pt', model, optimizer, epoch=epoch)

Example (Inference):
    >>> with ema.apply(model):
    ...     output = model(test_input)  # Uses EMA weights

Example (Resume Training):
    >>> model = MyModel()
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ema, extras = EMA.load('checkpoint_50.pt', model, optimizer)
    >>> start_epoch = extras['epoch']

References:
    - Polyak, B.T. & Juditsky, A.B. (1992). "Acceleration of Stochastic
      Approximation by Averaging"
    - Tarvainen, A. & Valpola, H. (2017). "Mean teachers are better role
      models" (EMA in semi-supervised learning)
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, Optional

import torch
from torch import nn


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated as an
    exponential moving average of the training parameters. The EMA weights
    typically produce better results for generative models.

    The update rule is:
        θ_ema = decay * θ_ema + (1 - decay) * θ_current

    With decay=0.9999, this is roughly a moving average over the last
    10,000 updates.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate. Higher = slower updates. Default: 0.9999.
        warmup_steps: Number of steps to linearly ramp up decay from
            warmup_decay to decay. Helps early training. Default: 0.
        warmup_decay: Starting decay value during warmup. Default: 0.9.
        include_buffers: Whether to also track model buffers (e.g.,
            BatchNorm running stats). Default: True.

    Key Methods:
        update(model): Update EMA from current model weights (call after optimizer.step())
        apply(model): Context manager to temporarily use EMA weights for inference
        save(path, model, ...): Save checkpoint with model and EMA state
        load(path, model, ...): Class method to load checkpoint and create EMA

    Example:
        >>> model = nn.Linear(10, 10)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> ema = EMA(model, decay=0.999)
        >>>
        >>> # Training
        >>> optimizer.step()
        >>> ema.update(model)
        >>>
        >>> # Inference
        >>> with ema.apply(model):
        ...     output = model(x)
        >>>
        >>> # Checkpointing
        >>> ema.save('checkpoint.pt', model, optimizer, epoch=10)
        >>> ema, extras = EMA.load('checkpoint.pt', model, optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        warmup_decay: float = 0.9,
        include_buffers: bool = True,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if not 0.0 <= warmup_decay <= 1.0:
            raise ValueError(f"warmup_decay must be in [0, 1], got {warmup_decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.warmup_decay = warmup_decay
        self.include_buffers = include_buffers
        self.step_count = 0

        # Store shadow copies of parameters
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}

        self._initialize(model)

    def _initialize(self, model: nn.Module) -> None:
        """Initialize shadow parameters from model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

        if self.include_buffers:
            for name, buffer in model.named_buffers():
                # Only track floating point buffers (skip integer buffers like num_batches_tracked)
                if buffer is not None and buffer.is_floating_point():
                    self.shadow_buffers[name] = buffer.data.clone()

    def _get_decay(self) -> float:
        """Get current decay value, accounting for warmup."""
        if self.warmup_steps == 0 or self.step_count >= self.warmup_steps:
            return self.decay

        # Linear interpolation during warmup
        progress = self.step_count / self.warmup_steps
        return self.warmup_decay + progress * (self.decay - self.warmup_decay)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters from current model parameters.

        Call this after each optimizer.step().

        Args:
            model: The model with updated parameters.
        """
        decay = self._get_decay()
        self.step_count += 1

        with torch.no_grad():
            # Update parameters
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # θ_ema = decay * θ_ema + (1 - decay) * θ
                    self.shadow_params[name].lerp_(param.data, 1 - decay)

            # Update buffers (only floating point ones)
            if self.include_buffers:
                for name, buffer in model.named_buffers():
                    if buffer is not None and buffer.is_floating_point() and name in self.shadow_buffers:
                        self.shadow_buffers[name].lerp_(buffer.data, 1 - decay)

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into a model.

        This permanently modifies the model's parameters. Use `apply()` for
        temporary application.

        Args:
            model: The model to copy parameters into.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])

            if self.include_buffers:
                for name, buffer in model.named_buffers():
                    if buffer is not None and buffer.is_floating_point() and name in self.shadow_buffers:
                        buffer.data.copy_(self.shadow_buffers[name])

    @contextmanager
    def apply(self, model: nn.Module):
        """Temporarily apply EMA parameters to model.

        Use as a context manager to temporarily swap in EMA weights for
        inference, then restore original weights afterward.

        Args:
            model: The model to temporarily modify.

        Yields:
            None

        Example:
            >>> with ema.apply(model):
            ...     # model now has EMA weights
            ...     output = model(input)
            >>> # model has original weights again
        """
        # Backup current parameters
        backup_params: Dict[str, torch.Tensor] = {}
        backup_buffers: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    backup_params[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[name])

            if self.include_buffers:
                for name, buffer in model.named_buffers():
                    if buffer is not None and buffer.is_floating_point() and name in self.shadow_buffers:
                        backup_buffers[name] = buffer.data.clone()
                        buffer.data.copy_(self.shadow_buffers[name])

        try:
            yield
        finally:
            # Restore original parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup_params:
                        param.data.copy_(backup_params[name])

                if self.include_buffers:
                    for name, buffer in model.named_buffers():
                        if buffer is not None and buffer.is_floating_point() and name in backup_buffers:
                            buffer.data.copy_(backup_buffers[name])

    def state_dict(self) -> Dict:
        """Return EMA state for checkpointing.

        Returns:
            Dictionary containing all EMA state.

        Example:
            >>> torch.save({
            ...     'model': model.state_dict(),
            ...     'ema': ema.state_dict(),
            ... }, 'checkpoint.pt')
        """
        return {
            'shadow_params': self.shadow_params,
            'shadow_buffers': self.shadow_buffers,
            'decay': self.decay,
            'warmup_steps': self.warmup_steps,
            'warmup_decay': self.warmup_decay,
            'step_count': self.step_count,
            'include_buffers': self.include_buffers,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load EMA state from checkpoint.

        Args:
            state_dict: State dictionary from `state_dict()`.

        Example:
            >>> checkpoint = torch.load('checkpoint.pt')
            >>> ema.load_state_dict(checkpoint['ema'])
        """
        self.shadow_params = state_dict['shadow_params']
        self.shadow_buffers = state_dict.get('shadow_buffers', {})
        self.decay = state_dict['decay']
        self.warmup_steps = state_dict.get('warmup_steps', 0)
        self.warmup_decay = state_dict.get('warmup_decay', 0.9)
        self.step_count = state_dict['step_count']
        self.include_buffers = state_dict.get('include_buffers', True)

    def save(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **extras,
    ) -> None:
        """Save model, EMA, and optionally optimizer to a checkpoint file.

        Convenience method that handles the common checkpoint pattern.

        Args:
            path: File path to save the checkpoint.
            model: The model to save.
            optimizer: Optional optimizer to include in checkpoint.
            **extras: Additional items to include in the checkpoint.

        Example:
            >>> ema.save('checkpoint.pt', model)
            >>> ema.save('checkpoint.pt', model, optimizer)
            >>> ema.save('checkpoint.pt', model, optimizer, epoch=10, loss=0.5)
        """
        checkpoint = {
            'model': model.state_dict(),
            'ema': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint.update(extras)
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[torch.device] = None,
    ) -> tuple['EMA', Dict]:
        """Load model, EMA, and optionally optimizer from a checkpoint file.

        Class method that creates an EMA instance and loads the checkpoint.

        Args:
            path: File path to load the checkpoint from.
            model: The model to load weights into (modified in-place).
            optimizer: Optional optimizer to load state into (modified in-place).
            map_location: Device to map tensors to (passed to torch.load).

        Returns:
            Tuple of (ema, extras) where extras is a dict of any additional
            items in the checkpoint (e.g., epoch, loss).

        Example:
            >>> ema, extras = EMA.load('checkpoint.pt', model)
            >>> print(extras.get('epoch'))
            >>>
            >>> ema, extras = EMA.load('checkpoint.pt', model, optimizer)
            >>> epoch = extras['epoch']
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        # Load model weights
        model.load_state_dict(checkpoint['model'])

        # Create EMA and load its state
        ema = cls(model)
        ema.load_state_dict(checkpoint['ema'])

        # Load optimizer if provided
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # Collect extras (everything except model, ema, optimizer)
        extras = {
            k: v for k, v in checkpoint.items()
            if k not in ('model', 'ema', 'optimizer')
        }

        return ema, extras

    def to(self, device: torch.device) -> 'EMA':
        """Move EMA parameters to a device.

        Args:
            device: Target device.

        Returns:
            self for chaining.
        """
        self.shadow_params = {
            name: param.to(device) for name, param in self.shadow_params.items()
        }
        self.shadow_buffers = {
            name: buffer.to(device) for name, buffer in self.shadow_buffers.items()
        }
        return self

    @property
    def current_decay(self) -> float:
        """Get the current decay value (accounts for warmup)."""
        return self._get_decay()

    def __repr__(self) -> str:
        return (
            f"EMA(decay={self.decay}, warmup_steps={self.warmup_steps}, "
            f"step_count={self.step_count}, n_params={len(self.shadow_params)})"
        )


def create_ema_model(model: nn.Module) -> nn.Module:
    """Create a separate copy of a model for EMA.

    This is an alternative to the EMA class that maintains a full model copy.
    More memory intensive but allows using the EMA model directly.

    Args:
        model: The model to copy.

    Returns:
        A deep copy of the model with requires_grad=False.

    Example:
        >>> model = MyModel()
        >>> ema_model = create_ema_model(model)
        >>> # Update EMA manually:
        >>> update_ema_model(model, ema_model, decay=0.9999)
    """
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


def update_ema_model(
    model: nn.Module,
    ema_model: nn.Module,
    decay: float = 0.9999,
) -> None:
    """Update an EMA model from the training model.

    Alternative to the EMA class for when you want to maintain a full model copy.

    Args:
        model: The training model with current parameters.
        ema_model: The EMA model to update.
        decay: EMA decay rate.

    Example:
        >>> for batch in dataloader:
        ...     loss = model(batch).mean()
        ...     loss.backward()
        ...     optimizer.step()
        ...     update_ema_model(model, ema_model, decay=0.9999)
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.lerp_(param, 1 - decay)

        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.lerp_(buffer, 1 - decay)


__all__ = [
    "EMA",
    "create_ema_model",
    "update_ema_model",
]
