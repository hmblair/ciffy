"""Early stopping utilities for training.

Provides EarlyStopper class that monitors a metric and signals when
training should stop due to lack of improvement.

Example:
    >>> from ciffy.nn import EarlyStopper
    >>> stopper = EarlyStopper(patience=10, min_delta=1e-4)
    >>> for epoch in range(100):
    ...     val_loss = validate()
    ...     if stopper.should_stop(val_loss):
    ...         print(f"Early stopping at epoch {epoch}")
    ...         break
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopperState:
    """Serializable state for checkpointing."""

    best_value: float
    counter: int
    stopped_epoch: int | None


class EarlyStopper:
    """Early stopping based on validation metrics.

    Monitors a metric and signals when training should stop if no improvement
    is observed for `patience` consecutive epochs.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for metrics where lower is better (loss),
              'max' for metrics where higher is better (accuracy).

    Example:
        >>> stopper = EarlyStopper(patience=10, min_delta=1e-4)
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     if stopper.should_stop(val_loss):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         print(f"Best loss: {stopper.best_value}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # Initialize state
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.stopped_epoch: int | None = None

    def should_stop(self, value: float, epoch: int = 0) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value to check.
            epoch: Current epoch number (for tracking stopped_epoch).

        Returns:
            True if training should stop, False otherwise.
        """
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True

        return False

    def _is_improvement(self, value: float) -> bool:
        """Check if value represents an improvement."""
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        else:
            return value > self.best_value + self.min_delta

    def state_dict(self) -> dict:
        """Get state for checkpointing.

        Returns:
            Dictionary containing stopper state.
        """
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint.

        Args:
            state: Dictionary containing stopper state.
        """
        self.best_value = state["best_value"]
        self.counter = state["counter"]
        self.stopped_epoch = state.get("stopped_epoch")

    def reset(self) -> None:
        """Reset stopper to initial state."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.stopped_epoch = None


__all__ = [
    "EarlyStopper",
    "EarlyStopperState",
]
