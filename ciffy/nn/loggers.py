"""
Metrics loggers for training.

Provides logging backends for tracking training metrics. All loggers
implement the MetricsLogger protocol from base_trainer.

Example:
    >>> from ciffy.nn.loggers import WandbLogger
    >>> from ciffy.nn.vae.trainer import VAETrainer, VAEConfig
    >>>
    >>> config = VAEConfig.from_yaml("config.yaml")
    >>> logger = WandbLogger(project="my-project", config=config.to_dict())
    >>> trainer = VAETrainer(config, model, dataset, logger=logger)
    >>> trainer.train()
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)

# Check for wandb availability
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbLogger:
    """Weights & Biases metrics logger.

    Logs training metrics to wandb for visualization and experiment tracking.

    Example:
        >>> logger = WandbLogger(
        ...     project="polymer-vae",
        ...     config={"lr": 1e-4, "epochs": 100},
        ...     name="experiment-1",
        ... )
        >>> for epoch in range(100):
        ...     metrics = train_epoch(...)
        ...     logger.log(metrics, step=epoch)
        >>> logger.finish()

    Args:
        project: Wandb project name.
        config: Configuration dictionary to log.
        name: Run name. None for auto-generated.
        group: Group name for organizing related runs.
        tags: List of tags for the run.
        notes: Notes/description for the run.
        **kwargs: Additional arguments passed to wandb.init().
    """

    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        **kwargs: Any,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required for WandbLogger. "
                "Install with: pip install wandb"
            )

        self.run = wandb.init(
            project=project,
            config=config,
            name=name,
            group=group,
            tags=tags,
            notes=notes,
            **kwargs,
        )
        self._finished = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics for a given step.

        Args:
            metrics: Dictionary of metric names to values.
            step: Training step or epoch number.
        """
        if self._finished:
            return

        # Filter out non-numeric values and internal keys
        log_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                log_metrics[key] = value

        wandb.log(log_metrics, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Log summary metrics (shown in runs table).

        Args:
            metrics: Dictionary of summary metric names to values.
        """
        if self._finished:
            return

        for key, value in metrics.items():
            wandb.run.summary[key] = value

    def finish(self) -> None:
        """Finalize the wandb run."""
        if not self._finished:
            wandb.finish()
            self._finished = True

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()


class NoOpLogger:
    """No-operation logger that discards all metrics.

    Use this when logging is disabled or for testing.

    Example:
        >>> logger = NoOpLogger()
        >>> logger.log({"loss": 0.5}, step=0)  # Does nothing
        >>> logger.finish()  # Does nothing
    """

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Discard metrics (no-op)."""
        pass

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Discard summary metrics (no-op)."""
        pass

    def finish(self) -> None:
        """No-op finish."""
        pass

    def __enter__(self) -> "NoOpLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def create_logger(
    project: str | None = None,
    config: dict[str, Any] | None = None,
    enabled: bool = True,
    **kwargs: Any,
) -> WandbLogger | NoOpLogger:
    """Create an appropriate logger based on configuration.

    Args:
        project: Wandb project name. None to disable wandb.
        config: Configuration dictionary to log.
        enabled: Whether logging is enabled.
        **kwargs: Additional arguments passed to WandbLogger.

    Returns:
        WandbLogger if project is specified and enabled, else NoOpLogger.

    Example:
        >>> logger = create_logger(
        ...     project="my-project",
        ...     config={"lr": 1e-4},
        ...     enabled=True,
        ... )
    """
    if not enabled or project is None:
        return NoOpLogger()

    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, falling back to NoOpLogger")
        return NoOpLogger()

    return WandbLogger(project=project, config=config, **kwargs)


__all__ = [
    "WandbLogger",
    "NoOpLogger",
    "create_logger",
    "WANDB_AVAILABLE",
]
