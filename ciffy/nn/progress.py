"""
Training progress display utilities.

Provides researcher-friendly progress output with time estimates,
trend indicators, and meaningful metrics for molecular structure models.

Example:
    >>> from ciffy.nn.progress import TrainingProgress
    >>>
    >>> progress = TrainingProgress(n_epochs=100, metrics=["loss", "rmsd"])
    >>> for epoch in range(100):
    ...     metrics = train_epoch(...)
    ...     progress.update(epoch, metrics)
    >>> progress.finish()
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass
class MetricHistory:
    """Tracks history of a single metric for trend analysis."""

    name: str
    values: list[float] = field(default_factory=list)
    window_size: int = 10  # For trend calculation

    def add(self, value: float) -> None:
        """Add a new value."""
        self.values.append(value)

    @property
    def current(self) -> float | None:
        """Get current (latest) value."""
        return self.values[-1] if self.values else None

    @property
    def best(self) -> float | None:
        """Get best (minimum) value."""
        return min(self.values) if self.values else None

    @property
    def trend(self) -> str:
        """Get trend indicator based on recent values.

        Returns:
            '↘' if improving (decreasing), '↗' if worsening,
            '→' if stable, '' if not enough data.
        """
        if len(self.values) < 3:
            return ""

        recent = self.values[-self.window_size:]
        if len(recent) < 3:
            return ""

        # Compare first and second halves
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / mid
        second_half = sum(recent[mid:]) / (len(recent) - mid)

        # Calculate relative change
        if first_half == 0:
            return ""

        change = (second_half - first_half) / abs(first_half)

        if change < -0.01:  # More than 1% improvement
            return "↘"
        elif change > 0.01:  # More than 1% worsening
            return "↗"
        else:
            return "→"

    def trend_description(self) -> str:
        """Get human-readable trend description."""
        trend = self.trend
        if trend == "↘":
            return "improving"
        elif trend == "↗":
            return "worsening"
        elif trend == "→":
            return "stable"
        return ""


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_eta(seconds: float) -> str:
    """Format ETA in human-readable form."""
    if seconds < 0:
        return "unknown"
    return format_duration(seconds)


class TrainingProgress:
    """Tracks and displays training progress.

    Provides:
    - Time per epoch and ETA
    - Metric trends with visual indicators
    - Clear formatting for researchers

    Example:
        >>> progress = TrainingProgress(n_epochs=100)
        >>> for epoch in range(100):
        ...     loss = train_one_epoch()
        ...     progress.update(epoch, {"loss": loss, "rmsd": 0.5})
        >>> progress.finish()
    """

    def __init__(
        self,
        n_epochs: int,
        model_name: str = "Model",
        show_header: bool = True,
        update_every: int = 1,
        file: TextIO | None = None,
    ):
        """Initialize progress tracker.

        Args:
            n_epochs: Total number of epochs.
            model_name: Name to display in header.
            show_header: Whether to print header at start.
            update_every: Print update every N epochs.
            file: Output file (default: stdout).
        """
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.update_every = update_every
        self.file = file or sys.stdout

        self.start_time: float | None = None
        self.epoch_times: list[float] = []
        self.last_epoch_start: float | None = None
        self.metrics: dict[str, MetricHistory] = {}

        if show_header:
            self._print_header()

    def _print_header(self) -> None:
        """Print training header."""
        print(f"\nTraining {self.model_name}", file=self.file)
        print("─" * 60, file=self.file)

    def _print(self, msg: str) -> None:
        """Print message with carriage return for in-place updates."""
        print(f"\r{msg}", end="", file=self.file, flush=True)

    def _println(self, msg: str) -> None:
        """Print message with newline."""
        print(msg, file=self.file)

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        if self.start_time is None:
            self.start_time = time.time()
        self.last_epoch_start = time.time()

    def end_epoch(self) -> None:
        """Mark the end of an epoch and record timing."""
        if self.last_epoch_start is not None:
            elapsed = time.time() - self.last_epoch_start
            self.epoch_times.append(elapsed)
            self.last_epoch_start = None

    def update(
        self,
        epoch: int,
        metrics: dict[str, float],
        force: bool = False,
    ) -> None:
        """Update progress with new metrics.

        Args:
            epoch: Current epoch (0-indexed).
            metrics: Dictionary of metric name -> value.
            force: Force print even if not at update interval.
        """
        # Record metrics
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = MetricHistory(name)
            self.metrics[name].add(value)

        # End epoch timing if not already done
        self.end_epoch()

        # Check if we should print
        epoch_1 = epoch + 1  # 1-indexed for display
        should_print = force or (epoch_1 % self.update_every == 0) or (epoch_1 == self.n_epochs)

        if should_print:
            self._print_progress(epoch_1, metrics)

    def _print_progress(self, epoch: int, metrics: dict[str, float]) -> None:
        """Print progress line."""
        # Calculate timing
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.n_epochs - epoch
            eta = avg_epoch_time * remaining_epochs
            elapsed = time.time() - (self.start_time or time.time())
            time_str = f"{format_duration(avg_epoch_time)}/epoch │ ETA {format_eta(eta)}"
        else:
            time_str = ""
            elapsed = 0

        # Build progress bar
        progress_pct = epoch / self.n_epochs
        bar_width = 20
        filled = int(bar_width * progress_pct)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build metrics string
        metric_parts = []
        for name, value in metrics.items():
            history = self.metrics.get(name)
            trend = history.trend if history else ""

            # Format value based on magnitude
            if abs(value) < 0.01:
                val_str = f"{value:.2e}"
            elif abs(value) < 10:
                val_str = f"{value:.4f}"
            else:
                val_str = f"{value:.2f}"

            # Add unit for known metrics
            if "rmsd" in name.lower():
                val_str += "Å"

            metric_parts.append(f"{name}: {val_str}{trend}")

        metrics_str = " │ ".join(metric_parts)

        # Print line
        line = f"Epoch {epoch:3d}/{self.n_epochs} │ {bar} │ {time_str}"
        if metrics_str:
            line += f"\n  {metrics_str}"

        self._println(line)

    def finish(self, final_metrics: dict[str, float] | None = None) -> None:
        """Print final summary.

        Args:
            final_metrics: Optional final metrics to display.
        """
        if self.start_time:
            total_time = time.time() - self.start_time
        else:
            total_time = 0

        print("─" * 60, file=self.file)
        print(f"✓ Training complete in {format_duration(total_time)}", file=self.file)

        # Print final metrics with best values
        if final_metrics:
            print("\nFinal metrics:", file=self.file)
            for name, value in final_metrics.items():
                history = self.metrics.get(name)
                best = history.best if history else None

                # Format value
                if "rmsd" in name.lower():
                    val_str = f"{value:.4f}Å"
                    if best and best < value:
                        val_str += f" (best: {best:.4f}Å)"
                elif abs(value) < 0.01:
                    val_str = f"{value:.2e}"
                else:
                    val_str = f"{value:.4f}"

                print(f"  {name}: {val_str}", file=self.file)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with timing and metric statistics.
        """
        total_time = time.time() - (self.start_time or time.time())
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0

        summary = {
            "total_time": total_time,
            "avg_epoch_time": avg_epoch_time,
            "n_epochs": len(self.epoch_times),
        }

        for name, history in self.metrics.items():
            summary[f"{name}_final"] = history.current
            summary[f"{name}_best"] = history.best
            summary[f"{name}_trend"] = history.trend_description()

        return summary


class ResidueTrainingProgress:
    """Progress display for multi-residue flow training.

    Shows progress across multiple residue types with per-residue metrics.
    """

    def __init__(
        self,
        residues: list[str],
        n_epochs: int,
        file: TextIO | None = None,
    ):
        """Initialize multi-residue progress.

        Args:
            residues: List of residue names.
            n_epochs: Epochs per residue.
            file: Output file.
        """
        self.residues = residues
        self.n_epochs = n_epochs
        self.file = file or sys.stdout

        self.current_residue: str | None = None
        self.current_progress: TrainingProgress | None = None
        self.results: dict[str, dict[str, Any]] = {}

        self._print_header()

    def _print_header(self) -> None:
        """Print overall header."""
        print(f"\n{'═' * 60}", file=self.file)
        print(f"Training flow models for {len(self.residues)} residue type(s)", file=self.file)
        print(f"Residues: {', '.join(self.residues)}", file=self.file)
        print(f"{'═' * 60}", file=self.file)

    def start_residue(self, residue: str, n_train: int, n_test: int, n_atoms: int) -> TrainingProgress:
        """Start training a new residue.

        Args:
            residue: Residue name.
            n_train: Number of training instances.
            n_test: Number of test instances.
            n_atoms: Atoms per residue.

        Returns:
            TrainingProgress for this residue.
        """
        self.current_residue = residue

        print(f"\n[{residue}] {n_train:,} train, {n_test:,} test instances ({n_atoms} atoms)",
              file=self.file)

        self.current_progress = TrainingProgress(
            n_epochs=self.n_epochs,
            model_name=f"{residue} flow",
            show_header=False,
            update_every=max(1, self.n_epochs // 4),  # ~4 updates
            file=self.file,
        )

        return self.current_progress

    def finish_residue(
        self,
        residue: str,
        metrics: dict[str, float],
    ) -> None:
        """Finish training a residue.

        Args:
            residue: Residue name.
            metrics: Final metrics.
        """
        if self.current_progress:
            self.current_progress.finish(metrics)

        self.results[residue] = metrics
        self.current_residue = None
        self.current_progress = None

    def finish(self) -> None:
        """Print final summary table."""
        print(f"\n{'═' * 60}", file=self.file)
        print("TRAINING SUMMARY", file=self.file)
        print(f"{'═' * 60}", file=self.file)

        # Header
        print(f"{'Residue':<8} {'Train':<8} {'Test':<8} {'PCA RMSD':<12} {'Test RMSD':<12} {'Var%':<8}",
              file=self.file)
        print("─" * 60, file=self.file)

        # Results
        for residue, metrics in self.results.items():
            n_train = metrics.get("n_train", 0)
            n_test = metrics.get("n_test", 0)
            pca_rmsd = metrics.get("pca_rmsd", 0)
            test_rmsd = metrics.get("test_rmsd", 0)
            var_exp = metrics.get("var_explained", 0) * 100

            print(
                f"{residue:<8} {n_train:<8} {n_test:<8} "
                f"{pca_rmsd:.4f}Å{'':<5} {test_rmsd:.4f}Å{'':<5} {var_exp:.1f}%",
                file=self.file,
            )


__all__ = [
    "MetricHistory",
    "TrainingProgress",
    "ResidueTrainingProgress",
    "format_duration",
    "format_eta",
]
