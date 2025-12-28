"""
Training diagnostics for debugging optimization issues.

Provides utilities to track gradients, parameters, activations, and learning rates
during training to diagnose issues like vanishing/exploding gradients, dead neurons,
or learning rate problems.

Quick Start - Standalone Usage:
    >>> from ciffy.nn import TrainingDiagnostics, DiagnosticsConfig
    >>>
    >>> config = DiagnosticsConfig(track_gradients=True, track_parameters=True)
    >>> diagnostics = TrainingDiagnostics(model, config)
    >>>
    >>> for epoch in range(100):
    ...     optimizer.zero_grad()
    ...     loss = model(batch).mean()
    ...     loss.backward()
    ...     # Compute after backward, before optimizer.step()
    ...     metrics = diagnostics.compute(optimizer)
    ...     optimizer.step()
    ...     logger.log(metrics, step=epoch)
    >>> diagnostics.cleanup()

Quick Start - Integrated with Trainer:
    Add diagnostics config to your training config:

    >>> from ciffy.nn import DiagnosticsConfig
    >>>
    >>> config.diagnostics = DiagnosticsConfig(
    ...     track_gradients=True,
    ...     track_parameters=True,
    ...     track_learning_rate=True,
    ... )
    >>> trainer = MyTrainer(config)
    >>> trainer.train()  # Diagnostics automatically logged

    Or in YAML config:

    .. code-block:: yaml

        diagnostics:
          track_gradients: true
          track_parameters: true
          track_learning_rate: true
          gradient_per_layer: true
          log_frequency: 1

Metrics Produced:
    Gradient metrics (track_gradients=True):
        - diag/grad/norm: Global L2 norm of all gradients
        - diag/grad/max: Maximum absolute gradient value
        - diag/grad/mean: Mean absolute gradient value
        - diag/grad/pct_zero: Percentage of zero gradients
        - diag/grad/layer/{name}: Per-layer gradient norms

    Parameter metrics (track_parameters=True):
        - diag/param/norm: Global L2 norm of all parameters
        - diag/param/mean: Mean parameter value
        - diag/param/std: Standard deviation of parameters
        - diag/param/layer/{name}: Per-layer parameter norms
        - diag/param/delta/{name}: Percent change from initialization

    Activation metrics (track_activations=True):
        - diag/act/layer/{name}/mean: Mean activation value
        - diag/act/layer/{name}/dead_pct: Percentage of dead (zero) activations
        - diag/act/layer/{name}/saturated_pct: Percentage near saturation

    Learning rate (track_learning_rate=True):
        - diag/lr: Current learning rate

Quick Diagnosis:
    For a one-time gradient health check:

    >>> from ciffy.nn import diagnose_gradients
    >>> loss.backward()
    >>> report = diagnose_gradients(model)
    >>> if not report['healthy']:
    ...     print(f"Issues: {report['issues']}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

__all__ = [
    "GradientTracker",
    "ParameterTracker",
    "ActivationTracker",
    "LearningRateTracker",
    "TrainingDiagnostics",
    "DiagnosticsConfig",
]


@dataclass
class DiagnosticsConfig:
    """Configuration for training diagnostics.

    Args:
        track_gradients: Whether to track gradient statistics.
        track_parameters: Whether to track parameter statistics.
        track_activations: Whether to track activation statistics.
        track_learning_rate: Whether to track learning rate.
        gradient_per_layer: Track per-layer gradient norms (vs global only).
        parameter_per_layer: Track per-layer parameter stats (vs global only).
        activation_layers: List of layer names/types to track activations for.
            If None, tracks all layers with activations.
        prefix: Prefix for metric names (e.g., "train/" or "diag/").
        log_frequency: Log every N steps (1 = every step).
    """

    track_gradients: bool = True
    track_parameters: bool = True
    track_activations: bool = False  # Off by default (requires hooks)
    track_learning_rate: bool = True
    gradient_per_layer: bool = True
    parameter_per_layer: bool = True
    activation_layers: list[str] | None = None
    prefix: str = "diag/"
    log_frequency: int = 1


class GradientTracker:
    """Track gradient statistics during training.

    Computes gradient norms globally and per-layer to detect:
    - Vanishing gradients (very small norms)
    - Exploding gradients (very large norms)
    - Dead layers (zero gradients)

    Example:
        >>> tracker = GradientTracker(model, per_layer=True)
        >>> loss.backward()
        >>> metrics = tracker.compute()
        >>> # {'grad/norm': 0.5, 'grad/max': 1.2, 'grad/layer/fc1': 0.3, ...}
    """

    def __init__(
        self,
        model: "nn.Module",
        per_layer: bool = True,
        prefix: str = "grad/",
    ):
        """Initialize gradient tracker.

        Args:
            model: PyTorch model to track.
            per_layer: Whether to compute per-layer statistics.
            prefix: Prefix for metric names.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GradientTracker")

        self.model = model
        self.per_layer = per_layer
        self.prefix = prefix

    def compute(self) -> dict[str, float]:
        """Compute gradient statistics after backward pass.

        Call this after loss.backward() but before optimizer.step().

        Returns:
            Dictionary with gradient metrics:
            - {prefix}norm: Global L2 norm of all gradients
            - {prefix}max: Maximum absolute gradient value
            - {prefix}min: Minimum absolute gradient value (non-zero)
            - {prefix}mean: Mean absolute gradient value
            - {prefix}num_zero: Number of parameters with zero gradients
            - {prefix}layer/{name}: Per-layer L2 norms (if per_layer=True)
        """
        metrics = {}
        all_grads = []
        layer_norms = {}
        num_zero = 0
        num_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()
            grad_flat = grad.flatten()
            all_grads.append(grad_flat)

            # Count zero gradients
            num_zero += (grad_flat == 0).sum().item()
            num_params += grad_flat.numel()

            # Per-layer norm
            if self.per_layer:
                layer_norm = grad.norm(2).item()
                # Simplify layer name: remove common prefixes
                simple_name = self._simplify_name(name)
                layer_norms[simple_name] = layer_norm

        if not all_grads:
            # No gradients computed yet
            return {f"{self.prefix}norm": 0.0}

        # Concatenate all gradients
        all_grads = torch.cat(all_grads)

        # Global statistics
        metrics[f"{self.prefix}norm"] = all_grads.norm(2).item()
        metrics[f"{self.prefix}max"] = all_grads.abs().max().item()

        nonzero_grads = all_grads[all_grads != 0]
        if len(nonzero_grads) > 0:
            metrics[f"{self.prefix}min"] = nonzero_grads.abs().min().item()
            metrics[f"{self.prefix}mean"] = nonzero_grads.abs().mean().item()
        else:
            metrics[f"{self.prefix}min"] = 0.0
            metrics[f"{self.prefix}mean"] = 0.0

        metrics[f"{self.prefix}num_zero"] = num_zero
        metrics[f"{self.prefix}pct_zero"] = 100.0 * num_zero / max(num_params, 1)

        # Add per-layer norms
        if self.per_layer:
            for name, norm in layer_norms.items():
                metrics[f"{self.prefix}layer/{name}"] = norm

        return metrics

    def _simplify_name(self, name: str) -> str:
        """Simplify parameter name for cleaner logging."""
        # Remove .weight, .bias suffixes
        for suffix in [".weight", ".bias"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name


class ParameterTracker:
    """Track parameter statistics during training.

    Monitors parameter values to detect:
    - Weight explosion (very large values)
    - Weight vanishing (very small values)
    - Dead neurons (constant weights)

    Example:
        >>> tracker = ParameterTracker(model, per_layer=True)
        >>> metrics = tracker.compute()
        >>> # {'param/norm': 150.0, 'param/mean': 0.01, 'param/layer/fc1': 5.2, ...}
    """

    def __init__(
        self,
        model: "nn.Module",
        per_layer: bool = True,
        prefix: str = "param/",
    ):
        """Initialize parameter tracker.

        Args:
            model: PyTorch model to track.
            per_layer: Whether to compute per-layer statistics.
            prefix: Prefix for metric names.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ParameterTracker")

        self.model = model
        self.per_layer = per_layer
        self.prefix = prefix

        # Store initial norms for comparison
        self._initial_norms: dict[str, float] = {}
        self._capture_initial()

    def _capture_initial(self) -> None:
        """Capture initial parameter norms for delta tracking."""
        for name, param in self.model.named_parameters():
            simple_name = self._simplify_name(name)
            self._initial_norms[simple_name] = param.detach().norm(2).item()

    def compute(self) -> dict[str, float]:
        """Compute parameter statistics.

        Returns:
            Dictionary with parameter metrics:
            - {prefix}norm: Global L2 norm of all parameters
            - {prefix}max: Maximum absolute parameter value
            - {prefix}min: Minimum absolute parameter value
            - {prefix}mean: Mean parameter value
            - {prefix}std: Standard deviation of parameters
            - {prefix}layer/{name}: Per-layer L2 norms (if per_layer=True)
            - {prefix}delta/{name}: Change from initial norm (if per_layer=True)
        """
        metrics = {}
        all_params = []
        layer_stats = {}

        for name, param in self.model.named_parameters():
            param_data = param.detach()
            param_flat = param_data.flatten()
            all_params.append(param_flat)

            if self.per_layer:
                simple_name = self._simplify_name(name)
                current_norm = param_data.norm(2).item()
                layer_stats[simple_name] = {
                    "norm": current_norm,
                    "mean": param_flat.mean().item(),
                    "std": param_flat.std().item() if param_flat.numel() > 1 else 0.0,
                    "max": param_flat.abs().max().item(),
                }
                # Delta from initial
                if simple_name in self._initial_norms:
                    initial = self._initial_norms[simple_name]
                    if initial > 0:
                        layer_stats[simple_name]["delta_pct"] = (
                            100.0 * (current_norm - initial) / initial
                        )

        if not all_params:
            return {}

        # Concatenate all parameters
        all_params = torch.cat(all_params)

        # Global statistics
        metrics[f"{self.prefix}norm"] = all_params.norm(2).item()
        metrics[f"{self.prefix}max"] = all_params.abs().max().item()
        metrics[f"{self.prefix}min"] = all_params.abs().min().item()
        metrics[f"{self.prefix}mean"] = all_params.mean().item()
        metrics[f"{self.prefix}std"] = all_params.std().item()

        # Add per-layer stats
        if self.per_layer:
            for name, stats in layer_stats.items():
                metrics[f"{self.prefix}layer/{name}"] = stats["norm"]
                if "delta_pct" in stats:
                    metrics[f"{self.prefix}delta/{name}"] = stats["delta_pct"]

        return metrics

    def _simplify_name(self, name: str) -> str:
        """Simplify parameter name for cleaner logging."""
        for suffix in [".weight", ".bias"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name


class ActivationTracker:
    """Track activation statistics using forward hooks.

    Monitors activation values to detect:
    - Dead ReLU neurons (always zero)
    - Saturated sigmoid/tanh (always near bounds)
    - Exploding activations

    Example:
        >>> tracker = ActivationTracker(model, layer_types=[nn.ReLU, nn.GELU])
        >>> output = model(input)  # Hooks capture activations
        >>> metrics = tracker.compute()
        >>> # {'act/layer/relu1/mean': 0.5, 'act/layer/relu1/dead_pct': 5.2, ...}
        >>> tracker.remove_hooks()  # Clean up when done
    """

    def __init__(
        self,
        model: "nn.Module",
        layer_types: list[type] | None = None,
        layer_names: list[str] | None = None,
        prefix: str = "act/",
    ):
        """Initialize activation tracker.

        Args:
            model: PyTorch model to track.
            layer_types: List of layer types to track (e.g., [nn.ReLU, nn.GELU]).
                If None and layer_names is None, tracks common activation layers.
            layer_names: List of specific layer names to track.
            prefix: Prefix for metric names.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ActivationTracker")

        self.model = model
        self.prefix = prefix
        self._hooks: list = []
        self._activations: dict[str, "torch.Tensor"] = {}

        # Default to common activation layers
        if layer_types is None and layer_names is None:
            layer_types = [nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh]

        self._register_hooks(layer_types, layer_names)

    def _register_hooks(
        self,
        layer_types: list[type] | None,
        layer_names: list[str] | None,
    ) -> None:
        """Register forward hooks on target layers."""

        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._activations[name] = output.detach()

            return hook

        for name, module in self.model.named_modules():
            should_track = False

            if layer_names is not None and name in layer_names:
                should_track = True
            elif layer_types is not None:
                for layer_type in layer_types:
                    if isinstance(module, layer_type):
                        should_track = True
                        break

            if should_track:
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)

    def compute(self) -> dict[str, float]:
        """Compute activation statistics from last forward pass.

        Returns:
            Dictionary with activation metrics per tracked layer:
            - {prefix}layer/{name}/mean: Mean activation value
            - {prefix}layer/{name}/std: Standard deviation
            - {prefix}layer/{name}/max: Maximum absolute value
            - {prefix}layer/{name}/dead_pct: Percentage of dead (zero) activations
            - {prefix}layer/{name}/saturated_pct: Percentage near saturation bounds
        """
        metrics = {}

        for name, act in self._activations.items():
            act_flat = act.flatten().float()

            layer_prefix = f"{self.prefix}layer/{name}"

            metrics[f"{layer_prefix}/mean"] = act_flat.mean().item()
            metrics[f"{layer_prefix}/std"] = act_flat.std().item()
            metrics[f"{layer_prefix}/max"] = act_flat.abs().max().item()

            # Dead neurons (exactly zero)
            dead_count = (act_flat == 0).sum().item()
            metrics[f"{layer_prefix}/dead_pct"] = 100.0 * dead_count / max(act_flat.numel(), 1)

            # Saturated neurons (near -1 or 1 for tanh, near 0 or 1 for sigmoid)
            saturated = ((act_flat.abs() > 0.99) | (act_flat.abs() < 0.01)).sum().item()
            metrics[f"{layer_prefix}/saturated_pct"] = (
                100.0 * saturated / max(act_flat.numel(), 1)
            )

        return metrics

    def clear(self) -> None:
        """Clear stored activations."""
        self._activations.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


class LearningRateTracker:
    """Track learning rate from optimizer and scheduler.

    Extracts current learning rate(s) for logging, supporting:
    - Single learning rate
    - Per-parameter-group learning rates
    - Learning rate schedulers

    Example:
        >>> tracker = LearningRateTracker()
        >>> metrics = tracker.compute(optimizer, scheduler)
        >>> # {'lr': 0.001} or {'lr/group0': 0.001, 'lr/group1': 0.0001}
    """

    def __init__(self, prefix: str = ""):
        """Initialize learning rate tracker.

        Args:
            prefix: Prefix for metric names.
        """
        self.prefix = prefix

    def compute(
        self,
        optimizer: Any,
        scheduler: Any | None = None,
    ) -> dict[str, float]:
        """Extract current learning rate(s).

        Args:
            optimizer: PyTorch optimizer.
            scheduler: Optional learning rate scheduler.

        Returns:
            Dictionary with learning rate metrics:
            - {prefix}lr: Learning rate (if single group)
            - {prefix}lr/group{i}: Per-group learning rates (if multiple)
        """
        metrics = {}

        # Get LR from optimizer param groups
        lrs = [group["lr"] for group in optimizer.param_groups]

        if len(lrs) == 1:
            metrics[f"{self.prefix}lr"] = lrs[0]
        else:
            for i, lr in enumerate(lrs):
                metrics[f"{self.prefix}lr/group{i}"] = lr

        # Also get last_lr from scheduler if available
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            try:
                last_lrs = scheduler.get_last_lr()
                if len(last_lrs) == 1:
                    metrics[f"{self.prefix}lr_scheduled"] = last_lrs[0]
            except Exception:
                pass  # Some schedulers don't support get_last_lr

        return metrics


class TrainingDiagnostics:
    """Unified training diagnostics combining all trackers.

    Provides a single interface to track gradients, parameters, activations,
    and learning rates during training.

    Example:
        >>> diagnostics = TrainingDiagnostics(model)
        >>> for epoch in range(100):
        ...     for batch in dataloader:
        ...         optimizer.zero_grad()
        ...         loss = model(batch).mean()
        ...         loss.backward()
        ...         # Compute diagnostics after backward, before optimizer step
        ...         diag_metrics = diagnostics.compute(optimizer)
        ...         optimizer.step()
        ...         logger.log({**train_metrics, **diag_metrics}, step=step)
        >>> diagnostics.cleanup()  # Remove activation hooks

    Args:
        model: PyTorch model to track.
        config: Diagnostics configuration. If None, uses defaults.
    """

    def __init__(
        self,
        model: "nn.Module",
        config: DiagnosticsConfig | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TrainingDiagnostics")

        self.model = model
        self.config = config or DiagnosticsConfig()
        self._step = 0

        # Initialize trackers
        prefix = self.config.prefix

        self.gradient_tracker = (
            GradientTracker(
                model,
                per_layer=self.config.gradient_per_layer,
                prefix=f"{prefix}grad/",
            )
            if self.config.track_gradients
            else None
        )

        self.parameter_tracker = (
            ParameterTracker(
                model,
                per_layer=self.config.parameter_per_layer,
                prefix=f"{prefix}param/",
            )
            if self.config.track_parameters
            else None
        )

        self.activation_tracker = (
            ActivationTracker(
                model,
                layer_names=self.config.activation_layers,
                prefix=f"{prefix}act/",
            )
            if self.config.track_activations
            else None
        )

        self.lr_tracker = (
            LearningRateTracker(prefix=prefix)
            if self.config.track_learning_rate
            else None
        )

    def compute(
        self,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, float]:
        """Compute all enabled diagnostic metrics.

        Call this after loss.backward() but before optimizer.step().

        Args:
            optimizer: PyTorch optimizer (required for LR tracking).
            scheduler: Optional learning rate scheduler.

        Returns:
            Dictionary with all diagnostic metrics.
        """
        self._step += 1

        # Check if we should log this step
        if self._step % self.config.log_frequency != 0:
            return {}

        metrics = {}

        # Gradient statistics
        if self.gradient_tracker is not None:
            metrics.update(self.gradient_tracker.compute())

        # Parameter statistics
        if self.parameter_tracker is not None:
            metrics.update(self.parameter_tracker.compute())

        # Activation statistics
        if self.activation_tracker is not None:
            metrics.update(self.activation_tracker.compute())
            self.activation_tracker.clear()

        # Learning rate
        if self.lr_tracker is not None and optimizer is not None:
            metrics.update(self.lr_tracker.compute(optimizer, scheduler))

        return metrics

    def cleanup(self) -> None:
        """Clean up resources (remove activation hooks)."""
        if self.activation_tracker is not None:
            self.activation_tracker.remove_hooks()

    def __enter__(self) -> "TrainingDiagnostics":
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()

    def summary(self) -> str:
        """Return a summary of what diagnostics are enabled."""
        parts = []
        if self.gradient_tracker:
            parts.append("gradients")
        if self.parameter_tracker:
            parts.append("parameters")
        if self.activation_tracker:
            parts.append("activations")
        if self.lr_tracker:
            parts.append("learning_rate")
        return f"TrainingDiagnostics({', '.join(parts)})"


def diagnose_gradients(model: "nn.Module") -> dict[str, Any]:
    """Quick diagnostic check for gradient health.

    Call after loss.backward() to get a health report.

    Args:
        model: Model with computed gradients.

    Returns:
        Dictionary with:
        - healthy: Whether gradients look healthy
        - issues: List of detected issues
        - stats: Gradient statistics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for diagnose_gradients")

    tracker = GradientTracker(model, per_layer=True)
    stats = tracker.compute()

    issues = []
    healthy = True

    grad_norm = stats.get("grad/norm", 0)
    grad_max = stats.get("grad/max", 0)
    pct_zero = stats.get("grad/pct_zero", 0)

    # Check for issues
    if grad_norm == 0:
        issues.append("All gradients are zero - check loss computation")
        healthy = False
    elif grad_norm < 1e-7:
        issues.append(f"Vanishing gradients: norm={grad_norm:.2e}")
        healthy = False
    elif grad_norm > 1e4:
        issues.append(f"Exploding gradients: norm={grad_norm:.2e}")
        healthy = False

    if grad_max > 1e3:
        issues.append(f"Very large gradient values: max={grad_max:.2e}")
        healthy = False

    if pct_zero > 50:
        issues.append(f"Many zero gradients: {pct_zero:.1f}%")
        healthy = False

    # Check per-layer
    dead_layers = []
    for key, value in stats.items():
        if key.startswith("grad/layer/") and value == 0:
            layer_name = key.replace("grad/layer/", "")
            dead_layers.append(layer_name)

    if dead_layers:
        issues.append(f"Dead layers (zero gradients): {dead_layers}")
        healthy = False

    return {
        "healthy": healthy,
        "issues": issues,
        "stats": stats,
    }
