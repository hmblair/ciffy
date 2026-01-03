"""Coordinate diffusion model for polymer structure generation.

This module provides the CoordinateDiffusionModel, which performs diffusion
directly on atom coordinates. Unlike latent diffusion, this approach works
in the full coordinate space, conditioned on sequence via PolymerEmbedding.

Example:
    >>> import ciffy
    >>> from ciffy.nn.diffusion import CoordinateDiffusionModel, CoordinateDiffusionConfig
    >>>
    >>> config = CoordinateDiffusionConfig()
    >>> model = CoordinateDiffusionModel(config)
    >>>
    >>> # Training step
    >>> loss, metrics = model.training_step(polymer)
    >>>
    >>> # Generate from template (PolymerGenerativeModel protocol)
    >>> template = ciffy.load("structure.cif").poly()
    >>> samples = model.sample(template, n_samples=5, num_steps=50)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from ciffy import Polymer

import numpy as np

from .coordinate_denoiser import CoordinateDenoiser, CoordinateDenoiserConfig
from .process import CosineNoiseSchedule, DiffusionProcess, LinearNoiseSchedule
from ..model_registry import register_model


@dataclass
class CoordinateDiffusionConfig:
    """Configuration for the coordinate diffusion model.

    Attributes:
        denoiser: Configuration for the transformer denoiser.
        num_timesteps: Number of diffusion steps.
        noise_schedule: Type of noise schedule ('cosine' or 'linear').
        coord_scale: Scale factor for coordinates (helps with noise levels).
    """

    denoiser: CoordinateDenoiserConfig = field(default_factory=CoordinateDenoiserConfig)
    num_timesteps: int = 1000
    noise_schedule: str = "cosine"
    coord_scale: float = 0.1  # Scale down coords for better noise dynamics


@register_model("coordinate_diffusion")
class CoordinateDiffusionModel(nn.Module):
    """Coordinate diffusion model for polymer structure generation.

    Implements the PolymerGenerativeModel protocol for interoperability.

    Performs diffusion directly on atom coordinates, conditioned on sequence
    via PolymerEmbedding (residue type + element type at atom level).

    Training:
        model.training_step(polymer) -> loss, metrics

    Sampling (protocol-compliant):
        model.sample(template, n_samples=5) -> list[Polymer]

    Example:
        >>> import ciffy
        >>> config = CoordinateDiffusionConfig()
        >>> model = CoordinateDiffusionModel(config)
        >>>
        >>> # Training
        >>> polymer = ciffy.load("structure.cif").poly().torch()
        >>> loss, metrics = model.training_step(polymer)
        >>>
        >>> # Sampling (returns list of Polymers)
        >>> template = ciffy.load("structure.cif").poly()
        >>> samples = model.sample(template, n_samples=5, num_steps=50)
    """

    def __init__(self, config: CoordinateDiffusionConfig) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CoordinateDiffusionModel")
        super().__init__()

        self.config = config

        # Ensure denoiser config has matching timesteps
        config.denoiser.num_timesteps = config.num_timesteps

        # Create denoiser
        self.denoiser = CoordinateDenoiser(config.denoiser)

        # Create diffusion process
        if config.noise_schedule == "cosine":
            schedule = CosineNoiseSchedule(config.num_timesteps)
        else:
            schedule = LinearNoiseSchedule(config.num_timesteps)
        self.diffusion = DiffusionProcess(schedule)

    @property
    def device(self) -> "torch.device":
        """Get device of model parameters."""
        return next(self.denoiser.parameters()).device

    def _scale_coords(self, coords: "torch.Tensor") -> "torch.Tensor":
        """Scale coordinates for diffusion."""
        return coords * self.config.coord_scale

    def _unscale_coords(self, coords: "torch.Tensor") -> "torch.Tensor":
        """Unscale coordinates after diffusion."""
        return coords / self.config.coord_scale

    def training_step(
        self,
        polymer: "Polymer",
        mask: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", dict[str, float]]:
        """Compute training loss for a single polymer.

        Args:
            polymer: Polymer object (must be torch backend).
            mask: Optional padding mask.

        Returns:
            Tuple of (loss, metrics dict).
        """
        coords = polymer.coordinates  # (N, 3)

        # Scale coordinates
        coords_scaled = self._scale_coords(coords)

        # Add batch dimension
        coords_batch = coords_scaled.unsqueeze(0)  # (1, N, 3)

        # Sample random timestep
        t = self.diffusion.schedule.random_timestep((1,))
        t = t.to(coords.device)

        # Forward diffusion (add noise)
        noise, noisy_coords = self.diffusion.forward_diffusion(coords_batch, t)

        # Predict noise
        pred_noise = self.denoiser(noisy_coords, t, polymer, mask)

        # MSE loss
        loss = F.mse_loss(pred_noise, noise)

        metrics = {
            "loss": loss.item(),
            "noise_mse": loss.item(),
            "timestep": t.item(),
        }

        return loss, metrics

    def training_step_batch(
        self,
        coords_batch: "torch.Tensor",
        polymers: list["Polymer"],
        mask: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", dict[str, float]]:
        """Compute training loss for a batch.

        Note: For coordinate diffusion, batching is tricky because each
        polymer may have different number of atoms. Use padding + mask.

        Args:
            coords_batch: (batch, max_atoms, 3) padded coordinates.
            polymers: List of Polymer objects for conditioning.
            mask: Padding mask (batch, max_atoms). True = padding.

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        batch_size = coords_batch.shape[0]

        # Scale coordinates
        coords_scaled = self._scale_coords(coords_batch)

        # Sample random timesteps
        t = self.diffusion.schedule.random_timestep((batch_size,))
        t = t.to(coords_batch.device)

        # Forward diffusion
        noise, noisy_coords = self.diffusion.forward_diffusion(coords_scaled, t)

        # For batched processing, we need to handle each polymer separately
        # due to PolymerEmbedding requiring a Polymer object
        pred_noises = []
        for i, polymer in enumerate(polymers):
            pred = self.denoiser(
                noisy_coords[i:i+1],
                t[i:i+1],
                polymer,
                mask[i:i+1] if mask is not None else None,
            )
            pred_noises.append(pred)
        pred_noise = torch.cat(pred_noises, dim=0)

        # MSE loss with masking
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(pred_noise)
            pred_noise_masked = pred_noise.masked_fill(mask_expanded, 0.0)
            noise_masked = noise.masked_fill(mask_expanded, 0.0)
            n_valid = (~mask).sum() * 3  # 3 coords per atom
            loss = (pred_noise_masked - noise_masked).pow(2).sum() / n_valid
        else:
            loss = F.mse_loss(pred_noise, noise)

        metrics = {
            "loss": loss.item(),
            "noise_mse": loss.item(),
            "mean_timestep": t.float().mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def _sample_coords(
        self,
        polymer: "Polymer",
        n_samples: int = 1,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        progress: bool = False,
    ) -> list["torch.Tensor"]:
        """Generate coordinate tensors via reverse diffusion.

        Args:
            polymer: Template polymer for conditioning.
            n_samples: Number of samples to generate.
            num_steps: DDIM steps (None = full DDPM with all timesteps).
            eta: DDIM stochasticity (0 = deterministic).
            progress: Show progress bar.

        Returns:
            List of (N, 3) coordinate tensors.
        """
        n_atoms = polymer.size()
        device = self.device

        # Start from noise
        x = torch.randn(n_samples, n_atoms, 3, device=device)

        # Get timesteps
        if num_steps is None:
            timesteps = self.diffusion.timesteps()
        else:
            timesteps = self.diffusion.get_sampling_timesteps(num_steps)

        iterator = timesteps.tolist()
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Sampling")
            except ImportError:
                pass

        # Reverse diffusion
        for i, t in enumerate(iterator):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

            # Predict noise for each sample (same conditioning)
            pred_noises = []
            for j in range(n_samples):
                pred = self.denoiser(x[j:j+1], t_batch[j:j+1], polymer)
                pred_noises.append(pred)
            pred_noise = torch.cat(pred_noises, dim=0)

            # DDIM or DDPM step
            if num_steps is not None:
                t_prev = timesteps[i + 1].item() if i < len(timesteps) - 1 else 0
                x = self.diffusion.ddim_step(x, pred_noise, t, t_prev, eta=eta)
            else:
                x = self.diffusion.ddpm_step(x, pred_noise, t)

        # Unscale coordinates
        samples = [self._unscale_coords(x[i]) for i in range(n_samples)]

        return samples

    def sample(
        self,
        template: "Polymer",
        n_samples: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ) -> list["Polymer"]:
        """
        Generate polymer conformations from a template.

        This method implements the PolymerGenerativeModel protocol, enabling
        this model to be used interchangeably with other generative models.

        Args:
            template: Template Polymer with sequence and topology information.
                Must have torch backend for the denoiser.
            n_samples: Number of independent conformations to generate.
            temperature: Sampling temperature (currently unused, reserved).
            **kwargs: Passed to internal sampling (e.g., num_steps, eta, progress).

        Returns:
            List of n_samples Polymers with generated coordinates.

        Example:
            >>> model = CoordinateDiffusionModel(config)
            >>> template = ciffy.load("structure.cif").poly()
            >>> samples = model.sample(template, n_samples=10)
        """
        # Convert to torch if needed
        if template.backend != "torch":
            template = template.torch().to(self.device)

        # Sample coordinates
        coords_list = self._sample_coords(template, n_samples, **kwargs)

        # Convert to numpy Polymers with template metadata
        template_np = template.numpy()
        return [
            template_np.copy(coordinates=coords.cpu().numpy())
            for coords in coords_list
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Unified Save/Load (SaveableModel protocol)
    # ─────────────────────────────────────────────────────────────────────────

    def get_save_state(self) -> tuple[dict[str, "torch.Tensor"], dict[str, Any]]:
        """Get state for unified save format.

        Returns:
            Tuple of (tensors_dict, config_dict) for safetensors serialization.
        """
        tensors = {}

        # Denoiser tensors
        for k, v in self.denoiser.state_dict().items():
            tensors[f"denoiser.{k}"] = v

        # Config
        config = {
            "num_timesteps": self.config.num_timesteps,
            "noise_schedule": self.config.noise_schedule,
            "coord_scale": self.config.coord_scale,
            "denoiser": asdict(self.config.denoiser),
        }

        return tensors, config

    @classmethod
    def from_save_state(
        cls,
        tensors: dict[str, "torch.Tensor"],
        config: dict[str, Any],
        device: str = "cpu",
    ) -> "CoordinateDiffusionModel":
        """Reconstruct model from unified save format.

        Args:
            tensors: Loaded tensors dict with prefixed keys.
            config: Loaded config dict.
            device: Device to load model to.

        Returns:
            Reconstructed CoordinateDiffusionModel.
        """
        # Build config
        denoiser_config = CoordinateDenoiserConfig(**config["denoiser"])
        model_config = CoordinateDiffusionConfig(
            denoiser=denoiser_config,
            num_timesteps=config["num_timesteps"],
            noise_schedule=config["noise_schedule"],
            coord_scale=config.get("coord_scale", 0.1),
        )

        # Create model
        model = cls(model_config)

        # Load denoiser weights
        denoiser_tensors = {
            k[len("denoiser."):]: v
            for k, v in tensors.items()
            if k.startswith("denoiser.")
        }
        model.denoiser.load_state_dict(denoiser_tensors)

        return model.to(device)


__all__ = [
    "CoordinateDiffusionConfig",
    "CoordinateDiffusionModel",
]
