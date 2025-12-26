"""Latent diffusion model for polymer structure generation.

This module provides the LatentDiffusionModel, which performs diffusion in the
latent space of a pre-trained PolymerFlowModel. The model learns to denoise
latent representations of polymer structures, enabling generation of new
structures by sampling from noise and iteratively denoising.

Similar to Stable Diffusion, this approach:
- Performs diffusion in a compressed latent space (n_residues, 12) instead of coordinates
- Uses a frozen pre-trained flow model for encoding/decoding
- Trains a transformer denoiser conditioned on residue sequence

Example:
    >>> from ciffy.nn.diffusion import LatentDiffusionModel, LatentDiffusionConfig
    >>>
    >>> config = LatentDiffusionConfig()
    >>> model = LatentDiffusionModel(config)
    >>>
    >>> # Training step
    >>> loss, metrics = model.training_step(coords, sequence)
    >>>
    >>> # Generate structure
    >>> coords = model.sample(sequence, n_samples=1, num_steps=50)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

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

from .latent_denoiser import LatentDenoiser, LatentDenoiserConfig
from .process import CosineNoiseSchedule, DiffusionProcess, LinearNoiseSchedule


@dataclass
class LatentDiffusionConfig:
    """Configuration for the latent diffusion model.

    Attributes:
        denoiser: Configuration for the transformer denoiser.
        num_timesteps: Number of diffusion steps.
        noise_schedule: Type of noise schedule ('cosine' or 'linear').
        flow_model_path: Path to pre-trained PolymerFlowModel (None uses default).
        freeze_flow_model: Whether to freeze the flow model weights.
    """

    denoiser: LatentDenoiserConfig = field(default_factory=LatentDenoiserConfig)
    num_timesteps: int = 1000
    noise_schedule: str = "cosine"
    flow_model_path: Optional[str] = None
    freeze_flow_model: bool = True


class LatentDiffusionModel(nn.Module):
    """Latent diffusion model for polymer structure generation.

    Combines:
        - Pre-trained PolymerFlowModel for encoding/decoding coordinates
        - DiffusionProcess for forward/reverse diffusion
        - LatentDenoiser (transformer) for noise prediction

    Training:
        model.training_step(coords, sequence) -> loss, metrics

    Sampling:
        model.sample(sequence, n_samples=1) -> coords

    Example:
        >>> config = LatentDiffusionConfig()
        >>> model = LatentDiffusionModel(config)
        >>>
        >>> # Training
        >>> coords = polymer.coordinates  # (N, 3)
        >>> sequence = polymer.sequence   # (n_residues,)
        >>> loss, metrics = model.training_step(coords, sequence)
        >>>
        >>> # Sampling
        >>> sampled_coords = model.sample(sequence, n_samples=5)
    """

    def __init__(
        self,
        config: LatentDiffusionConfig,
        flow_model: Optional["PolymerFlowModel"] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LatentDiffusionModel")
        super().__init__()

        self.config = config

        # Import here to avoid circular imports
        from ciffy.nn.flow import PolymerFlowModel, load_pretrained

        # Load or use provided flow model
        if flow_model is not None:
            self.flow_model = flow_model
        elif config.flow_model_path is not None:
            self.flow_model = PolymerFlowModel.load(config.flow_model_path)
        else:
            # Load default pretrained model
            self.flow_model = load_pretrained("rna")

        # Freeze flow model if configured
        if config.freeze_flow_model:
            for model in self.flow_model.residue_models.values():
                for param in model.parameters():
                    param.requires_grad = False

        # Ensure denoiser config matches flow model
        config.denoiser.latent_dim = self.flow_model.latent_dim
        config.denoiser.num_timesteps = config.num_timesteps

        # Create denoiser
        self.denoiser = LatentDenoiser(config.denoiser)

        # Create diffusion process
        if config.noise_schedule == "cosine":
            schedule = CosineNoiseSchedule(config.num_timesteps)
        else:
            schedule = LinearNoiseSchedule(config.num_timesteps)
        self.diffusion = DiffusionProcess(schedule)

        # Track latent dim for convenience
        self.latent_dim = self.flow_model.latent_dim

    @property
    def device(self) -> "torch.device":
        """Get device of model parameters."""
        return next(self.denoiser.parameters()).device

    def encode(
        self,
        coords: "torch.Tensor",
        sequence: "torch.Tensor | np.ndarray",
    ) -> "torch.Tensor":
        """Encode coordinates to latent space.

        Args:
            coords: (N, 3) atom coordinates.
            sequence: (n_residues,) residue type indices.

        Returns:
            (n_residues, latent_dim) latent vectors.
        """
        with torch.no_grad():
            return self.flow_model.encode(coords, sequence)

    def decode(
        self,
        latents: "torch.Tensor",
        sequence: "torch.Tensor | np.ndarray",
    ) -> "torch.Tensor":
        """Decode latents to coordinates.

        Args:
            latents: (n_residues, latent_dim) latent vectors.
            sequence: (n_residues,) residue type indices.

        Returns:
            (N, 3) atom coordinates.
        """
        with torch.no_grad():
            return self.flow_model.decode(latents, sequence)

    def training_step(
        self,
        coords: "torch.Tensor",
        sequence: "torch.Tensor | np.ndarray",
        mask: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", dict[str, float]]:
        """Compute training loss for a single sample.

        Args:
            coords: (N, 3) atom coordinates.
            sequence: (n_residues,) residue type indices.
            mask: Optional padding mask.

        Returns:
            Tuple of (loss, metrics dict).
        """
        # Convert sequence to tensor if needed
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence).long().to(coords.device)

        # Encode to latent space
        latents = self.encode(coords, sequence)  # (n_res, latent_dim)

        # Add batch dimension
        latents = latents.unsqueeze(0)  # (1, n_res, latent_dim)
        sequence = sequence.unsqueeze(0)  # (1, n_res)

        # Sample random timestep
        t = self.diffusion.schedule.random_timestep((1,))
        t = t.to(latents.device)

        # Forward diffusion
        noise, noisy_latents = self.diffusion.forward_diffusion(latents, t)

        # Predict noise
        pred_noise = self.denoiser(noisy_latents, t, sequence, mask)

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
        latents_batch: "torch.Tensor",
        sequence_batch: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", dict[str, float]]:
        """Compute training loss for a batch (pre-encoded latents).

        For efficiency, encode all samples before training and pass
        batched latents directly.

        Args:
            latents_batch: (batch, n_residues, latent_dim) latent vectors.
            sequence_batch: (batch, n_residues) residue type indices.
            mask: Optional padding mask (batch, n_residues).

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        batch_size = latents_batch.shape[0]

        # Sample random timesteps
        t = self.diffusion.schedule.random_timestep((batch_size,))
        t = t.to(latents_batch.device)

        # Forward diffusion
        noise, noisy_latents = self.diffusion.forward_diffusion(latents_batch, t)

        # Predict noise
        pred_noise = self.denoiser(noisy_latents, t, sequence_batch, mask)

        # MSE loss (optionally mask padded positions)
        if mask is not None:
            # Expand mask to latent dimension
            mask_expanded = mask.unsqueeze(-1).expand_as(pred_noise)
            # Zero out padded positions
            pred_noise_masked = pred_noise.masked_fill(mask_expanded, 0.0)
            noise_masked = noise.masked_fill(mask_expanded, 0.0)
            # Count non-padded positions
            n_valid = (~mask).sum() * pred_noise.shape[-1]
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
    def sample(
        self,
        sequence: "torch.Tensor | np.ndarray",
        n_samples: int = 1,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        progress: bool = False,
    ) -> Union["torch.Tensor", list["torch.Tensor"]]:
        """Generate polymer structures via reverse diffusion.

        Args:
            sequence: (n_residues,) residue type indices.
            n_samples: Number of samples to generate.
            num_steps: DDIM steps (None = full DDPM with all timesteps).
            eta: DDIM stochasticity (0 = deterministic).
            progress: Show progress bar.

        Returns:
            If n_samples=1: (N, 3) coordinates.
            Otherwise: List of (N, 3) coordinate tensors.
        """
        # Convert sequence to tensor if needed
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence).long()

        n_res = len(sequence)
        device = self.device

        # Batch sequence
        seq_batch = sequence.unsqueeze(0).expand(n_samples, -1).to(device)

        # Start from noise
        x = torch.randn(n_samples, n_res, self.latent_dim, device=device)

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

            # Predict noise
            pred_noise = self.denoiser(x, t_batch, seq_batch)

            # DDIM or DDPM step
            if num_steps is not None:
                t_prev = timesteps[i + 1].item() if i < len(timesteps) - 1 else 0
                x = self.diffusion.ddim_step(x, pred_noise, t, t_prev, eta=eta)
            else:
                x = self.diffusion.ddpm_step(x, pred_noise, t)

        # Decode to coordinates
        samples = []
        for i in range(n_samples):
            coords = self.decode(x[i], sequence.cpu().numpy())
            samples.append(coords)

        if n_samples == 1:
            return samples[0]
        return samples

    def sample_to_polymer(
        self,
        template: "Polymer",
        n_samples: int = 1,
        **kwargs,
    ) -> Union["Polymer", list["Polymer"]]:
        """Generate polymers with same metadata as template.

        Args:
            template: Polymer to use as metadata template.
            n_samples: Number of samples.
            **kwargs: Passed to sample().

        Returns:
            New Polymer(s) with generated coordinates.
        """
        sequence = torch.tensor(template.sequence, device=self.device)
        coords_list = self.sample(sequence, n_samples, **kwargs)

        if n_samples == 1:
            coords_list = [coords_list]

        polymers = []
        for coords in coords_list:
            coords_np = coords.cpu().numpy()
            polymers.append(template.with_coordinates(coords_np))

        if n_samples == 1:
            return polymers[0]
        return polymers

    def sample_from_sequence(
        self,
        sequence: str,
        n_samples: int = 1,
        id: str = "sampled",
        **kwargs,
    ) -> Union["Polymer", list["Polymer"]]:
        """
        Sample polymer conformations directly from a sequence string.

        Generates a template Polymer from the sequence string and samples
        new conformations via the diffusion process.

        Args:
            sequence: Sequence string (e.g., "acgu" for RNA, "MGKLF" for protein).
            n_samples: Number of conformations to generate.
            id: PDB ID for the generated polymers.
            **kwargs: Passed to sample() (e.g., num_steps, temperature).

        Returns:
            If n_samples=1: Single Polymer with generated coordinates.
            If n_samples>1: List of Polymers.

        Example:
            >>> model = LatentDiffusionModel(config)
            >>> polymer = model.sample_from_sequence("acgu", num_steps=50)
            >>> polymer.write("sampled.cif")
        """
        from ciffy.template import from_sequence

        # Create template with correct atoms for the flow model
        template = from_sequence(
            sequence,
            atoms=self.flow_model.atom_filter,
            id=id,
        )

        return self.sample_to_polymer(template, n_samples, **kwargs)

    def reconstruct(
        self,
        coords: "torch.Tensor",
        sequence: "torch.Tensor | np.ndarray",
        num_steps: int = 50,
    ) -> "torch.Tensor":
        """Encode, add noise, and denoise to test reconstruction.

        Useful for evaluating model quality without full sampling.

        Args:
            coords: (N, 3) atom coordinates.
            sequence: (n_residues,) residue type indices.
            num_steps: DDIM denoising steps.

        Returns:
            (N, 3) reconstructed coordinates.
        """
        # Convert sequence to tensor if needed
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence).long()

        # Encode
        latents = self.encode(coords, sequence)

        # Add moderate noise (halfway through schedule)
        t_mid = self.config.num_timesteps // 2
        t = torch.tensor([t_mid], device=latents.device)
        noise, noisy_latents = self.diffusion.forward_diffusion(
            latents.unsqueeze(0), t
        )

        # Denoise using DDIM starting from t_mid
        timesteps = self.diffusion.get_sampling_timesteps(num_steps)
        # Filter to only timesteps <= t_mid
        timesteps = timesteps[timesteps <= t_mid]

        x = noisy_latents
        seq_batch = sequence.unsqueeze(0).to(self.device)

        for i, t_curr in enumerate(timesteps.tolist()):
            t_batch = torch.tensor([t_curr], device=self.device)
            pred_noise = self.denoiser(x, t_batch, seq_batch)

            t_prev = timesteps[i + 1].item() if i < len(timesteps) - 1 else 0
            x = self.diffusion.ddim_step(x, pred_noise, t_curr, t_prev, eta=0.0)

        # Decode
        return self.decode(x.squeeze(0), sequence.cpu().numpy())


__all__ = [
    "LatentDiffusionConfig",
    "LatentDiffusionModel",
]
