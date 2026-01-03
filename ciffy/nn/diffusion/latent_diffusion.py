"""Latent diffusion model for polymer structure generation.

This module provides the LatentDiffusionModel, which performs diffusion in the
latent space of a pre-trained PolymerModel (flow, VAE, or consolidated). The
model learns to denoise latent representations of polymer structures, enabling
generation of new structures by sampling from noise and iteratively denoising.

Similar to Stable Diffusion, this approach:
- Performs diffusion in a compressed latent space (n_residues, latent_dim) instead of coordinates
- Uses a frozen pre-trained encoder/decoder model (any PolymerModel)
- Trains a transformer denoiser conditioned on residue sequence

Example:
    >>> import ciffy
    >>> from ciffy.nn.diffusion import LatentDiffusionModel, LatentDiffusionConfig
    >>>
    >>> config = LatentDiffusionConfig(encoder_path="outputs/models/flow")
    >>> model = LatentDiffusionModel(config)
    >>>
    >>> # Training step
    >>> loss, metrics = model.training_step(coords, sequence)
    >>>
    >>> # Generate from template (PolymerGenerativeModel protocol)
    >>> template = ciffy.load("structure.cif").poly()
    >>> samples = model.sample(template, n_samples=5, num_steps=50)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

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
    from ciffy.nn.polymer import PolymerModel

import numpy as np

from .latent_denoiser import LatentDenoiser, LatentDenoiserConfig
from .process import CosineNoiseSchedule, DiffusionProcess, LinearNoiseSchedule
from ..model_registry import register_model


@dataclass
class LatentDiffusionConfig:
    """Configuration for the latent diffusion model.

    Attributes:
        denoiser: Configuration for the transformer denoiser.
        num_timesteps: Number of diffusion steps.
        noise_schedule: Type of noise schedule ('cosine' or 'linear').
        encoder_path: Path to pre-trained PolymerModel (flow, vae, or consolidated).
        freeze_encoder: Whether to freeze the encoder/decoder weights.
        loss_weighting: Timestep loss weighting strategy:
            - "none": No weighting (default, all timesteps equal).
            - "snr": Weight by signal-to-noise ratio (alphabar/(1-alphabar)).
            - "min_snr_5": Min-SNR weighting capped at 5 (recommended).
    """

    denoiser: LatentDenoiserConfig = field(default_factory=LatentDenoiserConfig)
    num_timesteps: int = 1000
    noise_schedule: str = "cosine"
    encoder_path: Optional[str] = None
    freeze_encoder: bool = True
    loss_weighting: str = "none"


@register_model("latent_diffusion")
class LatentDiffusionModel(nn.Module):
    """Latent diffusion model for polymer structure generation.

    Implements the PolymerGenerativeModel protocol for interoperability.

    Combines:
        - Pre-trained PolymerModel (flow, VAE, or consolidated) for encoding/decoding
        - DiffusionProcess for forward/reverse diffusion
        - LatentDenoiser (transformer) for noise prediction

    Training:
        model.training_step(coords, sequence) -> loss, metrics

    Sampling (protocol-compliant):
        model.sample(template, n_samples=5) -> list[Polymer]

    Example:
        >>> import ciffy
        >>> config = LatentDiffusionConfig(encoder_path="outputs/models/flow")
        >>> model = LatentDiffusionModel(config)
        >>>
        >>> # Training
        >>> coords = polymer.coordinates  # (N, 3)
        >>> sequence = polymer.sequence   # (n_residues,)
        >>> loss, metrics = model.training_step(coords, sequence)
        >>>
        >>> # Sampling (returns list of Polymers)
        >>> template = ciffy.load("structure.cif").poly()
        >>> samples = model.sample(template, n_samples=5, num_steps=50)
    """

    def __init__(
        self,
        config: LatentDiffusionConfig,
        encoder_model: Optional["PolymerModel"] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LatentDiffusionModel")
        super().__init__()

        self.config = config

        # Import here to avoid circular imports
        from ciffy.nn.polymer import PolymerModel

        # Load or use provided encoder model
        if encoder_model is not None:
            self.encoder_model = encoder_model
        elif config.encoder_path is not None:
            self.encoder_model = PolymerModel.load(config.encoder_path)
        else:
            raise ValueError(
                "Must provide either encoder_model or config.encoder_path. "
                "Train a PolymerModel first using: python scripts/residue_models.py train"
            )

        # Freeze encoder if configured
        if config.freeze_encoder:
            for model in self.encoder_model.residue_models.values():
                for param in model.parameters():
                    param.requires_grad = False

        # Ensure denoiser config matches encoder model
        config.denoiser.latent_dim = self.encoder_model.latent_dim
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
        self.latent_dim = self.encoder_model.latent_dim

        # Store loss weighting strategy
        self._loss_weighting = config.loss_weighting

    def _get_loss_weights(self, t: "torch.Tensor") -> "torch.Tensor":
        """Compute loss weights for each timestep.

        Args:
            t: (batch,) timestep indices.

        Returns:
            (batch,) loss weights.
        """
        if self._loss_weighting == "none":
            return torch.ones_like(t, dtype=torch.float32)

        # Get SNR = alphabar / (1 - alphabar)
        alphabar = self.diffusion.schedule.alphabar(t)
        snr = alphabar / (1 - alphabar + 1e-8)

        if self._loss_weighting == "snr":
            # Weight proportional to SNR
            return snr
        elif self._loss_weighting == "min_snr_5":
            # Min-SNR weighting: min(SNR, gamma) / SNR where gamma=5
            # This reduces weight on low-noise (high-SNR) timesteps
            gamma = 5.0
            return torch.minimum(snr, torch.tensor(gamma, device=t.device)) / (snr + 1e-8)
        else:
            # Unknown weighting, fall back to uniform
            return torch.ones_like(t, dtype=torch.float32)

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
            return self.encoder_model.encode(coords, sequence)

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
            return self.encoder_model.decode(latents, sequence)

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

        # Get loss weights for each sample based on timestep
        weights = self._get_loss_weights(t)  # (batch,)

        # MSE loss (optionally mask padded positions)
        if mask is not None:
            # Expand mask to latent dimension
            mask_expanded = mask.unsqueeze(-1).expand_as(pred_noise)
            # Compute per-sample MSE
            sq_error = (pred_noise - noise).pow(2)
            sq_error = sq_error.masked_fill(mask_expanded, 0.0)
            # Sum over residues and latent dim, divide by valid count per sample
            n_valid_per_sample = (~mask).sum(dim=1, keepdim=True) * pred_noise.shape[-1]
            sample_loss = sq_error.sum(dim=(1, 2)) / n_valid_per_sample.squeeze()
            # Apply timestep weights and average
            loss = (sample_loss * weights).mean()
        else:
            # Compute per-sample MSE
            sample_loss = (pred_noise - noise).pow(2).mean(dim=(1, 2))
            # Apply timestep weights and average
            loss = (sample_loss * weights).mean()

        metrics = {
            "loss": loss.item(),
            "noise_mse": loss.item(),
            "mean_timestep": t.float().mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def _sample_coords(
        self,
        sequence: "torch.Tensor | np.ndarray",
        n_samples: int = 1,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        progress: bool = False,
    ) -> list["torch.Tensor"]:
        """Generate coordinate tensors via reverse diffusion (internal method).

        Args:
            sequence: (n_residues,) residue type indices.
            n_samples: Number of samples to generate.
            num_steps: DDIM steps (None = full DDPM with all timesteps).
            eta: DDIM stochasticity (0 = deterministic).
            progress: Show progress bar.

        Returns:
            List of (N, 3) coordinate tensors.
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
                Accepts either numpy or torch backend; output matches input.
            n_samples: Number of independent conformations to generate.
            temperature: Sampling temperature. For diffusion models, this is
                currently ignored (reserved for future use). Default 1.0.
            **kwargs: Passed to internal sampling (e.g., num_steps, eta, progress).

        Returns:
            List of n_samples Polymers with generated coordinates (same backend as template).

        Raises:
            ValueError: If template contains unsupported residues.

        Example:
            >>> model = LatentDiffusionModel(config)
            >>> template = ciffy.load("structure.cif").poly()
            >>> samples = model.sample(template, n_samples=10)
        """
        # Get sequence and validate against flow model
        sequence = template.sequence

        # Sample coordinates
        coords_list = self._sample_coords(sequence, n_samples, **kwargs)

        # Check if template atom count matches encoder model expectations
        expected_atoms = sum(
            self.encoder_model.residue_models[str(int(r))].n_atoms
            for r in sequence
        )

        # Convert coords to match template backend
        use_torch = template.backend == "torch"
        if use_torch:
            coords_list = [coords.to(template.coordinates.device) for coords in coords_list]
        else:
            coords_list = [coords.cpu().numpy() for coords in coords_list]

        if template.size() == expected_atoms:
            # Template matches - use copy for efficiency
            return [template.copy(coordinates=coords) for coords in coords_list]
        else:
            # Template has different atoms (e.g., missing atoms) - build fresh polymers
            from ciffy.polymer import from_sequence

            # Create template with encoder model's expected atoms
            encoder_template = from_sequence(
                template.sequence_str(),
                atoms=self.encoder_model.atom_filter,
                id=template.pdb_id,
            )
            if use_torch:
                encoder_template = encoder_template.torch().to(template.coordinates.device)
            return [encoder_template.copy(coordinates=coords) for coords in coords_list]

    def sample_from_sequence(
        self,
        sequence: str,
        n_samples: int = 1,
        id: str = "sampled",
        **kwargs,
    ) -> list["Polymer"]:
        """
        Sample polymer conformations directly from a sequence string.

        Convenience method that creates a template from a sequence string
        and calls sample().

        Args:
            sequence: Sequence string (e.g., "acgu" for RNA, "MGKLF" for protein).
            n_samples: Number of conformations to generate.
            id: PDB ID for the generated polymers.
            **kwargs: Passed to sample() (e.g., num_steps, eta, progress).

        Returns:
            List of Polymers with generated coordinates.

        Example:
            >>> model = LatentDiffusionModel(config)
            >>> polymers = model.sample_from_sequence("acgu", num_steps=50)
            >>> polymers[0].write("sampled.cif")
        """
        from ciffy.polymer import from_sequence

        # Create template with correct atoms for the encoder model
        template = from_sequence(
            sequence,
            atoms=self.encoder_model.atom_filter,
            id=id,
        )

        return self.sample(template, n_samples, **kwargs)

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

    # ─────────────────────────────────────────────────────────────────────────
    # Unified Save/Load (SaveableModel protocol)
    # ─────────────────────────────────────────────────────────────────────────

    def get_save_state(self) -> tuple[dict[str, "torch.Tensor"], dict[str, Any]]:
        """Get state for unified save format.

        Only saves denoiser weights. The encoder model path is stored in config
        and must be loaded separately.

        Returns:
            Tuple of (tensors_dict, config_dict) for safetensors serialization.
        """
        tensors = {}

        # Only save denoiser tensors (encoder is loaded from path)
        for k, v in self.denoiser.state_dict().items():
            tensors[k] = v

        # Config - store encoder path, not full encoder state
        config = {
            "num_timesteps": self.config.num_timesteps,
            "noise_schedule": self.config.noise_schedule,
            "freeze_encoder": self.config.freeze_encoder,
            "encoder_path": self.config.encoder_path,
            "denoiser": asdict(self.config.denoiser),
        }

        return tensors, config

    @classmethod
    def from_save_state(
        cls,
        tensors: dict[str, "torch.Tensor"],
        config: dict[str, Any],
        device: str = "cpu",
        encoder_model: Optional["PolymerModel"] = None,
    ) -> "LatentDiffusionModel":
        """Reconstruct model from unified save format.

        Args:
            tensors: Loaded tensors dict (denoiser weights).
            config: Loaded config dict.
            device: Device to load model to.
            encoder_model: Optional pre-loaded encoder. If None, loads from
                config["encoder_path"].

        Returns:
            Reconstructed LatentDiffusionModel.
        """
        from ciffy.nn.polymer import PolymerModel

        # Load encoder model from path if not provided
        if encoder_model is None:
            encoder_path = config.get("encoder_path")
            if encoder_path is None:
                raise ValueError(
                    "No encoder_path in config and no encoder_model provided. "
                    "Either provide encoder_model or ensure config has encoder_path."
                )
            encoder_model = PolymerModel.load(encoder_path, device=device)

        # Build config
        denoiser_config = LatentDenoiserConfig(**config["denoiser"])
        model_config = LatentDiffusionConfig(
            denoiser=denoiser_config,
            num_timesteps=config["num_timesteps"],
            noise_schedule=config["noise_schedule"],
            freeze_encoder=config["freeze_encoder"],
            encoder_path=config.get("encoder_path"),
        )

        # Create model
        model = cls(model_config, encoder_model=encoder_model)

        # Load denoiser weights
        model.denoiser.load_state_dict(tensors)

        return model.to(device)


__all__ = [
    "LatentDiffusionConfig",
    "LatentDiffusionModel",
]
