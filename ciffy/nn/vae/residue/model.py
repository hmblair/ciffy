"""
VAE model for residue conformations.

End-to-end learning without PCA - the encoder learns to compress
coordinates + transforms directly to a latent space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from ciffy.backend import convert_backend, cat, stack
from ciffy.nn.hub import HubMixin

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue, AtomGroup
    from ciffy.geometry import FrameIndices


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ResidueVAEConfig:
    """Configuration for ResidueVAE.

    Args:
        latent_dim: Dimensionality of latent space.
        hidden_dims: Hidden layer dimensions for encoder/decoder.
        beta: Weight for KL divergence loss (beta-VAE).
        dropout: Dropout probability (0 to disable).
        use_input_norm: Learn input mean/std normalization (improves reconstruction).
        use_residual: Add residual connections in decoder.
        separate_heads: Use separate decoder heads for coords vs transforms.
    """

    latent_dim: int = 12
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    beta: float = 1.0
    dropout: float = 0.0
    use_input_norm: bool = True
    use_residual: bool = True
    separate_heads: bool = True


# =============================================================================
# ResidueVAE
# =============================================================================


class InputNorm(nn.Module):
    """
    Learnable input normalization (like BatchNorm but with data-dependent init).

    On first forward pass, initializes to normalize input to zero mean, unit std.
    After initialization, scale and bias become learnable parameters.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    def initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0, correction=0).clamp(min=1e-6)
            self.bias.copy_(-mean / std)
            self.scale.copy_(1.0 / std)
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.initialize(x)
        return x * self.scale + self.bias

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Unnormalize output back to original scale."""
        return (y - self.bias) / self.scale


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and SiLU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidueVAE(nn.Module, HubMixin):
    """
    Variational Autoencoder for residue conformations.

    This model learns the joint distribution of residue coordinates AND
    the SE(3) transform to the next residue in the chain, matching the
    interface of ResidueFlowModel for use with PolymerModel.

    Unlike ResidueFlowModel (which uses PCA + normalizing flows), this
    model learns the dimensionality reduction end-to-end via a neural
    network encoder-decoder.

    The representation is: [coords_flat (n_atoms*3), transform (6)]
    where transform = [axis-angle (3), translation (3)] defines the relative
    position and orientation of the next residue's P atom.

    Architecture options (enabled by default for better reconstruction):
    - Input normalization: Learn mean/std to normalize input features
    - Residual connections: Skip connections in decoder layers
    - Separate heads: Different output heads for coords vs transforms

    Attributes:
        latent_dim: Dimensionality of latent space.
        n_atoms: Number of atoms per residue.
        residue: The source residue type.
        atoms: AtomGroup subset containing the atoms used.

    Example:
        >>> # Train using Lightning
        >>> from ciffy.nn.lightning import ResidueVAEModule, ResidueDataModule
        >>> dm = ResidueDataModule(cif_paths, residue=Residue.A)
        >>> module = ResidueVAEModule(config, residue=Residue.A)
        >>> trainer.fit(module, dm)
        >>> model = module.get_model()
        >>>
        >>> # Decode to get coordinates and transform
        >>> coords, transform = model.decode(z)
        >>>
        >>> # Use with PolymerModel (works with VAE too!)
        >>> from ciffy.nn import PolymerModel
        >>> polymer_model = PolymerModel({Residue.A: model, ...})
    """

    _hub_model_type = "residue-vae"

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        residue: "Residue",
        atom_indices: list[int],
        dropout: float = 0.0,
        use_input_norm: bool = True,
        use_residual: bool = True,
        separate_heads: bool = True,
    ):
        """
        Initialize ResidueVAE.

        Args:
            input_dim: Input dimension (n_atoms * 3 + 6 for coords + transform).
            latent_dim: Latent space dimensionality.
            hidden_dims: Hidden layer dimensions for encoder/decoder.
            residue: Residue type this model handles.
            atom_indices: List of atom type indices in column order.
            dropout: Dropout probability.
            use_input_norm: Learn input normalization (like ActNorm).
            use_residual: Add residual connections in decoder.
            separate_heads: Use separate heads for coords vs transforms.
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_atoms = (input_dim - 6) // 3
        self.residue = residue
        self._atom_indices = atom_indices
        self._hidden_dims = hidden_dims
        self._dropout = dropout
        self._use_input_norm = use_input_norm
        self._use_residual = use_residual
        self._separate_heads = separate_heads

        # Cached properties
        self._atoms_group: "AtomGroup | None" = None
        self._frame_indices: "FrameIndices | None" = None

        # Input normalization (optional but recommended)
        self.input_norm = InputNorm(input_dim) if use_input_norm else None

        # Build encoder: input -> hidden layers -> (mu, logvar)
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
            ])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Build decoder
        decoder_hidden = list(reversed(hidden_dims))

        if use_residual and len(decoder_hidden) >= 2:
            # Use residual blocks for better gradient flow
            # First layer projects from latent to hidden
            self.decoder_input = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden[0]),
                nn.LayerNorm(decoder_hidden[0]),
                nn.SiLU(),
            )

            # Residual blocks at the largest hidden dim
            self.decoder_residual = nn.Sequential(*[
                ResidualBlock(decoder_hidden[0], decoder_hidden[0], dropout)
                for _ in range(2)  # 2 residual blocks
            ])

            # Project down through remaining dims
            decoder_layers = []
            in_dim = decoder_hidden[0]
            for h_dim in decoder_hidden[1:]:
                decoder_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.SiLU(),
                ])
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
                in_dim = h_dim
            self.decoder = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
            self._decoder_out_dim = decoder_hidden[-1]
        else:
            # Simple sequential decoder (original behavior)
            self.decoder_input = None
            self.decoder_residual = None
            decoder_layers = []
            in_dim = latent_dim
            for h_dim in decoder_hidden:
                decoder_layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.SiLU(),
                ])
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
                in_dim = h_dim
            self.decoder = nn.Sequential(*decoder_layers)
            self._decoder_out_dim = decoder_hidden[-1]

        # Output heads
        n_coord_dims = self.n_atoms * 3
        if separate_heads:
            # Separate heads for coords and transforms (different scales)
            self.fc_coords = nn.Linear(self._decoder_out_dim, n_coord_dims)
            self.fc_transform = nn.Linear(self._decoder_out_dim, 6)
            self.fc_out = None
            # Initialize near zero
            nn.init.zeros_(self.fc_coords.weight)
            nn.init.zeros_(self.fc_coords.bias)
            nn.init.zeros_(self.fc_transform.weight)
            nn.init.zeros_(self.fc_transform.bias)
        else:
            # Single output layer (original behavior)
            self.fc_coords = None
            self.fc_transform = None
            self.fc_out = nn.Linear(self._decoder_out_dim, input_dim)
            nn.init.zeros_(self.fc_out.weight)
            nn.init.zeros_(self.fc_out.bias)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties (compatible with ResidueFlowModel)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Device where model parameters reside."""
        return self.fc_mu.weight.device

    @property
    def atoms(self) -> "AtomGroup":
        """AtomGroup subset containing the atoms used by this model."""
        if self._atoms_group is None:
            self._atoms_group = self.residue.subset(set(self._atom_indices))
        return self._atoms_group

    @property
    def frame_indices(self) -> "FrameIndices | None":
        """FrameIndices for glycosidic frame alignment (cached).

        Returns None if the model's atoms don't include the required
        atoms for frame computation.
        """
        if self._frame_indices is None:
            from ciffy.geometry import FrameIndices

            atoms_array = np.array(self._atom_indices, dtype=np.int64)
            try:
                self._frame_indices = FrameIndices.from_atoms(atoms_array, self.residue)
            except ValueError:
                return None
        return self._frame_indices

    # ─────────────────────────────────────────────────────────────────────────
    # VAE Core Methods
    # ─────────────────────────────────────────────────────────────────────────

    def encode_distribution(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: (N, input_dim) extended representation [coords_flat, transform].

        Returns:
            mu: (N, latent_dim) mean of latent distribution.
            logvar: (N, latent_dim) log-variance of latent distribution.
        """
        if self.input_norm is not None:
            x = self.input_norm(x)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick: sample z = mu + sigma * epsilon.

        Args:
            mu: (N, latent_dim) distribution mean.
            logvar: (N, latent_dim) log-variance.

        Returns:
            z: (N, latent_dim) sampled latent vectors.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_flat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to flat output.

        Args:
            z: (N, latent_dim) latent vectors.

        Returns:
            (N, input_dim) reconstructed [coords_flat, transform].
        """
        # Apply decoder with optional residual blocks
        if self.decoder_input is not None:
            h = self.decoder_input(z)
            h = self.decoder_residual(h)
            h = self.decoder(h)
        else:
            h = self.decoder(z)

        # Apply output heads
        if self._separate_heads:
            coords = self.fc_coords(h)
            transform = self.fc_transform(h)
            output = torch.cat([coords, transform], dim=-1)
        else:
            output = self.fc_out(h)

        # Unnormalize output if input normalization was used
        if self.input_norm is not None:
            output = self.input_norm.inverse(output)

        return output

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            x: (N, input_dim) extended representation.

        Returns:
            recon: (N, input_dim) reconstructed output.
            mu: (N, latent_dim) latent mean.
            logvar: (N, latent_dim) latent log-variance.
        """
        mu, logvar = self.encode_distribution(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode_flat(z)
        return recon, mu, logvar

    # ─────────────────────────────────────────────────────────────────────────
    # Interface Compatible with ResidueFlowModel
    # ─────────────────────────────────────────────────────────────────────────

    def encode(
        self,
        coords: "torch.Tensor | np.ndarray",
        next_coords: "torch.Tensor | np.ndarray | None" = None,
    ) -> torch.Tensor:
        """
        Encode coordinates to latent space (ResidueGenerativeCore protocol).

        Assumes coordinates are already aligned to the glycosidic frame.
        For raw coordinates, use PolymerModel which handles alignment.

        Args:
            coords: (n_atoms, 3) or (N, n_atoms, 3) pre-aligned coordinates.
            next_coords: Ignored (kept for protocol compatibility).

        Returns:
            (latent_dim,) or (N, latent_dim) latent vectors (mean).
        """
        coords_t = convert_backend(coords, self.fc_mu.weight).float()

        # Handle single sample
        single = coords_t.dim() == 2
        if single:
            coords_t = coords_t.unsqueeze(0)

        z = self.encode_aligned(coords_t)

        if single:
            return z.squeeze(0)
        return z

    def encode_aligned(
        self,
        aligned_coords: torch.Tensor,
        transforms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode pre-aligned coordinates directly.

        Args:
            aligned_coords: (N, n_atoms, 3) or (N, n_atoms*3) aligned coordinates.
            transforms: (N, 6) SE(3) transforms. If None, uses zeros.

        Returns:
            (N, latent_dim) latent vectors (mean).
        """
        if aligned_coords.dim() == 3:
            aligned_coords = aligned_coords.reshape(aligned_coords.shape[0], -1)

        if transforms is None:
            transforms = torch.zeros(
                aligned_coords.shape[0], 6, device=aligned_coords.device
            )

        extended = torch.cat([aligned_coords, transforms], dim=-1)
        mu, _ = self.encode_distribution(extended)
        return mu

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent vectors to coordinates and transforms.

        Args:
            z: (N, latent_dim) latent vectors.

        Returns:
            coords: (N, n_atoms, 3) residue coordinates.
            transforms: (N, 6) SE(3) transforms [axis-angle, translation].
        """
        extended = self.decode_flat(z)
        n_coord_dims = self.n_atoms * 3

        coords_flat = extended[:, :n_coord_dims]
        transforms = extended[:, n_coord_dims:]

        coords = coords_flat.reshape(-1, self.n_atoms, 3)
        return coords, transforms

    def sample(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample new conformations with link transforms.

        Returns:
            coords: (N, n_atoms, 3) sampled coordinates.
            transforms: (N, 6) sampled SE(3) transforms.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            return self.decode(z)

    # ─────────────────────────────────────────────────────────────────────────
    # Save/Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model to directory."""
        import json
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        tensors = {k: v.cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(tensors, path / "tensors.safetensors")

        import ciffy
        config = {
            "version": ciffy.__version__,
            "model_type": self._hub_model_type,
            "residue_name": self.residue.name,
            "atom_indices": [int(x) for x in self._atom_indices],
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self._hidden_dims,
            "dropout": self._dropout,
            "use_input_norm": self._use_input_norm,
            "use_residual": self._use_residual,
            "separate_heads": self._separate_heads,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "ResidueVAE":
        """
        Load model from directory.

        Args:
            path: Directory containing saved model.
            device: Device to load model to.

        Returns:
            Loaded ResidueVAE.
        """
        import json
        from safetensors.torch import load_file
        from ciffy.biochemistry import Residue

        path = Path(path)

        tensors = load_file(path / "tensors.safetensors", device=device)
        with open(path / "config.json") as f:
            config = json.load(f)

        residue = getattr(Residue, config["residue_name"])

        model = cls(
            input_dim=config["input_dim"],
            latent_dim=config["latent_dim"],
            hidden_dims=config["hidden_dims"],
            residue=residue,
            atom_indices=config["atom_indices"],
            dropout=config.get("dropout", 0.0),
            use_input_norm=config.get("use_input_norm", False),
            use_residual=config.get("use_residual", False),
            separate_heads=config.get("separate_heads", False),
        ).to(device)

        model.load_state_dict(tensors)
        return model


def _residue_vae_repr(self) -> str:
    return (
        f"ResidueVAE({self.residue.name}, "
        f"atoms={self.n_atoms}, "
        f"latent_dim={self.latent_dim})"
    )


ResidueVAE.__repr__ = _residue_vae_repr


__all__ = ["ResidueVAE", "ResidueVAEConfig"]
