"""
PCA + Flow model for residue conformations.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ciffy.utils import atoms_to_col_map

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue, AtomGroup

# Pre-computed constant for Gaussian log-prob
_LOG_2PI = math.log(2 * math.pi)


# =============================================================================
# Flow Components
# =============================================================================


class ActNorm(nn.Module):
    """
    Activation normalization with data-dependent initialization.

    On the first forward pass, initializes scale and bias to normalize
    the input to zero mean and unit variance. After initialization,
    these become learnable parameters.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))
        # Cache exp(log_scale) to avoid recomputing
        self._cached_scale: torch.Tensor | None = None

    def initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            self.bias.copy_(-x.mean(dim=0))
            # Use correction=0 (biased estimator) to avoid warning when batch_size=1
            # The clamp ensures we never get log(0) even with degenerate batches
            std = x.std(dim=0, correction=0).clamp(min=1e-6)
            self.log_scale.copy_(-torch.log(std))
            self.initialized.fill_(True)
            self._cached_scale = None  # Invalidate cache

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self.initialize(x)
        scale = torch.exp(self.log_scale)
        y = (x + self.bias) * scale
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.exp(-self.log_scale) - self.bias


class CouplingNetwork(nn.Module):
    """MLP for computing scale and translation in coupling layers."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Initialize output near zero for stable training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _rational_quadratic_spline(
    x: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    inverse: bool = False,
    bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rational-quadratic spline transform.

    Based on Neural Spline Flows (Durkan et al., 2019).

    Args:
        x: Input tensor (..., d).
        widths: Unnormalized bin widths (..., d, K).
        heights: Unnormalized bin heights (..., d, K).
        derivatives: Unnormalized derivatives at knots (..., d, K+1).
        inverse: If True, compute inverse transform.
        bound: Symmetric bound for the spline domain [-bound, bound].
        min_bin_width: Minimum bin width.
        min_bin_height: Minimum bin height.
        min_derivative: Minimum derivative at knots.

    Returns:
        y: Transformed tensor (..., d).
        log_det: Log determinant of Jacobian (...,).
    """
    K = widths.shape[-1]

    # Normalize widths and heights to sum to 2*bound with minimum bin sizes
    # After softmax: sums to 1
    # After adjustment: min_bin + (2*bound - K*min_bin) * softmax sums to 2*bound
    widths = torch.softmax(widths, dim=-1)
    widths = min_bin_width + (2 * bound - K * min_bin_width) * widths
    heights = torch.softmax(heights, dim=-1)
    heights = min_bin_height + (2 * bound - K * min_bin_height) * heights

    # Derivatives must be positive
    derivatives = min_derivative + torch.nn.functional.softplus(derivatives)

    # Cumulative widths and heights (knot positions)
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = torch.nn.functional.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (cumwidths - bound)  # Shift to [-bound, bound]

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = torch.nn.functional.pad(cumheights, (1, 0), value=0.0)
    cumheights = (cumheights - bound)

    # Handle values outside the spline domain with identity
    inside_mask = (x >= -bound) & (x <= bound)

    # Find which bin each x falls into
    # For inverse, use cumheights since input is y-space
    x_clamped = torch.clamp(x, -bound + 1e-6, bound - 1e-6)
    if inverse:
        bin_idx = torch.searchsorted(cumheights[..., 1:].contiguous(), x_clamped.unsqueeze(-1))
    else:
        bin_idx = torch.searchsorted(cumwidths[..., 1:].contiguous(), x_clamped.unsqueeze(-1))
    bin_idx = bin_idx.squeeze(-1).clamp(0, K - 1)

    # Gather bin parameters
    # Input is always 2D (batch, dim) from coupling layers
    n_batch = x.shape[0]
    d = x.shape[-1]

    # Index into each dimension
    idx_expanded = bin_idx.unsqueeze(-1)  # (n_batch, d, 1)
    w = widths.gather(-1, idx_expanded).squeeze(-1)  # (n_batch, d)
    h = heights.gather(-1, idx_expanded).squeeze(-1)
    xk = cumwidths.gather(-1, idx_expanded).squeeze(-1)
    yk = cumheights.gather(-1, idx_expanded).squeeze(-1)
    dk = derivatives.gather(-1, idx_expanded).squeeze(-1)
    dk1 = derivatives.gather(-1, idx_expanded + 1).squeeze(-1)

    # Slope of the linear segment
    s = h / w

    if inverse:
        # Inverse transform: given y, find x
        y = x_clamped
        y_rel = y - yk

        # Quadratic coefficients for inverse
        a = h * (s - dk) + y_rel * (dk + dk1 - 2 * s)
        b = h * dk - y_rel * (dk + dk1 - 2 * s)
        c = -s * y_rel

        # Solve quadratic: xi = (-b + sqrt(b^2 - 4ac)) / (2a)
        discriminant = b ** 2 - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0)  # Numerical stability
        xi = (-b + torch.sqrt(discriminant)) / (2 * a + 1e-8)
        xi = torch.clamp(xi, 0, 1)

        x_out = xi * w + xk

        # Log derivative (inverse of forward)
        numerator = s ** 2 * (dk1 * xi ** 2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
        denominator = (s + (dk + dk1 - 2 * s) * xi * (1 - xi)) ** 2
        log_det = -torch.log(numerator / (denominator + 1e-8) + 1e-8).sum(dim=-1)
    else:
        # Forward transform: given x, find y
        xi = (x_clamped - xk) / w

        # Rational quadratic formula
        numerator = h * (s * xi ** 2 + dk * xi * (1 - xi))
        denominator = s + (dk + dk1 - 2 * s) * xi * (1 - xi)
        y_out = yk + numerator / (denominator + 1e-8)

        # Log derivative
        numerator_deriv = s ** 2 * (dk1 * xi ** 2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
        log_det = torch.log(numerator_deriv / (denominator ** 2 + 1e-8) + 1e-8).sum(dim=-1)
        x_out = y_out

    # Apply identity for values outside domain
    x_out = torch.where(inside_mask, x_out, x)
    # log_det already has shape (n_batch,) after sum(dim=-1)

    return x_out, log_det


class SplineCoupling(nn.Module):
    """
    Neural spline coupling layer using rational-quadratic splines.

    More expressive than affine coupling - can learn arbitrary monotonic
    transformations within each bin. Uses the same amount of compute per
    layer but captures more complex distributions.

    Based on Neural Spline Flows (Durkan et al., 2019).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        even_mask: bool = True,
        n_bins: int = 8,
        bound: float = 3.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        self.bound = bound

        # Pre-compute integer indices instead of boolean mask (faster indexing)
        mask = torch.arange(dim) % 2 == (0 if even_mask else 1)
        self.register_buffer("masked_idx", torch.where(mask)[0])
        self.register_buffer("unmasked_idx", torch.where(~mask)[0])

        n_masked = len(self.masked_idx)
        n_unmasked = len(self.unmasked_idx)
        self.n_unmasked = n_unmasked

        # Output: widths (K), heights (K), derivatives (K+1) per unmasked dim
        n_params = n_unmasked * (3 * n_bins + 1)
        self.net = CouplingNetwork(n_masked, n_params, hidden_dim)

    def _get_spline_params(self, context: torch.Tensor):
        """Extract spline parameters from network output."""
        params = self.net(context)
        K = self.n_bins

        # Reshape to (batch, n_unmasked, 3K+1)
        params = params.reshape(-1, self.n_unmasked, 3 * K + 1)

        widths = params[..., :K]
        heights = params[..., K:2*K]
        derivatives = params[..., 2*K:]

        return widths, heights, derivatives

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use index_select for faster indexing than boolean masks
        x_a = x.index_select(1, self.masked_idx)
        x_b = x.index_select(1, self.unmasked_idx)

        widths, heights, derivatives = self._get_spline_params(x_a)

        y_b, log_det = _rational_quadratic_spline(
            x_b, widths, heights, derivatives,
            inverse=False, bound=self.bound
        )

        # Scatter results back - use index_copy_ for efficiency
        y = x.clone()  # Start with x (preserves x_a in place)
        y.scatter_(1, self.unmasked_idx.expand(x.shape[0], -1), y_b)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_a = y.index_select(1, self.masked_idx)
        y_b = y.index_select(1, self.unmasked_idx)

        widths, heights, derivatives = self._get_spline_params(y_a)

        x_b, _ = _rational_quadratic_spline(
            y_b, widths, heights, derivatives,
            inverse=True, bound=self.bound
        )

        x = y.clone()
        x.scatter_(1, self.unmasked_idx.expand(y.shape[0], -1), x_b)
        return x


# =============================================================================
# PCA + Flow Model
# =============================================================================


class PCAFlow(nn.Module):
    """
    PCA for dimensionality reduction + normalizing flow for density estimation.

    Uses neural spline flows (rational-quadratic splines) for expressive
    density modeling with good Gaussianity properties.

    The model is exactly invertible: decode(encode(x)) reconstructs x
    with error bounded only by PCA truncation.

    Args:
        V: PCA components matrix (k, d) where k is latent dim, d is coord dim.
        mean: Mean coordinates (d,).
        n_layers: Number of flow layers (ActNorm + SplineCoupling pairs).
        hidden_dim: Hidden dimension in coupling networks.
        bound: Tanh bound (in std devs) for decode(). None (default) disables
               bounding, preserving exact invertibility.
        n_bins: Number of spline bins (default 8).
        spline_bound: Spline domain bound (default 3.0).
    """

    def __init__(
        self,
        V: torch.Tensor,
        mean: torch.Tensor,
        n_layers: int = 4,
        hidden_dim: int = 56,
        bound: float | None = None,
        n_bins: int = 8,
        spline_bound: float = 3.0,
    ):
        super().__init__()
        self.k = V.shape[0]  # Latent dimension
        self.d = V.shape[1]  # Coordinate dimension (n_atoms * 3 + 6)
        self.bound = bound

        # PCA parameters (fixed, not learned)
        self.register_buffer("V", V)
        self.register_buffer("mean", mean)

        # Flow layers: alternating ActNorm + SplineCoupling
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ActNorm(self.k))
            self.layers.append(SplineCoupling(
                self.k, hidden_dim, even_mask=(i % 2 == 0),
                n_bins=n_bins, bound=spline_bound
            ))

    def coords_to_pca(self, x: torch.Tensor) -> torch.Tensor:
        """Project coordinates to PCA space."""
        flat = x.reshape(-1, self.d)
        return (flat - self.mean) @ self.V.T

    def pca_to_flat(self, pca: torch.Tensor) -> torch.Tensor:
        """Reconstruct flat coordinates from PCA (approximate due to truncation)."""
        return pca @ self.V + self.mean

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode coordinates to latent space.

        Args:
            x: Coordinates (N, d) flat representation.

        Returns:
            z: Latent vectors (N, k).
            log_det: Log Jacobian determinant (N,).
        """
        h = self.coords_to_pca(x)
        log_det = torch.zeros(h.shape[0], device=h.device)

        for layer in self.layers:
            h, ld = layer(h)
            log_det = log_det + ld

        return h, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to flat coordinates.

        Args:
            z: Latent vectors (N, k).

        Returns:
            Flat coordinates (N, d).
        """
        if self.bound is not None:
            z = self.bound * torch.tanh(z / self.bound)

        h = z
        for layer in reversed(self.layers):
            h = layer.inverse(h)
        return self.pca_to_flat(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates to latent space."""
        z, _ = self.forward(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to flat coordinates (N, d)."""
        return self.inverse(z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of coordinates under the model."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + _LOG_2PI).sum(dim=-1)
        return log_pz + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample new flat coordinates from the learned distribution."""
        z = torch.randn(n_samples, self.k, device=self.V.device)
        return self.decode(z)


# =============================================================================
# JIT Decoder
# =============================================================================


class _JITDecoder(nn.Module):
    """
    JIT-compatible decoder wrapper.

    Extracts just the decode path with bound baked in as a constant,
    avoiding conditionals that complicate tracing.
    """

    def __init__(self, flow: PCAFlow, n_atoms: int):
        super().__init__()
        self.k = flow.k
        self.n_atoms = n_atoms
        self.bound: float | None = flow.bound

        # Copy buffers
        self.register_buffer("V", flow.V.clone())
        self.register_buffer("mean", flow.mean.clone())

        # Copy layers in reverse order (TorchScript doesn't support reversed())
        import copy
        self.layers = nn.ModuleList([
            copy.deepcopy(layer) for layer in reversed(list(flow.layers))
        ])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vectors to (coords, transforms)."""
        if self.bound is not None:
            z = self.bound * torch.tanh(z / self.bound)

        h = z
        for layer in self.layers:
            h = layer.inverse(h)

        # PCA to flat coords
        flat = h @ self.V + self.mean

        # Split into coords and transforms
        n_coord_dims = self.n_atoms * 3
        coords_flat = flat[:, :n_coord_dims]
        transforms = flat[:, n_coord_dims:]
        coords = coords_flat.reshape(-1, self.n_atoms, 3)

        return coords, transforms


# =============================================================================
# ResidueFlowModel
# =============================================================================


@dataclass
class ResidueFlowConfig:
    """Configuration for ResidueFlowModel."""

    latent_dim: int = 12
    n_layers: int = 8
    hidden_dim: int = 64
    bound: float | None = None
    min_coverage: float = 0.9


class ResidueFlowModel(nn.Module):
    """
    Residue flow model that captures conformation and backbone link geometry.

    This model learns the joint distribution of residue coordinates AND
    the SE(3) transform to the next residue in the chain. This enables
    sampling residue conformations with realistic backbone connectivity.

    The representation is: [coords_flat (n_atoms*3), transform (6)]
    where transform = [axis-angle (3), translation (3)] defines the relative
    position and orientation of the next residue's P atom.

    As an nn.Module subclass, this model can be composed into larger networks
    and its parameters will be automatically included in optimizer updates.

    Attributes:
        flow: The underlying PCAFlow model (registered as submodule).
        residue: The source residue type.
        atoms: AtomGroup subset containing the atoms used.
        n_atoms: Number of atoms per residue.

    Example:
        >>> model = ResidueFlowModel.from_structures(cif_paths, Residue.A)
        >>> coords, transform = model.decode(z)
        >>> # Position next residue using the transform
        >>> from ciffy.nn.residue_flow import position_next_residue
        >>> coords2 = position_next_residue(coords, ref_coords, transform, atoms, residue)

        # Can be used as part of a larger module:
        >>> class MolecularModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.residue_flow = ResidueFlowModel.load("model_dir")
        ...         self.head = nn.Linear(12, 64)
        >>> # Parameters from residue_flow are automatically included
        >>> optimizer = Adam(molecular_model.parameters())
    """

    def __init__(
        self,
        flow: PCAFlow,
        residue: "Residue",
        atom_indices: list[int],
        n_atoms: int,
        jit: bool = False,
    ):
        super().__init__()
        # Register flow as submodule - parameters auto-included in .parameters()
        self.flow = flow

        # Non-tensor metadata (not registered as buffers/parameters)
        self.residue = residue
        self._atom_indices = atom_indices
        self.n_atoms = n_atoms
        self._atoms_group: "AtomGroup | None" = None
        self._jit_decoder: torch.jit.ScriptModule | None = None

        # Cached geometry projector (built lazily, invalidated on device change)
        self._geometry_projector: callable | None = None
        self._geometry_projector_device: torch.device | None = None

        # Pre-resolve frame column indices for fast frame computation
        self._init_frame_indices()

        if jit:
            self._compile_jit()

    def _init_frame_indices(self) -> None:
        """
        Pre-resolve frame column indices from atom names.

        This converts the string-based FrameDefinition to integer indices
        once at model initialization, enabling fast vectorized frame
        computation at runtime without Python attribute lookups.
        """
        from ciffy.biochemistry.linking import LINKING_BY_TYPE

        atom_to_col = atoms_to_col_map(self._atom_indices)
        link_def = LINKING_BY_TYPE.get(self.residue.molecule_type)

        if link_def is not None:
            # Pre-resolve prev frame (outgoing) indices
            self._prev_frame_cols = link_def.prev_frame.resolve(
                self.residue, atom_to_col
            )
            self._prev_z_toward_origin = link_def.prev_frame.z_toward_origin

            # Pre-resolve next frame (incoming) indices
            self._next_frame_cols = link_def.next_frame.resolve(
                self.residue, atom_to_col
            )
            self._next_z_toward_origin = link_def.next_frame.z_toward_origin
        else:
            # Non-polymer residue types (ligands, etc.) - no linking
            self._prev_frame_cols = None
            self._next_frame_cols = None
            self._prev_z_toward_origin = True
            self._next_z_toward_origin = True

    def _compile_jit(self) -> None:
        """Compile the decoder to TorchScript for faster inference."""
        self.flow.eval()
        decoder = _JITDecoder(self.flow, self.n_atoms)
        decoder.eval()
        self._jit_decoder = torch.jit.script(decoder)

    @property
    def is_jit(self) -> bool:
        """Whether the decoder is JIT-compiled."""
        return self._jit_decoder is not None

    @property
    def atoms(self) -> "AtomGroup":
        """AtomGroup subset containing the atoms used by this model."""
        if self._atoms_group is None:
            self._atoms_group = self.residue.subset(set(self._atom_indices))
        return self._atoms_group

    # ─────────────────────────────────────────────────────────────────────────
    # Frame Properties (for inter-residue positioning)
    # ─────────────────────────────────────────────────────────────────────────
    # These expose pre-resolved frame column indices for use by PolymerFlowModel.
    # TODO: Consider extracting frame computation to a geometry helper module
    # rather than coupling it with flow models.

    @property
    def prev_frame_cols(self) -> tuple[int, int, int | None] | None:
        """
        Pre-resolved column indices for the outgoing (prev) frame.

        Returns (origin_col, z_ref_col, perp_ref_col) or None if not a polymer.
        Used to compute the coordinate frame at the linking atom (e.g., O3' or C).
        """
        return self._prev_frame_cols

    @property
    def next_frame_cols(self) -> tuple[int, int, int | None] | None:
        """
        Pre-resolved column indices for the incoming (next) frame.

        Returns (origin_col, z_ref_col, perp_ref_col) or None if not a polymer.
        Used to compute the coordinate frame at the linking atom (e.g., P or N).
        """
        return self._next_frame_cols

    @property
    def prev_z_toward_origin(self) -> bool:
        """Whether the Z-axis points toward the origin for the outgoing frame."""
        return self._prev_z_toward_origin

    @property
    def next_z_toward_origin(self) -> bool:
        """Whether the Z-axis points toward the origin for the incoming frame."""
        return self._next_z_toward_origin

    @classmethod
    def from_structures(
        cls,
        cif_paths: list[Path],
        residue: "Residue",
        config: ResidueFlowConfig | None = None,
        n_epochs: int = 200,
        device: str = "cpu",
        verbose: bool = True,
    ) -> "ResidueFlowModel":
        """
        Train a model from CIF structures.

        Args:
            cif_paths: List of paths to CIF files.
            residue: Residue type to extract.
            config: Model configuration.
            n_epochs: Number of training epochs.
            device: Device to train on.
            verbose: Print progress.

        Returns:
            Trained ResidueFlowModel.
        """
        from .data import extract_residues_with_links
        from .train import train_pca_flow

        if config is None:
            config = ResidueFlowConfig()

        # Extract residues with link transforms
        if verbose:
            print(f"Extracting {residue.name} residues with links...")
        coords, transforms, atoms = extract_residues_with_links(
            cif_paths, residue, min_coverage=config.min_coverage, verbose=verbose
        )

        n_instances = len(coords)
        n_atoms = len(atoms)

        if verbose:
            print(f"Dataset: {n_instances} instances, {n_atoms} atoms")

        # Create extended representation (coords + SE(3) transforms)
        coords_flat = coords.reshape(n_instances, -1)
        extended = np.concatenate([coords_flat, transforms], axis=1)

        if verbose:
            print(f"Extended representation: {extended.shape[1]} dims ({n_atoms}*3 + 6)")

        # Train flow model
        flow, info = train_pca_flow(
            extended,
            latent_dim=config.latent_dim,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            bound=config.bound,
            n_epochs=n_epochs,
            device=device,
            verbose=verbose,
        )

        return cls(
            flow=flow,
            residue=residue,
            atom_indices=atoms,
            n_atoms=n_atoms,
        )

    def encode(
        self,
        coords: "torch.Tensor",
        transforms: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """
        Encode coordinates and transforms to latent space.

        Args:
            coords: (N, n_atoms, 3) or (N, n_atoms*3) coordinates.
            transforms: (N, 6) SE(3) transforms. If None, uses zeros.

        Returns:
            (N, latent_dim) latent vectors.
        """
        # Flatten coords if needed
        if coords.dim() == 3:
            coords = coords.reshape(coords.shape[0], -1)

        # Add transforms
        if transforms is None:
            transforms = torch.zeros(coords.shape[0], 6, device=coords.device)

        extended = torch.cat([coords, transforms], dim=-1)
        return self.flow.encode(extended)

    def decode(
        self,
        z: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Decode latent vectors to coordinates and transforms.

        Args:
            z: (N, latent_dim) latent vectors.

        Returns:
            coords: (N, n_atoms, 3) residue coordinates.
            transforms: (N, 6) SE(3) transforms [axis-angle, translation].
        """
        if self._jit_decoder is not None:
            return self._jit_decoder(z)

        extended = self.flow.decode(z)
        n_coord_dims = self.n_atoms * 3

        coords_flat = extended[:, :n_coord_dims]
        transforms = extended[:, n_coord_dims:]

        coords = coords_flat.reshape(-1, self.n_atoms, 3)
        return coords, transforms

    def sample(self, n_samples: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Sample new conformations with link transforms.

        Returns:
            coords: (N, n_atoms, 3) sampled coordinates.
            transforms: (N, 6) sampled SE(3) transforms.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.flow.k, device=self.flow.V.device)
            return self.decode(z)

    def save(self, path: str | Path) -> None:
        """Save model to directory."""
        import json
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        tensors = {k: v.cpu().contiguous() for k, v in self.flow.state_dict().items()}
        save_file(tensors, path / "tensors.safetensors")

        import ciffy
        config = {
            "version": ciffy.__version__,
            "residue_name": self.residue.name,
            "atom_indices": [int(x) for x in self._atom_indices],
            "n_atoms": self.n_atoms,
            "n_layers": len(self.flow.layers) // 2,
            "hidden_dim": self.flow.layers[1].net.net[0].out_features,
            "bound": float(self.flow.bound) if self.flow.bound is not None else None,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        jit: bool = False,
    ) -> "ResidueFlowModel":
        """
        Load model from directory.

        Args:
            path: Directory containing saved model.
            device: Device to load model to.
            jit: Whether to JIT-compile the decoder for faster inference.

        Returns:
            Loaded ResidueFlowModel.
        """
        import json
        from safetensors.torch import load_file
        from ciffy.biochemistry import Residue

        path = Path(path)

        tensors = load_file(path / "tensors.safetensors", device=device)
        with open(path / "config.json") as f:
            config = json.load(f)

        V = tensors["V"].float()
        mean = tensors["mean"].float()

        flow = PCAFlow(
            V, mean,
            n_layers=config["n_layers"],
            hidden_dim=config["hidden_dim"],
            bound=config["bound"],
        ).to(device)
        flow.load_state_dict(tensors)

        residue = getattr(Residue, config["residue_name"])

        return cls(
            flow=flow,
            residue=residue,
            atom_indices=config["atom_indices"],
            n_atoms=config["n_atoms"],
            jit=jit,
        )

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self.flow.k

    # ─────────────────────────────────────────────────────────────────────────
    # Device Management
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Device where model parameters reside."""
        return self.flow.V.device

    def _apply(self, fn):
        """
        Override _apply to clear cached state when moving devices.

        This is called by to(), cuda(), cpu(), etc.
        """
        # Clear cached projector - will be rebuilt for new device
        self._geometry_projector = None
        self._geometry_projector_device = None

        # Move JIT decoder if present
        if self._jit_decoder is not None:
            self._jit_decoder = self._jit_decoder._apply(fn)

        return super()._apply(fn)

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry Projection
    # ─────────────────────────────────────────────────────────────────────────

    def _get_geometry_projector(self, device: torch.device) -> callable:
        """
        Get cached geometry projector for device, building if needed.

        The projector is cached and reused for subsequent calls on the same
        device. Moving to a different device invalidates the cache.
        """
        if (self._geometry_projector is None or
                self._geometry_projector_device != device):
            self._geometry_projector = self._build_geometry_projector(device)
            self._geometry_projector_device = device
        return self._geometry_projector

    def _build_geometry_projector(self, device: torch.device) -> callable:
        """
        Build a Newton projector for bond length constraints.

        Uses the residue's bond definitions and ideal coordinates to compute
        target bond lengths. Only includes bonds where both atoms are present
        in the model's atom subset.

        Returns a function that projects coordinates onto ideal bond lengths
        using Gauss-Newton optimization. This preserves conformational diversity
        while fixing local geometry errors.
        """
        residue = self.residue
        atom_indices = self._atom_indices
        n_atoms = len(atom_indices)

        # Map from global atom value to local index in this model's subset
        global_to_model_local = {a: i for i, a in enumerate(atom_indices)}
        atom_set = set(atom_indices)

        # Get ideal coordinates and bonds from residue definition
        ideal_coords = residue.ideal  # (n_residue_atoms, 3)
        bonds = residue.bonds  # (n_bonds, 2) numpy array with local indices

        # Build mapping: residue local index -> global atom value
        local_to_global = {atom.local: int(atom) for atom in residue}

        # Build bond constraint data from residue's bond definitions
        # Only include bonds where both atoms are in our subset
        bond_pairs = []
        bond_targets = []

        for local1, local2 in bonds:
            # Map residue local indices to global atom values
            global1 = local_to_global[int(local1)]
            global2 = local_to_global[int(local2)]

            if global1 in atom_set and global2 in atom_set:
                # Get local indices in the model's atom subset
                model_i = global_to_model_local[global1]
                model_j = global_to_model_local[global2]
                bond_pairs.append((model_i, model_j))

                # Compute ideal bond length from ideal coordinates
                # Use residue local indices for ideal_coords access
                pos1 = ideal_coords[local1]
                pos2 = ideal_coords[local2]
                ideal_length = float(np.linalg.norm(pos2 - pos1))
                bond_targets.append(ideal_length)

        n_bonds = len(bond_targets)

        if n_bonds == 0:
            # No bonds to project - return identity function
            def identity(coords: torch.Tensor) -> torch.Tensor:
                return coords
            return identity

        # Pre-move constraint tensors to target device (cached for reuse)
        bond_pairs_d = torch.tensor(bond_pairs, device=device, dtype=torch.long)
        bond_targets_d = torch.tensor(bond_targets, device=device, dtype=torch.float32)

        # Pre-compute constant index tensors
        bond_idx = torch.arange(n_bonds, device=device)
        dims = torch.arange(3, device=device)

        def newton_step(coords: torch.Tensor) -> torch.Tensor:
            """Single vectorized Newton step. coords: (n_atoms, 3)"""
            residuals = torch.zeros(n_bonds, device=device)
            J = torch.zeros(n_bonds, n_atoms * 3, device=device)

            # Vectorized bond length constraints
            a1_idx = bond_pairs_d[:, 0]  # (n_bonds,)
            a2_idx = bond_pairs_d[:, 1]  # (n_bonds,)
            diff = coords[a2_idx] - coords[a1_idx]  # (n_bonds, 3)
            lengths = torch.norm(diff, dim=1)  # (n_bonds,)
            residuals = lengths - bond_targets_d
            units = diff / (lengths.unsqueeze(1) + 1e-8)  # (n_bonds, 3)

            # Build Jacobian for bonds (fully vectorized)
            row_idx = bond_idx.unsqueeze(1).expand(-1, 3).reshape(-1)
            col_a1 = (a1_idx.unsqueeze(1) * 3 + dims).reshape(-1)
            col_a2 = (a2_idx.unsqueeze(1) * 3 + dims).reshape(-1)
            J[row_idx, col_a1] = -units.reshape(-1)
            J[row_idx, col_a2] = units.reshape(-1)

            # Gauss-Newton: dx = -J^T @ (J @ J^T)^{-1} @ residuals
            JJT = J @ J.T
            y = torch.linalg.solve(JJT, residuals)
            dx = -J.T @ y

            return coords + dx.reshape(n_atoms, 3)

        return newton_step

    def project_geometry(
        self,
        coords: "torch.Tensor",
        n_steps: int = 2,
    ) -> "torch.Tensor":
        """
        Project coordinates onto ideal bond length and angle constraints.

        Uses Gauss-Newton optimization to correct local geometry while
        preserving overall conformation. Typically 2 steps are sufficient
        for sub-0.01Å bond length accuracy.

        The geometry projector is cached per device for efficiency.

        Args:
            coords: (N, n_atoms, 3) or (n_atoms, 3) coordinates.
            n_steps: Number of Newton steps (default 2).

        Returns:
            Projected coordinates with same shape as input.
        """
        newton_step = self._get_geometry_projector(coords.device)

        single = coords.dim() == 2
        if single:
            coords = coords.unsqueeze(0)

        projected = []
        for i in range(coords.shape[0]):
            c = coords[i]
            for _ in range(n_steps):
                c = newton_step(c)
            projected.append(c)

        result = torch.stack(projected)
        return result[0] if single else result

    def __repr__(self) -> str:
        return (
            f"ResidueFlowModel({self.residue.name}, "
            f"atoms={self.n_atoms}, "
            f"latent_dim={self.flow.k})"
        )
