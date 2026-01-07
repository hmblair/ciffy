"""
PCA + Quantile Spline residue model.

A simple, fast residue encoder that:
1. Projects coordinates to PCA space (linear dimensionality reduction)
2. Applies per-dimension quantile splines to Gaussianize marginals

Properties:
- Fast: O(1) encode/decode (matrix multiply + spline lookup)
- Differentiable: Both PCA and splines are differentiable
- N(0,1) marginals: Each latent dimension has standard normal marginal
- Good reconstruction: <0.1Å RMSD with sufficient dimensions

Note: Sampling from N(0,1) independently produces poor geometry because
dimensions have nonlinear dependencies. This model is designed for use
with an autoregressive model that learns the proper joint distribution.

Example:
    >>> from ciffy.nn.pca_quantile import PCAQuantileResidueModel, fit_pca_quantile
    >>> from ciffy.nn.polymer import PolymerModel
    >>>
    >>> # Fit models for each residue type
    >>> models = {}
    >>> for res in [Residue.A, Residue.C, Residue.G, Residue.U]:
    ...     models[res] = fit_pca_quantile(cif_paths, res, latent_dim=12)
    >>>
    >>> # Use with PolymerModel
    >>> polymer_model = PolymerModel(models)
    >>> latents = polymer_model.encode_polymer(polymer)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


@dataclass
class PCAQuantileConfig:
    """Configuration for PCA + Quantile Spline model."""

    latent_dim: int = 12
    """Number of PCA components / latent dimensions."""

    n_knots: int = 32
    """Number of knots for quantile splines."""

    transform_scale: float = 1.0
    """Scale factor for SE(3) transforms during training."""


class QuantileSpline(nn.Module):
    """
    Monotonic spline for marginal Gaussianization.

    Fits a piecewise linear function that maps empirical quantiles
    to N(0,1) quantiles. Differentiable for gradient-based optimization.
    """

    def __init__(self, n_knots: int = 32):
        super().__init__()
        self.n_knots = n_knots
        self.register_buffer("x_knots", torch.zeros(n_knots))
        self.register_buffer("y_knots", torch.zeros(n_knots))

    def fit(self, data: torch.Tensor) -> None:
        """Fit spline to match empirical quantiles to N(0,1)."""
        n = len(data)
        sorted_data, _ = torch.sort(data)
        probs = torch.linspace(
            0.5 / self.n_knots, 1 - 0.5 / self.n_knots, self.n_knots
        )
        indices = (probs * n).long().clamp(0, n - 1)
        self.x_knots = sorted_data[indices].clone()
        self.y_knots = torch.tensor(
            stats.norm.ppf(probs.numpy()), dtype=torch.float32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to N(0,1) space."""
        idx = torch.searchsorted(self.x_knots[:-1], x.contiguous())
        idx = idx.clamp(1, self.n_knots - 1)

        x0 = self.x_knots[idx - 1]
        x1 = self.x_knots[idx]
        y0 = self.y_knots[idx - 1]
        y1 = self.y_knots[idx]

        t = (x - x0) / (x1 - x0 + 1e-8)
        t = t.clamp(0, 1)
        return y0 + t * (y1 - y0)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Transform from N(0,1) space back to data space."""
        idx = torch.searchsorted(self.y_knots[:-1], y.contiguous())
        idx = idx.clamp(1, self.n_knots - 1)

        y0 = self.y_knots[idx - 1]
        y1 = self.y_knots[idx]
        x0 = self.x_knots[idx - 1]
        x1 = self.x_knots[idx]

        t = (y - y0) / (y1 - y0 + 1e-8)
        t = t.clamp(0, 1)
        return x0 + t * (x1 - x0)


class _AtomSet:
    """Wrapper for atom indices to match PolymerModel interface."""

    def __init__(self, atom_indices: np.ndarray):
        self._indices = atom_indices

    def index(self) -> list[int]:
        return self._indices.tolist()


class PCAQuantileResidueModel(nn.Module):
    """
    PCA + Quantile Spline residue model.

    Encodes residue conformations to a latent space with N(0,1) marginals.
    Uses PCA for linear dimensionality reduction, then per-dimension
    quantile splines for Gaussianization.

    Args:
        V: PCA projection matrix (latent_dim, data_dim).
        mean: Data mean for centering (data_dim,).
        n_atoms: Number of atoms in this residue type.
        atom_indices: Array of atom indices used.
        residue: The Residue type this model handles.
        config: Model configuration.
    """

    _hub_model_type = "pca_quantile"

    def __init__(
        self,
        V: torch.Tensor,
        mean: torch.Tensor,
        n_atoms: int,
        atom_indices: np.ndarray,
        residue: "Residue",
        config: PCAQuantileConfig | None = None,
    ):
        super().__init__()

        if config is None:
            config = PCAQuantileConfig(latent_dim=V.shape[0])

        self.config = config
        self.register_buffer("V", V)
        self.register_buffer("mean", mean)
        self._n_atoms = n_atoms
        self._atom_indices = atom_indices
        self._residue = residue

        self.splines = nn.ModuleList([
            QuantileSpline(n_knots=config.n_knots)
            for _ in range(config.latent_dim)
        ])

        # For PolymerModel compatibility
        self.atoms = _AtomSet(atom_indices)

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this residue type."""
        return self._n_atoms

    @property
    def residue(self) -> "Residue":
        """The Residue type this model handles."""
        return self._residue

    @property
    def latent_dim(self) -> int:
        """Latent space dimensionality."""
        return self.config.latent_dim

    @property
    def device(self) -> torch.device:
        """Device where model parameters reside."""
        return self.V.device

    @property
    def transform_scale(self) -> float:
        """Scale factor for transforms."""
        return self.config.transform_scale

    def fit(self, data: torch.Tensor) -> None:
        """
        Fit quantile splines to training data.

        Args:
            data: (N, n_atoms*3 + 6) flattened coordinates + transforms.
        """
        # Apply transform scaling
        if self.transform_scale != 1.0:
            n_coord_dims = self._n_atoms * 3
            data = data.clone()
            data[:, n_coord_dims:] = data[:, n_coord_dims:] * self.transform_scale

        pca_latent = (data - self.mean) @ self.V.T
        for i, spline in enumerate(self.splines):
            spline.fit(pca_latent[:, i])

    def encode(self, coords: torch.Tensor, transforms: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encode coordinates to latent space.

        Args:
            coords: (N, n_atoms, 3) or (N, n_atoms*3) coordinates.
            transforms: (N, 6) SE(3) transforms. If None, uses zeros.

        Returns:
            (N, latent_dim) latent vectors with N(0,1) marginals.
        """
        # Flatten coords if needed
        if coords.dim() == 3:
            coords = coords.reshape(coords.shape[0], -1)

        # Add transforms
        if transforms is None:
            transforms = torch.zeros(coords.shape[0], 6, device=coords.device)

        # Scale transforms
        if self.transform_scale != 1.0:
            transforms = transforms * self.transform_scale

        data = torch.cat([coords, transforms], dim=-1)
        pca_latent = (data - self.mean) @ self.V.T

        z = torch.stack([
            self.splines[i](pca_latent[:, i])
            for i in range(self.latent_dim)
        ], dim=1)

        return z

    def decode(
        self, z: torch.Tensor, project: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent vectors to coordinates and transforms.

        Args:
            z: (N, latent_dim) latent vectors.
            project: If True, project coordinates to satisfy bond constraints.

        Returns:
            coords: (N, n_atoms, 3) residue coordinates.
            transforms: (N, 6) SE(3) transforms.
        """
        # Inverse spline transform
        pca_latent = torch.stack([
            self.splines[i].inverse(z[:, i])
            for i in range(self.latent_dim)
        ], dim=1)

        # Inverse PCA
        flat = pca_latent @ self.V + self.mean

        n_coord_dims = self._n_atoms * 3
        coords = flat[:, :n_coord_dims].reshape(-1, self._n_atoms, 3)
        transforms = flat[:, n_coord_dims:]

        # Unscale transforms
        if self.transform_scale != 1.0:
            transforms = transforms / self.transform_scale

        if project:
            coords = self.project_geometry(coords)

        return coords, transforms

    def sample(
        self, n_samples: int, temperature: float = 1.0, project: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from N(0,1) prior.

        Note: Independent sampling produces poor geometry because dimensions
        have nonlinear dependencies. Use with an autoregressive model for
        proper sampling.

        Args:
            n_samples: Number of samples to generate.
            temperature: Scales the latent noise (default 1.0).
            project: If True, project coordinates to satisfy bond constraints.

        Returns:
            coords: (N, n_atoms, 3) sampled coordinates.
            transforms: (N, 6) sampled SE(3) transforms.
        """
        z = torch.randn(n_samples, self.latent_dim, device=self.device) * temperature
        return self.decode(z, project=project)

    def project_geometry(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Project coordinates to satisfy bond length constraints.

        TODO: Implement bond projection.

        Args:
            coords: (N, n_atoms, 3) coordinates.

        Returns:
            (N, n_atoms, 3) projected coordinates.
        """
        # Placeholder - return unchanged for now
        return coords

    def save(self, path: str | Path) -> None:
        """
        Save model to directory.

        Args:
            path: Directory to save model.
        """
        from safetensors.torch import save_file
        import ciffy

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        tensors = {}
        tensors["V"] = self.V.cpu().contiguous()
        tensors["mean"] = self.mean.cpu().contiguous()
        for i, spline in enumerate(self.splines):
            tensors[f"spline_{i}_x_knots"] = spline.x_knots.cpu().contiguous()
            tensors[f"spline_{i}_y_knots"] = spline.y_knots.cpu().contiguous()

        save_file(tensors, path / "tensors.safetensors")

        # Save config
        config = {
            "version": ciffy.__version__,
            "model_type": self._hub_model_type,
            "residue_name": self._residue.name,
            "atom_indices": [int(x) for x in self._atom_indices],
            "n_atoms": self._n_atoms,
            "latent_dim": self.config.latent_dim,
            "n_knots": self.config.n_knots,
            "transform_scale": self.config.transform_scale,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "PCAQuantileResidueModel":
        """
        Load model from directory.

        Args:
            path: Directory containing saved model.
            device: Device to load model to.

        Returns:
            Loaded PCAQuantileResidueModel.
        """
        from safetensors.torch import load_file
        from ciffy.biochemistry import Residue

        path = Path(path)

        with open(path / "config.json") as f:
            config_dict = json.load(f)

        tensors = load_file(path / "tensors.safetensors", device=device)

        residue = getattr(Residue, config_dict["residue_name"])
        atom_indices = np.array(config_dict["atom_indices"], dtype=np.int64)

        config = PCAQuantileConfig(
            latent_dim=config_dict["latent_dim"],
            n_knots=config_dict["n_knots"],
            transform_scale=config_dict.get("transform_scale", 1.0),
        )

        model = cls(
            V=tensors["V"],
            mean=tensors["mean"],
            n_atoms=config_dict["n_atoms"],
            atom_indices=atom_indices,
            residue=residue,
            config=config,
        )

        # Load spline knots
        for i, spline in enumerate(model.splines):
            spline.x_knots = tensors[f"spline_{i}_x_knots"]
            spline.y_knots = tensors[f"spline_{i}_y_knots"]

        return model


def fit_pca_quantile(
    cif_paths: list[str | Path],
    residue: "Residue",
    latent_dim: int = 12,
    n_knots: int = 32,
    transform_scale: float = 1.0,
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> PCAQuantileResidueModel:
    """
    Fit a PCA + Quantile Spline model for a residue type.

    Args:
        cif_paths: List of CIF file paths for training data.
        residue: Residue type to fit (e.g., Residue.A).
        latent_dim: Number of PCA components.
        n_knots: Number of knots for quantile splines.
        transform_scale: Scale factor for SE(3) transforms.
        min_coverage: Minimum atom coverage to include a residue.
        verbose: Whether to print progress.

    Returns:
        Fitted PCAQuantileResidueModel.

    Example:
        >>> from ciffy.nn.pca_quantile import fit_pca_quantile
        >>> from ciffy.biochemistry import Residue
        >>>
        >>> model = fit_pca_quantile(
        ...     cif_paths=["data/*.cif"],
        ...     residue=Residue.A,
        ...     latent_dim=12,
        ... )
        >>> model.save("models/A")
    """
    from ciffy.nn.flow.residue.data import extract_residues_with_links, compute_pca
    from ciffy.operations.metrics import rmsd

    if verbose:
        print(f"Fitting PCA+Quantile model for {residue.name}")

    # Extract data
    coords, transforms, atom_indices = extract_residues_with_links(
        cif_paths, residue, min_coverage=min_coverage, verbose=False
    )

    if len(coords) == 0:
        raise ValueError(f"No data found for residue {residue.name}")

    n_samples = len(coords)
    n_atoms = coords.shape[1]

    # Flatten and concatenate
    coords_flat = coords.reshape(n_samples, -1)
    data = np.concatenate([coords_flat, transforms * transform_scale], axis=1)
    data = data.astype(np.float32)

    # Compute PCA
    V, mean, S, var_explained = compute_pca(data, n_components=latent_dim)

    if verbose:
        print(f"  Samples: {n_samples}, Atoms: {n_atoms}")
        print(f"  PCA variance explained: {var_explained[latent_dim-1]*100:.1f}%")

    # Create model
    config = PCAQuantileConfig(
        latent_dim=latent_dim,
        n_knots=n_knots,
        transform_scale=transform_scale,
    )

    model = PCAQuantileResidueModel(
        V=torch.tensor(V, dtype=torch.float32),
        mean=torch.tensor(mean, dtype=torch.float32),
        n_atoms=n_atoms,
        atom_indices=atom_indices,
        residue=residue,
        config=config,
    )

    # Fit splines
    data_t = torch.tensor(data)
    model.fit(data_t)

    # Evaluate
    if verbose:
        with torch.no_grad():
            coords_t = torch.tensor(coords)
            transforms_t = torch.tensor(transforms, dtype=torch.float32)
            z = model.encode(coords_t, transforms_t)
            coords_recon, _ = model.decode(z)
            rmsds = rmsd(coords_recon, coords_t).numpy()

        z_np = z.numpy()
        print(f"  RMSD: {rmsds.mean():.4f} ± {rmsds.std():.4f} Å")
        print(f"  <0.1Å: {100*(rmsds < 0.1).mean():.1f}%")
        print(f"  Latent: μ={np.abs(z_np.mean(0)).mean():.4f}, σ={z_np.std(0).mean():.4f}")

    return model


def fit_all_residues(
    cif_paths: list[str | Path],
    residues: str = "ACGU",
    latent_dim: int = 12,
    n_knots: int = 32,
    transform_scale: float = 1.0,
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> dict["Residue", PCAQuantileResidueModel]:
    """
    Fit PCA + Quantile models for multiple residue types.

    Args:
        cif_paths: List of CIF file paths for training data.
        residues: String of residue characters (e.g., "ACGU").
        latent_dim: Number of PCA components.
        n_knots: Number of knots for quantile splines.
        transform_scale: Scale factor for SE(3) transforms.
        min_coverage: Minimum atom coverage to include a residue.
        verbose: Whether to print progress.

    Returns:
        Dict mapping Residue -> PCAQuantileResidueModel.

    Example:
        >>> from ciffy.nn.pca_quantile import fit_all_residues
        >>> from ciffy.nn.polymer import PolymerModel
        >>>
        >>> models = fit_all_residues(cif_paths, residues="ACGU")
        >>> polymer_model = PolymerModel(models)
    """
    from ciffy.biochemistry import Residue

    models = {}
    for char in residues:
        res = getattr(Residue, char)
        models[res] = fit_pca_quantile(
            cif_paths=cif_paths,
            residue=res,
            latent_dim=latent_dim,
            n_knots=n_knots,
            transform_scale=transform_scale,
            min_coverage=min_coverage,
            verbose=verbose,
        )

    return models
