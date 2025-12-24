"""
PCA + Flow model for residue conformations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ciffy.utils.enum_base import IndexEnum, PairEnum

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


# =============================================================================
# Atom Subset Enum
# =============================================================================


def create_atom_subset(residue: "Residue", atom_indices: list[int]) -> type[IndexEnum]:
    """
    Create an IndexEnum subset containing only the specified atoms from a residue.

    The returned enum has the same interface as IndexEnum:
        - Iteration over members
        - index() → array of values
        - list() → list of names
        - dict() → name → value mapping
        - revdict() → value → name mapping

    Args:
        residue: The source residue (e.g., Residue.A).
        atom_indices: List of atom indices to include.

    Returns:
        A new IndexEnum class with only the specified atoms.

    Example:
        >>> atoms = create_atom_subset(Residue.A, [0, 1, 5, 10])
        >>> list(atoms)  # [<Atoms.C1p: 0>, <Atoms.C2p: 1>, ...]
        >>> atoms.index()  # array([0, 1, 5, 10])
    """
    # Build name → value mapping for the subset
    atom_set = set(atom_indices)
    members = {}
    for member in residue:
        if member.value in atom_set:
            members[member.name] = member.value

    # Create the subset enum dynamically
    return IndexEnum(f"{residue.name}Atoms", members)


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

    def initialize(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            self.bias.copy_(-x.mean(dim=0))
            self.log_scale.copy_(-torch.log(x.std(dim=0).clamp(min=1e-6)))
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self.initialize(x)
        y = (x + self.bias) * torch.exp(self.log_scale)
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


class AffineCoupling(nn.Module):
    """
    Affine coupling layer.

    Splits input into two halves and transforms one conditioned on the other:
        y_a = x_a
        y_b = x_b * exp(s(x_a)) + t(x_a)

    This is exactly invertible by construction.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, even_mask: bool = True):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", torch.arange(dim) % 2 == (0 if even_mask else 1))

        n_masked = int(self.mask.sum())
        n_unmasked = dim - n_masked
        self.net = CouplingNetwork(n_masked, 2 * n_unmasked, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_a = x[:, self.mask]
        x_b = x[:, ~self.mask]

        st = self.net(x_a)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 0.5  # Bound scale for stability

        y_b = x_b * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        y = torch.empty_like(x)
        y[:, self.mask] = x_a
        y[:, ~self.mask] = y_b
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_a = y[:, self.mask]
        y_b = y[:, ~self.mask]

        st = self.net(y_a)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 0.5

        x_b = (y_b - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x[:, self.mask] = y_a
        x[:, ~self.mask] = x_b
        return x


# =============================================================================
# PCA + Flow Model
# =============================================================================


class PCAFlow(nn.Module):
    """
    PCA for dimensionality reduction + normalizing flow for density estimation.

    The model is exactly invertible: decode(encode(x)) reconstructs x
    with error bounded only by PCA truncation.

    Args:
        V: PCA components matrix (k, d) where k is latent dim, d is coord dim.
        mean: Mean coordinates (d,).
        n_layers: Number of flow layers (ActNorm + Coupling pairs).
        hidden_dim: Hidden dimension in coupling networks.
        bound: Tanh bound (in std devs) for decode(). Prevents extrapolation.
               None disables bounding.
    """

    def __init__(
        self,
        V: torch.Tensor,
        mean: torch.Tensor,
        n_layers: int = 8,
        hidden_dim: int = 64,
        bound: float | None = 3.0,
    ):
        super().__init__()
        self.k = V.shape[0]  # Latent dimension
        self.d = V.shape[1]  # Coordinate dimension (n_atoms * 3)
        self.n_atoms = self.d // 3
        self.bound = bound

        # PCA parameters (fixed, not learned)
        self.register_buffer("V", V)
        self.register_buffer("mean", mean)

        # Flow layers: alternating ActNorm + Coupling
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ActNorm(self.k))
            self.layers.append(AffineCoupling(self.k, hidden_dim, even_mask=(i % 2 == 0)))

    def coords_to_pca(self, x: torch.Tensor) -> torch.Tensor:
        """Project coordinates to PCA space."""
        flat = x.reshape(-1, self.d)
        return (flat - self.mean) @ self.V.T

    def pca_to_coords(self, pca: torch.Tensor) -> torch.Tensor:
        """Reconstruct coordinates from PCA (approximate due to truncation)."""
        flat = pca @ self.V + self.mean
        return flat.reshape(-1, self.n_atoms, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode coordinates to latent space.

        Args:
            x: Coordinates (N, n_atoms, 3) or (N, n_atoms*3).

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
        Decode latent vectors to coordinates.

        Args:
            z: Latent vectors (N, k).

        Returns:
            Coordinates (N, n_atoms, 3).
        """
        if self.bound is not None:
            z = self.bound * torch.tanh(z / self.bound)

        h = z
        for layer in reversed(self.layers):
            h = layer.inverse(h)
        return self.pca_to_coords(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates to latent space."""
        z, _ = self.forward(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to coordinates."""
        return self.inverse(z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of coordinates under the model."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample new coordinates from the learned distribution."""
        z = torch.randn(n_samples, self.k, device=self.V.device)
        return self.decode(z)


class _JITDecoder(nn.Module):
    """
    JIT-compatible decoder wrapper.

    Extracts just the decode path with bound baked in as a constant,
    avoiding conditionals that complicate tracing.
    """

    def __init__(self, flow: PCAFlow):
        super().__init__()
        self.k = flow.k
        self.n_atoms = flow.n_atoms
        self.bound: float | None = flow.bound

        # Copy buffers
        self.register_buffer("V", flow.V.clone())
        self.register_buffer("mean", flow.mean.clone())

        # Copy layers in reverse order (TorchScript doesn't support reversed())
        import copy
        self.layers = nn.ModuleList([
            copy.deepcopy(layer) for layer in reversed(list(flow.layers))
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to coordinates."""
        if self.bound is not None:
            z = self.bound * torch.tanh(z / self.bound)

        h = z
        for layer in self.layers:
            h = layer.inverse(h)

        # PCA to coords
        flat = h @ self.V + self.mean
        return flat.reshape(-1, self.n_atoms, 3)


# =============================================================================
# High-Level Model Wrapper
# =============================================================================


@dataclass
class ResidueFlowConfig:
    """Configuration for ResidueFlowModel."""

    latent_dim: int = 12
    n_layers: int = 8
    hidden_dim: int = 64
    bound: float | None = 3.0
    min_coverage: float = 0.9


class ResidueFlowModel:
    """
    High-level wrapper for training and using residue flow models.

    This class handles data extraction, alignment, PCA computation,
    and flow training in a single interface.

    Attributes:
        flow: The underlying PCAFlow model.
        residue: The source residue type (e.g., Residue.A).
        atoms: IndexEnum subset containing only the atoms used by this model.
               Provides full IndexEnum interface: index(), list(), dict(), etc.
        pca_rmsd: Reconstruction RMSD from PCA truncation.
        var_explained: Fraction of variance explained by the latent dimensions.

    Example:
        >>> model = ResidueFlowModel.from_structures(cif_paths, Residue.A)
        >>> model.atoms.list()  # ['C1p', 'C2p', 'O2p', ...]
        >>> model.atoms.index()  # array([0, 1, 2, ...])
        >>> len(model.atoms)  # 22
        >>> samples = model.sample(100)
        >>> model.save("adenosine_flow.pt")
    """

    def __init__(
        self,
        flow: PCAFlow,
        residue: "Residue",
        atom_indices: list[int],
        pca_rmsd: float,
        var_explained: float,
        jit: bool = False,
    ):
        self.flow = flow
        self.residue = residue
        self._atom_indices = atom_indices
        self._atoms_enum: type[IndexEnum] | None = None  # Lazy-created
        self.pca_rmsd = pca_rmsd
        self.var_explained = var_explained
        self._jit_decoder: torch.jit.ScriptModule | None = None

        if jit:
            self._compile_jit()

    @property
    def atoms(self) -> type[IndexEnum]:
        """
        IndexEnum subset containing only the atoms used by this model.

        Provides the full IndexEnum interface:
            - len(atoms): Number of atoms
            - atoms.list(): List of atom names
            - atoms.index(): Array of atom indices
            - atoms.dict(): Name → index mapping
            - atoms.revdict(): Index → name mapping
            - Iteration: for atom in atoms

        Returns:
            IndexEnum class with the subset of atoms.
        """
        if self._atoms_enum is None:
            self._atoms_enum = create_atom_subset(self.residue, self._atom_indices)
        return self._atoms_enum

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
        from .data import extract_residues, align_to_frame, compute_pca
        from .train import train_pca_flow

        if config is None:
            config = ResidueFlowConfig()

        # Extract and align
        if verbose:
            print(f"Extracting {residue.name} residues...")
        coords, atoms = extract_residues(
            cif_paths, residue, min_coverage=config.min_coverage, verbose=verbose
        )
        coords = align_to_frame(coords, atoms, residue)

        if verbose:
            print(f"Dataset: {len(coords)} instances, {len(atoms)} atoms")

        # Train
        flow, info = train_pca_flow(
            coords,
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
            pca_rmsd=info["pca_rmsd"],
            var_explained=info["var_explained"],
        )

    def _compile_jit(self) -> None:
        """Compile the decoder to TorchScript for faster inference."""
        self.flow.eval()
        decoder = _JITDecoder(self.flow)
        decoder.eval()
        self._jit_decoder = torch.jit.script(decoder)

    @property
    def is_jit(self) -> bool:
        """Whether the decoder is JIT-compiled."""
        return self._jit_decoder is not None

    def encode(self, coords: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode coordinates to latent space."""
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        coords = coords.to(self.flow.V.device)
        return self.flow.encode(coords)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to coordinates."""
        if self._jit_decoder is not None:
            return self._jit_decoder(z)
        return self.flow.decode(z)

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample new conformations."""
        with torch.no_grad():
            z = torch.randn(n_samples, self.flow.k, device=self.flow.V.device)
            samples = self.decode(z)
        return samples.cpu().numpy()

    def save(self, path: str | Path) -> None:
        """
        Save model to directory.

        Creates a directory containing:
            - tensors.safetensors: Model weights (V, mean, flow parameters)
            - config.json: Metadata (residue, atoms, metrics)

        Args:
            path: Directory path to save to. Created if it doesn't exist.
        """
        import json
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save flow state dict (includes V and mean as buffers)
        tensors = {k: v.cpu().contiguous() for k, v in self.flow.state_dict().items()}
        save_file(tensors, path / "tensors.safetensors")

        # Save metadata as JSON (convert numpy types to Python types)
        config = {
            "residue_name": self.residue.name,
            "atom_indices": [int(x) for x in self._atom_indices],
            "n_layers": len(self.flow.layers) // 2,
            "hidden_dim": self.flow.layers[1].net.net[0].out_features,
            "bound": float(self.flow.bound) if self.flow.bound is not None else None,
            "pca_rmsd": float(self.pca_rmsd),
            "var_explained": float(self.var_explained),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        jit: bool = True,
    ) -> "ResidueFlowModel":
        """
        Load model from directory.

        Args:
            path: Directory containing tensors.safetensors and config.json.
            device: Device to load model onto.
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
            pca_rmsd=config["pca_rmsd"],
            var_explained=config["var_explained"],
            jit=jit,
        )

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self.flow.k

    def __repr__(self) -> str:
        return (
            f"ResidueFlowModel({self.residue.name}, "
            f"atoms={len(self._atom_indices)}, "
            f"latent_dim={self.flow.k}, "
            f"var={self.var_explained*100:.1f}%, "
            f"rmsd={self.pca_rmsd:.3f}Å)"
        )
