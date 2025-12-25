"""
PCA + Flow model for residue conformations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ciffy.utils.enum_base import IndexEnum

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
            # Use correction=0 (biased estimator) to avoid warning when batch_size=1
            # The clamp ensures we never get log(0) even with degenerate batches
            std = x.std(dim=0, correction=0).clamp(min=1e-6)
            self.log_scale.copy_(-torch.log(std))
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
        bound: Tanh bound (in std devs) for decode(). None (default) disables
               bounding, preserving exact invertibility.
    """

    def __init__(
        self,
        V: torch.Tensor,
        mean: torch.Tensor,
        n_layers: int = 8,
        hidden_dim: int = 64,
        bound: float | None = None,
    ):
        super().__init__()
        self.k = V.shape[0]  # Latent dimension
        self.d = V.shape[1]  # Coordinate dimension (n_atoms * 3 + 6)
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
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
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


class ResidueFlowModel:
    """
    Residue flow model that captures conformation and backbone link geometry.

    This model learns the joint distribution of residue coordinates AND
    the SE(3) transform to the next residue in the chain. This enables
    sampling residue conformations with realistic backbone connectivity.

    The representation is: [coords_flat (n_atoms*3), transform (6)]
    where transform = [axis-angle (3), translation (3)] defines the relative
    position and orientation of the next residue's P atom.

    Attributes:
        flow: The underlying PCAFlow model.
        residue: The source residue type.
        atoms: IndexEnum subset containing the atoms used.
        n_atoms: Number of atoms per residue.
        pca_rmsd: Reconstruction RMSD from PCA truncation.
        var_explained: Fraction of variance explained.

    Example:
        >>> model = ResidueFlowModel.from_structures(cif_paths, Residue.A)
        >>> coords, transform = model.decode(z)
        >>> # Position next residue using the transform
        >>> from ciffy.nn.residue_flow import position_next_residue
        >>> coords2 = position_next_residue(coords, ref_coords, transform, atoms, residue)
    """

    def __init__(
        self,
        flow: PCAFlow,
        residue: "Residue",
        atom_indices: list[int],
        n_atoms: int,
        pca_rmsd: float,
        var_explained: float,
        jit: bool = False,
    ):
        self.flow = flow
        self.residue = residue
        self._atom_indices = atom_indices
        self.n_atoms = n_atoms
        self._atoms_enum: type[IndexEnum] | None = None
        self.pca_rmsd = pca_rmsd
        self.var_explained = var_explained
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

        atom_to_col = {a: i for i, a in enumerate(self._atom_indices)}
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
    def atoms(self) -> type[IndexEnum]:
        """IndexEnum subset containing the atoms used by this model."""
        if self._atoms_enum is None:
            self._atoms_enum = create_atom_subset(self.residue, self._atom_indices)
        return self._atoms_enum

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
            pca_rmsd=info["pca_rmsd"],
            var_explained=info["var_explained"],
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
            pca_rmsd=config["pca_rmsd"],
            var_explained=config["var_explained"],
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

    def to(self, device: str | torch.device) -> "ResidueFlowModel":
        """
        Move model to specified device.

        Clears cached geometry projector (will be rebuilt on first use).

        Args:
            device: Target device (e.g., "cpu", "cuda", "cuda:0").

        Returns:
            Self for method chaining.
        """
        device = torch.device(device) if isinstance(device, str) else device
        self.flow = self.flow.to(device)
        if self._jit_decoder is not None:
            self._jit_decoder = self._jit_decoder.to(device)
        # Clear cached projector - will be rebuilt for new device
        self._geometry_projector = None
        self._geometry_projector_device = None
        return self

    def cuda(self, device_id: int = 0) -> "ResidueFlowModel":
        """Move model to CUDA device."""
        return self.to(f"cuda:{device_id}")

    def cpu(self) -> "ResidueFlowModel":
        """Move model to CPU."""
        return self.to("cpu")

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

        # Map from atom enum value to local index in this model's subset
        atom_to_local = {a: i for i, a in enumerate(atom_indices)}
        atom_set = set(atom_indices)

        # Get ideal coordinates and bonds from residue definition
        ideal_coords = residue.ideal  # (n_residue_atoms, 3)
        bonds = residue.bonds  # PairEnum of (atom1, atom2) pairs

        # Build bond constraint data from residue's bond definitions
        # Only include bonds where both atoms are in our subset
        bond_pairs = []
        bond_targets = []

        for atom1, atom2 in bonds:
            if atom1.value in atom_set and atom2.value in atom_set:
                # Get local indices in the model's atom subset
                local_i = atom_to_local[atom1.value]
                local_j = atom_to_local[atom2.value]
                bond_pairs.append((local_i, local_j))

                # Compute ideal bond length from ideal coordinates
                # Use .local for 0-indexed access into ideal_coords
                pos1 = ideal_coords[atom1.local]
                pos2 = ideal_coords[atom2.local]
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
            f"latent_dim={self.flow.k}, "
            f"var={self.var_explained*100:.1f}%, "
            f"rmsd={self.pca_rmsd:.3f}Å)"
        )
