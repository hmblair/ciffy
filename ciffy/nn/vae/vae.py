"""
Polymer Variational Autoencoder for backbone dihedrals.

Provides a VAE that encodes polymer backbone conformations (via dihedral angles)
into a global latent space and can decode back to generate new conformations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

from .encoder import DihedralEncoder
from .decoder import DihedralDecoder
from .losses import VAELoss

from ...types import DihedralType, Molecule, Scale
from ...types.dihedral import (
    PROTEIN_BACKBONE,
    RNA_BACKBONE_EXTENDED,
    MAX_DIHEDRALS_PER_RESIDUE,
)
from ..model_registry import register_model

if TYPE_CHECKING:
    from ...polymer import Polymer


# Mapping from Molecule enum to type index (for dihedral type selection)
MOLECULE_TO_INDEX = {
    Molecule.PROTEIN: 0,
    Molecule.PROTEIN_D: 0,  # Treat D-amino acids same as L
    Molecule.RNA: 1,
    Molecule.DNA: 2,
}


@register_model("PolymerVAE")
class PolymerVAE(nn.Module if TORCH_AVAILABLE else object):
    """
    Variational Autoencoder for polymer backbone conformations.

    Encodes backbone dihedral angles into a global latent vector and
    decodes back to dihedral distributions. Supports both proteins
    and nucleic acids (RNA/DNA).

    The VAE operates on dihedral angles rather than Cartesian coordinates,
    which provides several advantages:
    - Lower dimensionality (3-7 angles per residue vs 3*N_atoms coordinates)
    - Built-in bond length and angle constraints
    - Natural handling of rotational invariance

    The encoder and decoder use residue embeddings via PolymerEmbedding,
    allowing the model to learn sequence-dependent conformational preferences
    (e.g., prolines prefer certain φ/ψ regions).

    Example:
        >>> from ciffy.nn.vae import PolymerVAE
        >>> import ciffy
        >>>
        >>> # Create VAE
        >>> vae = PolymerVAE(latent_dim=64).cuda()
        >>>
        >>> # Encode a polymer
        >>> polymer = ciffy.load("structure.cif", backend="torch").to("cuda")
        >>> z_mu, z_logvar = vae.encode(polymer)
        >>>
        >>> # Reconstruct from mean latent
        >>> recon = vae.decode(z_mu, polymer)
        >>>
        >>> # Sample new conformations
        >>> samples = vae.sample(polymer, n_samples=10, temperature=1.0)
        >>>
        >>> # Interpolate between two conformations
        >>> interp = vae.interpolate(polymer1, polymer2, n_steps=10)

    Args:
        latent_dim: Dimension of latent space z
        hidden_dim: Transformer hidden dimension
        residue_dim: Dimension of residue embeddings (default: hidden_dim // 4)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        beta: KL divergence weight (for beta-VAE training)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        residue_dim: Optional[int] = None,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        beta: float = 1.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PolymerVAE. Install with: pip install torch"
            )
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = DihedralEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            residue_dim=residue_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.decoder = DihedralDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            residue_dim=residue_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.loss_fn = VAELoss(beta=beta)

    def _get_molecule_type_index(self, polymer: "Polymer") -> int:
        """Determine molecule type index from polymer."""
        mol_types = polymer.molecule_type
        mol_type_val = mol_types[0].item() if hasattr(mol_types[0], "item") else mol_types[0]

        # Convert to Molecule enum if needed
        if isinstance(mol_type_val, int):
            mol_type = Molecule(mol_type_val)
        else:
            mol_type = mol_type_val

        if mol_type not in MOLECULE_TO_INDEX:
            raise ValueError(
                f"Unsupported molecule type: {mol_type}. "
                f"Supported types: {list(MOLECULE_TO_INDEX.keys())}"
            )

        return MOLECULE_TO_INDEX[mol_type]

    def _get_dihedral_types(self, mol_type_idx: int) -> list[DihedralType]:
        """Get list of dihedral types for a molecule type."""
        if mol_type_idx == 0:  # Protein
            return list(PROTEIN_BACKBONE)
        else:  # RNA or DNA
            return list(RNA_BACKBONE_EXTENDED)

    def _extract_dihedrals(
        self, polymer: "Polymer"
    ) -> tuple["torch.Tensor", "torch.Tensor", int]:
        """
        Extract backbone dihedrals from polymer in unified format.

        Returns:
            dihedrals: (L, D) tensor of dihedral angles (D=7, padded for proteins)
            mask: (L, D) boolean mask (True = valid dihedral)
            mol_type_idx: Molecule type index (0=protein, 1=RNA, 2=DNA)
        """
        mol_type_idx = self._get_molecule_type_index(polymer)
        dihedral_types = self._get_dihedral_types(mol_type_idx)
        n_residues = polymer.size(Scale.RESIDUE)
        device = polymer.coordinates.device

        # Initialize output tensors
        dihedrals = torch.zeros(n_residues, MAX_DIHEDRALS_PER_RESIDUE, device=device)
        mask = torch.zeros(
            n_residues, MAX_DIHEDRALS_PER_RESIDUE, dtype=torch.bool, device=device
        )

        # Extract each dihedral type
        for i, dtype in enumerate(dihedral_types):
            try:
                values = polymer.dihedral(dtype)
                if len(values) > 0:
                    # The dihedral() method returns values only for residues that
                    # have that dihedral defined. We need to map these back to
                    # residue positions. The mapping depends on the dihedral type.
                    n_vals = len(values)

                    # Determine which residues have this dihedral
                    # PHI: starts at residue 1 (residue 0 has no phi)
                    # PSI, OMEGA: end at residue n-2 (residue n-1 has no psi/omega)
                    # ALPHA: starts at residue 1 (needs previous O3')
                    # EPSILON, ZETA: end at residue n-2 (needs next P/O5')
                    if dtype == DihedralType.PHI:
                        start_idx = 1
                    elif dtype == DihedralType.ALPHA:
                        start_idx = 1
                    else:
                        start_idx = 0

                    end_idx = start_idx + n_vals
                    if end_idx > n_residues:
                        end_idx = n_residues
                        n_vals = end_idx - start_idx

                    dihedrals[start_idx:end_idx, i] = values[:n_vals]
                    mask[start_idx:end_idx, i] = True
            except (ValueError, KeyError, RuntimeError):
                # Dihedral type not available for this structure
                pass

        return dihedrals, mask, mol_type_idx

    def _set_dihedrals(
        self,
        polymer: "Polymer",
        dihedrals: "torch.Tensor",
        mask: "torch.Tensor",
        mol_type_idx: int,
    ) -> "Polymer":
        """
        Set backbone dihedrals on polymer from unified format.

        Args:
            polymer: Polymer to modify
            dihedrals: (L, D) tensor of dihedral angles
            mask: (L, D) boolean mask (True = valid)
            mol_type_idx: Molecule type index

        Returns:
            Modified polymer
        """
        dihedral_types = self._get_dihedral_types(mol_type_idx)

        # Set each dihedral type
        for i, dtype in enumerate(dihedral_types):
            if mask[:, i].any():
                valid_mask = mask[:, i]
                values = dihedrals[:, i][valid_mask]
                try:
                    polymer.set_dihedral(dtype, values)
                except (ValueError, KeyError, RuntimeError):
                    # Dihedral type may not be settable for this structure
                    pass

        return polymer

    def encode(self, polymer: "Polymer") -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Encode polymer to latent distribution parameters.

        Args:
            polymer: Polymer object (must be torch backend)

        Returns:
            mu: (latent_dim,) mean of latent distribution
            logvar: (latent_dim,) log-variance of latent distribution
        """
        dihedrals, mask, _ = self._extract_dihedrals(polymer)
        return self.encoder(dihedrals, mask, polymer)

    def decode(
        self,
        z: "torch.Tensor",
        polymer: "Polymer",
        sample: bool = False,
        temperature: float = 1.0,
    ) -> "Polymer":
        """
        Decode latent vector to polymer with new dihedrals.

        Args:
            z: (latent_dim,) latent vector
            polymer: Template polymer (provides structure and sequence)
            sample: If True, sample from distribution; if False, use mean
            temperature: Sampling temperature (only used if sample=True)

        Returns:
            New polymer with decoded dihedrals
        """
        _, mask, mol_type_idx = self._extract_dihedrals(polymer)

        if sample:
            decoded_dihedrals = self.decoder.sample(
                z, polymer, mask, temperature
            )
        else:
            mu, _ = self.decoder(z, polymer, mask)
            decoded_dihedrals = mu

        # Create copy of polymer and set new dihedrals
        result = polymer.with_coordinates(polymer.coordinates.clone())
        result = self._set_dihedrals(result, decoded_dihedrals, mask, mol_type_idx)

        # Trigger coordinate reconstruction from internal coordinates
        _ = result.coordinates

        return result

    def reparameterize(
        self,
        mu: "torch.Tensor",
        logvar: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Reparameterization trick for differentiable sampling.

        z = mu + std * epsilon, where epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        polymer: "Polymer",
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
    ]:
        """
        Full forward pass for training.

        Args:
            polymer: Input polymer

        Returns:
            mu_pred: (L, D) Predicted mean dihedrals
            kappa_pred: (L, D) Predicted concentrations
            target: (L, D) Target dihedrals
            z_mu: (latent_dim,) Latent mean
            z_logvar: (latent_dim,) Latent log-variance
            mask: (L, D) Boolean mask for valid dihedrals
        """
        # Extract target dihedrals
        target, mask, _ = self._extract_dihedrals(polymer)

        # Encode
        z_mu, z_logvar = self.encoder(target, mask, polymer)

        # Sample z using reparameterization
        z = self.reparameterize(z_mu, z_logvar)

        # Decode
        mu_pred, kappa_pred = self.decoder(z, polymer, mask)

        return mu_pred, kappa_pred, target, z_mu, z_logvar, mask

    def compute_loss(
        self,
        polymer: "Polymer",
    ) -> dict[str, "torch.Tensor"]:
        """
        Compute VAE loss for a single polymer.

        Args:
            polymer: Input polymer

        Returns:
            Dictionary with keys:
                'loss': Total loss (reconstruction + beta * KL)
                'recon_loss': Reconstruction loss (von Mises NLL)
                'kl_loss': KL divergence
        """
        mu_pred, kappa_pred, target, z_mu, z_logvar, mask = self.forward(polymer)
        return self.loss_fn(mu_pred, kappa_pred, target, z_mu, z_logvar, mask)

    def sample(
        self,
        polymer: "Polymer",
        n_samples: int = 1,
        temperature: float = 1.0,
    ) -> list["Polymer"]:
        """
        Sample new conformations for a polymer.

        Generates conformations by sampling from the prior N(0, I) in latent
        space and decoding. The template polymer provides the sequence and
        topology; only the dihedral angles are modified.

        Args:
            polymer: Template polymer (provides sequence and structure)
            n_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            List of polymers with sampled conformations
        """
        samples = []
        device = polymer.coordinates.device

        for _ in range(n_samples):
            # Sample z from prior N(0, I)
            z = torch.randn(self.latent_dim, device=device)

            # Decode to polymer
            result = self.decode(z, polymer, sample=True, temperature=temperature)
            samples.append(result)

        return samples

    def interpolate(
        self,
        polymer1: "Polymer",
        polymer2: "Polymer",
        n_steps: int = 10,
    ) -> list["Polymer"]:
        """
        Interpolate between two conformations in latent space.

        Encodes both polymers to latent space, performs linear interpolation,
        and decodes each interpolated point. The polymers should have the
        same sequence and topology.

        Args:
            polymer1: Starting conformation
            polymer2: Ending conformation
            n_steps: Number of interpolation steps (including endpoints)

        Returns:
            List of polymers along the interpolation path
        """
        # Encode both polymers (use mean, not sample)
        z1_mu, _ = self.encode(polymer1)
        z2_mu, _ = self.encode(polymer2)

        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, n_steps, device=z1_mu.device)
        results = []

        for alpha in alphas:
            z = (1 - alpha) * z1_mu + alpha * z2_mu
            result = self.decode(z, polymer1, sample=False)
            results.append(result)

        return results

    def reconstruct(
        self,
        polymer: "Polymer",
        sample_latent: bool = False,
    ) -> "Polymer":
        """
        Reconstruct a polymer through the VAE.

        Encodes the polymer to latent space and decodes back. Useful for
        testing reconstruction quality.

        Args:
            polymer: Input polymer to reconstruct
            sample_latent: If True, sample from latent distribution;
                          if False, use the mean (deterministic)

        Returns:
            Reconstructed polymer
        """
        z_mu, z_logvar = self.encode(polymer)

        if sample_latent:
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z = z_mu

        return self.decode(z, polymer, sample=False)
