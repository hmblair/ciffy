"""
VAE-specific training configuration and trainer.

Provides VAEConfig and VAETrainer for training Polymer VAE models.

Example:
    >>> from ciffy.nn.vae.trainer import VAEConfig, VAETrainer
    >>>
    >>> config = VAEConfig.from_yaml("config.yaml")
    >>> trainer = VAETrainer(config)
    >>> result = trainer.train()
    >>> print(f"Best loss: {result['best_loss']:.4f}")
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

from ..base_trainer import (
    BaseConfig,
    BaseTrainer,
    MetricsLogger,
    OutputConfig,
    TrainingConfig,
    WandbConfig,
)
from ..training import (
    BetaScheduler,
    get_device,
    get_worker_init_fn,
    polymer_collate_fn,
    set_seed,
)
from ..loggers import create_logger
from ..dataset import PolymerDataset
from .vae import PolymerVAE

# Import types for loss function validation
try:
    from ...types import Molecule, Scale

    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False
    Molecule = None
    Scale = None

logger = logging.getLogger(__name__)


# =============================================================================
# VAE Configuration
# =============================================================================


@dataclass
class VAEModelConfig:
    """VAE model architecture configuration.

    Attributes:
        latent_dim: Dimension of the latent space.
        hidden_dim: Hidden dimension for transformer layers.
        num_layers: Number of transformer layers in encoder/decoder.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        beta: KL weight for VAE loss (ELBO).
        beta_schedule: Beta annealing schedule ('constant', 'linear', 'cosine', 'cyclical').
        beta_warmup_epochs: Epochs to reach target beta. None = half of total epochs.
        beta_cycles: Number of cycles for cyclical schedule.
    """

    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    beta: float = 1.0
    beta_schedule: str = "constant"
    beta_warmup_epochs: int | None = None
    beta_cycles: int = 4


@dataclass
class DataConfig:
    """Data loading configuration.

    Attributes:
        data_dir: Directory containing CIF/PDB files.
        scale: Scale for splitting polymers ('chain' or 'molecule').
        min_atoms: Minimum atoms per structure. None for no limit.
        max_atoms: Maximum atoms per structure. None for no limit.
        molecule_types: List of molecule types to include (e.g., ['protein', 'rna']).
        exclude_ids: List of structure IDs to exclude.
        limit: Maximum number of samples (for overfitting tests).
    """

    data_dir: str = "./data"
    scale: str = "chain"
    min_atoms: int | None = None
    max_atoms: int | None = None
    molecule_types: list[str] | None = None
    exclude_ids: list[str] | None = None
    limit: int | None = None


@dataclass
class VAEConfig(BaseConfig):
    """Full VAE training configuration.

    Combines base configuration (training, output, wandb) with
    VAE-specific model and data configuration.

    Example:
        >>> config = VAEConfig.from_yaml("vae_config.yaml")
        >>> print(config.model.latent_dim)  # 64
        >>> print(config.training.epochs)    # 100
    """

    model: VAEModelConfig = field(default_factory=VAEModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def _from_dict(cls, data: dict) -> "VAEConfig":
        """Create config from dictionary, handling nested dataclasses."""
        kwargs = {}

        # Handle all fields defined on VAEConfig and BaseConfig
        all_fields = {f.name: f for f in fields(cls)}

        for name, f in all_fields.items():
            if name in data:
                value = data[name]
                if hasattr(f.type, "__dataclass_fields__"):
                    kwargs[name] = cls._dict_to_dataclass(f.type, value)
                else:
                    kwargs[name] = value

        return cls(**kwargs)


# =============================================================================
# VAE Trainer
# =============================================================================


class VAETrainer(BaseTrainer):
    """Trainer for Polymer VAE models.

    Extends BaseTrainer with VAE-specific functionality:
    - Beta scheduling (KL annealing)
    - Sample generation during training
    - VAE-specific loss function with polymer validation

    Example:
        >>> config = VAEConfig.from_yaml("config.yaml")
        >>> trainer = VAETrainer(config)
        >>> result = trainer.train()

    Or with manual setup:
        >>> config = VAEConfig.from_yaml("config.yaml")
        >>> model = PolymerVAE(latent_dim=64)
        >>> dataset = PolymerDataset("./data")
        >>> trainer = VAETrainer(config, model=model, dataset=dataset)
        >>> result = trainer.train()
    """

    config: VAEConfig  # Type hint for IDE support

    def __init__(
        self,
        config: VAEConfig,
        model: PolymerVAE | None = None,
        dataset: PolymerDataset | None = None,
        device: "torch.device | None" = None,
        logger: MetricsLogger | None = None,
        quiet: bool = False,
        experiment_name: str | None = None,
    ):
        """Initialize the VAE trainer.

        Args:
            config: VAE training configuration.
            model: Optional pre-created model. If None, created from config.
            dataset: Optional pre-created dataset. If None, created from config.
            device: Device to train on. If None, uses config.training.device.
            logger: Optional metrics logger (e.g., WandbLogger).
            quiet: If True, suppress progress bars and reduce logging.
            experiment_name: Optional name suffix for output directories.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for VAETrainer")

        self.experiment_name = experiment_name

        # Setup device first (needed for model creation)
        if device is None:
            self._device = get_device(config.training.device)
        else:
            self._device = device

        # Create model if not provided
        if model is None:
            model = self._create_model(config.model)

        # Create dataset if not provided
        if dataset is None:
            dataset = self._create_dataset(config.data)

        # Create logger from config if not provided
        if logger is None and config.wandb.project is not None and config.wandb.enabled:
            logger = create_logger(
                project=config.wandb.project,
                config=config.to_dict(),
                name=config.wandb.name or experiment_name,
                group=config.wandb.group,
                enabled=config.wandb.enabled,
            )

        # Initialize base trainer
        super().__init__(
            config=config,
            model=model,
            dataset=dataset,
            device=self._device,
            logger=logger,
            quiet=quiet,
        )

        # Setup beta scheduler
        warmup = config.model.beta_warmup_epochs
        if warmup is None:
            warmup = config.training.epochs // 2

        self.beta_scheduler = BetaScheduler(
            schedule=config.model.beta_schedule,
            target_beta=config.model.beta,
            warmup_epochs=warmup,
            total_epochs=config.training.epochs,
            n_cycles=config.model.beta_cycles,
        )

        # Adjust output directories for experiment name
        if experiment_name:
            self.checkpoint_dir = self.checkpoint_dir / experiment_name
            self.sample_dir = self.sample_dir / experiment_name
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.sample_dir.mkdir(parents=True, exist_ok=True)
            self.best_checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"

        if not quiet:
            logger.info(f"Beta schedule: {self.beta_scheduler}")

    def _create_model(self, config: VAEModelConfig) -> PolymerVAE:
        """Create VAE model from configuration."""
        return PolymerVAE(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            beta=config.beta,
        )

    def _create_dataset(self, config: DataConfig) -> PolymerDataset:
        """Create polymer dataset from configuration."""
        if not TYPES_AVAILABLE:
            raise ImportError("ciffy.types is required for dataset creation")

        scale = Scale.CHAIN if config.scale.lower() == "chain" else Scale.MOLECULE

        molecule_types = None
        if config.molecule_types:
            molecule_types = tuple(Molecule[mt.upper()] for mt in config.molecule_types)

        return PolymerDataset(
            directory=config.data_dir,
            scale=scale,
            min_atoms=config.min_atoms,
            max_atoms=config.max_atoms,
            molecule_types=molecule_types,
            exclude_ids=config.exclude_ids,
            limit=config.limit,
            backend="torch",
        )

    def create_optimizer(self) -> "optim.Optimizer":
        """Create AdamW optimizer for VAE training."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

    def create_dataloader(self) -> "DataLoader":
        """Create DataLoader for polymer training."""
        return DataLoader(
            self.dataset,
            batch_size=1,  # Always 1 for variable-size polymers
            shuffle=True,
            num_workers=self.config.training.num_workers,
            collate_fn=polymer_collate_fn,
            worker_init_fn=get_worker_init_fn(self.config.training.seed),
            pin_memory=(str(self.device) != "cpu"),
            persistent_workers=(self.config.training.num_workers > 0),
        )

    def create_loss_fn(self) -> Callable[["nn.Module", Any], dict[str, "torch.Tensor"]]:
        """Create VAE loss function with polymer validation."""
        if not TYPES_AVAILABLE:
            raise ImportError("ciffy.types is required for loss function")

        device = self.device
        supported_types = (Molecule.PROTEIN, Molecule.PROTEIN_D, Molecule.RNA, Molecule.DNA)

        def vae_loss_fn(model: PolymerVAE, polymer: Any) -> dict[str, "torch.Tensor"]:
            # Strip non-polymer atoms (ligands, water, etc.)
            polymer = polymer.poly()

            # Skip too-small polymers
            if polymer.size(Scale.RESIDUE) < 2:
                return {"loss": torch.tensor(float("nan"))}

            # Skip unsupported molecule types
            mol_type_val = polymer.molecule_type[0]
            if hasattr(mol_type_val, "item"):
                mol_type_val = mol_type_val.item()
            mol_type = Molecule(mol_type_val)

            if mol_type not in supported_types:
                return {"loss": torch.tensor(float("nan"))}

            # Move to device and compute loss
            polymer = polymer.to(device)
            return model.compute_loss(polymer)

        return vae_loss_fn

    def on_epoch_start(self, epoch: int) -> None:
        """Update beta (KL weight) at the start of each epoch."""
        current_beta = self.beta_scheduler.get_beta(epoch)
        self.model.beta = current_beta

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Generate samples at the end of each epoch (if not quiet)."""
        if not self.quiet:
            self._generate_samples(epoch + 1)

    def _generate_samples(self, epoch: int) -> None:
        """Generate and save sample reconstructions and perturbations."""
        self.model.eval()

        # Select random valid structure
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        template = None
        for idx in indices:
            try:
                polymer = self.dataset[idx]
                if polymer is None:
                    continue
                polymer = polymer.poly()
                if polymer.size(Scale.RESIDUE) >= 2:
                    template = polymer.to(self.device)
                    break
            except Exception:
                continue

        if template is None:
            logger.warning("Could not find valid structure for sampling")
            return

        n_perturbations = self.config.output.n_perturbations
        perturbation_scale = self.config.output.perturbation_scale

        with torch.no_grad():
            # Save original template
            template_cpu = template.numpy()
            template_cpu.write(str(self.sample_dir / f"epoch{epoch:04d}_original.cif"))

            # Encode and reconstruct
            z_mu, z_logvar = self.model.encode(template)
            recon = self.model.decode(z_mu, template, sample=False)
            recon_cpu = recon.numpy()
            recon_cpu.write(str(self.sample_dir / f"epoch{epoch:04d}_reconstruction.cif"))

            # Generate perturbed samples
            for i in range(n_perturbations):
                noise = torch.randn_like(z_mu) * perturbation_scale
                z_perturbed = z_mu + noise
                perturbed = self.model.decode(z_perturbed, template, sample=False)
                perturbed_cpu = perturbed.numpy()
                perturbed_cpu.write(str(self.sample_dir / f"epoch{epoch:04d}_perturb{i+1}.cif"))

        logger.info(f"Saved {n_perturbations + 2} samples to {self.sample_dir}")

    def _log_epoch(self, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics including current beta."""
        current_beta = self.beta_scheduler.get_beta(epoch)

        parts = [f"Epoch {epoch + 1}/{total_epochs}"]

        if "loss" in metrics:
            parts.append(f"Loss: {metrics['loss']:.4f}")
        if "recon_loss" in metrics:
            parts.append(f"Recon: {metrics['recon_loss']:.4f}")
        if "kl_loss" in metrics:
            parts.append(f"KL: {metrics['kl_loss']:.4f}")

        parts.append(f"Beta: {current_beta:.4f}")

        if "n_samples" in metrics:
            parts.append(f"Samples: {int(metrics['n_samples'])}")
        if "n_skipped" in metrics and metrics["n_skipped"] > 0:
            parts.append(f"Skipped: {int(metrics['n_skipped'])}")

        logger.info(" | ".join(parts))


__all__ = [
    "VAEModelConfig",
    "DataConfig",
    "VAEConfig",
    "VAETrainer",
]
