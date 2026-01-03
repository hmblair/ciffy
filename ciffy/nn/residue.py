"""
Unified API for residue-level generative models.

This module provides a single entry point for training any residue-level model:
- ResidueFlowModel (PCA + normalizing flow)
- ResidueVAE (MLP encoder/decoder)
- ConsolidatedResidueVAE (shared encoder, per-residue decoders)
- More model types can be registered

Example usage:
    >>> from ciffy.nn import residue
    >>>
    >>> # Train flow models (default)
    >>> model = residue.train(["data/*.cif"], residues="ACGU", model_type="flow")
    >>>
    >>> # Train VAE models
    >>> model = residue.train(["data/*.cif"], residues="ACGU", model_type="vae")
    >>>
    >>> # Train consolidated VAE (shared encoder)
    >>> model = residue.train(["data/*.cif"], residues="ACGU", model_type="consolidated")
    >>>
    >>> # Sample from any trained model
    >>> polymer = model.sample_from_sequence("acgu")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.nn.polymer import PolymerModel


@dataclass
class ModelTypeInfo:
    """Information for a registered model type."""
    module_cls: type
    config_cls: type
    model_cls: type
    description: str
    consolidated: bool = False  # True if model trains all residues together


# Registry of model types
_MODEL_REGISTRY: dict[str, ModelTypeInfo] = {}


def register_model_type(
    name: str,
    module_cls: type,
    config_cls: type,
    model_cls: type,
    description: str = "",
    consolidated: bool = False,
) -> None:
    """Register a new model type for the unified training API.

    Args:
        name: Short name for the model type (e.g., "flow", "vae").
        module_cls: Lightning module class.
        config_cls: Full config dataclass.
        model_cls: The model class (for type hints and loading).
        description: Human-readable description.
        consolidated: If True, trains all residues together (shared encoder).
    """
    _MODEL_REGISTRY[name] = ModelTypeInfo(
        module_cls=module_cls,
        config_cls=config_cls,
        model_cls=model_cls,
        description=description,
        consolidated=consolidated,
    )


def _register_builtin_models():
    """Register built-in model types."""
    from .lightning.modules.residue_flow import (
        ResidueFlowModule,
        ResidueFlowFullConfig,
    )
    from .lightning.modules.residue_vae import (
        ResidueVAEModule,
        ResidueVAEFullConfig,
    )
    from .lightning.modules.consolidated_vae import (
        ConsolidatedVAEModule,
        ConsolidatedVAEFullConfig,
    )
    from .flow.residue.model import ResidueFlowModel
    from .vae.residue.model import ResidueVAE
    from .vae.residue.consolidated import ConsolidatedResidueVAE

    register_model_type(
        "flow",
        ResidueFlowModule,
        ResidueFlowFullConfig,
        ResidueFlowModel,
        "PCA + normalizing flow (exact density, interpretable latents)",
    )

    register_model_type(
        "vae",
        ResidueVAEModule,
        ResidueVAEFullConfig,
        ResidueVAE,
        "Variational autoencoder (learned compression, better reconstruction)",
    )

    register_model_type(
        "consolidated",
        ConsolidatedVAEModule,
        ConsolidatedVAEFullConfig,
        ConsolidatedResidueVAE,
        "Consolidated VAE (shared encoder, per-residue decoders, 4x data efficiency)",
        consolidated=True,
    )


# Register built-in models on import
_register_builtin_models()


def available_models() -> dict[str, str]:
    """Return available model types and their descriptions.

    Returns:
        Dict mapping model type names to descriptions.

    Example:
        >>> from ciffy import residue
        >>> residue.available_models()
        {'flow': 'PCA + normalizing flow ...', 'vae': 'Variational autoencoder ...'}
    """
    return {name: info.description for name, info in _MODEL_REGISTRY.items()}


def _resolve_paths(cif_paths: list[Union[str, Path]]) -> list[Path]:
    """Resolve CIF paths, handling globs and directories."""
    resolved = []
    for p in cif_paths:
        path = Path(p) if not isinstance(p, Path) else p
        if path.is_dir():
            resolved.extend(path.glob("*.cif"))
        elif "*" in str(p):
            resolved.extend(Path(".").glob(str(p)))
        else:
            resolved.append(path)
    return resolved


def _parse_residues(residues: Union[list[str], str]) -> list["Residue"]:
    """Parse residue specification to list of Residue enums."""
    from ciffy.biochemistry import Residue

    if isinstance(residues, str):
        return [getattr(Residue, c.upper()) for c in residues]
    else:
        return [
            getattr(Residue, r.upper()) if isinstance(r, str) else r
            for r in residues
        ]


def train(
    cif_paths: list[Union[str, Path]],
    residues: Union[list[str], str] = "ACGU",
    model_type: str = "flow",
    output_dir: Union[str, Path, None] = None,
    n_epochs: int = 200,
    latent_dim: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    accelerator: str = "auto",
    verbose: bool = True,
    **kwargs,
) -> "PolymerModel":
    """
    Train residue-level generative models on CIF structures.

    Unified interface for training any registered model type.
    Uses PyTorch Lightning for training.

    Args:
        cif_paths: Paths to CIF files for training data. Supports:
            - List of file paths
            - Glob patterns (e.g., ["data/*.cif"])
            - Directories (will find all .cif files)
        residues: Residue types to train. Can be:
            - String like "ACGU" (each character is a residue)
            - List of residue names ["A", "C", "G", "U"]
        model_type: Type of model to train. Options:
            - "flow": PCA + normalizing flow (default)
            - "vae": Variational autoencoder
            Use available_models() to see all options.
        output_dir: Where to save trained model (optional).
        n_epochs: Number of training epochs (default: 200).
        latent_dim: Latent space dimension (default: 12).
        batch_size: Training batch size (default: 256).
        lr: Learning rate (default: 1e-3).
        accelerator: Device for training ("auto", "cpu", "gpu", "mps").
        verbose: Whether to show progress bars.
        **kwargs: Model-specific arguments:
            - flow: n_layers, hidden_dim, use_rotation
            - vae: hidden_dims, beta, free_bits, use_residual

    Returns:
        Trained PolymerModel that can encode/decode/sample polymers.

    Example:
        >>> from ciffy import residue
        >>>
        >>> # Train flow models
        >>> model = residue.train(
        ...     ["data/*.cif"],
        ...     residues="ACGU",
        ...     model_type="flow",
        ...     output_dir="models/rna_flow",
        ... )
        >>>
        >>> # Train VAE models
        >>> model = residue.train(
        ...     ["data/*.cif"],
        ...     residues="ACGU",
        ...     model_type="vae",
        ...     output_dir="models/rna_vae",
        ... )
        >>>
        >>> # Sample from trained model
        >>> polymer = model.sample_from_sequence("acguacgu")
        >>> polymer.write("output.cif")
    """
    import lightning as L
    from .polymer import PolymerModel
    from .lightning import FlowDataModule as ResidueDataModule
    from .config import TrainingConfig

    # Validate model type
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")

    model_info = _MODEL_REGISTRY[model_type]

    # Resolve paths
    resolved_paths = _resolve_paths(cif_paths)
    if not resolved_paths:
        raise ValueError(f"No CIF files found in {cif_paths}")

    # Parse residues
    residue_list = _parse_residues(residues)

    if verbose:
        print(f"Training {model_type} models on {len(resolved_paths)} CIF files")
        print(f"  Residues: {[r.name for r in residue_list]}")

    # Build config based on model type
    config = _build_config(model_info.config_cls, latent_dim, batch_size, lr, n_epochs, **kwargs)

    # Consolidated models train all residues together
    if model_info.consolidated:
        return _train_consolidated(
            model_info, config, resolved_paths, residue_list,
            n_epochs, batch_size, accelerator, output_dir, verbose
        )

    # Per-residue models train each residue type separately
    residue_models = {}

    for res in residue_list:
        if verbose:
            print(f"\nTraining {model_type} for {res.name}...")

        try:
            # Create data module (shared across model types)
            dm = ResidueDataModule(
                cif_paths=list(resolved_paths),
                residue=res,
                batch_size=batch_size,
            )

            # Create Lightning module
            module = model_info.module_cls(config, res)

            # Create trainer
            trainer = L.Trainer(
                max_epochs=n_epochs,
                accelerator=accelerator,
                enable_progress_bar=verbose,
                enable_model_summary=verbose,
                logger=False,
            )

            # Train
            trainer.fit(module, dm)

            # Get trained model
            model = module.get_model()
            residue_models[res] = model

            if verbose:
                print(f"  {res.name}: {model.n_atoms} atoms, latent_dim={model.latent_dim}")

            # Save individual model if output_dir specified
            if output_dir:
                model_path = Path(output_dir) / res.name
                model.save(model_path)

        except Exception as e:
            if verbose:
                print(f"  {res.name}: Failed - {e}")
            raise

    if not residue_models:
        raise ValueError("No models were trained successfully")

    # Create PolymerModel
    polymer_model = PolymerModel(residue_models)

    # Save if requested
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        polymer_model.save(output_dir)
        if verbose:
            print(f"\nSaved PolymerModel to {output_dir}")

    return polymer_model


def _train_consolidated(
    model_info: ModelTypeInfo,
    config,
    resolved_paths: list[Path],
    residue_list: list["Residue"],
    n_epochs: int,
    batch_size: int,
    accelerator: str,
    output_dir: Union[str, Path, None],
    verbose: bool,
) -> "PolymerModel":
    """Train a consolidated model (all residues together)."""
    import lightning as L
    from .polymer import PolymerModel
    from .lightning.data import ConsolidatedDataModule

    if verbose:
        print(f"\nTraining consolidated model for all residues...")

    # Create consolidated data module
    dm = ConsolidatedDataModule(
        cif_paths=list(resolved_paths),
        residues=residue_list,
        batch_size=batch_size,
    )

    # Create Lightning module
    module = model_info.module_cls(config, residue_list)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=n_epochs,
        accelerator=accelerator,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        logger=False,
    )

    # Train
    trainer.fit(module, dm)

    # Get trained model
    consolidated_model = module.get_model()

    if verbose:
        n_params = sum(p.numel() for p in consolidated_model.parameters())
        print(f"  Trained consolidated model: {n_params:,} parameters")
        for res in residue_list:
            n_atoms = len(consolidated_model._residue_atoms[res])
            print(f"    {res.name}: {n_atoms} atoms")

    # Convert to PolymerModel via as_residue_models()
    residue_models = consolidated_model.as_residue_models()
    polymer_model = PolymerModel(residue_models)

    # Save if requested
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Save the consolidated model
        consolidated_model.save(Path(output_dir) / "consolidated")
        # Also save as PolymerModel for compatibility
        polymer_model.save(output_dir)
        if verbose:
            print(f"\nSaved PolymerModel to {output_dir}")

    return polymer_model


def _build_config(config_cls: type, latent_dim: int, batch_size: int, lr: float, n_epochs: int, **kwargs):
    """Build config for a model type, handling type-specific parameters."""
    from .config import TrainingConfig

    # Extract model-specific kwargs
    config_cls_name = config_cls.__name__

    if "Flow" in config_cls_name:
        from .lightning.modules.residue_flow import (
            ResidueFlowModelConfig,
            ResidueFlowDataConfig,
        )
        model_config = ResidueFlowModelConfig(
            latent_dim=latent_dim,
            n_layers=kwargs.get("n_layers", 6),
            hidden_dim=kwargs.get("hidden_dim", 64),
            use_rotation=kwargs.get("use_rotation", True),
        )
        data_config = ResidueFlowDataConfig(batch_size=batch_size)
        return config_cls(
            model=model_config,
            data=data_config,
            training=TrainingConfig(lr=lr, epochs=n_epochs),
        )

    elif "Consolidated" in config_cls_name:
        from .lightning.modules.consolidated_vae import (
            ConsolidatedVAEModelConfig,
            ConsolidatedVAEDataConfig,
        )
        model_config = ConsolidatedVAEModelConfig(
            latent_dim=latent_dim,
            d_model=kwargs.get("d_model", 64),
            d_dist=kwargs.get("d_dist", 32),
            n_heads=kwargs.get("n_heads", 4),
            n_encoder_layers=kwargs.get("n_encoder_layers", 2),
            hidden_dims=kwargs.get("hidden_dims", [256, 128]),
            dropout=kwargs.get("dropout", 0.1),
            beta=kwargs.get("beta", 1.0),
            beta_warmup_epochs=kwargs.get("beta_warmup_epochs", 50),
            free_bits=kwargs.get("free_bits", 0.5),
            use_input_norm=kwargs.get("use_input_norm", True),
            use_residual=kwargs.get("use_residual", True),
            gamma=kwargs.get("gamma", 0.0),
            n_geom_samples=kwargs.get("n_geom_samples", 16),
        )
        data_config = ConsolidatedVAEDataConfig(batch_size=batch_size)
        return config_cls(
            model=model_config,
            data=data_config,
            training=TrainingConfig(lr=lr, epochs=n_epochs),
        )

    elif "VAE" in config_cls_name:
        from .lightning.modules.residue_vae import (
            ResidueVAEModelConfig,
            ResidueVAEDataConfig,
        )
        model_config = ResidueVAEModelConfig(
            latent_dim=latent_dim,
            hidden_dims=kwargs.get("hidden_dims", [256, 128]),
            beta=kwargs.get("beta", 1.0),
            free_bits=kwargs.get("free_bits", 0.5),
            use_residual=kwargs.get("use_residual", True),
            gamma=kwargs.get("gamma", 0.0),
            n_geom_samples=kwargs.get("n_geom_samples", 16),
        )
        data_config = ResidueVAEDataConfig(batch_size=batch_size)
        return config_cls(
            model=model_config,
            data=data_config,
            training=TrainingConfig(lr=lr, epochs=n_epochs),
        )

    else:
        # Generic fallback
        return config_cls()


def load(
    path: Union[str, Path],
    device: str = "cpu",
) -> "PolymerModel":
    """
    Load a trained PolymerModel from directory.

    Automatically detects model type (flow, vae, etc.) from saved config.

    Args:
        path: Directory containing saved model.
        device: Device to load model to.

    Returns:
        PolymerModel ready for encoding, decoding, and sampling.

    Example:
        >>> from ciffy.nn import residue
        >>> model = residue.load("models/rna_vae")
        >>> polymer = model.sample_from_sequence("acgu")
    """
    from .polymer import PolymerModel
    return PolymerModel.load(path, device=device)


__all__ = [
    "train",
    "load",
    "available_models",
    "register_model_type",
]
