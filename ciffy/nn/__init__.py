"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.

Modules:
    - dataset: PolymerDataset for loading CIF files
    - embedding: PolymerEmbedding for learnable embeddings
    - transformer: Modern transformer with Pre-LN, RoPE, SwiGLU
    - training: Reusable training utilities
    - vae: Variational autoencoder for polymer conformations
    - dense_network: Simple MLP building block
    - diffusion: Noise schedules and diffusion process utilities
    - ema: Exponential moving average for model weights
    - geometric: SO(3)-equivariant layers (optional, requires sphericart)
"""

from .dataset import PolymerDataset
from .dense_network import DenseNetwork
from .embedding import PolymerEmbedding
from .transformer import (
    Transformer,
    TransformerBlock,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)
from .training import (
    ExperimentResult,
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    polymer_collate_fn,
    get_worker_init_fn,
    BetaScheduler,
)
from .base_trainer import (
    BaseConfig,
    BaseTrainer,
    TrainingConfig,
    OutputConfig,
    WandbConfig,
    MetricsLogger,
)
from .loggers import (
    WandbLogger,
    NoOpLogger,
    create_logger,
)
from .experiment_runner import (
    run_experiments,
    format_results_table,
)
from .protocols import PolymerGenerativeModel, PolymerEncoder
from .model_registry import register_model, get_model_class
from .inference import load_model_from_checkpoint, load_vae, generate_samples
from .inference_config import InferenceConfig
from .inference_runner import InferenceResult, run_inference_jobs, format_inference_results_table
from .vae import PolymerVAE, DihedralEncoder, DihedralDecoder, VAETrainer, VAEConfig
from .diffusion import (
    FixedSinusoidalEmbedding,
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    DiffusionProcess,
    TimestepEmbedding,
)
from .ema import EMA, create_ema_model, update_ema_model

__all__ = [
    # Dataset
    "PolymerDataset",
    # Dense network
    "DenseNetwork",
    # Embedding
    "PolymerEmbedding",
    # Transformer components
    "Transformer",
    "TransformerBlock",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
    # Training utilities
    "set_seed",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "train_epoch",
    "polymer_collate_fn",
    "get_worker_init_fn",
    "BetaScheduler",
    # Base trainer framework
    "BaseConfig",
    "BaseTrainer",
    "TrainingConfig",
    "OutputConfig",
    "WandbConfig",
    "MetricsLogger",
    # Loggers
    "WandbLogger",
    "NoOpLogger",
    "create_logger",
    # Experiment running
    "ExperimentResult",
    "run_experiments",
    "format_results_table",
    # Inference protocols and models
    "PolymerGenerativeModel",
    "PolymerEncoder",
    "register_model",
    "get_model_class",
    # Inference utilities
    "load_model_from_checkpoint",
    "load_vae",
    "generate_samples",
    "InferenceConfig",
    "InferenceResult",
    "run_inference_jobs",
    "format_inference_results_table",
    # VAE
    "PolymerVAE",
    "DihedralEncoder",
    "DihedralDecoder",
    "VAETrainer",
    "VAEConfig",
    # Diffusion utilities
    "FixedSinusoidalEmbedding",
    "NoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    "DiffusionProcess",
    "TimestepEmbedding",
    # EMA utilities
    "EMA",
    "create_ema_model",
    "update_ema_model",
]

# Optional geometric deep learning module
# Requires sphericart: pip install ciffy[geometric]
try:
    from .geometric import (
        Repr,
        ProductRepr,
        Irrep,
        ProductIrrep,
        EquivariantLinear,
        EquivariantTransformer,
        EquivariantAttention,
        EquivariantTransformerBlock,
        SphericalHarmonic,
        RadialBasisFunctions,
        build_knn_graph,
    )
    GEOMETRIC_AVAILABLE = True
    __all__.extend([
        "Repr",
        "ProductRepr",
        "Irrep",
        "ProductIrrep",
        "EquivariantLinear",
        "EquivariantTransformer",
        "EquivariantAttention",
        "EquivariantTransformerBlock",
        "SphericalHarmonic",
        "RadialBasisFunctions",
        "build_knn_graph",
        "GEOMETRIC_AVAILABLE",
    ])
except ImportError:
    GEOMETRIC_AVAILABLE = False
    __all__.append("GEOMETRIC_AVAILABLE")
