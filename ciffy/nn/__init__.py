"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.

Modules:
    - layers: Reusable neural network building blocks (DenseNetwork, Transformer, etc.)
    - diffusion: Noise schedules, diffusion processes, and EMA utilities
    - runners: Multi-job experiment and inference runners
    - vae: Variational autoencoder for polymer conformations
    - geometric: SO(3)-equivariant layers (optional, requires sphericart)
"""

from .dataset import PolymerDataset

# Layers (moved from root to layers/)
from .layers import (
    DenseNetwork,
    PolymerEmbedding,
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

# Runners (moved from root to runners/)
from .runners import (
    run_experiments,
    format_results_table,
    InferenceResult,
    run_inference_jobs,
    format_inference_results_table,
)

from .protocols import PolymerGenerativeModel, PolymerEncoder
from .model_registry import register_model, get_model_class
from .inference import load_model_from_checkpoint, load_vae, generate_samples
from .inference_config import InferenceConfig
from .vae import PolymerVAE, DihedralEncoder, DihedralDecoder, VAETrainer, VAEConfig

# Diffusion (moved from root to diffusion/)
from .diffusion import (
    FixedSinusoidalEmbedding,
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    DiffusionProcess,
    TimestepEmbedding,
    EMA,
    create_ema_model,
    update_ema_model,
    DiffusionConfig,
    DiffusionTrainer,
)

__all__ = [
    # Dataset
    "PolymerDataset",
    # Layers (from layers/)
    "DenseNetwork",
    "PolymerEmbedding",
    # Transformer components (from layers/)
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
    # Experiment running (from runners/)
    "ExperimentResult",
    "run_experiments",
    "format_results_table",
    # Inference protocols and models
    "PolymerGenerativeModel",
    "PolymerEncoder",
    "register_model",
    "get_model_class",
    # Inference utilities (from runners/)
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
    # Diffusion utilities (from diffusion/)
    "FixedSinusoidalEmbedding",
    "NoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    "DiffusionProcess",
    "TimestepEmbedding",
    # EMA utilities (from diffusion/)
    "EMA",
    "create_ema_model",
    "update_ema_model",
    # Diffusion trainer (from diffusion/)
    "DiffusionConfig",
    "DiffusionTrainer",
]

# Optional geometric deep learning module
# Requires sphericart: pip install ciffy[geometric]
try:
    from .geometric import (
        Repr,
        ProductRepr,
        Irrep,
        ProductIrrep,
        MatrixOutput,
        LowRankMatrixOutput,
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
        "MatrixOutput",
        "LowRankMatrixOutput",
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
