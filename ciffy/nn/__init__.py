"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.

Modules:
    - layers: Reusable neural network building blocks (DenseNetwork, Transformer, etc.)
    - diffusion: Noise schedules, diffusion processes, and EMA utilities
    - config: Configuration framework for training
    - training: Training infrastructure and utilities
    - io: Model saving, loading, and Hub integration
    - flow: Normalizing flow models for polymer conformations
    - vae: Variational autoencoder models
    - autoregressive: Autoregressive models for polymer generation
    - geometric: SO(3)-equivariant layers (optional, requires sphericart)
"""

import torch


def configure_precision(
    tf32: bool = True,
    matmul_precision: str = "high",
) -> None:
    """Configure GPU precision settings for optimal performance.

    Enables TF32 tensor core operations and sets matrix multiplication precision.
    TF32 provides up to 3x speedup on Ampere+ GPUs (A100, RTX 30xx, RTX 40xx)
    with minimal accuracy loss for most deep learning workloads.

    Call this once at the start of your script before creating models.

    Args:
        tf32: Enable TF32 for matmul and cuDNN operations. Default True.
        matmul_precision: Precision for float32 matmuls. Options:
            - "highest": Full float32 precision (slowest, most accurate)
            - "high": TF32 for internal computations (good balance)
            - "medium": TF32 with reduced accumulation precision (fastest)

    Example:
        >>> import ciffy.nn as nn
        >>> nn.configure_precision(tf32=True, matmul_precision="high")
        >>> # Now all models will use TF32 tensor cores on compatible GPUs
    """
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision(matmul_precision)


# Core
from .dataset import PolymerDataset

# Layers
from .layers import (
    CausalMultiHeadAttention,
    CausalTransformer,
    CausalTransformerBlock,
    DenseNetwork,
    MultiHeadAttention,
    PolymerEmbedding,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    Transformer,
    TransformerBlock,
    create_causal_mask,
)

# Shared building blocks
from .blocks import (
    CoordinateDecoder,
    InputNorm,
    RBFDistanceEncoder,
    ResidualBlock,
)

# Config (from config/ submodule)
from .config import (
    BaseConfig,
    DataConfig,
    DiagnosticsConfig,
    InferenceConfig,
    MetricsLogger,
    OutputConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
    WandbConfig,
    get_device,
)

# Training (from training/ submodule)
from .training import (
    ActivationTracker,
    DataScalingSplit,
    DataSplit,
    GradientTracker,
    LearningRateTracker,
    NoOpLogger,
    ParameterTracker,
    TrainingDiagnostics,
    WandbLogger,
    create_logger,
    create_scaling_split,
    diagnose_gradients,
    split_by_structure,
)

# I/O (from io/ submodule)
from .io import (
    HubMixin,
    SaveableModel,
    generate_samples,
    get_cache_dir,
    get_model_class,
    get_model_info,
    list_registered_models,
    load_model,
    load_model_from_checkpoint,
    register_model,
    save_model,
    set_cache_dir,
)

# Protocols
from .protocols import PolymerEncoder, PolymerGenerativeModel, PolymerPropertyPredictor

# Diffusion
from .diffusion import (
    CoordinateDenoiser,
    CoordinateDenoiserConfig,
    CoordinateDiffusionConfig,
    CoordinateDiffusionModel,
    CosineNoiseSchedule,
    DiffusionConfig,
    DiffusionProcess,
    EMA,
    FixedSinusoidalEmbedding,
    LatentDenoiser,
    LatentDenoiserConfig,
    LatentDiffusionConfig,
    LatentDiffusionModel,
    LinearNoiseSchedule,
    NoiseSchedule,
    TimestepEmbedding,
    create_ema_model,
    update_ema_model,
)

# Residue-level encoder/decoder (new vectorized implementation)
from .residue import (
    ResidueDecoder,
    ResidueEncoder,
    ResidueVAE,
)

# Autoregressive models
from .autoregressive import (
    AtomARModel,
    AtomARModelConfig,
    CoordinateARModel,
    CoordinateARModelConfig,
    PolymerLatentARModel,
    ResidueLatentARModel,
    ResidueLatentARModelConfig,
)

__all__ = [
    # Precision configuration
    "configure_precision",
    # Dataset
    "PolymerDataset",
    # Layers
    "DenseNetwork",
    "PolymerEmbedding",
    "Transformer",
    "TransformerBlock",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLU",
    "CausalTransformer",
    "CausalTransformerBlock",
    "CausalMultiHeadAttention",
    "create_causal_mask",
    # Blocks
    "InputNorm",
    "ResidualBlock",
    "CoordinateDecoder",
    "RBFDistanceEncoder",
    # Config
    "BaseConfig",
    "DataConfig",
    "TrainingConfig",
    "OutputConfig",
    "WandbConfig",
    "MetricsLogger",
    "SchedulerConfig",
    "ValidationConfig",
    "InferenceConfig",
    "get_device",
    # Training
    "WandbLogger",
    "NoOpLogger",
    "create_logger",
    "GradientTracker",
    "ParameterTracker",
    "ActivationTracker",
    "LearningRateTracker",
    "TrainingDiagnostics",
    "DiagnosticsConfig",
    "diagnose_gradients",
    "DataSplit",
    "DataScalingSplit",
    "split_by_structure",
    "create_scaling_split",
    # I/O
    "save_model",
    "load_model",
    "get_model_info",
    "SaveableModel",
    "register_model",
    "get_model_class",
    "list_registered_models",
    "load_model_from_checkpoint",
    "generate_samples",
    "HubMixin",
    "get_cache_dir",
    "set_cache_dir",
    # Protocols
    "PolymerGenerativeModel",
    "PolymerEncoder",
    "PolymerPropertyPredictor",
    # Diffusion
    "FixedSinusoidalEmbedding",
    "NoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    "DiffusionProcess",
    "TimestepEmbedding",
    "EMA",
    "create_ema_model",
    "update_ema_model",
    "DiffusionConfig",
    "LatentDenoiserConfig",
    "LatentDenoiser",
    "LatentDiffusionConfig",
    "LatentDiffusionModel",
    "CoordinateDenoiserConfig",
    "CoordinateDenoiser",
    "CoordinateDiffusionConfig",
    "CoordinateDiffusionModel",
    # Residue-level encoder/decoder
    "ResidueEncoder",
    "ResidueDecoder",
    "ResidueVAE",
    # Autoregressive models
    "ResidueLatentARModel",
    "ResidueLatentARModelConfig",
    "PolymerLatentARModel",
    "CoordinateARModel",
    "CoordinateARModelConfig",
    "AtomARModel",
    "AtomARModelConfig",
]

# Optional geometric deep learning module
# Requires sphericart: pip install ciffy[geometric]
try:
    from .geometric import (
        EquivariantAttention,
        EquivariantLinear,
        EquivariantTransformer,
        EquivariantTransformerBlock,
        Irrep,
        LowRankMatrixOutput,
        MatrixOutput,
        ProductIrrep,
        ProductRepr,
        RadialBasisFunctions,
        Repr,
        SphericalHarmonic,
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
