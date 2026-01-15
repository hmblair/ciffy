"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning.

Generic modules:
    - layers: Reusable neural network building blocks (Transformer, MLP, etc.)
    - diffusion: Noise schedules, diffusion processes, and EMA utilities
    - flow: Normalizing flow architectures (RealNVP, NeuralSplineFlow)
    - config: Configuration framework for training
    - training: Training infrastructure and utilities
    - io: Model saving, loading, and Hub integration
    - geometric: SO(3)-equivariant layers (optional, requires sphericart)

Polymer-specific:
    - polymer: PolymerDataset, PolymerEmbedding

For polymer-specific models (ResidueVAE, AR models, diffusion models),
see science/rna-representation/nn/.
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


# Polymer-specific utilities (backward-compat re-exports)
from .polymer import PolymerDataset, PolymerEmbedding

# Layers
from .layers import (
    CausalTransformer,
    MLP,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    Transformer,
    TransformerBlock,
    create_causal_mask,
)

# Flow
from .flow import RealNVP, NeuralSplineFlow


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
    DataSplit,
    GradientTracker,
    LearningRateTracker,
    NoOpLogger,
    ParameterTracker,
    TrainingDiagnostics,
    WandbLogger,
    create_logger,
    diagnose_gradients,
    split_items,
    split_train_test,
    split_by_clusters,
    split_by_sequence_identity,
    split_by_sequence,
    split_to_directories,
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

# Diffusion (generic components only)
from .diffusion import (
    CosineNoiseSchedule,
    DiffusionProcess,
    EMA,
    FixedSinusoidalEmbedding,
    LinearNoiseSchedule,
    NoiseSchedule,
    TimestepEmbedding,
    create_ema_model,
    update_ema_model,
)

__all__ = [
    # Precision configuration
    "configure_precision",
    # Polymer utilities
    "PolymerDataset",
    "PolymerEmbedding",
    # Layers
    "MLP",
    "Transformer",
    "CausalTransformer",
    # Flow
    "RealNVP",
    "NeuralSplineFlow",
    # Diffusion (generic)
    "NoiseSchedule",
    "CosineNoiseSchedule",
    "DiffusionProcess",
    "EMA",
    # Training utilities
    "split_items",
    "DataSplit",
    # Protocols
    "PolymerGenerativeModel",
    "PolymerEncoder",
]

# Optional geometric deep learning module
# Requires sphericart: pip install ciffy[geometric]
try:
    from .geometric import EquivariantTransformer, RadialBasisFunctions
    GEOMETRIC_AVAILABLE = True
    __all__.extend(["EquivariantTransformer", "GEOMETRIC_AVAILABLE"])
except ImportError:
    GEOMETRIC_AVAILABLE = False
    __all__.append("GEOMETRIC_AVAILABLE")
