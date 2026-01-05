"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.

Modules:
    - layers: Reusable neural network building blocks (DenseNetwork, Transformer, etc.)
    - diffusion: Noise schedules, diffusion processes, and EMA utilities
    - runners: Multi-job experiment and inference runners
    - flow: Normalizing flow models for polymer conformations
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

# Shared building blocks for residue models
from .blocks import (
    InputNorm,
    ResidualBlock,
    CoordinateDecoder,
    RBFDistanceEncoder,
)

from .config import (
    BaseConfig,
    TrainingConfig,
    OutputConfig,
    WandbConfig,
    MetricsLogger,
    SchedulerConfig,
    ValidationConfig,
)
from .loggers import (
    WandbLogger,
    NoOpLogger,
    create_logger,
)
from .diagnostics import (
    GradientTracker,
    ParameterTracker,
    ActivationTracker,
    LearningRateTracker,
    TrainingDiagnostics,
    DiagnosticsConfig,
    diagnose_gradients,
)


from .protocols import PolymerGenerativeModel, PolymerEncoder, PolymerPropertyPredictor
from .model_registry import register_model, get_model_class, list_registered_models
from .model_io import save_model, load_model, get_model_info, SaveableModel
from .inference import load_model_from_checkpoint, generate_samples
from .inference_config import InferenceConfig
from .split import DataSplit, DataScalingSplit, split_by_structure, create_scaling_split

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
    # Latent diffusion
    LatentDenoiserConfig,
    LatentDenoiser,
    LatentDiffusionConfig,
    LatentDiffusionModel,
    # Coordinate diffusion
    CoordinateDenoiserConfig,
    CoordinateDenoiser,
    CoordinateDiffusionConfig,
    CoordinateDiffusionModel,
)

# Polymer model and protocol (generic, works with Flow/VAE)
from .polymer import PolymerModel, PolymerFlowModel, ResidueGenerativeCore

# VAE models
from .vae import (
    ConsolidatedResidueVAE,
    ConsolidatedVAEConfig,
)

# PCA + Quantile Spline model
from .pca_quantile import (
    PCAQuantileResidueModel,
    PCAQuantileConfig,
    fit_pca_quantile,
    fit_all_residues,
)

# Hub integration for model distribution
from .hub import HubMixin, get_cache_dir, set_cache_dir

# Unified residue model training API
from . import residue

__all__ = [
    # Precision configuration
    "configure_precision",
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
    # Config framework
    "BaseConfig",
    "TrainingConfig",
    "OutputConfig",
    "WandbConfig",
    "MetricsLogger",
    "SchedulerConfig",
    "ValidationConfig",
    # Loggers
    "WandbLogger",
    "NoOpLogger",
    "create_logger",
    # Training diagnostics
    "GradientTracker",
    "ParameterTracker",
    "ActivationTracker",
    "LearningRateTracker",
    "TrainingDiagnostics",
    "DiagnosticsConfig",
    "diagnose_gradients",
    # Inference protocols and models
    "PolymerGenerativeModel",
    "PolymerEncoder",
    "PolymerPropertyPredictor",
    "SaveableModel",
    "register_model",
    "get_model_class",
    "list_registered_models",
    # Model I/O
    "save_model",
    "load_model",
    "get_model_info",
    # Inference utilities
    "load_model_from_checkpoint",
    "generate_samples",
    "InferenceConfig",
    # Data splitting
    "DataSplit",
    "DataScalingSplit",
    "split_by_structure",
    "create_scaling_split",
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
    # Diffusion config (from diffusion/)
    "DiffusionConfig",
    # Latent diffusion (from diffusion/)
    "LatentDenoiserConfig",
    "LatentDenoiser",
    "LatentDiffusionConfig",
    "LatentDiffusionModel",
    # Coordinate diffusion (from diffusion/)
    "CoordinateDenoiserConfig",
    "CoordinateDenoiser",
    "CoordinateDiffusionConfig",
    "CoordinateDiffusionModel",
    # Polymer model (generic, works with Flow/VAE)
    "PolymerModel",
    "PolymerFlowModel",  # Deprecated alias
    "ResidueGenerativeCore",
    # VAE models
    "ConsolidatedResidueVAE",
    "ConsolidatedVAEConfig",
    # PCA + Quantile Spline model
    "PCAQuantileResidueModel",
    "PCAQuantileConfig",
    "fit_pca_quantile",
    "fit_all_residues",
    # Hub integration
    "HubMixin",
    "get_cache_dir",
    "set_cache_dir",
    # Unified residue training API
    "residue",
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
