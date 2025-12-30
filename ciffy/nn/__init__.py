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

from .data_validation import (
    DataCompatibilityReport,
    StructureExample,
    validate_flow_model_compatibility,
)
from .dataset import PolymerDataset
from .filtered_dataset import FilterConfig, FilteredPolymerDataset

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

# Flow models (triggers @register_model decorators)
from .flow import PolymerFlowModel

# Hub integration for model distribution
from .hub import HubMixin, get_cache_dir, set_cache_dir

__all__ = [
    # Data validation and filtering
    "DataCompatibilityReport",
    "StructureExample",
    "validate_flow_model_compatibility",
    "FilterConfig",
    "FilteredPolymerDataset",
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
    # Flow models
    "PolymerFlowModel",
    # Hub integration
    "HubMixin",
    "get_cache_dir",
    "set_cache_dir",
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
