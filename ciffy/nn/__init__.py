"""
Neural network utilities for ciffy.

Provides PyTorch-compatible modules for deep learning on molecular structures.

Modules:
    - dataset: PolymerDataset for loading CIF files
    - embedding: PolymerEmbedding for learnable embeddings
    - transformer: Modern transformer with Pre-LN, RoPE, SwiGLU
    - training: Reusable training utilities
    - vae: Variational autoencoder for polymer conformations
"""

from .dataset import PolymerDataset
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

__all__ = [
    # Dataset
    "PolymerDataset",
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
]
