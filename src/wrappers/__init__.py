"""ONN Wrappers - Production library integrations."""

from .hf_trainer_wrapper import HFTrainerWrapper, TrainingResult, train
from .model_loader import UniversalModelLoader, load_model, register_model
from .quantization_wrapper import QuantizationWrapper, quantize_model, QuantizationType
from .batched_sparse import BatchedSparseTrainer, LoRALayer
from .accelerate_wrapper import (
    AccelerateWrapper,
    AccelerateConfig,
    DistributedStrategy,
    auto_accelerate,
    get_deepspeed_config,
)

__all__ = [
    # HuggingFace Trainer
    "HFTrainerWrapper", "TrainingResult", "train",
    # Model Loading
    "UniversalModelLoader", "load_model", "register_model",
    # Quantization
    "QuantizationWrapper", "quantize_model", "QuantizationType",
    # Sparse Training
    "BatchedSparseTrainer", "LoRALayer",
    # Distributed Training (Accelerate/DeepSpeed/FSDP)
    "AccelerateWrapper", "AccelerateConfig", "DistributedStrategy",
    "auto_accelerate", "get_deepspeed_config",
]
