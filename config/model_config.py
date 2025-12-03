"""Model hyperparameters and architecture configuration.

Centralizes all model-related settings to avoid hardcoded values scattered
throughout the codebase. Supports loading from YAML/JSON for experiments.
Uses Pydantic for validation with dataclass fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

# Try Pydantic first (preferred), fallback to dataclass
try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    from dataclasses import dataclass


if PYDANTIC_AVAILABLE:
    class ModelConfig(BaseModel):
        """Configuration for HFCompatibleLM architecture with validation."""
        
        # Architecture
        vocab_size: int = Field(default=49280, ge=1000)
        hidden_size: int = Field(default=960, ge=128)
        num_layers: int = Field(default=32, ge=1, le=128)
        num_heads: int = Field(default=15, ge=1)
        num_kv_heads: int = Field(default=5, ge=1)
        intermediate_size: int = Field(default=2560, ge=128)
        max_seq_len: int = Field(default=2048, ge=128, le=32768)
        
        # Embeddings
        pad_token_id: int = Field(default=0, ge=0)
        rope_base: int = Field(default=10000, ge=1000)
        
        # Attention
        use_flash_attention: bool = True
        attention_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
        
        # Normalization
        rms_norm_eps: float = Field(default=1e-6, gt=0.0)
        
        @field_validator("num_kv_heads")
        @classmethod
        def validate_kv_heads(cls, v: int, info) -> int:
            num_heads = info.data.get("num_heads", 15)
            if v > num_heads:
                raise ValueError(f"num_kv_heads ({v}) cannot exceed num_heads ({num_heads})")
            if num_heads % v != 0:
                raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({v})")
            return v
        
        model_config = {"extra": "forbid"}  # Catch typos in config files
        
        def to_dict(self) -> dict:
            return self.model_dump()
        
        @classmethod
        def from_dict(cls, d: dict) -> "ModelConfig":
            return cls(**d)
        
        @classmethod
        def from_json(cls, path: Path) -> "ModelConfig":
            with open(path, "r") as f:
                return cls(**json.load(f))
        
        def save_json(self, path: Path) -> None:
            with open(path, "w") as f:
                json.dump(self.model_dump(), f, indent=2)

    class TrainingConfig(BaseModel):
        """Configuration for training runs with validation."""
        
        # Optimization
        learning_rate: float = Field(default=2e-5, gt=0.0, le=1.0)
        weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
        warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
        max_grad_norm: float = Field(default=1.0, gt=0.0)
        
        # Batch settings
        batch_size: int = Field(default=4, ge=1, le=1024)
        gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
        
        # Schedule
        epochs: int = Field(default=3, ge=1, le=1000)
        save_every: int = Field(default=0, ge=0)
        max_checkpoints: int = Field(default=3, ge=1, le=100)
        eval_steps: int = Field(default=500, ge=1)
        
        # Memory optimization
        use_amp: bool = True
        amp_dtype: str = Field(default="float16")
        gradient_checkpointing: bool = True
        
        # Advanced
        use_compile: bool = False
        compile_mode: str = Field(default="reduce-overhead")
        use_ema: bool = False
        ema_decay: float = Field(default=0.999, ge=0.9, le=1.0)
        label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
        
        @field_validator("amp_dtype")
        @classmethod
        def validate_amp_dtype(cls, v: str) -> str:
            if v not in {"float16", "bfloat16"}:
                raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
            return v
        
        model_config = {"extra": "forbid"}
        
        def to_dict(self) -> dict:
            return self.model_dump()
        
        @classmethod
        def from_dict(cls, d: dict) -> "TrainingConfig":
            return cls(**d)

    class InferenceConfig(BaseModel):
        """Configuration for inference/generation with validation."""
        
        # Sampling
        temperature: float = Field(default=0.7, gt=0.0, le=2.0)
        top_p: float = Field(default=0.9, gt=0.0, le=1.0)
        top_k: int = Field(default=50, ge=0)
        repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
        
        # Generation limits
        max_new_tokens: int = Field(default=256, ge=1, le=8192)
        min_new_tokens: int = Field(default=1, ge=0)
        
        # Optimization
        use_cache: bool = True  # KV-cache
        use_flash_attention: bool = True
        
        # Batching
        max_batch_size: int = Field(default=8, ge=1, le=128)
        batch_timeout_ms: int = Field(default=50, ge=1, le=1000)
        
        def to_dict(self) -> dict:
            return self.model_dump()

else:
    # Fallback to dataclass if pydantic not available
    @dataclass
    class ModelConfig:
        """Configuration for HFCompatibleLM architecture."""
        
        # Architecture
        vocab_size: int = 49280
        hidden_size: int = 960
        num_layers: int = 32
        num_heads: int = 15
        num_kv_heads: int = 5  # For grouped-query attention
        intermediate_size: int = 2560
        max_seq_len: int = 2048
        
        # Embeddings
        pad_token_id: int = 0
        rope_base: int = 10000
        
        # Attention
        use_flash_attention: bool = True
        attention_dropout: float = 0.0
        
        # Normalization
        rms_norm_eps: float = 1e-6
        
        def to_dict(self) -> dict:
            return {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "intermediate_size": self.intermediate_size,
                "max_seq_len": self.max_seq_len,
                "pad_token_id": self.pad_token_id,
                "rope_base": self.rope_base,
                "use_flash_attention": self.use_flash_attention,
                "attention_dropout": self.attention_dropout,
                "rms_norm_eps": self.rms_norm_eps,
            }
        
        @classmethod
        def from_dict(cls, d: dict) -> "ModelConfig":
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        
        @classmethod
        def from_json(cls, path: Path) -> "ModelConfig":
            with open(path, "r") as f:
                return cls.from_dict(json.load(f))
        
        def save_json(self, path: Path) -> None:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

    @dataclass
    class TrainingConfig:
        """Configuration for training runs."""
        
        # Optimization
        learning_rate: float = 2e-5
        weight_decay: float = 0.01
        warmup_ratio: float = 0.1
        max_grad_norm: float = 1.0
        
        # Batch settings
        batch_size: int = 4
        gradient_accumulation_steps: int = 4
        
        # Schedule
        epochs: int = 3
        save_every: int = 0  # 0 = only best/final
        max_checkpoints: int = 3
        eval_steps: int = 500
        
        # Memory optimization
        use_amp: bool = True
        amp_dtype: str = "float16"  # "float16" or "bfloat16"
        gradient_checkpointing: bool = True
        
        # Advanced
        use_compile: bool = False
        compile_mode: str = "reduce-overhead"
        use_ema: bool = False
        ema_decay: float = 0.999
        label_smoothing: float = 0.0
        
        def to_dict(self) -> dict:
            return {k: getattr(self, k) for k in self.__dataclass_fields__}
        
        @classmethod
        def from_dict(cls, d: dict) -> "TrainingConfig":
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @dataclass
    class InferenceConfig:
        """Configuration for inference/generation."""
        
        # Sampling
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.1
        
        # Generation limits
        max_new_tokens: int = 256
        min_new_tokens: int = 1
        
        # Optimization
        use_cache: bool = True  # KV-cache
        use_flash_attention: bool = True
        
        # Batching
        max_batch_size: int = 8
        batch_timeout_ms: int = 50  # Max wait for batching
        
        def to_dict(self) -> dict:
            return {k: getattr(self, k) for k in self.__dataclass_fields__}


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


def load_model_config(path: Optional[Path] = None) -> ModelConfig:
    """Load model config from JSON or return defaults."""
    if path and path.exists():
        return ModelConfig.from_json(path)
    return DEFAULT_MODEL_CONFIG


def load_training_config(path: Optional[Path] = None) -> TrainingConfig:
    """Load training config from JSON or return defaults."""
    if path and path.exists():
        with open(path, "r") as f:
            return TrainingConfig.from_dict(json.load(f))
    return DEFAULT_TRAINING_CONFIG
