"""Configuration Orchestrator - Intelligent configuration generation.

The brain of ONN 2.0: takes hardware profile and model requirements,
outputs optimal training configuration for HuggingFace Trainer and DeepSpeed.

Decision Logic:
1. Analyze hardware constraints (VRAM, RAM, CPU)
2. Determine optimal precision (fp32/fp16/bf16/int8/int4)
3. Calculate batch size and gradient accumulation
4. Select parallelism strategy (DDP, FSDP, DeepSpeed ZeRO)
5. Configure memory optimizations (gradient checkpointing, CPU offload)
6. Generate HuggingFace TrainingArguments
7. Generate DeepSpeed config if needed
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .hardware_profiler import HardwareProfile, HardwareProfiler, get_profiler


class Precision(Enum):
    """Training precision options."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class ParallelismStrategy(Enum):
    """Distributed training strategies."""
    NONE = "none"               # Single GPU, no parallelism
    DDP = "ddp"                 # Data Distributed Parallel
    FSDP = "fsdp"               # Fully Sharded Data Parallel
    DEEPSPEED_ZERO1 = "zero1"   # DeepSpeed ZeRO Stage 1
    DEEPSPEED_ZERO2 = "zero2"   # DeepSpeed ZeRO Stage 2
    DEEPSPEED_ZERO3 = "zero3"   # DeepSpeed ZeRO Stage 3


@dataclass
class QuantizationConfig:
    """Quantization settings."""
    enabled: bool = False
    precision: Precision = Precision.FP16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration."""
    enabled: bool = False
    stage: int = 2  # ZeRO stage
    offload_optimizer: bool = False
    offload_param: bool = False
    overlap_comm: bool = True
    reduce_bucket_size: int = 5e8
    allgather_bucket_size: int = 5e8
    
    def to_dict(self) -> Dict[str, Any]:
        """Generate DeepSpeed JSON config."""
        config = {
            "zero_optimization": {
                "stage": self.stage,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none",
                },
                "offload_param": {
                    "device": "cpu" if self.offload_param else "none",
                },
                "overlap_comm": self.overlap_comm,
                "reduce_bucket_size": self.reduce_bucket_size,
                "allgather_bucket_size": self.allgather_bucket_size,
            },
            "gradient_clipping": 1.0,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
        }
        
        if self.stage >= 2:
            config["zero_optimization"]["contiguous_gradients"] = True
        
        if self.stage >= 3:
            config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7
            config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e5
        
        return config


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Core settings
    device: str = "auto"
    num_devices: int = 1
    precision: Precision = Precision.FP16
    
    # Batch settings
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 1
    
    # Memory optimizations
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    
    # Parallelism
    parallelism: ParallelismStrategy = ParallelismStrategy.NONE
    deepspeed_config: Optional[DeepSpeedConfig] = None
    
    # Quantization
    quantization: Optional[QuantizationConfig] = None
    
    # Training settings
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Advanced
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Metadata
    config_reasoning: List[str] = field(default_factory=list)
    
    def to_hf_training_args(self) -> Dict[str, Any]:
        """Generate HuggingFace TrainingArguments kwargs."""
        args = {
            "per_device_train_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "num_train_epochs": self.num_epochs,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps if self.eval_steps > 0 else None,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "gradient_checkpointing": self.gradient_checkpointing,
            "report_to": [],  # We handle tracking ourselves
        }
        
        # Precision settings
        if self.precision == Precision.FP16:
            args["fp16"] = True
            args["bf16"] = False
        elif self.precision == Precision.BF16:
            args["fp16"] = False
            args["bf16"] = True
        else:
            args["fp16"] = False
            args["bf16"] = False
        
        # DeepSpeed
        if self.deepspeed_config and self.deepspeed_config.enabled:
            args["deepspeed"] = self.deepspeed_config.to_dict()
        
        return args
    
    def to_quantization_config(self) -> Optional[Dict[str, Any]]:
        """Generate bitsandbytes config for model loading."""
        if not self.quantization or not self.quantization.enabled:
            return None
        
        
        if self.quantization.load_in_4bit:
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": self.quantization.bnb_4bit_compute_dtype,
                "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
            }
        elif self.quantization.load_in_8bit:
            return {"load_in_8bit": True}
        
        return None
    
    def summary(self) -> str:
        """Get human-readable configuration summary."""
        eff_batch = self.effective_batch_size
        lines = [
            "Training Configuration:",
            f"  Device: {self.device} (x{self.num_devices})",
            f"  Precision: {self.precision.value}",
            f"  Batch: {self.per_device_batch_size} x {self.gradient_accumulation_steps} = {eff_batch}",
            f"  Gradient Checkpointing: {self.gradient_checkpointing}",
            f"  CPU Offload: {self.cpu_offload}",
            f"  Parallelism: {self.parallelism.value}",
        ]
        
        if self.quantization and self.quantization.enabled:
            lines.append(f"  Quantization: {'INT4' if self.quantization.load_in_4bit else 'INT8'}")
        
        if self.config_reasoning:
            lines.append("\nConfiguration Decisions:")
            for reason in self.config_reasoning:
                lines.append(f"  → {reason}")
        
        return "\n".join(lines)


class ConfigOrchestrator:
    """Intelligently generates optimal training configurations."""
    
    def __init__(
        self,
        profiler: Optional[HardwareProfiler] = None,
        target_effective_batch_size: int = 32,
    ):
        """Initialize orchestrator.
        
        Args:
            profiler: Hardware profiler instance (uses global if None).
            target_effective_batch_size: Default effective batch size goal.
        """
        self.profiler = profiler or get_profiler()
        self.target_batch_size = target_effective_batch_size
    
    def orchestrate(
        self,
        model_name_or_path: str,
        num_params: Optional[int] = None,
        task: str = "causal_lm",
        dataset_size: Optional[int] = None,
        max_seq_len: int = 512,
        prefer_device: Optional[str] = None,
        target_batch_size: Optional[int] = None,
        num_epochs: int = 3,
        learning_rate: Optional[float] = None,
        force_precision: Optional[str] = None,
        force_quantization: Optional[str] = None,
    ) -> TrainingConfig:
        """Generate optimal training configuration.
        
        Args:
            model_name_or_path: Model identifier (for size estimation).
            num_params: Explicit parameter count (overrides estimation).
            task: Training task ("causal_lm", "seq2seq", "classification").
            dataset_size: Number of training samples.
            max_seq_len: Maximum sequence length.
            prefer_device: User device preference.
            target_batch_size: Desired effective batch size.
            num_epochs: Training epochs.
            learning_rate: Learning rate (None = auto).
            force_precision: Force specific precision.
            force_quantization: Force quantization ("int4", "int8", None).
        
        Returns:
            Optimized TrainingConfig.
        """
        profile = self.profiler.profile()
        config = TrainingConfig()
        config.config_reasoning = []
        
        # Step 1: Estimate model size if not provided
        if num_params is None:
            num_params = self._estimate_model_params(model_name_or_path)
        
        config.config_reasoning.append(
            f"Model estimated at {num_params / 1e6:.1f}M parameters"
        )
        
        # Step 2: Select device
        config.device = self.profiler.get_optimal_device(
            prefer_device=prefer_device if prefer_device != "auto" else None
        )
        config.num_devices = len(profile.gpus) if profile.gpus else 1
        config.config_reasoning.append(f"Selected device: {config.device}")
        
        # Step 3: Determine precision and quantization
        available_vram = profile.available_vram_mb if profile.has_gpu else float("inf")
        config, required_vram = self._select_precision_and_quantization(
            config=config,
            profile=profile,
            num_params=num_params,
            max_seq_len=max_seq_len,
            available_vram=available_vram,
            force_precision=force_precision,
            force_quantization=force_quantization,
        )
        
        # Step 4: Calculate batch size
        target = target_batch_size or self.target_batch_size
        config = self._calculate_batch_size(
            config=config,
            profile=profile,
            num_params=num_params,
            max_seq_len=max_seq_len,
            available_vram=available_vram,
            required_vram=required_vram,
            target_batch_size=target,
        )
        
        # Step 5: Select parallelism strategy
        config = self._select_parallelism(
            config=config,
            profile=profile,
            num_params=num_params,
        )
        
        # Step 6: Memory optimizations
        config = self._configure_memory_optimizations(
            config=config,
            profile=profile,
            num_params=num_params,
            available_vram=available_vram,
        )
        
        # Step 7: Set training hyperparameters
        config.num_epochs = num_epochs
        config.learning_rate = learning_rate or self._auto_lr(num_params, config.effective_batch_size)
        config.config_reasoning.append(
            f"Learning rate: {config.learning_rate:.2e}"
        )
        
        return config
    
    def _estimate_model_params(self, model_name_or_path: str) -> int:
        """Estimate model parameters from name."""
        name_lower = model_name_or_path.lower()
        
        # Pattern matching for common models
        size_patterns = {
            "70b": 70_000_000_000,
            "65b": 65_000_000_000,
            "40b": 40_000_000_000,
            "34b": 34_000_000_000,
            "30b": 30_000_000_000,
            "13b": 13_000_000_000,
            "7b": 7_000_000_000,
            "3b": 3_000_000_000,
            "2.7b": 2_700_000_000,
            "1.5b": 1_500_000_000,
            "1.3b": 1_300_000_000,
            "1b": 1_000_000_000,
            "774m": 774_000_000,
            "350m": 350_000_000,
            "125m": 125_000_000,
            "small": 125_000_000,
            "medium": 350_000_000,
            "large": 774_000_000,
            "xl": 1_500_000_000,
            "xxl": 11_000_000_000,
            "nano": 8_000_000,
            "tiny": 32_000_000,
        }
        
        for pattern, params in size_patterns.items():
            if pattern in name_lower:
                return params
        
        # Default to medium-sized model
        return 350_000_000
    
    def _select_precision_and_quantization(
        self,
        config: TrainingConfig,
        profile: HardwareProfile,
        num_params: int,
        max_seq_len: int,
        available_vram: float,
        force_precision: Optional[str],
        force_quantization: Optional[str],
    ) -> tuple[TrainingConfig, float]:
        """Select optimal precision and quantization."""
        
        # Calculate VRAM requirements for each precision
        vram_fp32 = self.profiler.estimate_model_vram_mb(num_params, "fp32", True, True, 1, max_seq_len)
        vram_fp16 = self.profiler.estimate_model_vram_mb(num_params, "fp16", True, True, 1, max_seq_len)
        vram_int8 = self.profiler.estimate_model_vram_mb(num_params, "int8", True, True, 1, max_seq_len)
        vram_int4 = self.profiler.estimate_model_vram_mb(num_params, "int4", True, True, 1, max_seq_len)
        
        # Handle forced settings
        if force_quantization == "int4":
            config.precision = Precision.INT4
            config.quantization = QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
                precision=Precision.INT4,
            )
            config.config_reasoning.append("INT4 quantization forced by user")
            return config, vram_int4
        
        if force_quantization == "int8":
            config.precision = Precision.INT8
            config.quantization = QuantizationConfig(
                enabled=True,
                load_in_8bit=True,
                precision=Precision.INT8,
            )
            config.config_reasoning.append("INT8 quantization forced by user")
            return config, vram_int8
        
        if force_precision:
            precision_map = {
                "fp32": Precision.FP32,
                "fp16": Precision.FP16,
                "bf16": Precision.BF16,
            }
            config.precision = precision_map.get(force_precision, Precision.FP16)
            config.config_reasoning.append(f"Precision {force_precision} forced by user")
            return config, vram_fp16 if force_precision in ("fp16", "bf16") else vram_fp32
        
        # Auto-select based on available VRAM
        if available_vram >= vram_fp16 * 1.2:  # 20% safety margin
            # Use BF16 if supported, otherwise FP16
            if profile.supports_bf16:
                config.precision = Precision.BF16
                config.config_reasoning.append(
                    f"BF16 selected (need {vram_fp16:.0f}MB, have {available_vram:.0f}MB)"
                )
            else:
                config.precision = Precision.FP16
                config.config_reasoning.append(
                    f"FP16 selected (need {vram_fp16:.0f}MB, have {available_vram:.0f}MB)"
                )
            return config, vram_fp16
        
        elif available_vram >= vram_int8 * 1.2:
            config.precision = Precision.INT8
            config.quantization = QuantizationConfig(
                enabled=True,
                load_in_8bit=True,
                precision=Precision.INT8,
            )
            config.config_reasoning.append(
                f"INT8 quantization enabled (need {vram_int8:.0f}MB, have {available_vram:.0f}MB)"
            )
            return config, vram_int8
        
        elif available_vram >= vram_int4 * 1.2:
            config.precision = Precision.INT4
            config.quantization = QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
                precision=Precision.INT4,
            )
            config.config_reasoning.append(
                f"INT4 quantization enabled (need {vram_int4:.0f}MB, have {available_vram:.0f}MB)"
            )
            return config, vram_int4
        
        else:
            # Not enough VRAM even for INT4, will need CPU offloading
            config.precision = Precision.INT4
            config.quantization = QuantizationConfig(
                enabled=True,
                load_in_4bit=True,
                precision=Precision.INT4,
            )
            config.cpu_offload = True
            config.config_reasoning.append(
                "Extreme memory constraints: INT4 + CPU offload needed"
            )
            return config, vram_int4
    
    def _calculate_batch_size(
        self,
        config: TrainingConfig,
        profile: HardwareProfile,
        num_params: int,
        max_seq_len: int,
        available_vram: float,
        required_vram: float,
        target_batch_size: int,
    ) -> TrainingConfig:
        """Calculate optimal batch size and gradient accumulation."""
        
        # Start with batch size 1 and calculate how many we can fit
        remaining_vram = available_vram - required_vram
        
        # Estimate activation memory per sample
        # Very rough: ~4 bytes per element, hidden_dim * seq_len * num_layers
        hidden_dim = int(math.sqrt(num_params / 12))  # Rough estimate
        num_layers = max(1, num_params // (hidden_dim * hidden_dim * 12))
        activation_per_sample = (hidden_dim * max_seq_len * num_layers * 4) / (1024**2)
        
        # Calculate max batch size that fits in VRAM
        if config.device == "cpu":
            max_batch_size = 8  # CPU can handle more, but slowly
        else:
            max_batch_size = max(1, int(remaining_vram / activation_per_sample))
            max_batch_size = min(max_batch_size, 32)  # Cap at reasonable max
        
        # Set per-device batch size
        config.per_device_batch_size = max_batch_size
        
        # Calculate gradient accumulation to reach target
        if max_batch_size >= target_batch_size:
            config.gradient_accumulation_steps = 1
        else:
            config.gradient_accumulation_steps = max(1, target_batch_size // max_batch_size)
        
        config.effective_batch_size = config.per_device_batch_size * config.gradient_accumulation_steps
        
        config.config_reasoning.append(
            f"Batch size: {config.per_device_batch_size} x {config.gradient_accumulation_steps} "
            f"= {config.effective_batch_size} effective"
        )
        
        return config
    
    def _select_parallelism(
        self,
        config: TrainingConfig,
        profile: HardwareProfile,
        num_params: int,
    ) -> TrainingConfig:
        """Select distributed training strategy."""
        
        num_gpus = len(profile.gpus)
        
        if num_gpus <= 1:
            config.parallelism = ParallelismStrategy.NONE
            return config
        
        # Multi-GPU: choose between DDP and DeepSpeed
        if num_params > 1_000_000_000:  # > 1B params
            # Large model: use DeepSpeed ZeRO
            if config.cpu_offload:
                config.parallelism = ParallelismStrategy.DEEPSPEED_ZERO3
                config.deepspeed_config = DeepSpeedConfig(
                    enabled=True,
                    stage=3,
                    offload_optimizer=True,
                    offload_param=True,
                )
                config.config_reasoning.append(
                    "DeepSpeed ZeRO-3 with offloading for very large model"
                )
            else:
                config.parallelism = ParallelismStrategy.DEEPSPEED_ZERO2
                config.deepspeed_config = DeepSpeedConfig(
                    enabled=True,
                    stage=2,
                    offload_optimizer=False,
                )
                config.config_reasoning.append(
                    "DeepSpeed ZeRO-2 for large model"
                )
        else:
            # Smaller model: DDP is sufficient
            config.parallelism = ParallelismStrategy.DDP
            config.config_reasoning.append("DDP for multi-GPU training")
        
        return config
    
    def _configure_memory_optimizations(
        self,
        config: TrainingConfig,
        profile: HardwareProfile,
        num_params: int,
        available_vram: float,
    ) -> TrainingConfig:
        """Configure memory optimization strategies."""
        
        # Enable gradient checkpointing for large models or tight VRAM
        if num_params > 500_000_000 or available_vram < 8000:
            config.gradient_checkpointing = True
            config.config_reasoning.append(
                "Gradient checkpointing enabled for memory efficiency"
            )
        
        # Reduce dataloader workers if memory is tight
        if profile.memory and profile.memory.available_ram_mb < 8000:
            config.dataloader_num_workers = 2
            config.config_reasoning.append(
                "Reduced dataloader workers due to limited RAM"
            )
        
        return config
    
    def _auto_lr(self, num_params: int, batch_size: int) -> float:
        """Calculate learning rate based on model size and batch."""
        # Rough heuristic: smaller LR for larger models
        base_lr = 2e-4
        
        if num_params > 10_000_000_000:  # > 10B
            base_lr = 1e-5
        elif num_params > 1_000_000_000:  # > 1B
            base_lr = 5e-5
        elif num_params > 100_000_000:  # > 100M
            base_lr = 1e-4
        
        # Scale with batch size (linear scaling rule)
        lr = base_lr * math.sqrt(batch_size / 32)
        
        return lr


# Convenience function
def auto_configure(
    model_name_or_path: str,
    **kwargs
) -> TrainingConfig:
    """Auto-configure training for a model.
    
    Convenience wrapper around ConfigOrchestrator.orchestrate().
    """
    orchestrator = ConfigOrchestrator()
    return orchestrator.orchestrate(model_name_or_path, **kwargs)
