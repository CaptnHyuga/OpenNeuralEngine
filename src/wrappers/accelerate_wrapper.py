"""Accelerate Wrapper - Unified distributed training interface.

Wraps HuggingFace Accelerate which provides:
- DeepSpeed ZeRO (all stages)
- FSDP (Fully Sharded Data Parallel)
- Multi-GPU DDP
- Mixed precision training
- Gradient accumulation
- CPU offloading

Why Accelerate instead of raw DeepSpeed?
- Unified API for all distributed strategies
- Better HuggingFace ecosystem integration
- Handles boilerplate (device placement, gradient sync)
- Active maintenance by HuggingFace team
- Works seamlessly with Trainer

ONN adds:
- Hardware-aware auto-configuration
- Zero-config distributed training
- Automatic strategy selection
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# Check for accelerate
try:
    from accelerate import Accelerator, DistributedType
    from accelerate.utils import (
        DeepSpeedPlugin,
        FullyShardedDataParallelPlugin,
        set_seed,
    )
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

# ONN imports
try:
    from ..orchestration import get_hardware_profile, HardwareProfile
    ONN_ORCHESTRATION = True
except ImportError:
    ONN_ORCHESTRATION = False


class DistributedStrategy(Enum):
    """Distributed training strategies."""
    NONE = "none"                    # Single GPU
    DDP = "ddp"                      # DataDistributedParallel
    FSDP = "fsdp"                    # Fully Sharded Data Parallel
    DEEPSPEED_ZERO1 = "deepspeed_zero1"
    DEEPSPEED_ZERO2 = "deepspeed_zero2"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"


@dataclass
class AccelerateConfig:
    """Configuration for Accelerate wrapper."""
    
    # Strategy
    strategy: DistributedStrategy = DistributedStrategy.NONE
    
    # Mixed precision
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    
    # Gradient settings
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    
    # DeepSpeed specific
    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    
    # FSDP specific  
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    
    # General
    seed: int = 42
    logging_dir: Optional[str] = None


@dataclass
class DeepSpeedZeroConfig:
    """DeepSpeed ZeRO configuration generator."""
    
    stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 500_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Generate DeepSpeed JSON config."""
        config = {
            "zero_optimization": {
                "stage": self.stage,
                "overlap_comm": self.overlap_comm,
                "contiguous_gradients": self.contiguous_gradients,
                "reduce_bucket_size": self.reduce_bucket_size,
            },
            "gradient_clipping": 1.0,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "steps_per_print": 100,
        }
        
        # Optimizer offloading
        if self.offload_optimizer:
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        
        # Parameter offloading (ZeRO-3 only)
        if self.stage >= 3:
            config["zero_optimization"]["stage3_prefetch_bucket_size"] = self.stage3_prefetch_bucket_size
            config["zero_optimization"]["stage3_param_persistence_threshold"] = self.stage3_param_persistence_threshold
            
            if self.offload_param:
                config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
        
        return config
    
    def save(self, path: Union[str, Path]) -> str:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(path)


class AccelerateWrapper:
    """Hardware-aware distributed training wrapper using HuggingFace Accelerate.
    
    Automatically configures the optimal distributed strategy based on:
    - Number of GPUs
    - Available VRAM
    - Model size
    - Hardware capabilities
    
    Example:
        wrapper = AccelerateWrapper.auto_configure(model_size_gb=140)
        accelerator = wrapper.get_accelerator()
        
        model, optimizer, dataloader = accelerator.prepare(
            model, optimizer, dataloader
        )
        
        for batch in dataloader:
            outputs = model(**batch)
            accelerator.backward(outputs.loss)
            optimizer.step()
    """
    
    def __init__(self, config: Optional[AccelerateConfig] = None):
        """Initialize wrapper.
        
        Args:
            config: Accelerate configuration. Auto-configured if None.
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "accelerate not installed. "
                "Install with: pip install accelerate"
            )
        
        self.config = config or AccelerateConfig()
        self._accelerator: Optional[Accelerator] = None
        self._deepspeed_config_path: Optional[str] = None
    
    @classmethod
    def auto_configure(
        cls,
        model_size_gb: Optional[float] = None,
        num_params: Optional[int] = None,
        prefer_strategy: Optional[DistributedStrategy] = None,
    ) -> "AccelerateWrapper":
        """Auto-configure based on hardware and model size.
        
        Args:
            model_size_gb: Model size in GB (estimated if not provided).
            num_params: Number of model parameters.
            prefer_strategy: Preferred strategy (auto-selected if None).
            
        Returns:
            Configured AccelerateWrapper instance.
        """
        config = AccelerateConfig()
        
        # Get hardware profile
        profile = None
        if ONN_ORCHESTRATION:
            try:
                profile = get_hardware_profile()
            except Exception:
                pass
        
        # Estimate model size from params
        if model_size_gb is None and num_params:
            # Rough estimate: 2 bytes per param (fp16) + optimizer states
            model_size_gb = (num_params * 2 * 4) / (1024**3)  # 4x for Adam states
        
        # Determine available resources
        num_gpus = torch.cuda.device_count()
        total_vram_gb = 0
        supports_bf16 = False
        
        if profile and profile.has_gpu:
            total_vram_gb = profile.total_vram_mb / 1024
            supports_bf16 = profile.supports_bf16
        elif num_gpus > 0:
            total_vram_gb = sum(
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(num_gpus)
            )
        
        # Select mixed precision
        if supports_bf16:
            config.mixed_precision = "bf16"
        elif num_gpus > 0:
            config.mixed_precision = "fp16"
        else:
            config.mixed_precision = "no"
        
        # Select strategy based on hardware and model size
        if prefer_strategy:
            config.strategy = prefer_strategy
        else:
            config.strategy = cls._select_strategy(
                num_gpus=num_gpus,
                total_vram_gb=total_vram_gb,
                model_size_gb=model_size_gb or 0,
            )
        
        # Configure ZeRO stage and offloading
        if config.strategy in (
            DistributedStrategy.DEEPSPEED_ZERO2,
            DistributedStrategy.DEEPSPEED_ZERO3,
        ):
            if config.strategy == DistributedStrategy.DEEPSPEED_ZERO3:
                config.zero_stage = 3
                # Enable offloading if VRAM is tight
                if model_size_gb and model_size_gb > total_vram_gb * 0.7:
                    config.offload_optimizer = True
                if model_size_gb and model_size_gb > total_vram_gb * 0.9:
                    config.offload_param = True
            else:
                config.zero_stage = 2
                # Offload optimizer for large models
                if model_size_gb and model_size_gb > total_vram_gb * 0.5:
                    config.offload_optimizer = True
        
        # Enable gradient checkpointing for large models
        if model_size_gb and model_size_gb > total_vram_gb * 0.3:
            config.gradient_checkpointing = True
        
        return cls(config)
    
    @staticmethod
    def _select_strategy(
        num_gpus: int,
        total_vram_gb: float,
        model_size_gb: float,
    ) -> DistributedStrategy:
        """Select optimal distributed strategy."""
        
        # Single GPU or no GPU
        if num_gpus <= 1:
            if model_size_gb > total_vram_gb * 0.8:
                # Model too large - need ZeRO CPU offloading
                return DistributedStrategy.DEEPSPEED_ZERO3
            elif model_size_gb > total_vram_gb * 0.5:
                return DistributedStrategy.DEEPSPEED_ZERO2
            else:
                return DistributedStrategy.NONE
        
        # Multi-GPU
        per_gpu_vram = total_vram_gb / num_gpus
        
        if model_size_gb > total_vram_gb * 0.8:
            # Very large model - need aggressive sharding
            return DistributedStrategy.DEEPSPEED_ZERO3
        elif model_size_gb > per_gpu_vram * 0.5:
            # Large model - use ZeRO-2
            return DistributedStrategy.DEEPSPEED_ZERO2
        elif num_gpus >= 4:
            # Many GPUs, moderate model - FSDP or ZeRO-2
            return DistributedStrategy.FSDP
        else:
            # Small model, few GPUs - simple DDP
            return DistributedStrategy.DDP
    
    def get_accelerator(self) -> Accelerator:
        """Get configured Accelerator instance.
        
        Returns:
            Configured HuggingFace Accelerator.
        """
        if self._accelerator is not None:
            return self._accelerator
        
        kwargs = {
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "mixed_precision": self.config.mixed_precision,
        }
        
        # Configure DeepSpeed
        if self.config.strategy in (
            DistributedStrategy.DEEPSPEED_ZERO1,
            DistributedStrategy.DEEPSPEED_ZERO2,
            DistributedStrategy.DEEPSPEED_ZERO3,
        ):
            ds_config = DeepSpeedZeroConfig(
                stage=self.config.zero_stage,
                offload_optimizer=self.config.offload_optimizer,
                offload_param=self.config.offload_param,
            )
            
            # Save config to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(ds_config.to_dict(), f)
                self._deepspeed_config_path = f.name
            
            kwargs["deepspeed_plugin"] = DeepSpeedPlugin(
                hf_ds_config=ds_config.to_dict(),
                zero_stage=self.config.zero_stage,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                offload_optimizer_device="cpu" if self.config.offload_optimizer else "none",
                offload_param_device="cpu" if self.config.offload_param else "none",
            )
        
        # Configure FSDP
        elif self.config.strategy == DistributedStrategy.FSDP:
            try:
                from torch.distributed.fsdp import ShardingStrategy
                
                sharding_map = {
                    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                    "NO_SHARD": ShardingStrategy.NO_SHARD,
                }
                
                kwargs["fsdp_plugin"] = FullyShardedDataParallelPlugin(
                    sharding_strategy=sharding_map.get(
                        self.config.fsdp_sharding_strategy,
                        ShardingStrategy.FULL_SHARD
                    ),
                    cpu_offload=self.config.fsdp_cpu_offload,
                )
            except ImportError:
                # FSDP not available, fall back to DDP
                pass
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        self._accelerator = Accelerator(**kwargs)
        return self._accelerator
    
    def prepare(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        scheduler: Optional[Any] = None,
    ):
        """Prepare model, optimizer, and dataloader for distributed training.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer (optional).
            dataloader: DataLoader (optional).
            scheduler: Learning rate scheduler (optional).
            
        Returns:
            Prepared objects in same order as input.
        """
        accelerator = self.get_accelerator()
        
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
        
        # Prepare all objects
        objects_to_prepare = [model]
        if optimizer is not None:
            objects_to_prepare.append(optimizer)
        if dataloader is not None:
            objects_to_prepare.append(dataloader)
        if scheduler is not None:
            objects_to_prepare.append(scheduler)
        
        prepared = accelerator.prepare(*objects_to_prepare)
        
        # Return in same order
        if len(objects_to_prepare) == 1:
            return prepared
        return prepared
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with proper gradient handling.
        
        Args:
            loss: Loss tensor to backpropagate.
        """
        accelerator = self.get_accelerator()
        accelerator.backward(loss)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        output_dir: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Save model checkpoint with distributed handling.
        
        Args:
            model: Model to save.
            output_dir: Output directory.
            optimizer: Optimizer state to save (optional).
        """
        accelerator = self.get_accelerator()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        accelerator.wait_for_everyone()
        
        # Save model
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), output_dir / "model.pt")
        
        # Save optimizer
        if optimizer is not None:
            accelerator.save(optimizer.state_dict(), output_dir / "optimizer.pt")
        
        # Save config
        if accelerator.is_main_process:
            config_dict = {
                "strategy": self.config.strategy.value,
                "mixed_precision": self.config.mixed_precision,
                "zero_stage": self.config.zero_stage,
            }
            with open(output_dir / "accelerate_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
    
    def summary(self) -> str:
        """Get human-readable configuration summary."""
        lines = [
            "Accelerate Configuration:",
            f"  Strategy: {self.config.strategy.value}",
            f"  Mixed Precision: {self.config.mixed_precision}",
            f"  Gradient Accumulation: {self.config.gradient_accumulation_steps}",
            f"  Gradient Checkpointing: {self.config.gradient_checkpointing}",
        ]
        
        if "deepspeed" in self.config.strategy.value:
            lines.extend([
                f"  ZeRO Stage: {self.config.zero_stage}",
                f"  Offload Optimizer: {self.config.offload_optimizer}",
                f"  Offload Params: {self.config.offload_param}",
            ])
        
        if self.config.strategy == DistributedStrategy.FSDP:
            lines.extend([
                f"  FSDP Sharding: {self.config.fsdp_sharding_strategy}",
                f"  FSDP CPU Offload: {self.config.fsdp_cpu_offload}",
            ])
        
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def auto_accelerate(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    model_size_gb: Optional[float] = None,
) -> tuple:
    """One-liner to prepare model for distributed training.
    
    Automatically detects hardware and configures optimal strategy.
    
    Example:
        model, optimizer, dataloader = auto_accelerate(model, optimizer, dataloader)
        
        for batch in dataloader:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer (optional).
        dataloader: DataLoader (optional).
        model_size_gb: Estimated model size for better configuration.
        
    Returns:
        Tuple of prepared objects (model, optimizer, dataloader).
    """
    # Estimate model size if not provided
    if model_size_gb is None:
        num_params = sum(p.numel() for p in model.parameters())
        model_size_gb = (num_params * 2 * 4) / (1024**3)
    
    wrapper = AccelerateWrapper.auto_configure(model_size_gb=model_size_gb)
    print(wrapper.summary())
    
    return wrapper.prepare(model, optimizer, dataloader)


def get_deepspeed_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> Dict[str, Any]:
    """Generate DeepSpeed config dict.
    
    Useful for passing to HuggingFace Trainer or other tools.
    
    Args:
        stage: ZeRO stage (1, 2, or 3).
        offload_optimizer: Enable optimizer CPU offloading.
        offload_param: Enable parameter CPU offloading (stage 3 only).
        
    Returns:
        DeepSpeed configuration dictionary.
    """
    config = DeepSpeedZeroConfig(
        stage=stage,
        offload_optimizer=offload_optimizer,
        offload_param=offload_param,
    )
    return config.to_dict()
