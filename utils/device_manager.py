"""VRAM-aware device selection and memory management.

Automatically selects the best available device (CUDA > MPS > CPU) while
respecting VRAM constraints and providing graceful fallback on OOM.

Critical for training large models with limited GPU memory.
"""
from __future__ import annotations

import torch
from typing import Optional, Tuple
import warnings


def get_available_vram_mb(device: str = "cuda") -> float:
    """Get available VRAM in megabytes.
    
    Args:
        device: Device to check ("cuda", "mps", "cpu").
    
    Returns:
        Available VRAM in MB (inf for CPU).
    """
    if device == "cpu":
        return float("inf")
    
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return 0.0
        
        device_idx = 0
        if ":" in device:
            device_idx = int(device.split(":")[1])
        
        props = torch.cuda.get_device_properties(device_idx)
        total_memory = props.total_memory / (1024 ** 2)  # Convert to MB
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        
        return total_memory - allocated
    
    if device == "mps":
        # MPS (Apple Silicon) shares memory with system
        # Conservative estimate: assume 8GB available for ML workloads
        return 8192.0
    
    return 0.0


def select_device_with_vram_budget(
    vram_budget_mb: Optional[float] = None,
    prefer_device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[str, float]:
    """Select best device respecting VRAM constraints.
    
    Priority:
    1. User preference (if specified and valid)
    2. CUDA GPU (if available and has enough VRAM)
    3. MPS (Apple Silicon, if available)
    4. CPU (always available, unlimited "VRAM")
    
    Args:
        vram_budget_mb: Required VRAM in MB. None = auto-detect.
        prefer_device: User's preferred device ("cuda", "mps", "cpu").
        verbose: Print device selection reasoning.
    
    Returns:
        (device_name, available_vram_mb)
    """
    # Check user preference first
    if prefer_device:
        if prefer_device.startswith("cuda") and torch.cuda.is_available():
            vram = get_available_vram_mb(prefer_device)
            if vram_budget_mb is None or vram >= vram_budget_mb:
                if verbose:
                    print(f"✅ Using preferred device: {prefer_device} ({vram:.0f} MB VRAM)")
                return prefer_device, vram
            else:
                warnings.warn(
                    f"Requested device {prefer_device} has insufficient VRAM "
                    f"({vram:.0f} MB < {vram_budget_mb:.0f} MB required). Falling back.",
                    stacklevel=2,
                )
        
        elif prefer_device == "mps" and torch.backends.mps.is_available():
            vram = get_available_vram_mb("mps")
            if verbose:
                print(f"✅ Using preferred device: mps (~{vram:.0f} MB available)")
            return "mps", vram
        
        elif prefer_device == "cpu":
            if verbose:
                print("✅ Using CPU (as requested)")
            return "cpu", float("inf")
    
    # Auto-select: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
        vram = get_available_vram_mb(device)
        
        if vram_budget_mb is None or vram >= vram_budget_mb:
            if verbose:
                print(f"🚀 Auto-selected: {device} ({vram:.0f} MB VRAM available)")
            return device, vram
        else:
            if verbose:
                print(f"⚠️  CUDA available but insufficient VRAM ({vram:.0f} MB < {vram_budget_mb:.0f} MB)")
    
    if torch.backends.mps.is_available():
        device = "mps"
        vram = get_available_vram_mb(device)
        if verbose:
            print(f"🍎 Auto-selected: Apple Silicon MPS (~{vram:.0f} MB)")
        return device, vram
    
    # Fallback to CPU
    if verbose:
        print("💻 Falling back to CPU (no GPU detected or insufficient VRAM)")
    return "cpu", float("inf")


def estimate_model_vram_mb(
    num_params: int,
    precision: str = "fp32",
    batch_size: int = 1,
    seq_len: int = 512,
    gradient_checkpointing: bool = False,
) -> float:
    """Estimate VRAM requirements for a model.
    
    Args:
        num_params: Total model parameters.
        precision: "fp32", "fp16", or "int8".
        batch_size: Training batch size.
        seq_len: Sequence length.
        gradient_checkpointing: Whether gradient checkpointing is enabled.
    
    Returns:
        Estimated VRAM in MB.
    """
    # Bytes per parameter
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
    }.get(precision, 4)
    
    # Model weights
    model_mb = (num_params * bytes_per_param) / (1024 ** 2)
    
    # Gradients (same size as model in training)
    gradients_mb = model_mb
    
    # Optimizer state (2x model size for Adam)
    optimizer_mb = model_mb * 2
    
    # Activations (reduced with gradient checkpointing)
    activation_scale = 0.3 if gradient_checkpointing else 1.0
    activations_mb = (batch_size * seq_len * num_params * 0.01 * activation_scale) / (1024 ** 2)
    
    total_mb = model_mb + gradients_mb + optimizer_mb + activations_mb
    
    # Add 20% buffer for framework overhead
    return total_mb * 1.2


def enable_memory_efficient_mode(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Apply memory optimizations based on device.
    
    Args:
        model: The model to optimize.
        device: Target device.
    
    Returns:
        Optimized model.
    """
    if device.startswith("cuda"):
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
    
    elif device == "mps":
        # MPS-specific optimizations
        # Currently limited, but prepare for future enhancements
        pass
    
    return model


def safe_to_device(
    model: torch.nn.Module,
    device: str,
    vram_budget_mb: Optional[float] = None,
) -> torch.nn.Module:
    """Move model to device with OOM handling.
    
    Args:
        model: Model to move.
        device: Target device.
        vram_budget_mb: Expected VRAM usage (for validation).
    
    Returns:
        Model on device (or CPU if OOM occurred).
    """
    try:
        model = model.to(device)
        
        if device.startswith("cuda"):
            # Verify we didn't exceed budget
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            if vram_budget_mb and allocated > vram_budget_mb * 1.1:
                warnings.warn(
                    f"Model uses {allocated:.0f} MB, exceeds budget of {vram_budget_mb:.0f} MB. "
                    "May cause OOM during training.",
                    stacklevel=2,
                )
        
        return model
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            warnings.warn(f"OOM on {device}, falling back to CPU", stacklevel=2)
            torch.cuda.empty_cache() if device.startswith("cuda") else None
            return model.to("cpu")
        raise


def get_optimal_batch_size(
    model: torch.nn.Module,
    device: str,
    seq_len: int = 512,
    target_vram_utilization: float = 0.8,
) -> int:
    """Estimate optimal batch size for available VRAM.
    
    Args:
        model: The model.
        device: Target device.
        seq_len: Sequence length.
        target_vram_utilization: Use this fraction of available VRAM.
    
    Returns:
        Recommended batch size.
    """
    if device == "cpu":
        return 8  # CPU: small batches for efficiency
    
    available_vram = get_available_vram_mb(device)
    target_vram = available_vram * target_vram_utilization
    
    # Estimate single-sample VRAM usage
    num_params = sum(p.numel() for p in model.parameters())
    single_sample_mb = estimate_model_vram_mb(
        num_params=num_params,
        batch_size=1,
        seq_len=seq_len,
    )
    
    # Calculate batch size
    batch_size = max(1, int(target_vram / single_sample_mb))
    
    # Round to nearest power of 2 for efficiency
    import math
    batch_size = 2 ** math.floor(math.log2(batch_size))
    
    return max(1, min(batch_size, 64))  # Clamp to [1, 64]
