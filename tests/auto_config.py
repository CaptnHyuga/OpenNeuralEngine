"""Intelligent configuration system for SOTA Transformer Models.

Supports:
- Hardware auto-detection (GPU, VRAM, CPU cores)
- SOTA component configuration (RoPE, GQA, SwiGLU, RMSNorm)
- Memory-optimized presets for constrained devices
- Adaptive precision selection (BF16 > FP16 > FP32)
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.project_paths as paths  # noqa: E402


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    has_cuda: bool
    gpu_name: Optional[str]
    gpu_memory_gb: float
    cpu_count: int
    system_memory_gb: float
    device_type: str  # 'cuda' or 'cpu'
    compute_capability: Optional[Tuple[int, int]] = None
    supports_bf16: bool = False
    supports_flash_attention: bool = False
    
    def __str__(self):
        if self.has_cuda:
            extras = []
            if self.supports_bf16:
                extras.append("BF16")
            if self.supports_flash_attention:
                extras.append("FlashAttn")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            return (
                "GPU: "
                f"{self.gpu_name} ({self.gpu_memory_gb:.1f}GB VRAM){extra_str}, "
                f"CPU: {self.cpu_count} cores, RAM: {self.system_memory_gb:.1f}GB"
            )
        return f"CPU: {self.cpu_count} cores, RAM: {self.system_memory_gb:.1f}GB"


def detect_hardware() -> HardwareProfile:
    """Detect available hardware and return profile."""
    import psutil
    
    has_cuda = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = 0.0
    compute_capability = None
    supports_bf16 = False
    supports_flash_attention = False
    
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory_bytes / (1024**3)
        
        # Get compute capability
        compute_capability = torch.cuda.get_device_capability(0)
        
        # BF16 supported on Ampere+ (compute capability >= 8.0)
        supports_bf16 = compute_capability[0] >= 8
        
        # Flash Attention supported on Ampere+ with PyTorch 2.0+
        supports_flash_attention = (
            compute_capability[0] >= 8 and 
            hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        )
    
    cpu_count = psutil.cpu_count(logical=False) or 4
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    return HardwareProfile(
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cpu_count=cpu_count,
        system_memory_gb=system_memory_gb,
        device_type="cuda" if has_cuda else "cpu",
        compute_capability=compute_capability,
        supports_bf16=supports_bf16,
        supports_flash_attention=supports_flash_attention,
    )


def generate_adaptive_config(
    hardware: HardwareProfile,
    task: str = "lm",
    mode: str = "text",
    preset: str = "balanced"
) -> Dict:
    """Generate configuration for SOTA transformer with adaptive settings.
    
    Args:
        hardware: Detected hardware profile
        task: 'classification' or 'lm'
        mode: 'text' or 'multimodal'
        preset: 'ultra_low_vram', 'fast', 'balanced', 'quality', or 'sota'
    """
    
    # ---------------------------------------------------------
    # 1. BASE ARCHITECTURE
    # ---------------------------------------------------------
    config = {
        "model_name": "SNN-SOTA-Adaptive",
        "task": task,
        "vocab_size": 32000,
        
        # Base Architecture
        "hidden_dim": 768,
        "num_micro_layers": 24,
        "num_heads": 12,
        "embedding_dim": 768,
        
        # Fine-tuning defaults
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "warmup_steps": 50,
        
        # Data settings
        "data_source": "synthetic",
        "synthetic_task": "piece_parity",
        "synthetic_piece_separator": " [SEP] ",
        "synthetic_seed": 42,
    }

    # ---------------------------------------------------------
    # 2. SOTA COMPONENT FLAGS
    # ---------------------------------------------------------
    
    # Enable SOTA components by default (can be disabled for legacy)
    sota_config = {
        "use_rope": True,           # Rotary Position Embeddings
        "use_gqa": True,            # Grouped Query Attention
        "use_swiglu": True,         # SwiGLU activation
        "use_rmsnorm": True,        # RMSNorm (faster than LayerNorm)
        "use_flash_attention": hardware.supports_flash_attention,
        
        # RoPE configuration
        "rope_base": 10000.0,       # Standard (100000 for long-context)
        "rope_type": "standard",    # standard, ntk, yarn
        
        # FFN configuration (SwiGLU uses 2.67x intermediate)
        "ffn_activation": "swiglu",
        "ffn_intermediate_dim": None,  # Auto-compute
    }
    
    # ---------------------------------------------------------
    # 3. GQA CONFIGURATION (Memory-adaptive)
    # ---------------------------------------------------------
    # GQA ratio depends on available VRAM
    # Lower VRAM → more aggressive KV sharing
    
    if hardware.has_cuda:
        vram_gb = hardware.gpu_memory_gb
        
        if vram_gb < 4:
            # Ultra-low: 6:1 GQA (like Qwen 2B)
            num_kv_heads = 2
            gqa_ratio = "6:1"
        elif vram_gb < 8:
            # Low: 4:1 GQA
            num_kv_heads = 3
            gqa_ratio = "4:1"
        elif vram_gb < 16:
            # Medium: 3:1 GQA
            num_kv_heads = 4
            gqa_ratio = "3:1"
        else:
            # High: 2:1 GQA (still saves memory)
            num_kv_heads = 6
            gqa_ratio = "2:1"
    else:
        # CPU: Use moderate GQA
        num_kv_heads = 4
        gqa_ratio = "3:1"
    
    sota_config["num_kv_heads"] = num_kv_heads
    config["_gqa_ratio"] = gqa_ratio  # For display
    config.update(sota_config)

    # ---------------------------------------------------------
    # 4. ADAPTIVE HYPERPARAMETERS (Fit Model to Hardware)
    # ---------------------------------------------------------
    
    target_global_batch_size = 64 if preset == "quality" else 32
    
    if hardware.has_cuda:
        vram_gb = hardware.gpu_memory_gb
        
        if vram_gb >= 24:  # High-End (3090/4090/A100)
            batch_size = 32
            max_seq_len = 2048
            vram_budget = 20000
            gradient_checkpointing = False
            mixed_precision = "bf16" if hardware.supports_bf16 else "fp16"
            
        elif vram_gb >= 12:  # Mid-Range (3060/4070)
            batch_size = 8
            max_seq_len = 1024
            vram_budget = 10000
            gradient_checkpointing = False
            mixed_precision = "bf16" if hardware.supports_bf16 else "fp16"
            
        elif vram_gb >= 8:   # Entry (3050/2060)
            batch_size = 4
            max_seq_len = 512
            vram_budget = 6000
            gradient_checkpointing = True
            mixed_precision = "fp16"
            
        elif vram_gb >= 4:   # Low-End
            batch_size = 1
            max_seq_len = 256
            vram_budget = 3500
            gradient_checkpointing = True
            mixed_precision = "fp16"
            print("! WARNING: Low VRAM detected (<=4GB). Using aggressive optimizations.")
            
        else: # < 4GB
            batch_size = 1
            max_seq_len = 128
            vram_budget = 2000
            gradient_checkpointing = True
            mixed_precision = "fp16"
            print("! CRITICAL: Very low VRAM. Using maximum memory optimizations.")

    else:  # CPU Only
        ram_gb = hardware.system_memory_gb
        gradient_checkpointing = True
        mixed_precision = "fp32"  # No mixed precision on CPU
        
        if ram_gb >= 32:
            batch_size = 8
            max_seq_len = 512
            vram_budget = 8000
        elif ram_gb >= 16:
            batch_size = 4
            max_seq_len = 256
            vram_budget = 4000
        else:
            batch_size = 1
            max_seq_len = 128
            vram_budget = 2000
            
    # Calculate Gradient Accumulation to maintain training stability
    grad_accum_steps = max(1, target_global_batch_size // batch_size)
    
    config.update({
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "max_seq_len": max_seq_len,
        "vram_budget_mb": vram_budget,
        "chunk_size": 4,
        "gradient_checkpointing": gradient_checkpointing,
        "mixed_precision": mixed_precision,
    })

    # ---------------------------------------------------------
    # 5. PRESET-SPECIFIC SETTINGS
    # ---------------------------------------------------------
    
    if preset == "ultra_low_vram":
        # Maximum memory optimization
        train_samples, eval_samples = 1000, 200
        epochs = 3
        config.update({
            "gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "num_kv_heads": 2,  # Maximum GQA (6:1)
            "max_seq_len": min(config["max_seq_len"], 128),
            "batch_size": 1,
            "use_flash_attention": True,
        })
    elif preset == "fast":
        train_samples, eval_samples = 2000, 500
        epochs = 3
    elif preset == "quality":
        train_samples, eval_samples = 20000, 2000
        epochs = 10
    elif preset == "sota":
        # Full SOTA configuration with all optimizations
        train_samples, eval_samples = 50000, 5000
        epochs = 15
        config.update({
            "rope_base": 100000.0,  # Better long-context
            "rope_type": "ntk",     # NTK scaling for extrapolation
        })
    else:  # balanced
        train_samples, eval_samples = 10000, 1000
        epochs = 5
    
    config.update({
        "synthetic_train_samples": train_samples,
        "synthetic_eval_samples": eval_samples,
        "epochs": epochs,
    })

    # ---------------------------------------------------------
    # 6. MULTIMODAL SETTINGS
    # ---------------------------------------------------------
    if mode == "multimodal":
        config.update({
            "train_image_min_pieces": 2,
            "train_image_max_pieces": 4,
            "synthetic_image_height": 224,
            "synthetic_image_width": 224,
            "synthetic_image_channels": 3,
            # Vision encoder settings
            "vision_encoder_type": "lightweight",  # or "siglip" for quality
            "use_pixel_shuffle": True,
            "pixel_shuffle_factor": 2,
        })
    
    return config


def generate_legacy_config(
    hardware: HardwareProfile,
    task: str = "lm",
    mode: str = "text",
    preset: str = "balanced"
) -> Dict:
    """Generate legacy configuration without SOTA components.
    
    Use for backwards compatibility or comparison testing.
    """
    config = generate_adaptive_config(hardware, task, mode, preset)
    
    # Disable SOTA components
    config.update({
        "use_rope": False,
        "use_gqa": False,
        "use_swiglu": False,
        "use_rmsnorm": False,
        "use_flash_attention": False,
    })
    
    config["model_name"] = "SNN-Legacy"
    
    return config


def save_config(config: Dict, output_path: Path):
    """Save configuration to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SOTA config for SNN Training")
    parser.add_argument(
        "--output",
        type=str,
        default=str(paths.CONFIG_DIR / "auto_generated.json"),
        help="Where to save the generated config",
    )
    parser.add_argument("--mode", type=str, choices=["text", "multimodal"], default="text",
                       help="Model mode")
    parser.add_argument("--task", type=str, choices=["classification", "lm"], default="lm",
                       help="Task type")
    parser.add_argument("--preset", type=str, 
                       choices=["ultra_low_vram", "fast", "balanced", "quality", "sota", "legacy"],
                       default="balanced",
                       help="Training preset")
    parser.add_argument("--show-only", action="store_true",
                       help="Only show hardware info")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Hardware Detection")
    print("=" * 70)
    hardware = detect_hardware()
    print(hardware)
    print()
    
    if args.show_only:
        print("\nSOTA Capabilities:")
        print(f"  - BFloat16:       {'✓' if hardware.supports_bf16 else '✗'}")
        print(f"  - Flash Attention: {'✓' if hardware.supports_flash_attention else '✗'}")
        if hardware.compute_capability:
            print(f"  - Compute Cap:    {hardware.compute_capability[0]}.{hardware.compute_capability[1]}")
        return
    
    print("=" * 70)
    print(f"Generating SOTA Config ({args.preset})")
    print("=" * 70)
    
    if args.preset == "legacy":
        config = generate_legacy_config(hardware, task=args.task, mode=args.mode, preset="balanced")
    else:
        config = generate_adaptive_config(hardware, task=args.task, mode=args.mode, preset=args.preset)
    
    # Display configuration
    print("\n📊 Architecture:")
    print(f"  - Hidden Dim:     {config['hidden_dim']}")
    print(f"  - Layers:         {config['num_micro_layers']}")
    print(f"  - Heads:          {config['num_heads']}")
    
    print("\n🚀 SOTA Components:")
    print(f"  - RoPE:           {'✓' if config.get('use_rope', False) else '✗'}")
    print(f"  - GQA:            {'✓' if config.get('use_gqa', False) else '✗'} ({config.get('_gqa_ratio', 'N/A')})")
    print(f"  - SwiGLU:         {'✓' if config.get('use_swiglu', False) else '✗'}")
    print(f"  - RMSNorm:        {'✓' if config.get('use_rmsnorm', False) else '✗'}")
    print(f"  - Flash Attention: {'✓' if config.get('use_flash_attention', False) else '✗'}")
    
    print("\n💾 Memory Optimization:")
    print(f"  - Mixed Precision: {config.get('mixed_precision', 'fp32')}")
    print(f"  - Grad Checkpoint: {'✓' if config.get('gradient_checkpointing', False) else '✗'}")
    print(f"  - VRAM Budget:     {config['vram_budget_mb']} MB")
    
    print("\n⚙️ Training:")
    print(f"  - Batch Size:     {config['batch_size']}")
    print(f"  - Grad Accum:     {config['gradient_accumulation_steps']} steps")
    print(f"  - Effective Batch: {config['effective_batch_size']}")
    print(f"  - Max Seq Len:    {config['max_seq_len']}")
    print(f"  - Epochs:         {config['epochs']}")
    
    # Remove display-only keys before saving
    config.pop("_gqa_ratio", None)
    
    output_path = paths.resolve_path(args.output)
    save_config(config, output_path)
    print(f"\n✓ Saved configuration to {output_path}")

if __name__ == "__main__":
    main()