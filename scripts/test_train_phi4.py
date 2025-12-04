#!/usr/bin/env python
"""Test Training Script - Train phi-4 (15B) on 4GB VRAM.

This script demonstrates the 20× memory reduction capability of ONN
by training a 15B parameter model on a 4GB VRAM GPU.

Memory Breakdown:
- phi-4 in FP16: ~30GB (won't fit)
- phi-4 in INT4: ~7.5GB (still won't fit)
- phi-4 INT4 + QLoRA: ~3-4GB (fits!)

The magic combination:
1. INT4 quantization (4× reduction)
2. QLoRA (trains only 0.5% of parameters)
3. Gradient checkpointing (saves activation memory)
4. CPU offload optimizer (moves Adam states to RAM)
5. Small batch size + gradient accumulation

Usage:
    python scripts/test_train_phi4.py --model models/phi-4 --data data/Dataset/math.jsonl

    # Quick test with fewer samples
    python scripts/test_train_phi4.py --model models/phi-4 --data data/Dataset/math.jsonl --max-samples 100

    # Resume from checkpoint
    python scripts/test_train_phi4.py --model models/phi-4 --data data/Dataset/math.jsonl --resume
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if environment is ready for training."""
    logger.info("=" * 60)
    logger.info("Environment Check")
    logger.info("=" * 60)
    
    # CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be very slow on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {vram_gb:.2f} GB")
    
    # Check dependencies
    missing = []
    try:
        import transformers
        logger.info(f"transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        import peft
        logger.info(f"peft: {peft.__version__}")
    except ImportError:
        missing.append("peft")
    
    try:
        import bitsandbytes
        logger.info(f"bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        missing.append("bitsandbytes")
    
    try:
        import datasets
        logger.info(f"datasets: {datasets.__version__}")
    except ImportError:
        missing.append("datasets")
    
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    logger.info("=" * 60)
    return True


def estimate_memory(model_path: str, vram_gb: float):
    """Estimate memory requirements and feasibility."""
    from src.wrappers.qlora_trainer import QLoRAConfig, estimate_vram_for_model
    
    logger.info("=" * 60)
    logger.info("Memory Estimation")
    logger.info("=" * 60)
    
    # Estimate model size from config
    try:
        import json
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Estimate params from config
            hidden_size = config.get("hidden_size", 4096)
            num_layers = config.get("num_hidden_layers", 32)
            intermediate_size = config.get("intermediate_size", hidden_size * 4)
            vocab_size = config.get("vocab_size", 32000)
            
            # Rough parameter count
            # Attention: 4 * hidden * hidden per layer
            # MLP: 3 * hidden * intermediate per layer
            # Embeddings: vocab * hidden
            attn_params = 4 * hidden_size * hidden_size * num_layers
            mlp_params = 3 * hidden_size * intermediate_size * num_layers
            embed_params = vocab_size * hidden_size * 2  # input + output
            
            num_params = attn_params + mlp_params + embed_params
            logger.info(f"Estimated parameters: {num_params / 1e9:.2f}B")
        else:
            # Default estimate for unknown model
            num_params = 15_000_000_000  # Assume 15B
            logger.warning(f"Could not read config, assuming {num_params / 1e9:.0f}B parameters")
    except Exception as e:
        logger.warning(f"Error estimating model size: {e}")
        num_params = 15_000_000_000
    
    # Estimate VRAM with QLoRA
    estimate = estimate_vram_for_model(num_params, vram_gb, seq_length=256)
    
    logger.info(f"Available VRAM: {vram_gb:.2f} GB")
    logger.info(f"Estimated VRAM usage: {estimate['estimated_vram_gb']:.2f} GB")
    logger.info(f"Breakdown:")
    for key, value in estimate["breakdown"].items():
        logger.info(f"  - {key}: {value:.2f} GB")
    
    if estimate["fits_in_vram"]:
        logger.info("[OK] Model should fit in VRAM with QLoRA!")
    else:
        logger.warning("[WARNING] Model may not fit! Recommendations:")
        for rec in estimate["recommendations"]:
            logger.warning(f"  - {rec}")
    
    logger.info("=" * 60)
    return estimate["fits_in_vram"]


def train_model(
    model_path: str,
    data_path: str,
    output_dir: str = "./qlora_output",
    max_samples: int = None,
    num_epochs: int = 1,
    resume: bool = False,
    seq_length: int = 256,
    lora_r: int = 8,
    batch_size: int = 1,
    grad_accum: int = 16,
):
    """Train model with QLoRA.
    
    Args:
        model_path: Path to model.
        data_path: Path to training data.
        output_dir: Output directory.
        max_samples: Limit training samples.
        num_epochs: Number of epochs.
        resume: Resume from checkpoint.
        seq_length: Maximum sequence length.
        lora_r: LoRA rank.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
    """
    from src.wrappers.qlora_trainer import QLoRATrainer, QLoRAConfig
    
    logger.info("=" * 60)
    logger.info("Starting QLoRA Training")
    logger.info("=" * 60)
    
    # Create config for 4GB VRAM
    config = QLoRAConfig(
        # Ultra-low VRAM settings
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # INT4 quantization
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        
        # Memory optimizations
        gradient_checkpointing=True,
        cpu_offload_optimizer=True,
        cpu_offload_params=False,
        
        # Training settings
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        num_epochs=num_epochs,
        
        # Sequence
        max_seq_length=seq_length,
        
        # Saving/Logging
        save_steps=50,
        logging_steps=5,
    )
    
    # Log configuration
    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"LoRA rank: {config.lora_r}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"Batch size: {config.per_device_batch_size} x {config.gradient_accumulation_steps} = {effective_batch}")
    logger.info(f"Gradient checkpointing: {config.gradient_checkpointing}")
    logger.info(f"CPU offload optimizer: {config.cpu_offload_optimizer}")
    
    # Estimate memory
    memory_estimate = config.estimate_memory_usage(15_000_000_000)
    logger.info(f"Estimated VRAM: {memory_estimate['total_gb']:.2f} GB")
    
    # Create trainer
    trainer = QLoRATrainer(
        model_path=model_path,
        output_dir=output_dir,
        config=config,
    )
    
    # Find checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint_dirs = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoint_dirs:
            checkpoint = str(sorted(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))[-1])
            logger.info(f"Resuming from {checkpoint}")
    
    # Train!
    try:
        results = trainer.train(
            data_path=data_path,
            max_samples=max_samples,
            resume_from_checkpoint=checkpoint,
        )
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Final loss: {results['train_loss']:.4f}")
        logger.info(f"Runtime: {results['train_runtime']:.2f}s")
        logger.info(f"Samples/second: {results['samples_per_second']:.2f}")
        logger.info(f"Model saved to: {output_dir}")
        
        return results
        
    except torch.cuda.OutOfMemoryError:
        logger.error("=" * 60)
        logger.error("OUT OF MEMORY!")
        logger.error("=" * 60)
        logger.error("Try these fixes:")
        logger.error(f"  1. Reduce --seq-length (currently {seq_length})")
        logger.error(f"  2. Reduce --lora-r (currently {lora_r})")
        logger.error("  3. Close other GPU applications")
        logger.error("  4. Use an even smaller model")
        raise


def test_inference(
    model_path: str,
    adapter_path: str = None,
):
    """Test inference with the trained model."""
    from src.wrappers.low_vram_inference import LowVRAMInference
    
    logger.info("=" * 60)
    logger.info("Testing Inference")
    logger.info("=" * 60)
    
    # Create inferencer
    inferencer = LowVRAMInference(
        model_path=model_path,
        adapter_path=adapter_path,
        vram_limit_gb=4.0,
    )
    
    # Test prompts
    test_prompts = [
        "### Problem:\nSolve: 2x + 5 = 15\n\n### Answer:\n",
        "### Problem:\nWhat is the derivative of x^3?\n\n### Answer:\n",
        "### Problem:\nCalculate: 15% of 200\n\n### Answer:\n",
    ]
    
    logger.info("Loading model...")
    inferencer.load_model()
    
    # Report memory
    memory = inferencer.get_memory_usage()
    logger.info(f"VRAM used: {memory.get('vram_allocated_gb', 0):.2f} GB")
    
    # Generate
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nTest {i+1}:")
        logger.info(f"Prompt: {prompt[:50]}...")
        
        response = inferencer.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=False,
        )
        
        logger.info(f"Response: {response}")
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train large models on low-VRAM GPUs with QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 100 samples
  python scripts/test_train_phi4.py --model models/phi-4 --data data/Dataset/math.jsonl --max-samples 100
  
  # Full training run
  python scripts/test_train_phi4.py --model models/phi-4 --data data/Dataset/math.jsonl --epochs 3
  
  # Resume interrupted training
  python scripts/test_train_phi4.py --model models/phi-4 --data data/Dataset/math.jsonl --resume
  
  # Test inference only
  python scripts/test_train_phi4.py --model models/phi-4 --infer-only --adapter qlora_output/final_model
        """,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/phi-4",
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/Dataset/math.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./qlora_output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of training samples (for testing)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Maximum sequence length (reduce if OOM)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (reduce if OOM)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="Only run inference test (skip training)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter for inference",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip environment check",
    )
    
    args = parser.parse_args()
    
    # Environment check
    if not args.skip_check and not check_environment():
        sys.exit(1)
    
    # Get VRAM
    vram_gb = 4.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Memory estimation
    if not estimate_memory(args.model, vram_gb):
        logger.warning("Model may not fit in VRAM. Continuing with ultra-low settings...")
    
    # Training or inference
    if args.infer_only:
        adapter = args.adapter or Path(args.output) / "final_model"
        if Path(adapter).exists():
            test_inference(args.model, str(adapter))
        else:
            logger.warning(f"Adapter not found at {adapter}, running base model inference")
            test_inference(args.model, None)
    else:
        # Train
        train_model(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            max_samples=args.max_samples,
            num_epochs=args.epochs,
            resume=args.resume,
            seq_length=args.seq_length,
            lora_r=args.lora_r,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
        
        # Test inference after training
        logger.info("\nRunning inference test with trained model...")
        test_inference(args.model, str(Path(args.output) / "final_model"))


if __name__ == "__main__":
    main()
