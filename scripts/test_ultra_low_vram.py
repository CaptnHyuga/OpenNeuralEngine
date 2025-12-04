#!/usr/bin/env python
"""Test Ultra-Low VRAM Training - phi-4 (16B) on 4GB VRAM.

This script demonstrates ONN's custom memory optimizations that enable
training 16B+ parameter models on 4GB VRAM consumer GPUs.

The key insight: We don't need the whole model in VRAM at once!
- Stream transformer layers from disk one at a time
- Keep only LoRA adapters in VRAM (tiny!)
- Offload activations to CPU between layers

Usage:
    python scripts/test_ultra_low_vram.py --model models/phi-4 --data data/Dataset/math.jsonl
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ultra-Low VRAM Training")
    parser.add_argument("--model", "-m", default="models/phi-4", help="Model path")
    parser.add_argument("--data", "-d", default="data/Dataset/math.jsonl", help="Data path")
    parser.add_argument("--max-samples", type=int, default=50, help="Max training samples")
    parser.add_argument("--seq-length", type=int, default=64, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    
    args = parser.parse_args()
    
    # Check VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {vram:.2f} GB")
    else:
        logger.error("No CUDA GPU found!")
        return
    
    # Import our custom trainer
    from src.wrappers.ultra_low_vram import (
        UltraLowVRAMTrainer, 
        UltraLowVRAMConfig,
        get_gpu_memory_info,
    )
    
    logger.info("=" * 60)
    logger.info("Ultra-Low VRAM Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Seq length: {args.seq_length}")
    logger.info(f"LoRA rank: {args.lora_r}")
    
    # Configure for 4GB VRAM
    config = UltraLowVRAMConfig(
        vram_budget_gb=4.0,
        layers_in_vram=1,
        stream_from_disk=True,
        use_int4=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_target_modules=["q_proj", "v_proj"],  # Minimal
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        max_seq_length=args.seq_length,
        learning_rate=1e-4,
        num_epochs=1,
        offload_activations=True,
        activation_checkpointing=True,
        optimizer="sgd",  # Uses less memory than Adam
        use_amp=True,
    )
    
    # Create trainer
    trainer = UltraLowVRAMTrainer(
        model_path=args.model,
        config=config,
    )
    
    # Setup and check memory
    trainer.setup()
    mem = get_gpu_memory_info()
    logger.info(f"After setup: {mem['allocated']:.2f}GB / {mem['total']:.2f}GB VRAM used")
    
    # Train
    logger.info("Starting training...")
    try:
        results = trainer.train(
            data_path=args.data,
            max_samples=args.max_samples,
        )
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Final loss: {results['train_loss']:.4f}")
        logger.info(f"Steps: {results['num_steps']}")
        logger.info("=" * 60)
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("Out of Memory! Try reducing:")
        logger.error(f"  --seq-length {args.seq_length // 2}")
        logger.error(f"  --lora-r {max(1, args.lora_r // 2)}")
        raise
    
    # Test inference
    logger.info("\nTesting inference...")
    test_prompt = "Problem: What is 2 + 2?\nAnswer:"
    response = trainer.inference(test_prompt, max_new_tokens=20)
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
