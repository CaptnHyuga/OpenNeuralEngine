"""Test layer-parallel training."""

import sys
import logging
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    else:
        logger.error("No GPU!")
        return
    
    from src.wrappers.layer_parallel import (
        LayerParallelTrainer,
        LayerParallelConfig,
        load_math_data,
        get_mem,
    )
    
    logger.info("=" * 60)
    logger.info("Layer-Parallel Training")
    logger.info("Load weights ONCE per layer, process ALL samples!")
    logger.info("=" * 60)
    
    config = LayerParallelConfig(
        batch_size=8,  # Process 8 samples through each layer at once
        max_seq_length=32,
        lora_r=4,
        learning_rate=1e-4,
    )
    
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Seq length: {config.max_seq_length}")
    
    # Create trainer
    trainer = LayerParallelTrainer("models/phi-4", config)
    
    # Load data
    data = load_math_data("data/Dataset/math.jsonl", max_samples=16)
    logger.info(f"Loaded {len(data)} samples")
    
    # Train
    logger.info("\nStarting training...")
    results = trainer.train(data)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final loss: {results['loss']:.4f}")
    logger.info(f"VRAM: {get_mem():.2f}GB")
    logger.info("=" * 60)
    
    # Save
    trainer.save_lora("layer_parallel_output")


if __name__ == "__main__":
    main()
