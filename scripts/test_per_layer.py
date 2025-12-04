"""Test per-layer gradient trainer."""

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
    
    from src.wrappers.per_layer_grad import (
        PerLayerGradientTrainer,
        PerLayerConfig,
        get_mem,
    )
    
    logger.info("=" * 60)
    logger.info("Per-Layer Gradient Training")
    logger.info("Manual gradients, no computation graph!")
    logger.info("=" * 60)
    
    config = PerLayerConfig(
        max_seq_length=32,
        lora_r=4,
        learning_rate=1e-4,
    )
    
    trainer = PerLayerGradientTrainer("models/phi-4", config)
    
    # Load data
    import json
    texts = []
    with open("data/Dataset/math.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 8:  # More samples
                break
            item = json.loads(line)
            texts.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
    
    logger.info(f"Loaded {len(texts)} samples")
    
    # Train with batching
    results = trainer.train(texts, batch_size=4)  # Process 4 samples per weight load
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Done! Avg loss: {results['avg_loss']:.4f}")
    logger.info(f"Final VRAM: {get_mem():.2f}GB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
