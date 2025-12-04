"""Test hybrid cached training."""

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
    
    from src.wrappers.hybrid_cached import HybridTrainer, HybridConfig, get_mem
    
    logger.info("=" * 60)
    logger.info("Hybrid Cached Training")
    logger.info("Cache first/last layers, stream middle layers")
    logger.info("=" * 60)
    
    config = HybridConfig(
        max_seq_length=32,
        lora_r=4,
        cached_first_layers=1,  # Start conservative
        cached_last_layers=1,
    )
    
    trainer = HybridTrainer("models/phi-4", config)
    
    # Load data
    import json
    texts = []
    with open("data/Dataset/math.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 4:
                break
            item = json.loads(line)
            texts.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
    
    logger.info(f"\nLoaded {len(texts)} samples")
    
    # Train
    results = trainer.train(texts)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Done! Avg loss: {results['avg_loss']:.4f}")
    logger.info(f"Final VRAM: {get_mem():.2f}GB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
