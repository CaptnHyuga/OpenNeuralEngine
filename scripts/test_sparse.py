"""Test sparse layer training."""

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
    
    from src.wrappers.sparse_layer import SparseLayerTrainer, SparseConfig, get_mem
    
    logger.info("=" * 60)
    logger.info("Sparse Layer Training")
    logger.info("Compute only first 3 + last 5 layers = 8 layers instead of 40!")
    logger.info("=" * 60)
    
    config = SparseConfig(
        max_seq_length=64,
        lora_r=8,
        compute_first_n=3,
        compute_last_n=5,
    )
    
    trainer = SparseLayerTrainer("models/phi-4", config)
    
    # Load data
    import json
    texts = []
    with open("data/Dataset/math.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 8:  # More samples
                break
            item = json.loads(line)
            texts.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
    
    logger.info(f"\nLoaded {len(texts)} samples")
    
    # Train
    t0 = __import__('time').time()
    results = trainer.train(texts)
    total_time = __import__('time').time() - t0
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Done! Avg loss: {results['avg_loss']:.4f}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/len(texts):.1f}s/sample)")
    logger.info(f"Final VRAM: {get_mem():.2f}GB")
    logger.info("=" * 60)
    
    trainer.save_lora("sparse_output")


if __name__ == "__main__":
    main()
