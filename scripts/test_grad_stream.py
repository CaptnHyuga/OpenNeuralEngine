"""Test gradient streaming trainer."""

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
    
    from src.wrappers.gradient_streaming import (
        GradientStreamingTrainer,
        GradientStreamConfig,
        get_mem,
        clear_all,
    )
    
    config = GradientStreamConfig(
        checkpoint_every=4,
        lora_r=4,
        max_seq_length=16,
    )
    
    logger.info("Creating trainer...")
    trainer = GradientStreamingTrainer("models/phi-4", config)
    
    logger.info(f"VRAM after init: {get_mem():.2f}GB")
    
    # Load one sample
    import json
    with open("data/Dataset/math.jsonl") as f:
        item = json.loads(f.readline())
    text = f"Problem: {item['problem']}\nAnswer: {item['answer']}"
    
    inputs = trainer.tokenizer(
        text, truncation=True, max_length=16,
        padding="max_length", return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(trainer.device)
    
    logger.info("Running training step...")
    
    try:
        loss = trainer.simple_train_step(input_ids, input_ids.clone())
        logger.info(f"Loss: {loss:.4f}")
        logger.info(f"VRAM after step: {get_mem():.2f}GB")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    clear_all()
    
    # Try generation
    logger.info("\nTrying generation...")
    try:
        response = trainer.generate("2+2=", max_new_tokens=5)
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"Generation error: {e}")


if __name__ == "__main__":
    main()
