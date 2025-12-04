"""
Test the Streaming Compute Engine.

This demonstrates the revolutionary approach where weights flow through
like water - never accumulating, always computing.
"""

import sys
import logging
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Streaming Compute")
    parser.add_argument("--model", type=str, default="models/phi-4")
    parser.add_argument("--data", type=str, default="data/Dataset/math.jsonl")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=4)
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {gpu_mem:.2f} GB")
    else:
        logger.error("No GPU available!")
        return
    
    # Import streaming compute
    from src.wrappers.streaming_compute import (
        StreamingTransformer,
        StreamingConfig,
        get_gpu_memory,
    )
    
    logger.info("=" * 60)
    logger.info("Streaming Compute Engine Test")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Chunk size: {args.chunk_size} (rows per stream)")
    logger.info(f"LoRA rank: {args.lora_r}")
    
    # Create config
    config = StreamingConfig(
        weight_chunk_size=args.chunk_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
    )
    
    # Create streaming transformer
    logger.info("\nInitializing Streaming Transformer...")
    model = StreamingTransformer(args.model, config)
    
    mem = get_gpu_memory()
    logger.info(f"After init: {mem['allocated']:.2f}GB / {mem['total']:.2f}GB VRAM")
    
    # Load some training data
    logger.info(f"\nLoading data from {args.data}")
    import json
    data = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            item = json.loads(line)
            if "problem" in item and "answer" in item:
                text = f"Problem: {item['problem']}\nAnswer: {item['answer']}"
            elif "text" in item:
                text = item["text"]
            else:
                continue
            data.append(text)
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.layer_processor.get_lora_parameters(),
        lr=1e-4,
    )
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    total_loss = 0.0
    for i, text in enumerate(data):
        # Tokenize
        inputs = model.tokenizer(
            text,
            truncation=True,
            max_length=args.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        labels = input_ids.clone()
        
        # Training step
        try:
            loss = model.train_step(input_ids, labels, optimizer)
            total_loss += loss
            
            mem = get_gpu_memory()
            logger.info(f"Sample {i+1}/{len(data)}: loss={loss:.4f}, VRAM={mem['allocated']:.2f}GB")
            
        except Exception as e:
            logger.error(f"Error on sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final stats
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Average loss: {total_loss / len(data):.4f}")
    logger.info("=" * 60)
    
    # Save LoRA weights
    model.save_lora("streaming_output")
    
    # Test generation
    logger.info("\nTesting generation...")
    prompt = "Problem: What is 2 + 2?\nAnswer:"
    response = model.generate(prompt, max_new_tokens=20)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")
    
    # Final memory
    mem = get_gpu_memory()
    logger.info(f"\nFinal VRAM: {mem['allocated']:.2f}GB / {mem['total']:.2f}GB")


if __name__ == "__main__":
    main()
