"""
Test Micro-Streaming Compute.

The ultimate memory optimization where even embeddings are streamed!
"""

import sys
import logging
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phi-4")
    parser.add_argument("--data", default="data/Dataset/math.jsonl")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--lora-r", type=int, default=4)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Clear any existing allocations
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    from src.wrappers.micro_streaming import (
        MicroStreamingTransformer,
        MicroStreamConfig,
        get_gpu_memory,
        clear_memory,
    )
    
    logger.info("=" * 60)
    logger.info("Micro-Streaming Compute Test")
    logger.info("=" * 60)
    
    config = MicroStreamConfig(
        output_chunk_size=args.chunk_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
    )
    
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"LoRA rank: {args.lora_r}")
    
    # Create model
    model = MicroStreamingTransformer(args.model, config)
    
    mem = get_gpu_memory()
    logger.info(f"After model init: {mem['allocated']:.2f}GB VRAM")
    
    # Load data
    import json
    data = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            item = json.loads(line)
            if "problem" in item and "answer" in item:
                text = f"Problem: {item['problem']}\nAnswer: {item['answer']}"
            else:
                continue
            data.append(text)
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.get_lora_parameters(), lr=1e-4)
    
    logger.info("\nStarting training...")
    
    for i, text in enumerate(data):
        inputs = model.tokenizer(
            text, truncation=True, max_length=args.seq_length,
            padding="max_length", return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        
        try:
            loss = model.train_step(input_ids, input_ids.clone(), optimizer)
            mem = get_gpu_memory()
            logger.info(f"Sample {i+1}: loss={loss:.4f}, VRAM={mem['allocated']:.2f}GB")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break
        
        clear_memory()
    
    logger.info("\nDone!")
    
    # Test generation
    logger.info("\nGeneration test:")
    response = model.generate("Problem: What is 5 + 3?\nAnswer:", max_new_tokens=10)
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    main()
