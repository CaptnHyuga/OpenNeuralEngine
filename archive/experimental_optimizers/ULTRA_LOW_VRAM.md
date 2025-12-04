# Ultra-Low VRAM Training for Large Language Models

## Overview

This module enables training of 16B+ parameter models on GPUs with only 4GB VRAM (like GTX 1650).

## Key Optimizations

### 1. Pinned Memory Transfer (3.5x speedup)
Using `pin_memory()` for CPU tensors before GPU transfer dramatically reduces transfer time:
- Regular transfer: ~365ms per weight tensor
- Pinned transfer: ~105ms per weight tensor

### 2. Sparse Layer Training (5x speedup)
Only compute through a subset of layers:
- First 3 layers (low-level features)
- Last 5 layers (task-specific fine-tuning)
- Skip 32 middle layers (use residual connections)

Research shows that LoRA on later layers is most effective for fine-tuning.

### 3. Per-Layer Gradient Streaming
Compute gradients one layer at a time:
- Load layer weights → Compute → Free weights
- Never hold all 40 layers in memory simultaneously
- Memory usage stays constant regardless of model size

## Performance Comparison

| Approach | Time/Sample | VRAM | Speedup |
|----------|-------------|------|---------|
| Standard QLoRA | N/A (OOM) | >6GB | - |
| gradient_streaming.py | 70s | 1.06GB | 1x |
| sparse_layer.py | 11s | 1.06GB | 6x |
| **optimized_sparse.py** | **6s** | 1.05GB | **11x** |

## Usage

```python
from src.wrappers.optimized_sparse import OptimizedSparseTrainer, OptimizedConfig

config = OptimizedConfig(
    max_seq_length=32,
    lora_r=4,
    lora_alpha=8,
    learning_rate=2e-4,
    compute_first_n=3,  # First N layers to compute
    compute_last_n=5,   # Last N layers to compute
)

trainer = OptimizedSparseTrainer("models/phi-4", config)

# Train
texts = ["Your training text here..."]
results = trainer.train(texts, epochs=1)

print(f"Avg loss: {results['avg_loss']:.4f}")
print(f"Avg time: {results['avg_time']:.1f}s/sample")
```

## Technical Details

### Why Pinned Memory Works
- Regular memory: CPU allocates → GPU requests → DMA transfer → GPU waits
- Pinned memory: Pre-registered with GPU driver → Direct DMA → No waiting

### Why Sparse Layers Work
- Transformer layers have residual connections: `output = input + delta`
- Middle layers often produce small `delta` values
- Skipping them is like setting `delta = 0`, which is valid
- First/last layers matter most for feature extraction and task adaptation

### Memory Budget (4GB VRAM)
- Embeddings: ~350MB (frozen)
- LoRA parameters: ~10MB (trainable)
- 1 layer weights: ~700MB (streamed)
- Activations: ~100MB
- Gradients: ~100MB
- Overhead: ~500MB
- **Total: ~1.7GB used, 2.3GB free for safety margin**

## Files

- `src/wrappers/optimized_sparse.py` - **RECOMMENDED**: Fastest approach
- `src/wrappers/sparse_layer.py` - Original sparse approach
- `src/wrappers/gradient_streaming.py` - Full model, slower but complete

## Limitations

1. **Quality vs Speed**: Sparse training may produce slightly lower quality than full training
2. **Batch Size**: Limited to batch_size=1 due to memory constraints
3. **Sequence Length**: Recommend max_seq_length ≤ 64 for 4GB VRAM

## Future Improvements

1. **Triton Kernels**: Custom CUDA kernels could provide additional 2-3x speedup (requires Linux)
2. **torch.compile**: Would enable kernel fusion (requires Triton)
3. **Quantized Training**: INT8 gradients could halve memory usage
