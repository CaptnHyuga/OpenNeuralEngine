# ONN Universal Optimizer System

## Overview

The Universal Optimizer is a **hardware-agnostic, architecture-agnostic** system for training and inference of neural networks of **any size** on **any hardware**.

## Key Components

### 1. `universal_optimizer.py` - Auto-Configuration Engine
- **Hardware Profiler**: Measures GPU/CPU capabilities, bandwidth, memory
- **Model Analyzer**: Discovers architecture without assumptions
- **Configuration Finder**: Empirically finds optimal settings
- **Equitable Benchmarking**: Fair metrics normalized by model size

### 2. `universal_engine.py` - Training & Inference
- **Single Entry Point**: One class for both training and inference
- **Auto-Detection**: Figures out what you need automatically
- **Streaming Inference**: Memory-efficient generation
- **Batched Training**: Amortizes PCIe transfer cost

### 3. `batched_sparse.py` - Core Training Implementation
- **Sparse Layer Selection**: Train only critical layers
- **LoRA Adapters**: Memory-efficient fine-tuning
- **Chunked Loading**: Fits large models in small VRAM
- **Batch Optimization**: Higher throughput via batching

## Usage

### Quick Start
```python
from src.wrappers.universal_engine import UniversalEngine

# Auto-configures for your hardware
engine = UniversalEngine("models/phi-4")

# Training
data = [{"input": "Hello", "output": "World"} for _ in range(100)]
engine.train(data, epochs=1)

# Inference
text = engine.generate("Once upon a time")
```

### Manual Configuration
```python
from src.wrappers.batched_sparse import BatchedSparseTrainer

trainer = BatchedSparseTrainer(
    model_path="models/phi-4",
    chunk_size=3,           # Layers to load at once
    sparse_layers=[0,1,2,3,36,37,38,39],  # Which layers to train
    lora_rank=8,
)
```

### Run Auto-Optimizer
```python
from src.wrappers.universal_optimizer import UniversalOptimizer

optimizer = UniversalOptimizer("models/phi-4")
config = optimizer.find_optimal_config()
optimizer.save_config(config)
optimizer.benchmark()
```

## Performance Results (GTX 1650 4GB, phi-4 16B)

| Configuration | Per Sample | Speedup | VRAM |
|---------------|------------|---------|------|
| Original      | 3261ms     | 1x      | OOM  |
| Batched (8)   | 949ms      | 3.4x    | 1.5GB|
| Batched (64)  | 516ms      | 6.3x    | 1.5GB|
| Batched (128) | 490ms      | 6.7x    | 1.5GB|

## Architecture Support

The system supports **ANY** neural network architecture:

### Automatic Detection
- Transformer (GPT, LLaMA, Phi, etc.)
- CNN (ResNet, VGG, EfficientNet, etc.)
- RNN/LSTM
- Custom architectures
- Future architectures (generic discovery)

### How It Works
1. **No Hardcoded Assumptions**: Discovers layer structure by inspection
2. **Generic Weight Discovery**: Finds trainable parameters automatically
3. **Adaptive Chunking**: Fits any layer size in available memory
4. **Equitable Metrics**: Fair comparison across model sizes

## Equitable Benchmarking

Traditional metrics favor larger models. Our equitable scores normalize:

```
Efficiency Score = (1 - loss) / num_parameters * 1e10
Speed Score = samples_per_second / batch_size * 100
Memory Score = available_memory / peak_memory * 100
Overall = weighted_average(efficiency, speed, memory)
```

This allows fair comparison:
- 7B model with 90% accuracy vs 70B model with 95% accuracy
- The smaller model may have higher **efficiency** score

## Files

```
src/wrappers/
├── universal_optimizer.py   # Auto-configuration engine
├── universal_engine.py      # Training & inference wrapper
├── batched_sparse.py        # Core batched training
├── auto_optimizer.py        # Legacy optimizer
└── optimized_sparse.py      # Base sparse training

models/phi-4/
├── universal_config.json    # Generated optimal config
└── optimal_config.json      # Legacy config format
```

## Hardware Requirements

**Minimum**: 4GB VRAM (tested with GTX 1650)
**Supported**: Any CUDA GPU, CPU fallback available

The system adapts to your hardware:
- Low VRAM: Smaller chunk_size, more loads
- High bandwidth: Larger chunks, faster training
- Limited RAM: Memory-mapped loading

## Future Work

1. **Multi-GPU Distribution**: Shard model across devices
2. **Quantization Integration**: INT8/INT4 inference
3. **KV Cache Optimization**: Faster inference
4. **Continuous Learning**: Train while inferring
