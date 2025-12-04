# OpenNeuralEngine (ONN) Architecture

## Overview

ONN is a **low-VRAM training system** that enables training 16B+ parameter models on consumer GPUs (4GB VRAM).

## Quick Start

```bash
# Train on any dataset
python -m src.training.pipeline --dataset data/Dataset/math.jsonl --epochs 1

# Run benchmarks
python -m src.benchmarks.benchmark --quick

# View metrics
tensorboard --logdir output
```

## Project Structure

```
OpenNeuralEngine/
├── src/
│   ├── training/           # 🚀 Main training pipeline
│   │   └── pipeline.py     # Unified trainer (USE THIS)
│   ├── benchmarks/         # 📊 Benchmark suite
│   │   └── benchmark.py    # Speed/learning/memory tests
│   ├── wrappers/           # 🔧 Core implementations
│   │   ├── batched_sparse.py    # Sparse training with batching
│   │   ├── model_loader.py      # Model loading utilities
│   │   └── quantization_wrapper.py
│   ├── data_adapters/      # 📁 Dataset loaders
│   └── orchestration/      # ⚙️ System orchestration
├── data/Dataset/           # Training data
├── models/                 # Model weights (safetensors)
├── output/                 # Training outputs
├── benchmark_results/      # Benchmark outputs
└── archive/                # Old experimental code (kept for reference)
```

## Key Optimizations

### 1. Sparse Layer Training
Only train 8 of 40 layers (first 4 + last 4):
```python
sparse_layers = [0, 1, 2, 3, 36, 37, 38, 39]
```
**Why?** Early and late layers have most task-specific impact.

### 2. LoRA Adapters
Low-Rank Adaptation for memory efficiency:
- Only 5.57M trainable params (0.034% of 16B)
- Base weights stay frozen
- No gradient computation for base

### 3. Batched Weight Loading
Load weights ONCE, process MANY samples:
```
Traditional: Load → Process 1 sample → Free → Load → Process 1 sample → ...
Optimized:   Load → Process 32 samples → Free
```
**Result:** 7-13x speedup

### 4. Safetensors (Security)
- No `.pt` files (pickle can execute arbitrary code)
- Only `.safetensors` format for checkpoints
- Safe loading and saving

## Benchmark Results

| Metric | Optimized | Baseline | Improvement |
|--------|-----------|----------|-------------|
| Time/sample | 3.6ms | 5.8ms | 1.6x faster |
| Learning | ✅ YES | ✅ YES | Verified |
| Peak VRAM | 269MB | 269MB | Same |

## Supported Data Formats

- **JSONL**: `{"input": "...", "output": "..."}`
- **Parquet**: Any columns (processed as dict records)
- **JSON**: List of dicts or single dict

## Metrics & Monitoring

### Tensorboard
```bash
tensorboard --logdir output --port 6006
```

### Programmatic
```python
from src.benchmarks import BenchmarkSuite
suite = BenchmarkSuite("models/phi-4")
results = suite.run_all(quick=True)
```

## Security Notes

1. **No pickle files** - Only safetensors format
2. **No arbitrary code execution** - Safe weight loading
3. **Input validation** - Dataset schemas verified

## Archived Code

Experimental optimizers are in `archive/experimental_optimizers/`:
- 25+ experimental approaches tested
- Kept for reference/inspiration
- Not part of main pipeline

## API Reference

### UnifiedTrainer
```python
from src.training.pipeline import UnifiedTrainer, TrainingConfig

config = TrainingConfig(
    model_path="models/phi-4",
    batch_size=32,
    epochs=1,
    learning_rate=1e-4,
)
trainer = UnifiedTrainer(config)
trainer.train(dataset)
```

### BenchmarkSuite
```python
from src.benchmarks import BenchmarkSuite

suite = BenchmarkSuite("models/phi-4")
suite.run_all(quick=True)
suite.save_results()
```

## Hardware Requirements

- **Minimum**: 4GB VRAM (GTX 1650)
- **Recommended**: 8GB+ VRAM
- **CPU**: Works but slower
