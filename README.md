# OpenNeuralEngine
OpenNeuralEngine is an open-source initiative designed to democratize access to artificial intelligence. Our mission is to simplify the complexities of neural networks, making powerful machine learning tools accessible to developers, researchers, and hobbyists of all skill levels.

## Vision

**To make Neural Networks accessible to all.**

We believe that the power of AI should not be locked behind steep learning curves or expensive proprietary software. OpenNeuralEngine aims to provide a robust, intuitive, and high-performance framework that empowers everyone to build, train, and deploy neural models effortlessly.

## Key Features

*   **Zero-Configuration Training:** Auto-detects your hardware and makes optimal decisions. No manual VRAM management, batch size tuning, or quantization selection.
*   **Production-Grade Under The Hood:** Integrates DeepSpeed, FSDP, ZeRO for scaling to billions of parameters.
*   **Universal Data Support:** Drag-and-drop any dataset - text, images, audio, video, 3D meshes. ONN figures out the format automatically.
*   **Any Architecture:** CNNs, Transformers, RNNs, Diffusion models - all work with the same simple interface.
*   **One Command:** Same command trains a 7B model on a laptop OR a 70B model on a multi-GPU cluster.

## Installation

```bash
# Clone the repository
git clone https://github.com/CaptnHyuga/OpenNeuralEngine.git
cd OpenNeuralEngine

# Install with all features
pip install -e ".[all]"

# Or minimal install
pip install -e .
```

## Quick Start

### Training

```bash
# Train any HuggingFace model - ONN auto-configures everything
onn train --model gpt2 --dataset ./data.jsonl

# Large model on limited hardware - automatic quantization + offloading
onn train --model meta-llama/Llama-2-7b --dataset ./data/ --use-lora

# Distributed training with DeepSpeed ZeRO-3
onn train --model phi-4 --dataset ./data/ --deepspeed zero3

# Multi-GPU with automatic parallelism
onn train --model llama-70b --dataset ./data/ --nodes 4
```

### Inference & Generation

```bash
# Quick text generation
onn generate --model gpt2 --prompt "Once upon a time"

# Interactive inference mode
onn infer --model ./my-checkpoint/

# Start OpenAI-compatible API server
onn serve --model ./my-checkpoint/ --port 8000
# Then: curl http://localhost:8000/v1/completions -d '{"prompt":"Hello"}'
```

### Evaluation

```bash
# Quick evaluation (5-10 min)
onn eval --model ./checkpoint/ --preset quick

# Standard benchmarks (MMLU, HellaSwag, ARC, etc.)
onn eval --model ./checkpoint/ --preset standard

# Specific tasks
onn eval --model ./checkpoint/ --task mmlu,hellaswag,arc_challenge
```

### Utilities

```bash
# System information
onn info

# Benchmark your hardware
onn benchmark --type full --output results.json

# Analyze model architecture
onn analyze --model gpt2 --config

# Manage checkpoints
onn checkpoint list ./outputs
onn checkpoint info ./my-model/
onn checkpoint clean ./outputs --keep 3

# Merge LoRA adapter with base model
onn merge --base llama-7b --adapter ./lora-adapter/ --output ./merged-model/

# Inspect any dataset
onn data ./my-dataset/

# Export model
onn export --model ./checkpoint/ --format onnx
```

## Supported Data Formats

ONN automatically detects and loads:

| Type | Formats |
|------|---------|
| **Text** | `.txt`, `.json`, `.jsonl`, `.csv`, `.parquet` |
| **Images** | `.jpg`, `.png`, `.webp`, folder structures |
| **Audio** | `.wav`, `.mp3`, `.flac`, `.ogg` |
| **Video** | `.mp4`, `.avi`, `.mov` |
| **3D Meshes** | `.obj`, `.ply`, `.stl` |
| **Multimodal** | Auto-detected paired data by filename |

## Project Structure

```
OpenNeuralEngine/
├── onn.py              # Main CLI entry point
├── src/
│   ├── orchestration/  # Hardware profiler, config orchestrator
│   ├── wrappers/       # HF Trainer, DeepSpeed, quantization
│   ├── data_adapters/  # Universal data loading
│   ├── evaluation/     # Benchmarking (lm-eval wrapper)
│   ├── inference/      # vLLM/HF inference engine
│   └── training/       # Training utilities
├── config/             # Configuration files
├── scripts/            # Utility scripts
├── tests/              # Test suite (170+ tests)
└── docs/               # Documentation
```

## Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Detailed walkthrough
- **[Architecture](docs/ARCHITECTURE.md)** - Technical deep-dive
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Project roadmap

## Need Help?

- 🐛 [Report an Issue](https://github.com/CaptnHyuga/OpenNeuralEngine/issues)
- 💬 [Ask a Question](https://github.com/CaptnHyuga/OpenNeuralEngine/discussions)

---

**License:** MIT | **Maintained by:** OpenNeuralEngine Team