# OpenNeuralEngine 2.0 - Implementation Plan

## Vision
Transform ONN from an educational framework into a **production-grade democratic AI framework** that:
- Wraps and orchestrates best-in-class libraries (HuggingFace, DeepSpeed, bitsandbytes)
- Provides **zero-configuration intelligence** - auto-detects hardware and optimizes accordingly
- Supports **any architecture** (Transformers, CNNs, RNNs, GNNs, Diffusion, custom hybrids)
- Enables **true drag-and-drop** for any dataset format
- Works identically on a 4GB laptop OR a multi-node cluster

## Architecture: ONN as Smart Orchestration Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Simple CLI)                   │
│        onn train --model llama-70b --dataset my_data.jsonl       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ONN ORCHESTRATION LAYER                      │
│  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐    │
│  │  Hardware   │  │  Configuration  │  │  Resource         │    │
│  │  Profiler   │→ │  Orchestrator   │→ │  Monitor          │    │
│  └─────────────┘  └─────────────────┘  └───────────────────┘    │
│  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────┐    │
│  │  Universal  │  │  Universal      │  │  Experiment       │    │
│  │  Model      │  │  Data           │  │  Tracking         │    │
│  │  Loader     │  │  Adapters       │  │  (Aim)            │    │
│  └─────────────┘  └─────────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PRODUCTION LIBRARIES (Wrapped)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ HuggingFace │  │  DeepSpeed  │  │bitsandbytes │             │
│  │  Trainer    │  │  ZeRO 1-3   │  │ INT4/INT8   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   PyTorch   │  │    timm     │  │    vLLM     │             │
│  │   FSDP      │  │  (Vision)   │  │ (Inference) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## What to KEEP (ONN's Unique Value)

| File/Component | Reason |
|----------------|--------|
| `experiment_tracking.py` | Aim integration is excellent - ENHANCE not replace |
| `utils/device_manager.py` | Good foundation - EXTEND with full profiling |
| `utils/model_io.py` | Safetensors I/O is solid - KEEP |
| `utils/project_paths.py` | Path management works - KEEP |
| `builders.py` (partial) | Model factory pattern - ADAPT for universal models |
| `hf_compat.py` (partial) | HF interop foundation - EXTEND |
| Aim integration | Docker-based tracking - KEEP |
| Preset system | User-friendly defaults - KEEP and EXPAND |

## What to WRAP (Use Production Libraries)

| Current ONN Code | Wrap With | Reason |
|------------------|-----------|--------|
| `trainer.py` training loop | HuggingFace Trainer | Battle-tested, all optimizations built-in |
| Custom attention | PyTorch SDPA / Flash Attention 2 | Kernel-level optimized |
| Gradient accumulation | DeepSpeed ZeRO | Production-grade memory optimization |
| Basic quantization | bitsandbytes, GPTQ, AWQ | SOTA quantization |
| Custom layers | HF/timm implementations | Maintained by large teams |
| Inference serving | vLLM | Optimized inference engine |

## What to REMOVE (Obsolete After Wrapping)

| File | Reason |
|------|--------|
| `efficient_attention.py` | → Use PyTorch SDPA / Flash Attention |
| `rope.py` | → Use HF's RoPE implementations |
| `layers.py` (most) | → Use HF layers |
| `text_micro_layers.py` | → Use HF building blocks |
| `multimodal_micro_layers.py` | → Use HF multimodal |
| `multimodal_fusion.py` | → Use HF multimodal |
| `routed_layers.py` | → Use MoE from transformers |
| `heads.py` | → Use HF heads |
| `puzzle_model.py` | → Replace with universal model wrapper |
| `base_module.py` | → Not needed with HF base |
| `advanced_training.py` | → Built into HF Trainer |
| `precision.py` | → Handled by HF/DeepSpeed |
| `memory.py` | → Handled by DeepSpeed |
| `vision_encoders.py` | → Use timm/HF vision |
| `multimodal_model.py` | → Use HF multimodal |
| `benchmark.py` | → Integrate into new system |
| `online_benchmarks.py` | → Integrate into evaluator |
| `dod_utils.py` | → Not needed |
| `constants.py` | → Consolidate into config |

## What to ADD (Novel ONN Features)

| New Component | Purpose |
|---------------|---------|
| `src/orchestration/hardware_profiler.py` | Full hardware detection (VRAM, RAM, CPU, disk, network) |
| `src/orchestration/config_orchestrator.py` | Decision engine: hardware → optimal config |
| `src/orchestration/resource_monitor.py` | Runtime monitoring and adaptive adjustment |
| `src/wrappers/hf_trainer_wrapper.py` | Intelligent HF Trainer wrapper |
| `src/wrappers/deepspeed_wrapper.py` | Auto-configured DeepSpeed |
| `src/wrappers/quantization_wrapper.py` | Unified quantization interface |
| `src/wrappers/model_loader.py` | Universal model loading |
| `src/data_adapters/base.py` | Data adapter interface |
| `src/data_adapters/image.py` | Image dataset adapter |
| `src/data_adapters/audio.py` | Audio dataset adapter |
| `src/data_adapters/text.py` | Text dataset adapter |
| `src/data_adapters/video.py` | Video dataset adapter |
| `src/data_adapters/mesh.py` | 3D mesh dataset adapter |
| `src/data_adapters/multimodal.py` | Paired data detection |
| `onn.py` | New unified CLI |

---

## New Directory Structure

```
SNN/
├── onn.py                          # NEW: Unified CLI entry point
├── pyproject.toml                  # UPDATED: New dependencies
├── config/
│   ├── presets/                    # NEW: Model presets
│   │   ├── nano.json
│   │   ├── tiny.json
│   │   └── production.json
│   └── hardware_profiles/          # NEW: Cached hardware profiles
├── src/
│   ├── __init__.py
│   ├── orchestration/              # NEW: Core intelligence layer
│   │   ├── __init__.py
│   │   ├── hardware_profiler.py    # Hardware detection & profiling
│   │   ├── config_orchestrator.py  # Decision engine
│   │   └── resource_monitor.py     # Runtime monitoring
│   ├── wrappers/                   # NEW: Production library wrappers
│   │   ├── __init__.py
│   │   ├── hf_trainer_wrapper.py   # HuggingFace Trainer wrapper
│   │   ├── deepspeed_wrapper.py    # DeepSpeed configuration
│   │   ├── quantization_wrapper.py # bitsandbytes/GPTQ/AWQ
│   │   └── model_loader.py         # Universal model loading
│   ├── data_adapters/              # NEW: Universal data interface
│   │   ├── __init__.py
│   │   ├── base.py                 # Base adapter interface
│   │   ├── registry.py             # Adapter registration
│   │   ├── image.py                # Image folder/archive
│   │   ├── audio.py                # Audio files (wav, mp3, flac)
│   │   ├── text.py                 # Text (txt, json, csv, parquet)
│   │   ├── video.py                # Video (mp4, avi, mov)
│   │   ├── mesh.py                 # 3D meshes (obj, ply, stl)
│   │   └── multimodal.py           # Paired data detection
│   ├── tracking/                   # KEPT: Aim integration (moved)
│   │   ├── __init__.py
│   │   └── experiment_tracker.py   # Enhanced from experiment_tracking.py
│   └── inference/                  # NEW: Inference serving
│       ├── __init__.py
│       └── server.py               # vLLM-wrapped inference
├── utils/                          # KEPT: Utilities
│   ├── __init__.py
│   ├── device_manager.py           # KEPT: Base device utilities
│   ├── model_io.py                 # KEPT: Safetensors I/O
│   ├── project_paths.py            # KEPT: Path management
│   └── tokenization.py             # KEPT: Basic tokenization
├── scripts/                        # UPDATED: Simplified scripts
│   ├── launch_inference.py         # Inference UI launcher
│   └── export_model.py             # Export utilities
├── tests/                          # UPDATED: New test structure
│   ├── test_hardware_profiler.py
│   ├── test_config_orchestrator.py
│   ├── test_data_adapters.py
│   └── test_integration.py
└── data/                           # KEPT: Dataset storage
```

---

## Implementation Phases

### Phase 1: Core Orchestration (Priority: CRITICAL)
**Timeline: 2-3 weeks**

1. **Hardware Profiler** (~500 lines)
   - VRAM detection (NVIDIA, AMD, Apple Silicon)
   - RAM and CPU profiling
   - Disk speed benchmarking
   - Capability matrix builder
   - Profile caching

2. **Configuration Orchestrator** (~800 lines)
   - Hardware → optimal config mapping
   - HuggingFace TrainingArguments generation
   - DeepSpeed config auto-generation
   - Quantization strategy selection
   - Batch size calculation

3. **HuggingFace Trainer Wrapper** (~400 lines)
   - Wrap HF Trainer API
   - Auto-configuration from orchestrator
   - Checkpoint integration with Aim

### Phase 2: Universal Data Adapters (Priority: HIGH)
**Timeline: 2-3 weeks**

1. **Base Adapter System** (~200 lines)
   - Adapter interface definition
   - Registration system
   - Auto-detection router

2. **Built-in Adapters** (~1500 lines total)
   - Text: txt, json, jsonl, csv, parquet
   - Image: folder structure, archives
   - Audio: wav, mp3, flac, ogg
   - Video: mp4, avi (frame extraction)
   - 3D: obj, ply, stl
   - Multimodal: paired file detection

### Phase 3: Universal Model Loading (Priority: HIGH)
**Timeline: 1-2 weeks**

1. **Model Loader** (~400 lines)
   - HuggingFace models
   - Local PyTorch files (.pt, .safetensors)
   - Python model definitions
   - timm models
   - torchvision models
   - Architecture auto-detection

### Phase 4: New CLI & Integration (Priority: MEDIUM)
**Timeline: 1-2 weeks**

1. **Unified CLI** (~300 lines)
   - `onn train` command
   - `onn infer` command
   - `onn eval` command
   - `onn export` command

2. **Integration**
   - Update Aim tracking
   - Update tests
   - Update documentation

---

## Code Estimation

| Component | Lines of Code | Complexity |
|-----------|---------------|------------|
| Hardware Profiler | 500-700 | Medium |
| Config Orchestrator | 800-1000 | High |
| HF Trainer Wrapper | 400-500 | Medium |
| Data Adapters (all) | 1500-2000 | Medium |
| Model Loader | 400-500 | Medium |
| CLI Interface | 300-400 | Low |
| Resource Monitor | 300-400 | Medium |
| Tests & Docs | 500-700 | Low |
| **TOTAL NEW CODE** | **4700-6200** | - |

---

## Dependencies to Add

```toml
[project.dependencies]
# Core (existing)
torch = ">=2.0"
safetensors = ">=0.4"
aim = ">=3.0"

# NEW: Production wrappers
transformers = ">=4.35"
accelerate = ">=0.24"
deepspeed = ">=0.12"
bitsandbytes = ">=0.41"
peft = ">=0.6"
datasets = ">=2.14"

# NEW: Data adapters
pillow = ">=10.0"
librosa = ">=0.10"
opencv-python = ">=4.8"
trimesh = ">=4.0"  # 3D meshes
pandas = ">=2.0"
pyarrow = ">=14.0"

# NEW: Inference
vllm = ">=0.2"  # Optional for inference serving

# NEW: Model sources
timm = ">=0.9"
torchvision = ">=0.16"
```

---

## Example Usage After Implementation

```bash
# Train any HuggingFace model on any dataset - ONN figures everything out
onn train --model meta-llama/Llama-2-70b --dataset ./my_data/

# Framework automatically:
# ✓ Detects 8GB VRAM, 32GB RAM
# ✓ Enables INT4 quantization
# ✓ Sets batch_size=1, gradient_accumulation=32
# ✓ Uses DeepSpeed ZeRO-2
# ✓ Streams dataset (never loads all into RAM)
# ✓ Tracks everything in Aim
# ✓ Works perfectly!

# Train custom model
onn train --model ./my_model.py --dataset ./audio_3d_pairs/
# ✓ Auto-detects paired audio-mesh data
# ✓ Applies appropriate preprocessing
# ✓ Same one-command simplicity

# Inference
onn infer --model ./checkpoint.safetensors
# ✓ Launches web UI
# ✓ Optimal inference settings auto-configured

# Evaluate
onn eval --model ./checkpoint.safetensors --suite all
# ✓ Runs comprehensive benchmarks
```

---

## Success Criteria

1. **Simplicity**: Single command trains any model on any data
2. **Universality**: Works with any architecture (CNN, Transformer, hybrid)
3. **Scalability**: Same code works on 4GB laptop AND multi-node cluster
4. **Intelligence**: Zero manual configuration required
5. **Production-grade**: Wraps battle-tested libraries
6. **Tracking**: Full experiment tracking via Aim
7. **Extensibility**: Easy to add custom adapters/models
