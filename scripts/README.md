# Scripts Directory

Command-line tools and utilities for working with SNN models.

## 🚀 Main Scripts

### Training & Evaluation
- **`../train.py`** - Main training script (kept in root for convenience)
  ```bash
  python train.py --model-preset nano --epochs 3 --device auto
  ```

### Inference
- **`launch_aim_inference.py`** - Launch Aim-integrated inference UI
  ```bash
  python scripts/launch_aim_inference.py --port 8001
  ```
  
- **`aim_inference_extension.py`** - FastAPI inference server with conversation tracking
  ```bash
  python scripts/aim_inference_extension.py
  ```

- **`aim_multi_model_inference.py`** - Multi-model inference with model switching
  ```bash
  python scripts/aim_multi_model_inference.py --models gpt2 TinyLlama/TinyLlama-1.1B-Chat-v1.0
  ```

### Model Evaluation
- **`eval_model.py`** - Comprehensive model evaluation CLI
  ```bash
  python scripts/eval_model.py --suite all --output results.json
  ```

## 🔧 Utility Scripts

### Model Export & Optimization
- **`quantize_model.py`** - INT8 quantization for 4x smaller models
  ```bash
  python scripts/quantize_model.py src/Core_Models/Save/best_model.safetensors --output model_int8.safetensors --verify
  ```

- **`export_to_onnx.py`** - Export models to ONNX for production deployment
  ```bash
  python scripts/export_to_onnx.py model.safetensors model.onnx --optimize --verify
  ```

## 📁 Directory Structure

```
scripts/
├── README.md                        # This file
├── aim_inference_extension.py       # FastAPI inference server
├── aim_multi_model_inference.py     # Multi-model inference UI
├── launch_aim_inference.py          # Inference launcher
├── eval_model.py                    # Model evaluation CLI
├── quantize_model.py                # INT8 quantization
└── export_to_onnx.py                # ONNX export
```

## 🆕 Adding New Scripts

When adding new scripts:
1. Place in `scripts/` directory
2. Add clear docstring at the top
3. Include usage examples in `--help`
4. Update this README
5. Use `argparse` for CLI arguments
6. Follow the existing code style

---

**Last Updated:** December 3, 2025
