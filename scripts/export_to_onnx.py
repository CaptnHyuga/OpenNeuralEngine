"""Export SNN models to ONNX format for production deployment.

This is a CLI wrapper around utils/onnx_export.py.

Benefits of ONNX:
- 2-10x faster inference on CPU
- Cross-platform deployment (no PyTorch dependency)
- Smaller deployment packages (~200MB vs ~2GB with PyTorch)
- Compatible with ONNX Runtime, TensorRT, OpenVINO
"""
import argparse
from pathlib import Path
from utils.model_loading import load_model_from_checkpoint

from utils.onnx_export import export_to_onnx, optimize_onnx_model, ONNXInferenceSession


def main():
    parser = argparse.ArgumentParser(
        description="Export SNN models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python scripts/export_to_onnx.py \\
    src/Core_Models/Save/best_model.safetensors \\
    deployment/model.onnx
  
  # Export with optimization and verification
  python scripts/export_to_onnx.py \\
    src/Core_Models/Save/best_model.safetensors \\
    deployment/model.onnx \\
    --optimize \\
    --verify \\
    --benchmark
  
  # Custom input shape (batch=1, seq_len=512)
  python scripts/export_to_onnx.py \\
    src/Core_Models/Save/best_model.safetensors \\
    deployment/model.onnx \\
    --input-shape 1 512

Then use in inference:
  python launch_aim_inference.py --models deployment/model.onnx
"""
    )
    
    parser.add_argument("input_model", help="Path to input model (.safetensors or .pt)")
    parser.add_argument("output_onnx", help="Path to save ONNX model (.onnx)")
    parser.add_argument("--input-shape", nargs=2, type=int, default=[1, 256],
                       metavar=("BATCH", "SEQ_LEN"),
                       help="Input shape (batch_size, sequence_length)")
    parser.add_argument("--opset", type=int, default=17,
                       help="ONNX opset version (17 recommended)")
    parser.add_argument("--optimize", action="store_true",
                       help="Apply ONNX optimization passes")
    parser.add_argument("--verify", action="store_true",
                       help="Verify ONNX output matches PyTorch")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark ONNX vs PyTorch speed")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_model)
    output_path = Path(args.output_onnx)
    
    if not input_path.exists():
        print(f"❌ Error: Model not found: {input_path}")
        return 1
    
    print(f"📦 Loading model: {input_path}")
    try:
        config_candidate = None
        for candidate in (input_path.with_suffix(".json"), input_path.parent / "config.json"):
            if candidate.exists():
                config_candidate = str(candidate)
                break
        model, metadata = load_model_from_checkpoint(str(input_path), config_path=config_candidate)
        print(f"   Device: {metadata['device']}")
        print(f"   Output type: {metadata.get('output_spec', {}).get('type', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1    print(f"🟦 Exporting ONNX to: {output_path}")
    onnx_path = export_to_onnx(
      model,
      str(output_path),
      input_shape=tuple(args.input_shape),
      opset=args.opset,
      optimize=args.optimize,
      verify=args.verify,
    )
    print(f"✅ Saved: {onnx_path}")
    
    if args.benchmark:
      print("\n📊 Benchmarking ONNX vs PyTorch...")
      session = ONNXInferenceSession(str(output_path))
      # Minimal synthetic benchmark on CPU
      import torch
      tokens = torch.randint(0, 1000, (args.input_shape[0], args.input_shape[1]), dtype=torch.long)
      import time
      with torch.no_grad():
        start = time.perf_counter(); _ = model(tokens); pt_time = time.perf_counter() - start
      start = time.perf_counter(); _ = session(tokens); onnx_time = time.perf_counter() - start
      print(f"   PyTorch: {pt_time*1000:.2f} ms | ONNX: {onnx_time*1000:.2f} ms | Speedup: {pt_time/onnx_time:.2f}x")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
