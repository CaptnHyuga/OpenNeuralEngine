"""ONNX export and inference utilities for SNN models.

Provides:
- Export PyTorch models to ONNX format
- Run inference with ONNX Runtime
- Optimize ONNX models for production
- Verify ONNX output matches PyTorch

Benefits:
- 2-10x faster inference (CPU optimized)
- Cross-platform deployment
- Smaller model files
- No PyTorch dependency for inference
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 128),
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 17,
    verify: bool = True,
    optimize: bool = True,
) -> None:
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Shape of dummy input (batch_size, seq_len) for language models
        input_names: Names for input tensors (default: ['input_ids'])
        output_names: Names for output tensors (default: ['logits'])
        dynamic_axes: Dynamic axes for variable-length inputs
        opset_version: ONNX opset version (17 recommended for latest features)
        verify: Verify ONNX output matches PyTorch
        optimize: Apply ONNX optimization passes
        
    Example:
        >>> model = build_model_from_config(config)
        >>> export_to_onnx(model, 'model.onnx', input_shape=(1, 256))
    """
    try:
        import onnx  # noqa: F401 - Used for later operations
    except ImportError as e:
        raise ImportError("onnx not installed. Install with: pip install onnx") from e
    
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default names
    if input_names is None:
        input_names = ['input_ids']
    if output_names is None:
        output_names = ['logits']
    
    # Default dynamic axes (batch and sequence length)
    if dynamic_axes is None:
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        }
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, input_shape, dtype=torch.long)
    
    print("📦 Exporting model to ONNX...")
    print(f"   Input shape: {input_shape}")
    print(f"   Opset version: {opset_version}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print(f"✅ Exported to: {output_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # Optimize ONNX model
    if optimize:
        optimized_path = output_path.with_suffix('.optimized.onnx')
        optimize_onnx_model(str(output_path), str(optimized_path))
    
    # Verify output matches PyTorch
    if verify:
        verify_onnx_export(model, str(output_path), dummy_input)


def optimize_onnx_model(
    input_path: str,
    output_path: str,
) -> None:
    """Optimize ONNX model for faster inference.
    
    Applies:
    - Constant folding
    - Redundant node elimination
    - Graph optimization passes
    
    Args:
        input_path: Path to ONNX model
        output_path: Path to save optimized model
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer as _ort_optimizer  # noqa: F401
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping optimization")
        return
    
    print("⚙️  Optimizing ONNX model...")
    
    # Load model
    model = onnx.load(input_path)
    
    # Apply optimization passes
    from onnx import optimizer as onnx_optimizer
    passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_pad',
        'eliminate_nop_transpose',
        'eliminate_unused_initializer',
        'extract_constant_to_initializer',
        'fuse_add_bias_into_conv',
        'fuse_bn_into_conv',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv',
        'fuse_transpose_into_gemm',
    ]
    
    optimized_model = onnx_optimizer.optimize(model, passes)
    
    # Save optimized model
    onnx.save(optimized_model, output_path)
    
    # Compare sizes
    original_size = Path(input_path).stat().st_size / (1024 * 1024)
    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"✅ Optimized: {original_size:.2f} MB → {optimized_size:.2f} MB")
    print(f"   Saved to: {output_path}")


def verify_onnx_export(
    pytorch_model: nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """Verify ONNX model output matches PyTorch.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        dummy_input: Test input tensor
        tolerance: Maximum allowed difference
        
    Raises:
        AssertionError: If outputs differ by more than tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime not installed. Install with: pip install onnxruntime"
        ) from e
    
    print("🔍 Verifying ONNX export...")
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input)
    
    # Handle tuple outputs
    if isinstance(pytorch_output, tuple):
        pytorch_output = pytorch_output[0]
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    onnx_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = session.run(None, onnx_input)[0]
    
    # Compare outputs
    diff = torch.abs(torch.from_numpy(onnx_output) - pytorch_output).max().item()
    
    print(f"   Max difference: {diff:.8f}")
    
    if diff > tolerance:
        raise AssertionError(
            f"ONNX output differs from PyTorch by {diff:.8f} (tolerance: {tolerance})"
        )
    
    print("✅ ONNX output matches PyTorch")


def run_onnx_inference(
    onnx_path: str,
    input_ids: Union[List[int], torch.Tensor],
    providers: Optional[List[str]] = None,
) -> torch.Tensor:
    """Run inference with ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        input_ids: Input token IDs (list or tensor)
        providers: Execution providers (default: ['CPUExecutionProvider'])
                  Options: 'CUDAExecutionProvider', 'TensorrtExecutionProvider'
        
    Returns:
        Output logits as torch.Tensor
        
    Example:
        >>> logits = run_onnx_inference('model.onnx', [1, 2, 3, 4])
        >>> predictions = logits.argmax(dim=-1)
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "onnxruntime not installed. Install with: pip install onnxruntime"
        ) from e
    
    # Default to CPU
    if providers is None:
        providers = ['CPUExecutionProvider']
    
    # Create session
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Convert input to numpy
    if isinstance(input_ids, torch.Tensor):
        input_array = input_ids.numpy()
    else:
        input_array = np.array([input_ids], dtype=np.int64)
    
    # Ensure 2D shape (batch, seq_len)
    if input_array.ndim == 1:
        input_array = input_array[np.newaxis, :]
    
    # Run inference
    input_name = session.get_inputs()[0].name
    onnx_inputs = {input_name: input_array}
    outputs = session.run(None, onnx_inputs)
    
    # Convert back to torch
    return torch.from_numpy(outputs[0])


class ONNXInferenceSession:
    """Wrapper for ONNX inference sessions with caching and utilities."""
    
    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
    ):
        """Initialize ONNX inference session.
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers (default: CPU)
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            ) from e
        
        if providers is None:
            # Auto-detect best provider
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print("📦 ONNX session initialized")
        print(f"   Providers: {providers}")
        print(f"   Input: {self.input_name}")
        print(f"   Output: {self.output_name}")
    
    def __call__(self, input_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """Run inference (callable interface)."""
        import numpy as np
        
        # Convert to numpy
        if isinstance(input_ids, torch.Tensor):
            input_array = input_ids.numpy()
        else:
            input_array = np.array([input_ids], dtype=np.int64)
        
        # Ensure 2D
        if input_array.ndim == 1:
            input_array = input_array[np.newaxis, :]
        
        # Run
        outputs = self.session.run(None, {self.input_name: input_array})
        
        return torch.from_numpy(outputs[0])
    
    def benchmark(
        self,
        input_shape: Tuple[int, int] = (1, 128),
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference speed.
        
        Args:
            input_shape: Input shape (batch_size, seq_len)
            num_runs: Number of iterations
            
        Returns:
            Dict with timing statistics
        """
        import time
        import numpy as np
        
        # Create random input
        input_array = np.random.randint(0, 1000, input_shape, dtype=np.int64)
        
        # Warmup
        for _ in range(10):
            self.session.run(None, {self.input_name: input_array})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run(None, {self.input_name: input_array})
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            'mean_ms': float(times.mean()),
            'std_ms': float(times.std()),
            'min_ms': float(times.min()),
            'max_ms': float(times.max()),
            'median_ms': float(np.median(times)),
            'throughput_samples_per_sec': 1000.0 / times.mean(),
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export and run ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export model to ONNX
  python onnx_export.py export model.safetensors model.onnx
  
  # Export with custom input shape
  python onnx_export.py export model.safetensors model.onnx --input-shape 1 256
  
  # Optimize ONNX model
  python onnx_export.py optimize model.onnx model.optimized.onnx
  
  # Benchmark ONNX model
  python onnx_export.py benchmark model.onnx
  
  # Test inference
  python onnx_export.py infer model.onnx --input "1,2,3,4,5"
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export PyTorch model to ONNX')
    export_parser.add_argument('model_path', help='Path to PyTorch model (.pt, .safetensors)')
    export_parser.add_argument('output', help='Output ONNX file')
    export_parser.add_argument('--input-shape', nargs=2, type=int, default=[1, 128],
                              help='Input shape (batch_size seq_len)')
    export_parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    export_parser.add_argument('--no-verify', action='store_true', help='Skip verification')
    export_parser.add_argument('--no-optimize', action='store_true', help='Skip optimization')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize ONNX model')
    opt_parser.add_argument('input', help='Input ONNX file')
    opt_parser.add_argument('output', help='Output optimized ONNX file')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark ONNX model')
    bench_parser.add_argument('onnx_path', help='Path to ONNX model')
    bench_parser.add_argument('--input-shape', nargs=2, type=int, default=[1, 128])
    bench_parser.add_argument('--runs', type=int, default=100)
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run ONNX inference')
    infer_parser.add_argument('onnx_path', help='Path to ONNX model')
    infer_parser.add_argument('--input', required=True, help='Input IDs (comma-separated)')
    
    args = parser.parse_args()
    
    if args.command == 'export':
        # Load model
        
        # For now, require user to specify how to load
        print("❌ Automated export not yet implemented")
        print("💡 Use export_to_onnx() function in your code")
        
    elif args.command == 'optimize':
        optimize_onnx_model(args.input, args.output)
        
    elif args.command == 'benchmark':
        session = ONNXInferenceSession(args.onnx_path)
        results = session.benchmark(
            input_shape=tuple(args.input_shape),
            num_runs=args.runs,
        )
        
        print("\n📊 Benchmark Results:")
        for key, value in results.items():
            print(f"   {key}: {value:.4f}")
    
    elif args.command == 'infer':
        input_ids = [int(x) for x in args.input.split(',')]
        output = run_onnx_inference(args.onnx_path, input_ids)
        predictions = output.argmax(dim=-1)
        
        print(f"Input: {input_ids}")
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {predictions.tolist()}")
    
    else:
        parser.print_help()
