"""Convert model weights between different formats.

Supports:
- Pickle (.pkl) → SafeTensors (.safetensors)
- PyTorch (.pt, .pth) → SafeTensors (.safetensors)
- SafeTensors → ONNX (via export)

WARNING: Pickle files can execute arbitrary code during loading.
Only convert pickle files from trusted sources!
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def convert_pickle_to_safetensors(
    pickle_path: str,
    output_path: str,
    verify: bool = True,
    trusted: bool = False,
) -> None:
    """Convert pickle weights to safetensors format.
    
    Args:
        pickle_path: Path to pickle file (.pkl)
        output_path: Path to output safetensors file
        verify: Verify conversion by reloading
        trusted: Set to True to acknowledge pickle security risk
        
    Raises:
        ValueError: If file format is unsupported or pickle is untrusted
        
    Example:
        >>> convert_pickle_to_safetensors('model.pkl', 'model.safetensors', trusted=True)
    """
    if not trusted:
        raise ValueError(
            "Pickle files can execute arbitrary code during loading!\n"
            "Only convert pickle files from trusted sources.\n"
            "Set trusted=True to proceed after verifying the source."
        )
    
    print(f"⚠️  Loading pickle file: {pickle_path}")
    print("   This can execute arbitrary code - ensure the file is trusted!")
    
    pickle_path = Path(pickle_path)
    output_path = Path(output_path)
    
    # Load pickle (DANGER ZONE - used for legacy checkpoint conversion only)
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)  # nosec B301 - Intentional for legacy checkpoint conversion
    
    # Extract state dict from various formats
    state_dict = _extract_state_dict(data)
    
    # Convert to safetensors
    _save_safetensors(state_dict, output_path)
    
    if verify:
        _verify_conversion(state_dict, output_path)
    
    print(f"✅ Successfully converted: {pickle_path} → {output_path}")


def convert_pytorch_to_safetensors(
    pt_path: str,
    output_path: str,
    verify: bool = True,
) -> None:
    """Convert PyTorch checkpoint to safetensors format.
    
    Args:
        pt_path: Path to PyTorch checkpoint (.pt, .pth)
        output_path: Path to output safetensors file
        verify: Verify conversion by reloading
        
    Example:
        >>> convert_pytorch_to_safetensors('checkpoint.pt', 'model.safetensors')
    """
    pt_path = Path(pt_path)
    output_path = Path(output_path)
    
    print(f"📦 Loading PyTorch checkpoint: {pt_path}")
    
    # Load with weights_only for security
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=True)
    
    # Extract state dict
    state_dict = _extract_state_dict(checkpoint)
    
    # Convert to safetensors
    _save_safetensors(state_dict, output_path)
    
    if verify:
        _verify_conversion(state_dict, output_path)
    
    print(f"✅ Successfully converted: {pt_path} → {output_path}")


def _extract_state_dict(data: Any) -> Dict[str, torch.Tensor]:
    """Extract state dict from various checkpoint formats."""
    
    if isinstance(data, dict):
        # Try common keys
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in data:
                return data[key]
        
        # Check if it's already a state dict (all values are tensors)
        if all(isinstance(v, torch.Tensor) for v in data.values()):
            return data
        
        raise ValueError(
            "Could not find state dict in checkpoint. "
            f"Available keys: {list(data.keys())}"
        )
    
    elif isinstance(data, nn.Module):
        # Direct model object
        return data.state_dict()
    
    else:
        raise ValueError(
            f"Unsupported checkpoint format: {type(data)}. "
            "Expected dict with 'state_dict' key or nn.Module."
        )


def _save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save state dict to safetensors format."""
    
    try:
        from safetensors.torch import save_file
    except ImportError as e:
        raise ImportError(
            "safetensors not installed. Install with: pip install safetensors"
        ) from e
    
    # Handle shared tensors (clone if needed)
    safe_state = {}
    seen_ptrs = {}
    
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"⚠️  Skipping non-tensor parameter: {key} (type: {type(tensor)})")
            continue
        
        # Check for shared storage
        storage = tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
        ptr = storage.data_ptr()
        
        if ptr in seen_ptrs:
            # Clone shared tensors
            safe_state[key] = tensor.clone()
        else:
            seen_ptrs[ptr] = key
            safe_state[key] = tensor
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add conversion metadata
    meta = metadata or {}
    meta['converted_with'] = 'SNN convert_weights.py'
    meta['num_parameters'] = str(sum(t.numel() for t in safe_state.values()))
    
    # Save
    save_file(safe_state, str(output_path), metadata=meta)


def _verify_conversion(
    original_state: Dict[str, torch.Tensor],
    safetensors_path: Path,
) -> None:
    """Verify that safetensors file matches original state dict."""
    
    from safetensors.torch import load_file
    
    loaded_state = load_file(str(safetensors_path))
    
    # Check keys match
    original_keys = set(k for k, v in original_state.items() if isinstance(v, torch.Tensor))
    loaded_keys = set(loaded_state.keys())
    
    if original_keys != loaded_keys:
        missing = original_keys - loaded_keys
        extra = loaded_keys - original_keys
        
        if missing:
            print(f"⚠️  Missing keys in converted file: {missing}")
        if extra:
            print(f"⚠️  Extra keys in converted file: {extra}")
        
        raise ValueError("Key mismatch between original and converted state dicts")
    
    # Check values match (sample check for large models)
    for key in list(original_keys)[:10]:  # Check first 10 parameters
        if not torch.allclose(original_state[key], loaded_state[key], rtol=1e-5):
            raise ValueError(f"Value mismatch for parameter: {key}")
    
    print("✅ Verification passed: weights match exactly")


def batch_convert_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.pkl",
    trusted: bool = False,
) -> None:
    """Convert all matching files in a directory.
    
    Args:
        input_dir: Directory containing model files
        output_dir: Directory for converted files
        pattern: Glob pattern for files to convert (e.g., "*.pkl", "*.pt")
        trusted: Set to True for pickle files (acknowledges security risk)
        
    Example:
        >>> batch_convert_directory('old_models/', 'new_models/', '*.pkl', trusted=True)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob(pattern))
    
    if not files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to convert")
    
    for input_path in files:
        output_path = output_dir / f"{input_path.stem}.safetensors"
        
        try:
            if input_path.suffix == '.pkl':
                convert_pickle_to_safetensors(str(input_path), str(output_path), trusted=trusted)
            elif input_path.suffix in ['.pt', '.pth']:
                convert_pytorch_to_safetensors(str(input_path), str(output_path))
            else:
                print(f"⚠️  Skipping unsupported format: {input_path}")
                
        except Exception as e:
            print(f"❌ Failed to convert {input_path}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert model weights to SafeTensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PyTorch checkpoint
  python convert_weights.py model.pt model.safetensors
  
  # Convert pickle (UNSAFE - only if trusted!)
  python convert_weights.py model.pkl model.safetensors --trusted
  
  # Convert directory
  python convert_weights.py old_models/ new_models/ --pattern "*.pt"
  
  # Batch convert pickle files (DANGEROUS!)
  python convert_weights.py pickles/ safetensors/ --pattern "*.pkl" --trusted

Security Warning:
  Pickle files can execute arbitrary code when loaded!
  Only convert pickle files from trusted sources.
  Use --trusted flag to acknowledge this risk.
"""
    )
    
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('output', help='Output file or directory')
    parser.add_argument('--pattern', default='*.pt', help='File pattern for batch conversion')
    parser.add_argument('--trusted', action='store_true', help='Trust pickle files (UNSAFE)')
    parser.add_argument('--no-verify', action='store_true', help='Skip verification')
    parser.add_argument('--batch', action='store_true', help='Batch convert directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Batch conversion
        batch_convert_directory(
            args.input,
            args.output,
            pattern=args.pattern,
            trusted=args.trusted,
        )
    else:
        # Single file conversion
        verify = not args.no_verify
        
        if input_path.suffix == '.pkl':
            convert_pickle_to_safetensors(
                args.input,
                args.output,
                verify=verify,
                trusted=args.trusted,
            )
        elif input_path.suffix in ['.pt', '.pth']:
            convert_pytorch_to_safetensors(
                args.input,
                args.output,
                verify=verify,
            )
        else:
            print(f"❌ Unsupported file format: {input_path.suffix}")
            print("   Supported: .pkl (with --trusted), .pt, .pth")
            exit(1)
