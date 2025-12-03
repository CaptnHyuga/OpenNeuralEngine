"""Architecture inspection and auto-detection for pretrained models.

This module provides tools to automatically infer model architecture from:
1. Safetensors metadata (preferred - has embedded config)
2. State dict shape inspection (.pt, .pkl)
3. ONNX graph structure (.onnx)

Enables loading arbitrary pretrained weights without explicit config files,
critical for researchers working with diverse architectures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn


class ArchitectureSignature:
    """Represents the structural signature of a model architecture.
    
    Used to match unknown checkpoints to known architecture patterns.
    """
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        has_gqa: bool = False,
        has_rope: bool = False,
        has_swiglu: bool = False,
        has_rmsnorm: bool = False,
        is_multimodal: bool = False,
        modalities: Optional[List[str]] = None,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.has_gqa = has_gqa
        self.has_rope = has_rope
        self.has_swiglu = has_swiglu
        self.has_rmsnorm = has_rmsnorm
        self.is_multimodal = is_multimodal
        self.modalities = modalities or []
    
    def to_config(self) -> Dict[str, Any]:
        """Convert signature to a config dict for model building."""
        config = {}
        
        if self.vocab_size:
            config["vocab_size"] = self.vocab_size
        if self.embedding_dim:
            config["embedding_dim"] = self.embedding_dim
        if self.hidden_dim:
            config["hidden_dim"] = self.hidden_dim
        if self.num_layers:
            config["num_micro_layers"] = self.num_layers
        if self.num_heads:
            config["num_heads"] = self.num_heads
        if self.num_kv_heads:
            config["num_kv_heads"] = self.num_kv_heads
        if self.max_seq_len:
            config["max_seq_len"] = self.max_seq_len
        
        # SOTA flags
        config["use_gqa"] = self.has_gqa
        config["use_rope"] = self.has_rope
        config["use_swiglu"] = self.has_swiglu
        config["use_rmsnorm"] = self.has_rmsnorm
        
        # Multimodal
        if self.is_multimodal:
            config["multimodal"] = True
            config["modalities"] = self.modalities
        
        return config


def detect_architecture_from_safetensors(path: str) -> Optional[Dict[str, Any]]:
    """Extract architecture config from safetensors metadata.
    
    Safetensors allows embedding arbitrary JSON metadata in the file header.
    This is our preferred method for self-contained checkpoints.
    
    Args:
        path: Path to .safetensors file.
    
    Returns:
        Config dict if found, None otherwise.
    """
    try:
        from safetensors import safe_open
        
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            
            if metadata and "config" in metadata:
                # Config was embedded during save
                return json.loads(metadata["config"])
            
            # Try legacy keys
            if metadata and "model_config" in metadata:
                return json.loads(metadata["model_config"])
                
    except Exception:
        pass
    
    return None


def infer_architecture_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> ArchitectureSignature:
    """Infer architecture by analyzing state dict shapes and keys.
    
    This is our fallback for .pt and .pkl files without metadata.
    Uses heuristics based on common naming patterns and tensor shapes.
    
    Args:
        state_dict: Model state dictionary.
        verbose: Print inspection details.
    
    Returns:
        ArchitectureSignature with inferred parameters.
    """
    sig = ArchitectureSignature()
    
    # Find embedding layer
    for key, tensor in state_dict.items():
        if "embedding" in key.lower() and "weight" in key:
            if len(tensor.shape) == 2:
                sig.vocab_size = tensor.shape[0]
                sig.embedding_dim = tensor.shape[1]
                if verbose:
                    print(f"  Found embedding: vocab={sig.vocab_size}, dim={sig.embedding_dim}")
                break
    
    # Find hidden dimension (look for attention projections)
    for key, tensor in state_dict.items():
        if "attn" in key.lower() and ("q_proj" in key or "query" in key):
            if len(tensor.shape) == 2:
                sig.hidden_dim = tensor.shape[1]
                if verbose:
                    print(f"  Found hidden_dim from attention: {sig.hidden_dim}")
                break
    
    # Count transformer blocks
    layer_indices = set()
    for key in state_dict.keys():
        # Match patterns like "block_0", "layer.0", "h.0", etc.
        if "block" in key or "layer" in key or ".h." in key:
            parts = key.split(".")
            for part in parts:
                if part.isdigit():
                    layer_indices.add(int(part))
    
    if layer_indices:
        sig.num_layers = max(layer_indices) + 1
        if verbose:
            print(f"  Inferred num_layers: {sig.num_layers}")
    
    # Detect GQA (grouped query attention)
    for key in state_dict.keys():
        if "kv_proj" in key or "num_kv_heads" in key:
            sig.has_gqa = True
            if verbose:
                print("  Detected GQA architecture")
            break
    
    # Detect RoPE
    for key in state_dict.keys():
        if "rope" in key.lower() or "rotary" in key.lower():
            sig.has_rope = True
            if verbose:
                print("  Detected RoPE")
            break
    
    # Detect SwiGLU (look for gated FFN)
    for key in state_dict.keys():
        if "gate" in key.lower() and ("mlp" in key.lower() or "ffn" in key.lower()):
            sig.has_swiglu = True
            if verbose:
                print("  Detected SwiGLU/Gated FFN")
            break
    
    # Detect RMSNorm
    for key in state_dict.keys():
        if "rms_norm" in key.lower() or "rmsnorm" in key.lower():
            sig.has_rmsnorm = True
            if verbose:
                print("  Detected RMSNorm")
            break
    
    # Detect multimodal (vision encoders, fusion layers)
    for key in state_dict.keys():
        if "vision" in key.lower() or "image" in key.lower():
            sig.is_multimodal = True
            if "vision" not in sig.modalities:
                sig.modalities.append("vision")
        if "audio" in key.lower():
            sig.is_multimodal = True
            if "audio" not in sig.modalities:
                sig.modalities.append("audio")
    
    return sig


def infer_input_output_spec(
    model: nn.Module,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Probe model to determine input/output specifications.
    
    Performs a dry-run forward pass with synthetic inputs to discover:
    - Input shapes and dtypes
    - Output shapes and types (logits, embeddings, etc.)
    
    Critical for UI to handle unknown architectures gracefully.
    
    Args:
        model: The loaded model.
        device: Device to run probe on.
    
    Returns:
        (input_spec, output_spec) dictionaries.
    """
    model.eval()
    model.to(device)
    
    input_spec = {
        "type": "token_ids",
        "dtype": "long",
        "shape": ["batch", "seq_len"],
        "seq_len_range": [1, 2048],
    }
    
    output_spec = {
        "type": "unknown",
        "shape": None,
    }
    
    try:
        # Try common input shapes
        for seq_len in [8, 16, 32]:
            test_input = torch.randint(0, 1000, (1, seq_len), dtype=torch.long, device=device)
            
            with torch.no_grad():
                output = model(test_input)
            
            # Analyze output
            if isinstance(output, torch.Tensor):
                output_spec["shape"] = list(output.shape)
                
                # Classify output type
                if len(output.shape) == 3 and output.shape[-1] > 100:
                    output_spec["type"] = "logits"  # (batch, seq, vocab)
                elif len(output.shape) == 2:
                    if output.shape[-1] < 100:
                        output_spec["type"] = "classification"  # (batch, num_classes)
                    else:
                        output_spec["type"] = "embeddings"  # (batch, hidden_dim)
                
                break
            
            elif isinstance(output, tuple):
                # Model returns multiple outputs (logits, hidden_states, etc.)
                output_spec["type"] = "multi_output"
                output_spec["shapes"] = [list(o.shape) if isinstance(o, torch.Tensor) else None for o in output]
                break
    
    except Exception as e:
        output_spec["error"] = str(e)
    
    return input_spec, output_spec


def load_architecture_config(
    checkpoint_path: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Unified architecture detection from any checkpoint format.
    
    Priority:
    1. Safetensors metadata (if available)
    2. Sibling .json config file
    3. State dict inspection
    
    Args:
        checkpoint_path: Path to checkpoint (.safetensors, .pt, .pkl).
        verbose: Print detection steps.
    
    Returns:
        Config dictionary ready for model building.
    """
    path = Path(checkpoint_path)
    
    if verbose:
        print(f"🔍 Auto-detecting architecture: {path.name}")
    
    # Method 1: Safetensors metadata
    if path.suffix == ".safetensors":
        config = detect_architecture_from_safetensors(str(path))
        if config:
            if verbose:
                print("  ✅ Found config in safetensors metadata")
            return config
    
    # Method 2: Sibling config file
    for candidate in [path.with_suffix(".json"), path.parent / "config.json"]:
        if candidate.exists():
            if verbose:
                print(f"  ✅ Found config file: {candidate.name}")
            return json.loads(candidate.read_text())
    
    # Method 3: State dict inspection
    if verbose:
        print("  ⚙️  No metadata found, inspecting state dict...")
    
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(path))
    else:
        checkpoint = torch.load(  # nosec B614
            path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    sig = infer_architecture_from_state_dict(state_dict, verbose=verbose)
    config = sig.to_config()
    
    if verbose:
        print(f"  ✅ Inferred config: {json.dumps(config, indent=2)}")
    
    return config


# Registry for custom architecture builders
_ARCHITECTURE_REGISTRY: Dict[str, Any] = {}


def register_architecture(name: str, builder_fn):
    """Register a custom architecture builder.
    
    Example:
        @register_architecture("my_custom_transformer")
        def build_custom_transformer(config):
            return MyCustomTransformer(**config)
    """
    _ARCHITECTURE_REGISTRY[name] = builder_fn


def get_builder_for_architecture(arch_type: str):
    """Retrieve builder function for a registered architecture."""
    return _ARCHITECTURE_REGISTRY.get(arch_type)
