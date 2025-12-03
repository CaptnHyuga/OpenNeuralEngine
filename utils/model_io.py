"""Model I/O utilities for saving and loading models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import safetensors  # noqa: F401
    SAFETENSORS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SAFETENSORS_AVAILABLE = False


def save_model(
    model: nn.Module,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model weights to safetensors format with embedded config.
    
    Args:
        model: PyTorch model to save.
        path: Output path for the safetensors file.
        metadata: Optional metadata to include in the file.
        config: Optional architecture config for auto-loading.
    """
    try:
        from safetensors.torch import save_file
        
        state_dict = model.state_dict()
        safe_state = {}
        seen_ptrs = {}
        def _data_ptr(tensor: torch.Tensor) -> int:
            storage = tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
            return storage.data_ptr()
        for key, tensor in state_dict.items():
            ptr = _data_ptr(tensor)
            if ptr in seen_ptrs:
                # Clone shared tensors so safetensors can serialize safely
                safe_state[key] = tensor.clone()
            else:
                seen_ptrs[ptr] = key
                safe_state[key] = tensor
        # Convert metadata values to strings for safetensors
        meta = {}
        if metadata:
            for k, v in metadata.items():
                meta[k] = str(v)
        
        # Embed config for self-contained checkpoints
        if config:
            import json
            meta["config"] = json.dumps(config)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(safe_state, str(path), metadata=meta)
        
    except ImportError:
        # Fallback to torch.save
        path = Path(path).with_suffix(".pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }, path)


def load_model(
    model: nn.Module,
    path: Path,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load model weights from safetensors or pt file.
    
    Args:
        model: Model to load weights into.
        path: Path to checkpoint file.
        strict: If True, require exact match.
        
    Returns:
        Dictionary with loading statistics.
    """
    path = Path(path)
    
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(str(path))
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Load with partial matching
    model_state = model.state_dict()
    matched = 0
    missing = []
    unexpected = []
    
    for key in state_dict:
        if key in model_state:
            if state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]
                matched += 1
            else:
                missing.append(key)
        else:
            unexpected.append(key)
    
    for key in model_state:
        if key not in state_dict:
            missing.append(key)
    
    model.load_state_dict(model_state, strict=strict)
    
    return {
        "matched": matched,
        "missing": missing,
        "unexpected": unexpected,
    }
