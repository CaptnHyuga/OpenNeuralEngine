"""Shared test helpers for secure serialization routines."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def safe_load_weights(path: str | Path) -> Any:
    """Load Torch artifacts - prefer safetensors, fallback to torch for test artifacts.
    
    Note: torch.load fallback is only for test-generated temporary files.
    Production code should use safetensors exclusively.
    """
    path = Path(path)
    
    # Try safetensors first
    safetensors_path = path.with_suffix('.safetensors') if path.suffix != '.safetensors' else path
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            return load_file(safetensors_path)
        except ImportError:
            pass
    
    # Fallback for test-generated .pt files only
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)  # nosec B614 - test artifacts
    
    raise FileNotFoundError(f"No checkpoint found at {path}")
