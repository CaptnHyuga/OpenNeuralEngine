"""Unified model loading utility.

Loads a PuzzleModel from a checkpoint and optional config.
Prefers safetensors; supports .pt checkpoints as fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch

from utils.model_io import load_model as _load_weights
from utils.architecture_inspector import load_architecture_config, infer_input_output_spec
from utils.device_manager import select_device_with_vram_budget, safe_to_device
from src.Core_Models.builders import build_model_from_config


def load_model_from_checkpoint(
    checkpoint_path: str,
    *,
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    vram_budget_mb: Optional[float] = None,
    auto_detect_io: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Build model from config and load weights from checkpoint.

    Args:
        checkpoint_path: Path to `.safetensors` or `.pt` checkpoint.
        config_path: Optional path to JSON config to build the model.
        config: Optional in-memory config dict.
        device: Target device ("cuda", "mps", "cpu"). Auto-selected if None.
        vram_budget_mb: VRAM budget for device selection.
        auto_detect_io: Whether to infer input/output specs.

    Returns:
        (model, metadata) where metadata contains:
            - "config": Architecture config used
            - "device": Device model is loaded on
            - "input_spec": Input specification (if auto_detect_io=True)
            - "output_spec": Output specification (if auto_detect_io=True)
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Build the model from provided config or auto-detect
    if config is None and config_path is None:
        # Use architecture inspector for robust detection
        config = load_architecture_config(str(ckpt), verbose=True)
    model = build_model_from_config(config_path=config_path, config=config)

    # Load weights with partial matching for robustness
    _load_weights(model, ckpt, strict=False)
    
    # Smart device selection
    target_device, available_vram = select_device_with_vram_budget(
        vram_budget_mb=vram_budget_mb,
        prefer_device=device,
        verbose=True,
    )
    
    # Move to device with OOM protection
    model = safe_to_device(model, target_device, vram_budget_mb)
    
    # Prepare metadata
    metadata = {
        "config": config,
        "device": target_device,
        "available_vram_mb": available_vram,
    }
    
    # Auto-detect I/O specs for UI flexibility
    if auto_detect_io:
        input_spec, output_spec = infer_input_output_spec(model, device=target_device)
        metadata["input_spec"] = input_spec
        metadata["output_spec"] = output_spec
    
    return model, metadata
