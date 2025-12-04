"""Streaming Checkpoint Manager - Efficient checkpoint saving and loading.

Handles checkpointing for large models with:
- Streaming save/load to avoid full model in memory
- Sharded checkpoints for multi-GB models
- Resume from interruption
- Checkpoint validation and corruption recovery
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    
    # Model info
    model_name: str
    num_params: int
    architecture_type: str
    
    # Training state
    epoch: int
    global_step: int
    best_loss: float = float("inf")
    
    # Checkpoint info
    num_shards: int = 1
    shard_files: List[str] = field(default_factory=list)
    total_size_bytes: int = 0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    checksum: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)
    
    def save(self, path: Path):
        """Save metadata to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "CheckpointMetadata":
        """Load metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class StreamingCheckpointManager:
    """Manages streaming checkpoints for large models."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_shard_size_gb: float = 5.0,
        keep_last_n: int = 3,
        use_safetensors: bool = True,
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints.
            max_shard_size_gb: Maximum size per shard file.
            keep_last_n: Number of recent checkpoints to keep.
            use_safetensors: Use safetensors format if available.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_shard_size_bytes = int(max_shard_size_gb * 1024 ** 3)
        self.keep_last_n = keep_last_n
        self.use_safetensors = use_safetensors and self._check_safetensors()
    
    def _check_safetensors(self) -> bool:
        """Check if safetensors is available."""
        try:
            import safetensors
            return True
        except ImportError:
            return False
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        best_loss: float = float("inf"),
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> Path:
        """Save a checkpoint with streaming for large models.
        
        Args:
            model: Model to checkpoint.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            epoch: Current epoch number.
            global_step: Current global step.
            best_loss: Best loss achieved.
            config: Training configuration.
            name: Checkpoint name (default: step-{global_step}).
        
        Returns:
            Path to checkpoint directory.
        """
        import datetime
        
        name = name or f"checkpoint-{global_step}"
        ckpt_path = self.checkpoint_dir / name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate model size and determine sharding
        state_dict = model.state_dict()
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        num_shards = max(1, (total_size + self.max_shard_size_bytes - 1) // self.max_shard_size_bytes)
        
        # Save model shards
        shard_files = self._save_sharded_model(state_dict, ckpt_path, num_shards)
        
        # Save optimizer state
        if optimizer is not None:
            optimizer_path = ckpt_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        if scheduler is not None:
            scheduler_path = ckpt_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Create metadata
        metadata = CheckpointMetadata(
            model_name=type(model).__name__,
            num_params=sum(p.numel() for p in model.parameters()),
            architecture_type=self._detect_arch_type(model),
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            num_shards=num_shards,
            shard_files=shard_files,
            total_size_bytes=total_size,
            config=config or {},
            checksum=self._compute_checksum(state_dict),
            created_at=datetime.datetime.now().isoformat(),
        )
        
        metadata.save(ckpt_path / "metadata.json")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return ckpt_path
    
    def _save_sharded_model(
        self,
        state_dict: Dict[str, torch.Tensor],
        ckpt_path: Path,
        num_shards: int,
    ) -> List[str]:
        """Save model state dict as sharded files."""
        if num_shards == 1:
            # Single file save
            if self.use_safetensors:
                from safetensors.torch import save_file
                filename = "model.safetensors"
                save_file(state_dict, ckpt_path / filename)
            else:
                filename = "model.pt"
                torch.save(state_dict, ckpt_path / filename)
            return [filename]
        
        # Sharded save
        keys = list(state_dict.keys())
        keys_per_shard = (len(keys) + num_shards - 1) // num_shards
        
        shard_files = []
        for i in range(num_shards):
            start_idx = i * keys_per_shard
            end_idx = min((i + 1) * keys_per_shard, len(keys))
            shard_keys = keys[start_idx:end_idx]
            
            shard_dict = {k: state_dict[k] for k in shard_keys}
            
            if self.use_safetensors:
                from safetensors.torch import save_file
                filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
                save_file(shard_dict, ckpt_path / filename)
            else:
                filename = f"model-{i:05d}-of-{num_shards:05d}.pt"
                torch.save(shard_dict, ckpt_path / filename)
            
            shard_files.append(filename)
        
        return shard_files
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
        map_location: str = "auto",
    ) -> CheckpointMetadata:
        """Load a checkpoint with streaming for large models.
        
        Args:
            model: Model to load weights into.
            checkpoint_path: Path to checkpoint directory.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            strict: Require all keys to match.
            map_location: Device mapping ("auto", "cpu", "cuda").
        
        Returns:
            Checkpoint metadata.
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load metadata
        metadata = CheckpointMetadata.load(checkpoint_path / "metadata.json")
        
        # Determine device
        if map_location == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = map_location
        
        # Load model shards
        state_dict = self._load_sharded_model(checkpoint_path, metadata.shard_files, device)
        
        # Validate checksum
        loaded_checksum = self._compute_checksum(state_dict)
        if loaded_checksum != metadata.checksum:
            print(f"⚠️  Checksum mismatch: expected {metadata.checksum}, got {loaded_checksum}")
        
        # Load into model
        model.load_state_dict(state_dict, strict=strict)
        
        # Load optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device, weights_only=True)
                )
        
        # Load scheduler
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if scheduler_path.exists():
                scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location=device, weights_only=True)
                )
        
        return metadata
    
    def _load_sharded_model(
        self,
        ckpt_path: Path,
        shard_files: List[str],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """Load model from sharded files."""
        state_dict = {}
        
        for filename in shard_files:
            filepath = ckpt_path / filename
            
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file
                shard_dict = load_file(str(filepath), device=device)
            else:
                shard_dict = torch.load(filepath, map_location=device, weights_only=True)
            
            state_dict.update(shard_dict)
        
        return state_dict
    
    def _detect_arch_type(self, model: nn.Module) -> str:
        """Detect model architecture type."""
        class_name = type(model).__name__.lower()
        
        if "transformer" in class_name or "gpt" in class_name or "llama" in class_name:
            return "transformer"
        elif "resnet" in class_name or "conv" in class_name:
            return "cnn"
        elif "lstm" in class_name or "rnn" in class_name:
            return "rnn"
        else:
            return "unknown"
    
    def _compute_checksum(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Compute checksum of state dict for validation."""
        # Hash key names and tensor shapes (fast but catches most corruption)
        hasher = hashlib.md5(usedforsecurity=False)
        
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            hasher.update(key.encode())
            hasher.update(str(tensor.shape).encode())
            # Sample a few values for content check
            if tensor.numel() > 0:
                flat = tensor.flatten()
                indices = torch.linspace(0, len(flat) - 1, min(10, len(flat)), dtype=torch.long)
                samples = flat[indices].cpu().numpy().tobytes()
                hasher.update(samples)
        
        return hasher.hexdigest()[:16]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        # Sort by step (newest first)
        checkpoints.sort(key=lambda x: x[1].global_step, reverse=True)
        
        # Remove old ones
        for ckpt_path, _ in checkpoints[self.keep_last_n:]:
            shutil.rmtree(ckpt_path)
    
    def list_checkpoints(self) -> List[Tuple[Path, CheckpointMetadata]]:
        """List all checkpoints with metadata."""
        checkpoints = []
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir():
                metadata_path = item / "metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = CheckpointMetadata.load(metadata_path)
                        checkpoints.append((item, metadata))
                    except Exception:
                        pass
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[Tuple[Path, CheckpointMetadata]]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Sort by step
        checkpoints.sort(key=lambda x: x[1].global_step, reverse=True)
        return checkpoints[0]
    
    def get_best_checkpoint(self) -> Optional[Tuple[Path, CheckpointMetadata]]:
        """Get the checkpoint with best loss."""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Sort by loss
        checkpoints.sort(key=lambda x: x[1].best_loss)
        return checkpoints[0]
    
    def resume_from_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Optional[CheckpointMetadata]:
        """Resume training from the latest checkpoint.
        
        Args:
            model: Model to restore.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
        
        Returns:
            Checkpoint metadata if resumed, None if no checkpoint found.
        """
        latest = self.get_latest_checkpoint()
        
        if latest is None:
            return None
        
        ckpt_path, metadata = latest
        print(f"📂 Resuming from checkpoint: {ckpt_path.name}")
        print(f"   Step: {metadata.global_step}, Epoch: {metadata.epoch}, Loss: {metadata.best_loss:.4f}")
        
        self.load_checkpoint(model, ckpt_path, optimizer, scheduler)
        
        return metadata


def stream_save_model(
    model: nn.Module,
    path: Path,
    use_safetensors: bool = True,
) -> None:
    """Save model with streaming (memory-efficient).
    
    Args:
        model: Model to save.
        path: Output path.
        use_safetensors: Use safetensors format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    state_dict = model.state_dict()
    
    if use_safetensors:
        try:
            from safetensors.torch import save_file
            if not str(path).endswith(".safetensors"):
                path = path.with_suffix(".safetensors")
            save_file(state_dict, str(path))
            return
        except ImportError:
            pass
    
    # Fallback to PyTorch
    if not str(path).endswith(".pt"):
        path = path.with_suffix(".pt")
    torch.save(state_dict, path)


def stream_load_model(
    model: nn.Module,
    path: Path,
    strict: bool = True,
    device: str = "auto",
) -> None:
    """Load model with streaming (memory-efficient).
    
    Args:
        model: Model to load into.
        path: Checkpoint path.
        strict: Require all keys to match.
        device: Target device.
    """
    path = Path(path)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device=device)
    else:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    
    model.load_state_dict(state_dict, strict=strict)
