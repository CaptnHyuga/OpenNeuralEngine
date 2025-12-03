"""Data Adapter Registry - Auto-detection and registration system.

Manages adapter registration and automatic format detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import DataAdapter, DataAdapterResult


# Global adapter registry
_ADAPTERS: Dict[str, Type[DataAdapter]] = {}


def register_adapter(adapter_class: Type[DataAdapter] = None, *, name: str = None):
    """Register a data adapter class.
    
    Can be used as a decorator or called directly:
    
        @register_adapter
        class MyAdapter(DataAdapter):
            ...
        
        # Or with custom name:
        @register_adapter(name="my-adapter")
        class MyAdapter(DataAdapter):
            ...
    """
    def _register(cls: Type[DataAdapter]) -> Type[DataAdapter]:
        adapter_name = name or getattr(cls, "name", cls.__name__)
        _ADAPTERS[adapter_name] = cls
        return cls
    
    if adapter_class is not None:
        return _register(adapter_class)
    return _register


def get_adapter(name: str) -> Type[DataAdapter]:
    """Get adapter class by name."""
    if name not in _ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(_ADAPTERS.keys())}")
    return _ADAPTERS[name]


def list_adapters() -> List[str]:
    """List all registered adapter names."""
    return list(_ADAPTERS.keys())


def get_adapter_for_path(
    path: Path,
    hint: Optional[str] = None,
) -> Optional[DataAdapter]:
    """Find an adapter that can handle the given path.
    
    Args:
        path: Path to file or directory.
        hint: Optional adapter name hint.
    
    Returns:
        Instantiated adapter or None if no adapter can handle the path.
    """
    path = Path(path)
    
    # If hint provided, try that adapter first
    if hint and hint in _ADAPTERS:
        adapter = _ADAPTERS[hint]()
        if adapter.can_handle(path):
            return adapter
    
    # Try all adapters
    for _name, adapter_class in _ADAPTERS.items():
        try:
            adapter = adapter_class()
            if adapter.can_handle(path):
                return adapter
        except Exception:
            continue
    
    return None


def AUTO_DETECT(
    path: str,
    **kwargs,
) -> DataAdapterResult:
    """Auto-detect format and load data.
    
    Convenience function that finds the right adapter and loads data.
    
    Args:
        path: Path to file or directory.
        **kwargs: Arguments to pass to adapter.load().
    
    Returns:
        DataAdapterResult from the detected adapter.
    
    Raises:
        ValueError: If no adapter can handle the path.
    """
    path = Path(path)
    
    adapter = get_adapter_for_path(path)
    if adapter is None:
        raise ValueError(
            f"No adapter found for: {path}\n"
            f"Available adapters: {list_adapters()}"
        )
    
    return adapter.load(path, **kwargs)


def detect_data_type(path: Path) -> str:
    """Detect the type of data at the given path.
    
    Args:
        path: Path to check.
    
    Returns:
        Data type string ("text", "image", "audio", "video", "mesh", "unknown").
    """
    path = Path(path)
    
    # Check file extension
    ext = path.suffix.lower()
    
    text_exts = {".txt", ".json", ".jsonl", ".csv", ".parquet", ".tsv", ".md"}
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    mesh_exts = {".obj", ".ply", ".stl", ".glb", ".gltf"}
    
    if ext in text_exts:
        return "text"
    if ext in image_exts:
        return "image"
    if ext in audio_exts:
        return "audio"
    if ext in video_exts:
        return "video"
    if ext in mesh_exts:
        return "mesh"
    
    # If directory, check contents
    if path.is_dir():
        files = list(path.iterdir())[:100]  # Sample first 100
        extensions = [f.suffix.lower() for f in files if f.is_file()]
        
        if extensions:
            # Find most common category
            counts = {
                "text": sum(1 for e in extensions if e in text_exts),
                "image": sum(1 for e in extensions if e in image_exts),
                "audio": sum(1 for e in extensions if e in audio_exts),
                "video": sum(1 for e in extensions if e in video_exts),
                "mesh": sum(1 for e in extensions if e in mesh_exts),
            }
            
            if max(counts.values()) > 0:
                return max(counts, key=counts.get)
    
    return "unknown"


def detect_paired_data(path: Path) -> bool:
    """Check if path contains paired multimodal data.
    
    Looks for files with matching names but different extensions
    (e.g., audio1.wav + audio1.txt = paired audio-text data).
    
    Args:
        path: Directory path to check.
    
    Returns:
        True if paired data detected.
    """
    path = Path(path)
    if not path.is_dir():
        return False
    
    # Get all files
    files = [f for f in path.iterdir() if f.is_file()]
    
    # Group by stem (filename without extension)
    stems = {}
    for f in files:
        stem = f.stem
        if stem not in stems:
            stems[stem] = []
        stems[stem].append(f.suffix.lower())
    
    # Check for stems with multiple extensions
    paired_count = sum(1 for exts in stems.values() if len(set(exts)) > 1)
    
    # Consider it paired if >50% of stems have multiple extensions
    return paired_count > len(stems) * 0.5
