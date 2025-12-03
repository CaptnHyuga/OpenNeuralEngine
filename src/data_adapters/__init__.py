"""ONN Data Adapters - Universal data loading interface.

Provides a unified interface for loading any type of data:
- Text (txt, json, jsonl, csv, parquet)
- Images (jpg, png, webp, folder structure)
- Audio (wav, mp3, flac, ogg)
- Video (mp4, avi, mov)
- 3D Meshes (obj, ply, stl)
- Multimodal (paired data auto-detection)

Auto-detects data format and applies appropriate preprocessing.
"""
from .base import DataAdapter, DataAdapterResult
from .registry import (
    register_adapter,
    get_adapter_for_path,
    list_adapters,
    AUTO_DETECT,
)
from .text import TextAdapter
from .image import ImageAdapter
from .audio import AudioAdapter

__all__ = [
    "DataAdapter",
    "DataAdapterResult",
    "register_adapter",
    "get_adapter_for_path",
    "list_adapters",
    "AUTO_DETECT",
    "TextAdapter",
    "ImageAdapter",
    "AudioAdapter",
]
