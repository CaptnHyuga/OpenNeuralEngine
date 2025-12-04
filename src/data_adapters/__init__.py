"""ONN Data Adapters - Universal data loading interface.

Provides a unified interface for loading any type of data:
- Text (txt, json, jsonl, csv, parquet)
- Images (jpg, png, webp, folder structure)
- Audio (wav, mp3, flac, ogg)
- Video (mp4, avi, mov)
- 3D Meshes (obj, ply, stl)
- Multimodal (paired data auto-detection)
- Streaming (memory-efficient loading for large datasets)

Auto-detects data format and applies appropriate preprocessing.
"""
from .base import DataAdapter, DataAdapterResult
from .registry import (
    register_adapter,
    get_adapter_for_path,
    list_adapters,
    AUTO_DETECT,
    detect_data_type,
    detect_paired_data,
)
from .text import TextAdapter
from .image import ImageAdapter
from .audio import AudioAdapter
from .video import VideoAdapter
from .mesh import MeshAdapter
from .multimodal import MultimodalAdapter, detect_modalities, create_manifest
from .streaming import (
    StreamingDataLoader,
    StreamingConfig,
    StreamingJSONLDataset,
    StreamingParquetDataset,
    MemoryMappedDataset,
    create_streaming_loader,
    estimate_dataset_memory,
)

__all__ = [
    # Base
    "DataAdapter",
    "DataAdapterResult",
    # Registry
    "register_adapter",
    "get_adapter_for_path",
    "list_adapters",
    "AUTO_DETECT",
    "detect_data_type",
    "detect_paired_data",
    # Adapters
    "TextAdapter",
    "ImageAdapter",
    "AudioAdapter",
    "VideoAdapter",
    "MeshAdapter",
    "MultimodalAdapter",
    # Streaming
    "StreamingDataLoader",
    "StreamingConfig",
    "StreamingJSONLDataset",
    "StreamingParquetDataset",
    "MemoryMappedDataset",
    "create_streaming_loader",
    "estimate_dataset_memory",
    # Utilities
    "detect_modalities",
    "create_manifest",
]
