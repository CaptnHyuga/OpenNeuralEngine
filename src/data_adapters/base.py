"""Base Data Adapter - Interface for all data adapters.

All data adapters inherit from this base class and implement:
- can_handle(): Check if this adapter can handle a given path
- load(): Load data and return a PyTorch Dataset
- get_collate_fn(): Return custom collate function if needed
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import Dataset


@dataclass
class DataAdapterResult:
    """Result of data adapter loading."""
    
    dataset: Dataset
    adapter_name: str
    num_samples: int
    data_type: str  # "text", "image", "audio", "video", "mesh", "multimodal"
    features: Dict[str, Any] = field(default_factory=dict)
    collate_fn: Optional[Callable] = None
    
    # Preprocessing info
    preprocessing_applied: List[str] = field(default_factory=list)
    
    # Metadata
    source_path: Optional[str] = None
    format_detected: Optional[str] = None


class DataAdapter(ABC):
    """Base class for all data adapters.
    
    Subclasses must implement:
    - can_handle(): Return True if this adapter can load the given path
    - load(): Load data and return a DataAdapterResult
    
    Optional overrides:
    - get_collate_fn(): Return custom collate function
    - get_supported_extensions(): List of file extensions
    """
    
    name: str = "base"
    data_type: str = "unknown"
    supported_extensions: List[str] = []
    
    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Check if this adapter can handle the given path.
        
        Args:
            path: Path to file or directory.
        
        Returns:
            True if this adapter can load the data.
        """
        pass
    
    @abstractmethod
    def load(
        self,
        path: Path,
        **kwargs,
    ) -> DataAdapterResult:
        """Load data from path.
        
        Args:
            path: Path to file or directory.
            **kwargs: Adapter-specific options.
        
        Returns:
            DataAdapterResult containing dataset and metadata.
        """
        pass
    
    def get_collate_fn(self) -> Optional[Callable]:
        """Get custom collate function for DataLoader.
        
        Returns:
            Collate function or None for default.
        """
        return None
    
    def _check_extension(self, path: Path) -> bool:
        """Check if file has a supported extension."""
        if path.is_dir():
            return False
        return path.suffix.lower() in self.supported_extensions
    
    def _scan_directory(
        self,
        directory: Path,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Scan directory for files with given extensions.
        
        Args:
            directory: Directory to scan.
            extensions: List of extensions (including dot). None = use supported_extensions.
            recursive: Whether to scan subdirectories.
        
        Returns:
            List of matching file paths.
        """
        extensions = extensions or self.supported_extensions
        extensions_set = set(ext.lower() for ext in extensions)
        
        files = []
        if recursive:
            for f in directory.rglob("*"):
                if f.is_file() and f.suffix.lower() in extensions_set:
                    files.append(f)
        else:
            for f in directory.iterdir():
                if f.is_file() and f.suffix.lower() in extensions_set:
                    files.append(f)
        
        return sorted(files)


class SimpleDataset(Dataset):
    """Simple dataset wrapper for list of items."""
    
    def __init__(
        self,
        items: List[Any],
        transform: Optional[Callable] = None,
    ):
        self.items = items
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Any:
        item = self.items[idx]
        if self.transform:
            item = self.transform(item)
        return item


class MapDataset(Dataset):
    """Dataset that applies a mapping function to items."""
    
    def __init__(
        self,
        items: List[Any],
        map_fn: Callable[[Any], Any],
    ):
        self.items = items
        self.map_fn = map_fn
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Any:
        return self.map_fn(self.items[idx])


class StreamingDataset(Dataset):
    """Dataset that streams from a generator/iterator."""
    
    def __init__(
        self,
        generator_fn: Callable[[], Any],
        length: Optional[int] = None,
    ):
        self.generator_fn = generator_fn
        self._length = length
        self._cache = {}
    
    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("Streaming dataset has unknown length")
        return self._length
    
    def __getitem__(self, idx: int) -> Any:
        if idx not in self._cache:
            # For streaming, we need to materialize on demand
            # This is a simplified version - real implementation would be more sophisticated
            gen = self.generator_fn()
            for i, item in enumerate(gen):
                self._cache[i] = item
                if i == idx:
                    break
        return self._cache.get(idx)
