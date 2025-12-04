"""Multimodal Data Adapter - Load paired multimodal datasets.

Supports automatic detection and pairing of:
- Image + Text (CLIP-style)
- Audio + Text (ASR/TTS)
- Video + Text (captioning)
- Audio + Mesh (audio-to-3D)
- Image + Mesh (image-to-3D)
- Any combination of supported modalities

Features:
- Automatic pair detection via filename matching
- Support for manifest files (JSON/CSV)
- Cross-modal data alignment
- Flexible pairing strategies
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .base import DataAdapter, DataAdapterResult
from .registry import register_adapter, get_adapter_for_path


class PairedDataset(Dataset):
    """Dataset for paired multimodal data."""
    
    def __init__(
        self,
        pairs: List[Dict[str, Path]],
        modality_loaders: Dict[str, Callable],
        transforms: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize paired dataset.
        
        Args:
            pairs: List of dicts mapping modality name to file path.
                   e.g., [{"image": Path("img1.jpg"), "text": Path("img1.txt")}, ...]
            modality_loaders: Dict mapping modality to loading function.
            transforms: Optional transforms per modality.
        """
        self.pairs = pairs
        self.modality_loaders = modality_loaders
        self.transforms = transforms or {}
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        item = {}
        
        for modality, path in pair.items():
            if modality in self.modality_loaders:
                data = self.modality_loaders[modality](path)
                
                if modality in self.transforms:
                    data = self.transforms[modality](data)
                
                item[modality] = data
            else:
                # Fallback: store path
                item[f"{modality}_path"] = str(path)
        
        return item


# Default loaders for common modalities
def load_text(path: Path) -> str:
    """Load text file."""
    return path.read_text(encoding="utf-8").strip()


def load_image(path: Path):
    """Load image file."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def load_audio(path: Path) -> torch.Tensor:
    """Load audio file."""
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(path))
        return waveform
    except ImportError:
        # Return path for lazy loading
        return str(path)


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


DEFAULT_LOADERS = {
    "text": load_text,
    "caption": load_text,
    "transcription": load_text,
    "image": load_image,
    "audio": load_audio,
    "metadata": load_json,
}


@register_adapter
class MultimodalAdapter(DataAdapter):
    """Adapter for loading paired multimodal datasets."""
    
    name = "multimodal"
    data_type = "multimodal"
    
    # Extension to modality mapping
    MODALITY_EXTENSIONS = {
        # Text
        ".txt": "text",
        ".json": "metadata",
        ".caption": "caption",
        # Image
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".webp": "image",
        # Audio
        ".wav": "audio",
        ".mp3": "audio",
        ".flac": "audio",
        # Video
        ".mp4": "video",
        ".avi": "video",
        ".mov": "video",
        # 3D
        ".obj": "mesh",
        ".ply": "mesh",
        ".stl": "mesh",
        ".glb": "mesh",
    }
    
    def can_handle(self, path: Path) -> bool:
        """Check if path contains paired multimodal data."""
        path = Path(path)
        
        if path.is_file():
            # Check if it's a manifest file
            if path.suffix.lower() in (".json", ".csv", ".jsonl"):
                return self._is_manifest_file(path)
            return False
        
        if path.is_dir():
            return self._detect_paired_data(path)
        
        return False
    
    def _is_manifest_file(self, path: Path) -> bool:
        """Check if file is a multimodal manifest."""
        try:
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Check for paired data structure
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict):
                        keys = set(first.keys())
                        return len(keys) >= 2
            
            elif path.suffix.lower() == ".csv":
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    return len(fieldnames) >= 2
        except Exception:
            pass
        return False
    
    def _detect_paired_data(self, directory: Path) -> bool:
        """Detect if directory contains paired data."""
        files = [f for f in directory.iterdir() if f.is_file()]
        
        # Group files by stem
        stems = {}
        for f in files:
            stem = f.stem
            if stem not in stems:
                stems[stem] = set()
            
            modality = self.MODALITY_EXTENSIONS.get(f.suffix.lower())
            if modality:
                stems[stem].add(modality)
        
        # Check how many stems have multiple modalities
        paired_count = sum(1 for modalities in stems.values() if len(modalities) > 1)
        
        return paired_count >= len(stems) * 0.3  # At least 30% paired
    
    def load(
        self,
        path: Path,
        modality_loaders: Optional[Dict[str, Callable]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        primary_modality: Optional[str] = None,
        sample_limit: Optional[int] = None,
        **kwargs,
    ) -> DataAdapterResult:
        """Load paired multimodal data.
        
        Args:
            path: Path to directory or manifest file.
            modality_loaders: Custom loaders per modality.
            transforms: Transforms per modality.
            primary_modality: Primary modality for ordering.
            sample_limit: Maximum number of pairs to load.
            **kwargs: Additional arguments.
        
        Returns:
            DataAdapterResult with paired dataset.
        """
        path = Path(path)
        loaders = {**DEFAULT_LOADERS, **(modality_loaders or {})}
        
        if path.is_file():
            pairs, modalities = self._load_from_manifest(path)
        else:
            pairs, modalities = self._scan_for_pairs(path, primary_modality)
        
        if sample_limit and len(pairs) > sample_limit:
            pairs = pairs[:sample_limit]
        
        dataset = PairedDataset(
            pairs=pairs,
            modality_loaders=loaders,
            transforms=transforms,
        )
        
        return DataAdapterResult(
            dataset=dataset,
            adapter_name=self.name,
            num_samples=len(pairs),
            data_type=self.data_type,
            features={
                "modalities": list(modalities),
                "num_modalities": len(modalities),
            },
            source_path=str(path),
            format_detected="manifest" if path.is_file() else "paired_directory",
            preprocessing_applied=[],
            collate_fn=self.get_collate_fn(modalities),
        )
    
    def _load_from_manifest(
        self,
        manifest_path: Path,
    ) -> Tuple[List[Dict[str, Path]], set]:
        """Load pairs from manifest file."""
        pairs = []
        modalities = set()
        base_dir = manifest_path.parent
        
        if manifest_path.suffix.lower() == ".json":
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data:
                if isinstance(item, dict):
                    pair = {}
                    for key, value in item.items():
                        if isinstance(value, str):
                            # Check if it's a file path
                            potential_path = base_dir / value
                            if potential_path.exists():
                                pair[key] = potential_path
                                modalities.add(key)
                            else:
                                # Store as text directly
                                pair[f"{key}_text"] = value
                                modalities.add(f"{key}_text")
                    
                    if pair:
                        pairs.append(pair)
        
        elif manifest_path.suffix.lower() == ".csv":
            with open(manifest_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    pair = {}
                    for key, value in row.items():
                        if value:
                            potential_path = base_dir / value
                            if potential_path.exists():
                                pair[key] = potential_path
                                modalities.add(key)
                            else:
                                pair[f"{key}_text"] = value
                                modalities.add(f"{key}_text")
                    
                    if pair:
                        pairs.append(pair)
        
        elif manifest_path.suffix.lower() == ".jsonl":
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        pair = {}
                        for key, value in item.items():
                            if isinstance(value, str):
                                potential_path = base_dir / value
                                if potential_path.exists():
                                    pair[key] = potential_path
                                    modalities.add(key)
                                else:
                                    pair[f"{key}_text"] = value
                                    modalities.add(f"{key}_text")
                        
                        if pair:
                            pairs.append(pair)
        
        return pairs, modalities
    
    def _scan_for_pairs(
        self,
        directory: Path,
        primary_modality: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Path]], set]:
        """Scan directory for paired files by filename matching."""
        files = list(directory.iterdir())
        
        # Group files by stem
        stem_to_files: Dict[str, Dict[str, Path]] = {}
        
        for f in files:
            if not f.is_file():
                continue
            
            modality = self.MODALITY_EXTENSIONS.get(f.suffix.lower())
            if modality:
                stem = f.stem
                if stem not in stem_to_files:
                    stem_to_files[stem] = {}
                stem_to_files[stem][modality] = f
        
        # Filter to only stems with multiple modalities
        pairs = []
        all_modalities = set()
        
        for stem, modality_files in sorted(stem_to_files.items()):
            if len(modality_files) > 1:
                pairs.append(modality_files)
                all_modalities.update(modality_files.keys())
            elif primary_modality and primary_modality in modality_files:
                # Include even single-modality if it matches primary
                pairs.append(modality_files)
                all_modalities.update(modality_files.keys())
        
        return pairs, all_modalities
    
    def get_collate_fn(self, modalities: set) -> Optional[Callable]:
        """Get custom collate function for multimodal batches."""
        def multimodal_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            result = {}
            
            # Collect all keys from batch
            all_keys = set()
            for item in batch:
                all_keys.update(item.keys())
            
            for key in all_keys:
                values = [item.get(key) for item in batch]
                
                # Filter None values
                valid_values = [v for v in values if v is not None]
                
                if not valid_values:
                    continue
                
                # Stack tensors
                if isinstance(valid_values[0], torch.Tensor):
                    try:
                        result[key] = torch.stack(valid_values)
                    except Exception:
                        result[key] = valid_values
                
                # Collect strings
                elif isinstance(valid_values[0], str):
                    result[key] = valid_values
                
                # Collect PIL Images
                elif hasattr(valid_values[0], "mode"):  # PIL Image
                    result[key] = valid_values
                
                else:
                    result[key] = valid_values
            
            return result
        
        return multimodal_collate


# Convenience functions
def detect_modalities(path: Path) -> List[str]:
    """Detect what modalities are present in a path."""
    path = Path(path)
    modalities = set()
    
    extensions_map = MultimodalAdapter.MODALITY_EXTENSIONS
    
    if path.is_file():
        modality = extensions_map.get(path.suffix.lower())
        if modality:
            modalities.add(modality)
    
    elif path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                modality = extensions_map.get(f.suffix.lower())
                if modality:
                    modalities.add(modality)
    
    return sorted(modalities)


def create_manifest(
    directory: Path,
    output_path: Path,
    format: str = "json",
) -> None:
    """Create a manifest file from paired data directory.
    
    Args:
        directory: Directory with paired files.
        output_path: Path for output manifest.
        format: Output format ("json", "csv", "jsonl").
    """
    adapter = MultimodalAdapter()
    pairs, modalities = adapter._scan_for_pairs(directory, None)
    
    # Convert paths to relative strings
    manifest_data = []
    for pair in pairs:
        entry = {}
        for modality, path in pair.items():
            entry[modality] = str(path.relative_to(directory))
        manifest_data.append(entry)
    
    output_path = Path(output_path)
    
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
    
    elif format == "csv":
        if manifest_data:
            fieldnames = sorted(modalities)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in manifest_data:
                    writer.writerow(entry)
    
    elif format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in manifest_data:
                f.write(json.dumps(entry) + "\n")
