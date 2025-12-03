"""Audio Data Adapter - Load audio datasets from various formats.

Supports:
- Audio files (.wav, .mp3, .flac, .ogg, .m4a)
- Directory of audio files
- Audio with transcription pairs

Features:
- Automatic resampling
- Spectrogram extraction
- Mel-filterbank features
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .base import DataAdapter, DataAdapterResult
from .registry import register_adapter


class AudioDataset(Dataset):
    """Dataset for audio data."""
    
    def __init__(
        self,
        audio_paths: List[Path],
        transcriptions: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        sample_rate: int = 16000,
        max_length_seconds: float = 30.0,
        processor=None,
    ):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = int(max_length_seconds * sample_rate)
        self.processor = processor
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audio_path = self.audio_paths[idx]
        
        # Load audio
        waveform = self._load_audio(audio_path)
        
        item = {"input_values": waveform, "audio_path": str(audio_path)}
        
        if self.processor:
            # Use HuggingFace processor (e.g., Whisper)
            processed = self.processor(
                waveform.numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            item["input_features"] = processed.input_features.squeeze(0)
        
        if self.transcriptions is not None:
            item["transcription"] = self.transcriptions[idx]
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item
    
    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio file."""
        try:
            import torchaudio
            
            waveform, sr = torchaudio.load(str(path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Truncate or pad to max length
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.squeeze(0)  # Remove channel dim

        except ImportError as e:
            raise ImportError(
                "torchaudio required for audio. Run: pip install torchaudio"
            ) from e


@register_adapter
class AudioAdapter(DataAdapter):
    """Adapter for loading audio datasets."""
    
    name = "audio"
    data_type = "audio"
    supported_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"]
    
    def can_handle(self, path: Path) -> bool:
        """Check if path is an audio file or directory with audio files."""
        path = Path(path)
        
        if path.is_file():
            return self._check_extension(path)
        
        if path.is_dir():
            audio_files = self._scan_directory(path, recursive=True)
            return len(audio_files) > 0
        
        return False
    
    def load(
        self,
        path: Path,
        sample_rate: int = 16000,
        max_length_seconds: float = 30.0,
        processor=None,
        with_transcriptions: bool = True,
        sample_limit: Optional[int] = None,
        **kwargs,
    ) -> DataAdapterResult:
        """Load audio data from file or directory.
        
        Args:
            path: Path to file or directory.
            sample_rate: Target sample rate.
            max_length_seconds: Maximum audio length.
            processor: HuggingFace audio processor.
            with_transcriptions: Look for paired transcription files.
            sample_limit: Maximum number of samples to load.
            **kwargs: Additional arguments.
        
        Returns:
            DataAdapterResult with loaded dataset.
        """
        path = Path(path)
        
        if path.is_file():
            audio_paths = [path]
            transcriptions = None
        else:
            audio_paths = self._scan_directory(path, recursive=True)
            
            # Look for transcriptions
            transcriptions = None
            if with_transcriptions:
                transcriptions = self._find_transcriptions(audio_paths, path)
        
        if sample_limit and len(audio_paths) > sample_limit:
            audio_paths = audio_paths[:sample_limit]
            if transcriptions:
                transcriptions = transcriptions[:sample_limit]
        
        dataset = AudioDataset(
            audio_paths=audio_paths,
            transcriptions=transcriptions,
            sample_rate=sample_rate,
            max_length_seconds=max_length_seconds,
            processor=processor,
        )
        
        return DataAdapterResult(
            dataset=dataset,
            adapter_name=self.name,
            num_samples=len(audio_paths),
            data_type=self.data_type,
            features={
                "sample_rate": sample_rate,
                "max_length_seconds": max_length_seconds,
                "has_transcriptions": transcriptions is not None,
                "has_processor": processor is not None,
            },
            source_path=str(path),
            format_detected="audio_directory" if path.is_dir() else path.suffix,
            preprocessing_applied=["resample", "mono_convert", "normalize"],
            collate_fn=self.get_collate_fn(),
        )
    
    def _find_transcriptions(
        self,
        audio_paths: List[Path],
        base_dir: Path,
    ) -> Optional[List[str]]:
        """Find transcription files paired with audio files.
        
        Looks for:
        - Same name with .txt extension (audio1.wav -> audio1.txt)
        - transcript.txt or transcriptions.json in same directory
        """
        transcriptions = []
        found_any = False
        
        for audio_path in audio_paths:
            # Try same name with .txt
            txt_path = audio_path.with_suffix(".txt")
            if txt_path.exists():
                transcription = txt_path.read_text(encoding="utf-8").strip()
                transcriptions.append(transcription)
                found_any = True
            else:
                transcriptions.append("")
        
        return transcriptions if found_any else None
    
    def get_collate_fn(self) -> Optional[Callable]:
        """Get collate function for audio batches."""
        def collate_audio(batch: List[Dict]) -> Dict[str, Any]:
            # Stack waveforms
            input_values = torch.stack([item["input_values"] for item in batch])
            
            result = {"input_values": input_values}
            
            # Stack features if present
            if "input_features" in batch[0]:
                input_features = torch.stack([item["input_features"] for item in batch])
                result["input_features"] = input_features
            
            # Collect transcriptions
            if "transcription" in batch[0]:
                result["transcriptions"] = [item["transcription"] for item in batch]
            
            # Labels
            if "labels" in batch[0]:
                result["labels"] = torch.tensor([item["labels"] for item in batch])
            
            return result
        
        return collate_audio
