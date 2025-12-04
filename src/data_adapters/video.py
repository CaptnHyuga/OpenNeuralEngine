"""Video Data Adapter - Load video datasets from various formats.

Supports:
- Video files (.mp4, .avi, .mov, .mkv, .webm, .wmv)
- Directory of video files
- Video with caption/description pairs

Features:
- Frame extraction (uniform, random, keyframe)
- Temporal sampling strategies
- Multiple resolution support
- Clip generation for action recognition
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from .base import DataAdapter, DataAdapterResult
from .registry import register_adapter


class VideoDataset(Dataset):
    """Dataset for video data with frame extraction."""
    
    def __init__(
        self,
        video_paths: List[Path],
        labels: Optional[List[int]] = None,
        captions: Optional[List[str]] = None,
        num_frames: int = 16,
        frame_sample_rate: int = 4,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        processor=None,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize video dataset.
        
        Args:
            video_paths: List of paths to video files.
            labels: Optional list of class labels (for classification).
            captions: Optional list of text captions.
            num_frames: Number of frames to extract per video.
            frame_sample_rate: Sample every Nth frame.
            image_size: Target frame size.
            transform: Optional transform to apply to frames.
            processor: HuggingFace video processor (e.g., VideoMAE).
            class_names: List of class names for classification.
        """
        self.video_paths = video_paths
        self.labels = labels
        self.captions = captions
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.image_size = image_size
        self.transform = transform
        self.processor = processor
        self.class_names = class_names
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        item = {"video_path": str(video_path)}
        
        if self.processor:
            # Use HuggingFace processor
            processed = self.processor(
                frames,
                return_tensors="pt",
            )
            item["pixel_values"] = processed.pixel_values.squeeze(0)
        else:
            # Stack frames into tensor: (num_frames, C, H, W)
            if self.transform:
                frames = [self.transform(f) for f in frames]
            
            if isinstance(frames[0], torch.Tensor):
                item["pixel_values"] = torch.stack(frames)
            else:
                item["frames"] = frames
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        if self.captions is not None:
            item["caption"] = self.captions[idx]
        
        return item
    
    def _extract_frames(self, video_path: Path) -> List:
        """Extract frames from video file."""
        try:
            import decord
            from decord import VideoReader, cpu
            decord.bridge.set_bridge("torch")
            
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            
            # Calculate frame indices
            indices = self._get_frame_indices(total_frames)
            
            # Extract frames
            frames = vr.get_batch(indices).numpy()
            
            # Convert to PIL for transforms
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            
            return pil_frames
            
        except ImportError:
            # Fallback to OpenCV
            return self._extract_frames_cv2(video_path)
    
    def _extract_frames_cv2(self, video_path: Path) -> List:
        """Fallback frame extraction using OpenCV."""
        try:
            import cv2
            from PIL import Image
            
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            indices = self._get_frame_indices(total_frames)
            frames = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    frames.append(Image.fromarray(frame))
                else:
                    # Pad with last frame if failed
                    if frames:
                        frames.append(frames[-1])
            
            cap.release()
            
            # Pad to num_frames if needed
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else Image.new("RGB", (self.image_size, self.image_size)))
            
            return frames[:self.num_frames]
            
        except ImportError as e:
            raise ImportError(
                "decord or opencv-python required for video. "
                "Run: pip install decord opencv-python"
            ) from e
    
    def _get_frame_indices(self, total_frames: int) -> List[int]:
        """Calculate which frames to extract.
        
        Uses uniform temporal sampling with the specified sample rate.
        """
        # Calculate total span needed
        span = self.num_frames * self.frame_sample_rate
        
        if span > total_frames:
            # Video is shorter than needed, sample uniformly
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()
        else:
            # Sample from center or random start
            start = (total_frames - span) // 2
            indices = list(range(start, start + span, self.frame_sample_rate))
        
        return indices[:self.num_frames]


class StreamingVideoDataset(IterableDataset):
    """Streaming dataset for large video collections."""
    
    def __init__(
        self,
        video_dir: Path,
        num_frames: int = 16,
        frame_sample_rate: int = 4,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        extensions: Optional[List[str]] = None,
    ):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.image_size = image_size
        self.transform = transform
        self.extensions = extensions or [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Create temporary dataset for each video
        for video_path in self.video_dir.rglob("*"):
            if video_path.suffix.lower() in self.extensions:
                try:
                    ds = VideoDataset(
                        [video_path],
                        num_frames=self.num_frames,
                        frame_sample_rate=self.frame_sample_rate,
                        image_size=self.image_size,
                        transform=self.transform,
                    )
                    yield ds[0]
                except Exception:
                    # Skip problematic videos
                    continue


@register_adapter
class VideoAdapter(DataAdapter):
    """Adapter for loading video datasets."""
    
    name = "video"
    data_type = "video"
    supported_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".m4v", ".flv"]
    
    def can_handle(self, path: Path) -> bool:
        """Check if path is a video file or directory with videos."""
        path = Path(path)
        
        if path.is_file():
            return self._check_extension(path)
        
        if path.is_dir():
            video_files = self._scan_directory(path, recursive=True)
            return len(video_files) > 0
        
        return False
    
    def load(
        self,
        path: Path,
        num_frames: int = 16,
        frame_sample_rate: int = 4,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        processor=None,
        with_captions: bool = True,
        streaming: bool = False,
        sample_limit: Optional[int] = None,
        **kwargs,
    ) -> DataAdapterResult:
        """Load video data from file or directory.
        
        Args:
            path: Path to video file or directory.
            num_frames: Number of frames to extract per video.
            frame_sample_rate: Sample every Nth frame.
            image_size: Target frame resolution.
            transform: Optional transform for frames.
            processor: HuggingFace video processor.
            with_captions: Look for paired caption files.
            streaming: Use streaming dataset for large collections.
            sample_limit: Maximum number of videos to load.
            **kwargs: Additional arguments.
        
        Returns:
            DataAdapterResult with loaded dataset.
        """
        path = Path(path)
        
        if path.is_file():
            # Single video
            dataset = VideoDataset(
                [path],
                num_frames=num_frames,
                frame_sample_rate=frame_sample_rate,
                image_size=image_size,
                transform=transform,
                processor=processor,
            )
            return DataAdapterResult(
                dataset=dataset,
                adapter_name=self.name,
                num_samples=1,
                data_type=self.data_type,
                source_path=str(path),
                format_detected="single_video",
                features={
                    "num_frames": num_frames,
                    "frame_sample_rate": frame_sample_rate,
                    "image_size": image_size,
                },
            )
        
        # Directory
        if streaming:
            dataset = StreamingVideoDataset(
                path,
                num_frames=num_frames,
                frame_sample_rate=frame_sample_rate,
                image_size=image_size,
                transform=transform,
            )
            return DataAdapterResult(
                dataset=dataset,
                adapter_name=self.name,
                num_samples=-1,  # Unknown for streaming
                data_type=self.data_type,
                source_path=str(path),
                format_detected="streaming_video",
            )
        
        # Load all videos
        video_paths, labels, captions, class_names = self._scan_video_directory(
            path, with_captions
        )
        
        if sample_limit and len(video_paths) > sample_limit:
            video_paths = video_paths[:sample_limit]
            if labels:
                labels = labels[:sample_limit]
            if captions:
                captions = captions[:sample_limit]
        
        dataset = VideoDataset(
            video_paths=video_paths,
            labels=labels,
            captions=captions,
            num_frames=num_frames,
            frame_sample_rate=frame_sample_rate,
            image_size=image_size,
            transform=transform,
            processor=processor,
            class_names=class_names,
        )
        
        features = {
            "num_frames": num_frames,
            "frame_sample_rate": frame_sample_rate,
            "image_size": image_size,
            "num_classes": len(class_names) if class_names else 0,
            "has_captions": captions is not None,
        }
        
        return DataAdapterResult(
            dataset=dataset,
            adapter_name=self.name,
            num_samples=len(video_paths),
            data_type=self.data_type,
            features=features,
            source_path=str(path),
            format_detected="video_folder" if labels else "flat_videos",
            preprocessing_applied=["frame_extraction", "resize"],
        )
    
    def _scan_video_directory(
        self,
        directory: Path,
        with_captions: bool,
    ) -> Tuple[List[Path], Optional[List[int]], Optional[List[str]], Optional[List[str]]]:
        """Scan directory for videos and detect structure."""
        
        # Check for class folder structure
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        
        if subdirs:
            class_counts = {}
            for subdir in subdirs:
                videos = self._scan_directory(subdir, recursive=False)
                if videos:
                    class_counts[subdir.name] = videos
            
            if class_counts:
                # Classification structure
                class_names = sorted(class_counts.keys())
                class_to_idx = {name: idx for idx, name in enumerate(class_names)}
                
                video_paths = []
                labels = []
                
                for class_name, videos in sorted(class_counts.items()):
                    for vid_path in videos:
                        video_paths.append(vid_path)
                        labels.append(class_to_idx[class_name])
                
                captions = self._find_captions(video_paths, directory) if with_captions else None
                
                return video_paths, labels, captions, class_names
        
        # Flat structure
        video_paths = self._scan_directory(directory, recursive=True)
        captions = self._find_captions(video_paths, directory) if with_captions else None
        
        return video_paths, None, captions, None
    
    def _find_captions(
        self,
        video_paths: List[Path],
        base_dir: Path,
    ) -> Optional[List[str]]:
        """Find caption files paired with videos."""
        captions = []
        found_any = False
        
        # Check for centralized caption file
        caption_file = base_dir / "captions.json"
        if caption_file.exists():
            with open(caption_file, "r", encoding="utf-8") as f:
                caption_data = json.load(f)
            
            for vid_path in video_paths:
                key = vid_path.stem
                if key in caption_data:
                    captions.append(caption_data[key])
                    found_any = True
                else:
                    captions.append("")
            
            return captions if found_any else None
        
        # Look for individual caption files
        for vid_path in video_paths:
            txt_path = vid_path.with_suffix(".txt")
            if txt_path.exists():
                captions.append(txt_path.read_text(encoding="utf-8").strip())
                found_any = True
            else:
                captions.append("")
        
        return captions if found_any else None
    
    def get_collate_fn(self) -> Optional[Callable]:
        """Get custom collate function for video batches."""
        def video_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            result = {}
            
            # Stack pixel values
            if "pixel_values" in batch[0]:
                result["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
            
            # Stack labels if present
            if "labels" in batch[0]:
                result["labels"] = torch.tensor([b["labels"] for b in batch])
            
            # Collect captions
            if "caption" in batch[0]:
                result["captions"] = [b["caption"] for b in batch]
            
            # Collect paths
            result["video_paths"] = [b["video_path"] for b in batch]
            
            return result
        
        return video_collate
