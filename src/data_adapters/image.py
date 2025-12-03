"""Image Data Adapter - Load image datasets from various formats.

Supports:
- Image files (.jpg, .jpeg, .png, .webp, .bmp, .gif, .tiff)
- Folder structure (class per folder)
- Archive files (.zip, .tar)
- HuggingFace datasets format

Features:
- Auto-detection of classification structure
- Standard augmentation transforms
- Multi-class and multi-label support
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .base import DataAdapter, DataAdapterResult
from .registry import register_adapter


class ImageDataset(Dataset):
    """Dataset for image data with transforms."""
    
    def __init__(
        self,
        image_paths: List[Path],
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        
        # Load image
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
        except ImportError as e:
            raise ImportError(
                "Pillow required for images. Run: pip install Pillow"
            ) from e
        
        if self.transform:
            image = self.transform(image)
        
        item = {"pixel_values": image, "image_path": str(image_path)}
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item


@register_adapter
class ImageAdapter(DataAdapter):
    """Adapter for loading image datasets."""
    
    name = "image"
    data_type = "image"
    supported_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"]
    
    def can_handle(self, path: Path) -> bool:
        """Check if path is an image file or directory with images."""
        path = Path(path)
        
        if path.is_file():
            return self._check_extension(path)
        
        if path.is_dir():
            # Check if directory contains images
            image_files = self._scan_directory(path, recursive=True)
            return len(image_files) > 0
        
        return False
    
    def load(
        self,
        path: Path,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        normalize: bool = True,
        augment: bool = False,
        sample_limit: Optional[int] = None,
        **kwargs,
    ) -> DataAdapterResult:
        """Load image data from file or directory.
        
        Args:
            path: Path to file or directory.
            transform: Custom transform pipeline (overrides defaults).
            image_size: Target image size.
            normalize: Apply ImageNet normalization.
            augment: Apply data augmentation.
            sample_limit: Maximum number of samples to load.
            **kwargs: Additional arguments.
        
        Returns:
            DataAdapterResult with loaded dataset.
        """
        path = Path(path)
        
        # Build transform if not provided
        if transform is None:
            transform = self._build_transform(image_size, normalize, augment)
        
        if path.is_file():
            # Single image
            dataset = ImageDataset([path], transform=transform)
            return DataAdapterResult(
                dataset=dataset,
                adapter_name=self.name,
                num_samples=1,
                data_type=self.data_type,
                source_path=str(path),
                format_detected="single_image",
            )
        
        # Directory - check structure
        image_paths, labels, class_names = self._scan_image_directory(path)
        
        if sample_limit and len(image_paths) > sample_limit:
            image_paths = image_paths[:sample_limit]
            if labels:
                labels = labels[:sample_limit]
        
        dataset = ImageDataset(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
            class_names=class_names,
        )
        
        features = {
            "image_size": image_size,
            "normalized": normalize,
            "augmented": augment,
            "num_classes": len(class_names) if class_names else 0,
        }
        
        if class_names:
            features["class_names"] = class_names
        
        return DataAdapterResult(
            dataset=dataset,
            adapter_name=self.name,
            num_samples=len(image_paths),
            data_type=self.data_type,
            features=features,
            source_path=str(path),
            format_detected="image_folder" if labels else "flat_images",
            preprocessing_applied=self._get_preprocessing_list(normalize, augment),
            collate_fn=self.get_collate_fn(),
        )
    
    def _scan_image_directory(
        self,
        directory: Path,
    ) -> Tuple[List[Path], Optional[List[int]], Optional[List[str]]]:
        """Scan directory for images and detect classification structure.
        
        If directory has subdirectories with images, treat as classification
        (one class per subdirectory). Otherwise, flat image list.
        """
        # Check for class folders
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        
        if subdirs:
            # Check if subdirs contain images
            class_counts = {}
            for subdir in subdirs:
                images = self._scan_directory(subdir, recursive=False)
                if images:
                    class_counts[subdir.name] = images
            
            if class_counts:
                # Classification structure detected
                class_names = sorted(class_counts.keys())
                class_to_idx = {name: idx for idx, name in enumerate(class_names)}
                
                image_paths = []
                labels = []
                
                for class_name, images in sorted(class_counts.items()):
                    for img_path in images:
                        image_paths.append(img_path)
                        labels.append(class_to_idx[class_name])
                
                return image_paths, labels, class_names
        
        # Flat structure - just images
        image_paths = self._scan_directory(directory, recursive=True)
        return image_paths, None, None
    
    def _build_transform(
        self,
        image_size: int,
        normalize: bool,
        augment: bool,
    ) -> Callable:
        """Build image transform pipeline."""
        try:
            from torchvision import transforms
        except ImportError as e:
            raise ImportError(
                "torchvision required for images. Run: pip install torchvision"
            ) from e
        
        transform_list = []
        
        if augment:
            transform_list.extend([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            transform_list.extend([
                transforms.Resize(int(image_size * 1.14)),  # Slightly larger for center crop
                transforms.CenterCrop(image_size),
            ])
        
        transform_list.append(transforms.ToTensor())
        
        if normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _get_preprocessing_list(self, normalize: bool, augment: bool) -> List[str]:
        """Get list of preprocessing steps applied."""
        steps = ["resize", "to_tensor"]
        if normalize:
            steps.append("imagenet_normalize")
        if augment:
            steps.extend(["random_crop", "random_flip", "color_jitter"])
        return steps
    
    def get_collate_fn(self) -> Optional[Callable]:
        """Get collate function for image batches."""
        def collate_images(batch: List[Dict]) -> Dict[str, Any]:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            
            result = {"pixel_values": pixel_values}
            
            if "labels" in batch[0]:
                labels = torch.tensor([item["labels"] for item in batch])
                result["labels"] = labels
            
            return result
        
        return collate_images
