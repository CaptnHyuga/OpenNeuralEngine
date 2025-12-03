"""Text Data Adapter - Load text datasets from various formats.

Supports:
- Plain text files (.txt)
- JSON files (.json)
- JSON Lines (.jsonl)
- CSV files (.csv)
- Parquet files (.parquet)
- TSV files (.tsv)

Features:
- Auto-detection of text column in structured data
- Streaming for large files
- Tokenization integration
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import Dataset, IterableDataset

from .base import DataAdapter, DataAdapterResult
from .registry import register_adapter


class TextDataset(Dataset):
    """Dataset for text data with optional tokenization."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer=None,
        max_length: int = 512,
        labels: Optional[List[Any]] = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            item = {"text": text}
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large text files."""
    
    def __init__(
        self,
        file_path: Path,
        tokenizer=None,
        max_length: int = 512,
        text_column: str = "text",
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        suffix = self.file_path.suffix.lower()
        
        if suffix == ".jsonl":
            yield from self._iter_jsonl()
        elif suffix == ".json":
            yield from self._iter_json()
        elif suffix == ".csv":
            yield from self._iter_csv()
        else:
            yield from self._iter_text()
    
    def _iter_jsonl(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data.get(self.text_column, data.get("content", str(data)))
                    yield self._process_text(text, data)
    
    def _iter_json(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text = item.get(self.text_column, item.get("content", str(item)))
                    yield self._process_text(text, item)
    
    def _iter_csv(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(self.text_column, list(row.values())[0])
                yield self._process_text(text, row)
    
    def _iter_text(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield self._process_text(line.strip())
    
    def _process_text(
        self,
        text: str,
        original: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            item = {"text": text}
        
        if original:
            # Include label if present
            for key in ["label", "labels", "target"]:
                if key in original:
                    item["labels"] = original[key]
                    break
        
        return item


@register_adapter
class TextAdapter(DataAdapter):
    """Adapter for loading text datasets."""
    
    name = "text"
    data_type = "text"
    supported_extensions = [".txt", ".json", ".jsonl", ".csv", ".parquet", ".tsv", ".md"]
    
    def can_handle(self, path: Path) -> bool:
        """Check if path is a text file or directory with text files."""
        path = Path(path)
        
        if path.is_file():
            return self._check_extension(path)
        
        if path.is_dir():
            # Check if directory contains text files
            text_files = self._scan_directory(path, recursive=False)
            return len(text_files) > 0
        
        return False
    
    def load(
        self,
        path: Path,
        tokenizer=None,
        max_length: int = 512,
        text_column: str = "text",
        label_column: Optional[str] = None,
        streaming: bool = False,
        sample_limit: Optional[int] = None,
        **kwargs,
    ) -> DataAdapterResult:
        """Load text data from file or directory.
        
        Args:
            path: Path to file or directory.
            tokenizer: HuggingFace tokenizer for encoding.
            max_length: Maximum sequence length.
            text_column: Column name for text in structured files.
            label_column: Column name for labels (if any).
            streaming: Use streaming dataset for large files.
            sample_limit: Maximum number of samples to load.
            **kwargs: Additional arguments.
        
        Returns:
            DataAdapterResult with loaded dataset.
        """
        path = Path(path)
        
        if path.is_dir():
            return self._load_directory(
                path, tokenizer, max_length, text_column, 
                label_column, sample_limit
            )
        
        if streaming:
            dataset = StreamingTextDataset(
                path, tokenizer, max_length, text_column
            )
            # Can't easily count streaming dataset
            num_samples = -1
        else:
            texts, labels = self._load_file(path, text_column, label_column, sample_limit)
            dataset = TextDataset(texts, tokenizer, max_length, labels)
            num_samples = len(texts)
        
        return DataAdapterResult(
            dataset=dataset,
            adapter_name=self.name,
            num_samples=num_samples,
            data_type=self.data_type,
            features={
                "max_length": max_length,
                "has_tokenizer": tokenizer is not None,
                "has_labels": labels is not None if not streaming else False,
            },
            source_path=str(path),
            format_detected=path.suffix,
            preprocessing_applied=["tokenization"] if tokenizer else [],
        )
    
    def _load_file(
        self,
        path: Path,
        text_column: str,
        label_column: Optional[str],
        sample_limit: Optional[int],
    ) -> tuple[List[str], Optional[List[Any]]]:
        """Load texts and labels from a single file."""
        suffix = path.suffix.lower()
        
        texts = []
        labels = [] if label_column else None
        
        if suffix == ".jsonl":
            texts, labels = self._load_jsonl(path, text_column, label_column)
        elif suffix == ".json":
            texts, labels = self._load_json(path, text_column, label_column)
        elif suffix in (".csv", ".tsv"):
            texts, labels = self._load_csv(path, text_column, label_column, suffix)
        elif suffix == ".parquet":
            texts, labels = self._load_parquet(path, text_column, label_column)
        else:
            texts = self._load_plain_text(path)
        
        if sample_limit and len(texts) > sample_limit:
            texts = texts[:sample_limit]
            if labels:
                labels = labels[:sample_limit]
        
        return texts, labels if labels else None
    
    def _load_jsonl(
        self,
        path: Path,
        text_column: str,
        label_column: Optional[str],
    ) -> tuple[List[str], Optional[List[Any]]]:
        texts = []
        labels = [] if label_column else None
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data.get(text_column, data.get("content", ""))
                    if text:
                        texts.append(text)
                        if labels is not None and label_column in data:
                            labels.append(data[label_column])
        
        return texts, labels
    
    def _load_json(
        self,
        path: Path,
        text_column: str,
        label_column: Optional[str],
    ) -> tuple[List[str], Optional[List[Any]]]:
        texts = []
        labels = [] if label_column else None
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text = item.get(text_column, item.get("content", ""))
                else:
                    text = str(item)
                if text:
                    texts.append(text)
                    if labels is not None and isinstance(item, dict) and label_column in item:
                        labels.append(item[label_column])
        elif isinstance(data, dict):
            # Single document or nested structure
            if text_column in data:
                texts.append(data[text_column])
        
        return texts, labels
    
    def _load_csv(
        self,
        path: Path,
        text_column: str,
        label_column: Optional[str],
        suffix: str,
    ) -> tuple[List[str], Optional[List[Any]]]:
        texts = []
        labels = [] if label_column else None
        
        delimiter = "\t" if suffix == ".tsv" else ","
        
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Auto-detect text column if not found
            fieldnames = reader.fieldnames or []
            if text_column not in fieldnames:
                # Try common column names
                for name in ["text", "content", "sentence", "document"]:
                    if name in fieldnames:
                        text_column = name
                        break
                else:
                    # Use first column
                    text_column = fieldnames[0] if fieldnames else "text"
            
            for row in reader:
                text = row.get(text_column, "")
                if text:
                    texts.append(text)
                    if labels is not None and label_column and label_column in row:
                        labels.append(row[label_column])
        
        return texts, labels
    
    def _load_parquet(
        self,
        path: Path,
        text_column: str,
        label_column: Optional[str],
    ) -> tuple[List[str], Optional[List[Any]]]:
        try:
            import pandas as pd
            
            df = pd.read_parquet(path)
            
            # Auto-detect text column
            if text_column not in df.columns:
                for name in ["text", "content", "sentence", "document"]:
                    if name in df.columns:
                        text_column = name
                        break
                else:
                    text_column = df.columns[0]
            
            texts = df[text_column].tolist()
            labels = df[label_column].tolist() if label_column and label_column in df.columns else None
            
            return texts, labels

        except ImportError as e:
            raise ImportError(
                "pandas and pyarrow required for parquet. Run: pip install pandas pyarrow"
            ) from e
    
    def _load_plain_text(self, path: Path) -> List[str]:
        """Load plain text file (one document or line per document)."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # If file has many lines, treat each as a sample
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        if len(lines) > 10:
            return lines
        else:
            # Small file, treat as single document
            return [content]
    
    def _load_directory(
        self,
        directory: Path,
        tokenizer,
        max_length: int,
        text_column: str,
        label_column: Optional[str],
        sample_limit: Optional[int],
    ) -> DataAdapterResult:
        """Load all text files from a directory."""
        all_texts = []
        all_labels = []
        
        files = self._scan_directory(directory)
        
        for file_path in files:
            texts, labels = self._load_file(file_path, text_column, label_column, None)
            all_texts.extend(texts)
            if labels:
                all_labels.extend(labels)
        
        if sample_limit and len(all_texts) > sample_limit:
            all_texts = all_texts[:sample_limit]
            if all_labels:
                all_labels = all_labels[:sample_limit]
        
        labels = all_labels if all_labels else None
        dataset = TextDataset(all_texts, tokenizer, max_length, labels)
        
        return DataAdapterResult(
            dataset=dataset,
            adapter_name=self.name,
            num_samples=len(all_texts),
            data_type=self.data_type,
            features={
                "max_length": max_length,
                "has_tokenizer": tokenizer is not None,
                "has_labels": labels is not None,
                "num_files": len(files),
            },
            source_path=str(directory),
            format_detected="directory",
            preprocessing_applied=["tokenization"] if tokenizer else [],
        )
