"""Centralized helpers for resolving important project directories."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "Dataset"
EVAL_RUNS_DIR = DATA_DIR / "eval_runs"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = SRC_DIR / "Core_Models" / "Save"

PathLike = Union[str, Path]

def add_src_to_sys_path() -> None:
    """Ensure the src/ directory is available for absolute imports."""
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

def resolve_path(path_like: PathLike, *, base_dir: Path | None = None) -> Path:
    """Resolve a path relative to the project root if needed.

    Treat POSIX-style absolute paths (starting with '/') as absolute even on
    Windows where `Path.is_absolute()` may report False due to missing drive.
    """
    # Normalize to string for POSIX-style root detection
    if isinstance(path_like, Path):
        raw = str(path_like)
    else:
        raw = path_like
    if isinstance(path_like, Path):
        # POSIX-style absolute on Windows (no drive, has root '/' or '\\')
        if not path_like.drive and path_like.root in ('/', '\\'):
            return path_like
    if isinstance(raw, str) and (raw.startswith('/') or raw.startswith('\\')):
        return Path(raw)
    path = Path(path_like)  # original path object
    if path.is_absolute():
        return path
    return (base_dir or PROJECT_ROOT) / path

def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def resolve_model_dir(path_like: PathLike | None = None) -> Path:
    """Resolve and create the directory that stores model checkpoints."""
    directory = Path(path_like) if path_like else MODELS_DIR
    return ensure_dir(resolve_path(directory))

def resolve_eval_dir(path_like: PathLike | None = None) -> Path:
    """Resolve and create the directory for evaluation reports."""
    directory = Path(path_like) if path_like else EVAL_RUNS_DIR
    return ensure_dir(resolve_path(directory))

def resolve_dataset(path_like: PathLike | None = None) -> Path:
    """Resolve the dataset file path (does not create files)."""
    return resolve_path(path_like or (DATASET_DIR / "sample_train.jsonl"))

# Automatically expose src/ for any consumer that imports this helper.
add_src_to_sys_path()
