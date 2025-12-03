"""Reusable test harness for QuickTests.

Provides:
- TestConfig: central config for model path, device, limits
- TestSession: context manager for consistent section logging
- Utilities: package checks, model checks, dummy image, timing
- Simple assert helpers and colored output

Designed for script-style tests (no pytest/unittest required).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

# Add project root to path and expose helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.project_paths as paths  # noqa: E402

paths.add_src_to_sys_path()

# Colors for pretty output
class _C:
    H = "\033[95m"  # header
    B = "\033[94m"  # blue
    G = "\033[92m"  # green
    Y = "\033[93m"  # yellow
    R = "\033[91m"  # red
    E = "\033[0m"   # end


def _supports_color() -> bool:
    return sys.stdout.isatty()


def ctext(text: str, color: str) -> str:
    if _supports_color():
        return f"{color}{text}{_C.E}"
    return text


@dataclass
class TestConfig:
    model_dir: Path = paths.MODELS_DIR
    device: str = "auto"  # auto|cpu|cuda
    max_tokens: int = 64
    temperature: float = 0.3
    timeout_s: int = 60


class TestSession:
    """Context manager for consistent section logging and timing."""
    def __init__(self, title: str):
        self.title = title
        self.start = 0.0

    def __enter__(self):
        print("\n" + "=" * 70)
        print(ctext(self.title, _C.B))
        print("=" * 70)
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        if exc:
            print(ctext(f"✖ {self.title} FAILED: {exc}", _C.R))
            return False
        print(ctext(f"✔ {self.title} OK ({elapsed:.2f}s)", _C.G))
        return True


# Prevent pytest from trying to collect harness helpers as tests
TestConfig.__test__ = False  # type: ignore[attr-defined]
TestSession.__test__ = False  # type: ignore[attr-defined]


def timeit(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure execution time of a function and print it."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = (time.time() - t0) * 1000
        print(ctext(f"   ↳ {fn.__name__} finished in {dt:.1f} ms", _C.Y))
        return out
    return wrapper


# ---- Utilities ----

def require_package(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception as e:
        print(ctext(f"   (skipping: optional package '{name}' missing: {e})", _C.Y))
        return False


def ensure_model_exists(cfg: TestConfig) -> None:
    model_p = cfg.model_dir / "model.safetensors"
    tok_p = cfg.model_dir / "tokenizer.json"
    if not model_p.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_p}")
    if not tok_p.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_p}")


def make_dummy_image(size=(384, 384)):
    from PIL import Image
    img = Image.new("RGB", size, color=(127, 127, 127))
    return img


def get_device(cfg: TestConfig):
    import torch
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def log_ok(msg: str) -> None:
    print(ctext(f"✓ {msg}", _C.G))


def log_info(msg: str) -> None:
    print(ctext(f"→ {msg}", _C.B))


def assert_non_empty(text: str, label: str = "text") -> None:
    if not isinstance(text, str) or not text.strip():
        raise AssertionError(f"Expected non-empty {label}")


def run_guarded(name: str, fn: Callable[[], None]) -> bool:
    """Run a test section and return True/False instead of raising to stop suite."""
    try:
        with TestSession(name):
            fn()
        return True
    except Exception as e:
        print(ctext(f"✖ {name} failed: {e}", _C.R))
        return False
