"""Pytest configuration and shared fixtures for OpenNeuralEngine tests.

This file provides reusable fixtures and test configuration that can be used
across all test modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.project_paths as paths
paths.add_src_to_sys_path()

# Import after paths are set up - make optional
HFCompatibleLM = None
try:
    from Core_Models.hf_compat import HFCompatibleLM
except ImportError:
    pass


# ============================================================================
# Session-level Fixtures (run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def device() -> str:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Return the path to the model directory."""
    return paths.MODELS_DIR


@pytest.fixture(scope="session", autouse=True)
def ensure_dummy_checkpoint(model_dir: Path) -> Path:
    """Create a tiny checkpoint so API interfaces can load during tests."""
    checkpoint = model_dir / "model.safetensors"
    paths.ensure_dir(model_dir)

    if checkpoint.exists():
        return checkpoint

    # Skip if HFCompatibleLM not available
    if HFCompatibleLM is None:
        pytest.skip("HFCompatibleLM not available")
        return checkpoint

    torch.manual_seed(0)
    tiny_model = HFCompatibleLM(
        vocab_size=64,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=256,
        max_seq_len=64,
    )
    save_file(tiny_model.state_dict(), checkpoint)
    return checkpoint


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the data directory."""
    return paths.DATA_DIR


@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    """Return the path to the dataset directory."""
    return paths.DATASET_DIR


# ============================================================================
# Function-level Fixtures (run once per test function)
# ============================================================================

@pytest.fixture
def random_seed() -> int:
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def small_batch() -> dict:
    """Create a small batch of synthetic data for testing."""
    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }


@pytest.fixture
def tiny_model_config() -> dict:
    """Create a minimal model configuration for fast testing."""
    return {
        "vocab_size": 1000,
        "embedding_dim": 64,
        "hidden_dim": 64,
        "num_micro_layers": 2,
        "num_heads": 2,
        "dropout": 0.0,
        "max_seq_len": 128,
    }


@pytest.fixture
def model_exists(model_dir: Path) -> bool:
    """Check if a trained model checkpoint exists."""
    checkpoint_path = model_dir / "model.safetensors"
    return checkpoint_path.exists()


# ============================================================================
# Parametrize Helpers
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring a trained model checkpoint"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU acceleration"
    )
    config.addinivalue_line(
        "markers", "online: mark test as requiring network/API access"
    )
    config.addinivalue_line(
        "markers", "aim: mark test as Aim logging related"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    # Check if online tests should be skipped
    skip_online = pytest.mark.skip(reason="Online tests skipped by default. Use -m online to run.")
    
    for item in items:
        # Auto-mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Auto-mark slow tests
        if "slow" in item.nodeid.lower() or "training" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark API tests
        if "api" in item.nodeid.lower():
            item.add_marker(pytest.mark.api)
        
        # Skip online tests by default unless explicitly requested
        if "online" in item.keywords:
            # Only skip if -m online was not passed
            if not config.getoption("-m") or "online" not in config.getoption("-m"):
                item.add_marker(skip_online)


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def assert_shape():
    """Helper fixture to assert tensor shapes."""
    def _assert_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
        actual = tuple(tensor.shape)
        assert actual == expected_shape, (
            f"{name} shape mismatch: expected {expected_shape}, got {actual}"
        )
    return _assert_shape


@pytest.fixture
def assert_dtype():
    """Helper fixture to assert tensor dtypes."""
    def _assert_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: str = "tensor"):
        assert tensor.dtype == expected_dtype, (
            f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
        )
    return _assert_dtype


@pytest.fixture
def assert_device():
    """Helper fixture to assert tensor device."""
    def _assert_device(tensor: torch.Tensor, expected_device: str, name: str = "tensor"):
        actual = str(tensor.device).split(":")[0]  # Handle cuda:0 -> cuda
        assert actual == expected_device, (
            f"{name} device mismatch: expected {expected_device}, got {actual}"
        )
    return _assert_device


@pytest.fixture
def create_dummy_image():
    """Create a dummy image tensor for multimodal testing."""
    def _create(batch_size: int = 1, channels: int = 3, height: int = 224, width: int = 224):
        return torch.randn(batch_size, channels, height, width)
    return _create


@pytest.fixture
def create_dummy_text():
    """Create dummy text token IDs for testing."""
    def _create(batch_size: int = 1, seq_len: int = 32, vocab_size: int = 1000):
        return torch.randint(0, vocab_size, (batch_size, seq_len))
    return _create


# ============================================================================
# Skip Conditions
# ============================================================================

skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

skip_if_no_model = pytest.mark.skipif(
    not (paths.MODELS_DIR / "model.safetensors").exists(),
    reason="No trained model checkpoint found"
)


def requires_package(package_name: str):
    """Decorator to skip test if package is not available."""
    try:
        __import__(package_name)
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason=f"Package '{package_name}' not installed")
