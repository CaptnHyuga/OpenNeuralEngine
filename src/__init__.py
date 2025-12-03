"""OpenNeuralEngine 2.0 - Production-Grade Democratic AI Framework.

A unified framework for training any model on any data with automatic
hardware-aware configuration.

Main Components:
    - orchestration: Hardware profiling and configuration generation
    - wrappers: Production library wrappers (HuggingFace, DeepSpeed)
    - data_adapters: Universal data loading interface
    - tracking: Experiment tracking (Aim)

Quick Start:
    from src.orchestration import auto_configure
    from src.wrappers import train, load_model
    from src.data_adapters import AUTO_DETECT
    
    # Load any data
    dataset = AUTO_DETECT("./my_data/")
    
    # Train any model
    result = train("gpt2", dataset.dataset, num_epochs=3)
"""
from pathlib import Path

__version__ = "2.0.0"
__author__ = "OpenNeuralEngine Team"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "Core_Models" / "Save"
DATA_DIR = PROJECT_ROOT / "data"

# Lazy imports for faster startup
def _lazy_import():
    """Import main components on demand."""
    global orchestration, wrappers, data_adapters
    from . import orchestration
    from . import wrappers
    from . import data_adapters
    return orchestration, wrappers, data_adapters


__all__ = [
    "__version__",
    "PROJECT_ROOT",
    "MODELS_DIR", 
    "DATA_DIR",
    "orchestration",
    "wrappers",
    "data_adapters",
]
