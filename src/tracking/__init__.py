"""ONN Tracking Module - Metrics logging and experiment tracking.

Supports multiple tracking backends:
- AIM (via Docker) - aim_logger.py
- ClearML - clearml_tracker.py

ClearML Usage:
    from src.tracking.clearml_tracker import ClearMLTracker, init_clearml_task
    
    # Simple task initialization
    task = init_clearml_task("ONN Training", "experiment_1")
    
    # Full tracker with helper methods
    tracker = ClearMLTracker("ONN Training", "experiment_1")
    tracker.log_hyperparameters(config)
    tracker.log_metrics({"loss": 0.5}, step=100)
    tracker.upload_model("model.pt", "final_model")
    tracker.close()
"""
from .aim_logger import AIMDockerLogger, log_benchmark_to_aim

# Lazy import for ClearML to avoid ImportError if not installed
def get_clearml_tracker():
    """Get ClearMLTracker class (lazy import)."""
    from .clearml_tracker import ClearMLTracker
    return ClearMLTracker

def init_clearml_task(*args, **kwargs):
    """Initialize a ClearML task (lazy import wrapper)."""
    from .clearml_tracker import init_clearml_task as _init
    return _init(*args, **kwargs)

__all__ = [
    "AIMDockerLogger", 
    "log_benchmark_to_aim",
    "get_clearml_tracker",
    "init_clearml_task",
]
