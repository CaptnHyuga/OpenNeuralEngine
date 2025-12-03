"""Core model components for OpenNeuralEngine.

This package wraps production ML libraries and provides the ONN abstraction layer.

Note: Core model implementations have been refactored to use HuggingFace
Transformers, DeepSpeed, and other production libraries. The legacy
puzzle-based architecture has been deprecated in favor of wrapping
battle-tested frameworks.

Available modules:
- evaluator: Model evaluation utilities
- experiment_tracking: Aim-based experiment tracking
- hf_compat: HuggingFace compatibility layer
"""

# Only import what actually exists
_available_modules = {}

try:
    from .evaluator import Evaluator, EvalResult
    _available_modules["evaluator"] = True
except ImportError:
    Evaluator = None  # type: ignore
    EvalResult = None  # type: ignore
    _available_modules["evaluator"] = False

try:
    from .experiment_tracking import AimTracker
    _available_modules["experiment_tracking"] = True
except ImportError:
    AimTracker = None  # type: ignore
    _available_modules["experiment_tracking"] = False

try:
    from .hf_compat import HFCompatibleLM
    _available_modules["hf_compat"] = True
except ImportError:
    HFCompatibleLM = None  # type: ignore
    _available_modules["hf_compat"] = False


def check_available_modules():
    """Return dict of available Core_Models submodules."""
    return _available_modules.copy()


__all__ = [
    "Evaluator",
    "EvalResult",
    "AimTracker",
    "HFCompatibleLM",
    "check_available_modules",
]
