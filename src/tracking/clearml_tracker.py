"""
ClearML Experiment Tracking Module
==================================

Provides a unified, reusable ClearML integration for training, evaluation, and inference.
Designed for minimal code changes while providing comprehensive tracking.

Features:
- Automatic Task initialization with proper configuration capture
- Automatic hyperparameter logging from dataclasses/dicts
- Real-time metric logging with batching support
- Artifact management (models, checkpoints, datasets)
- Environment reproducibility (Docker, venv, packages)
- Stdout/stderr capture and exception logging (automatic)

Usage:
    from src.tracking.clearml_tracker import ClearMLTracker, init_clearml_task
    
    # Option 1: Simple initialization
    task = init_clearml_task("Training", "my_experiment")
    
    # Option 2: Full tracker with helpers
    tracker = ClearMLTracker("Training", "my_experiment")
    tracker.log_hyperparameters(config)
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
    tracker.upload_artifact("model", "path/to/model.pt")
    tracker.close()
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from contextlib import contextmanager

# Conditional ClearML import with graceful fallback
try:
    from clearml import Task, Logger, OutputModel, InputModel, Dataset
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None
    Logger = None
    OutputModel = None
    InputModel = None
    Dataset = None

# Type checking imports
if TYPE_CHECKING:
    from clearml import Task as TaskType, Logger as LoggerType
else:
    TaskType = Any
    LoggerType = Any

logger = logging.getLogger(__name__)


def is_clearml_available() -> bool:
    """Check if ClearML is installed and available."""
    return CLEARML_AVAILABLE


def init_clearml_task(
    project_name: str,
    task_name: str,
    task_type: str = "training",
    tags: Optional[List[str]] = None,
    reuse_last_task_id: bool = False,
    auto_connect_frameworks: bool = True,
    auto_connect_arg_parser: bool = True,
    auto_resource_monitoring: bool = True,
    output_uri: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialize a ClearML Task with sensible defaults.
    
    This is the primary entry point for ClearML integration. Call this at the
    start of your main() function or script.
    
    Args:
        project_name: ClearML project name (e.g., "ONN Training")
        task_name: Experiment name (e.g., "phi4-lora-experiment-v1")
        task_type: One of "training", "testing", "inference", "data_processing",
                   "application", "monitor", "controller", "optimizer", "service",
                   "qc", "custom"
        tags: Optional list of tags for filtering experiments
        reuse_last_task_id: If True, continues last unfinished task with same name
        auto_connect_frameworks: Auto-capture PyTorch, TensorBoard, etc.
        auto_connect_arg_parser: Auto-capture argparse arguments
        auto_resource_monitoring: Enable GPU/CPU/memory monitoring
        output_uri: Remote storage URI for artifacts (e.g., "s3://bucket/folder")
    
    Returns:
        ClearML Task object or None if ClearML is not available
    
    Example:
        task = init_clearml_task("ONN Training", "phi4-math-finetune", tags=["lora", "math"])
    """
    if not CLEARML_AVAILABLE:
        logger.warning("ClearML not available. Install with: pip install clearml")
        return None
    
    # Map task type string to ClearML TaskType
    task_type_map = {
        "training": Task.TaskTypes.training,
        "testing": Task.TaskTypes.testing,
        "inference": Task.TaskTypes.inference,
        "data_processing": Task.TaskTypes.data_processing,
        "application": Task.TaskTypes.application,
        "monitor": Task.TaskTypes.monitor,
        "controller": Task.TaskTypes.controller,
        "optimizer": Task.TaskTypes.optimizer,
        "service": Task.TaskTypes.service,
        "qc": Task.TaskTypes.qc,
        "custom": Task.TaskTypes.custom,
    }
    
    clearml_task_type = task_type_map.get(task_type, Task.TaskTypes.training)
    
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=clearml_task_type,
        tags=tags or [],
        reuse_last_task_id=reuse_last_task_id,
        auto_connect_frameworks=auto_connect_frameworks,
        auto_connect_arg_parser=auto_connect_arg_parser,
        auto_resource_monitoring=auto_resource_monitoring,
        output_uri=output_uri,
    )
    
    # Log environment information
    _log_environment_info(task)
    
    return task


def _log_environment_info(task: Any):
    """Log environment information for reproducibility."""
    import platform
    
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    
    # Log CUDA info if available
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_count"] = torch.cuda.device_count()
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
    except ImportError:
        pass
    
    task.connect(env_info, name="Environment")


class ClearMLTracker:
    """
    High-level ClearML tracker with convenient methods for ML experiments.
    
    This class provides a simplified interface for common tracking operations,
    with automatic batching and error handling.
    
    Example:
        tracker = ClearMLTracker("ONN Training", "experiment_1")
        tracker.log_hyperparameters({"lr": 1e-4, "batch_size": 32})
        
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            tracker.log_metric("loss", loss, step)
        
        tracker.upload_model("model.safetensors", "final_model")
        tracker.close()
    """
    
    def __init__(
        self,
        project_name: str = "ONN Training",
        task_name: str = "experiment",
        task_type: str = "training",
        tags: Optional[List[str]] = None,
        auto_connect_frameworks: bool = True,
        output_uri: Optional[str] = None,
    ):
        """Initialize ClearML tracker."""
        self.task = init_clearml_task(
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            tags=tags,
            auto_connect_frameworks=auto_connect_frameworks,
            output_uri=output_uri,
        )
        self._logger: Optional[Any] = None
        self._metrics_buffer: Dict[str, List[tuple]] = {}
        self._buffer_size = 50  # Flush every N metrics
        self._step_counters: Dict[str, int] = {}
    
    @property
    def logger(self) -> Optional[Any]:
        """Get ClearML Logger instance."""
        if self._logger is None and self.task is not None:
            self._logger = self.task.get_logger()
        return self._logger
    
    @property
    def is_active(self) -> bool:
        """Check if tracker is active and connected."""
        return self.task is not None
    
    def log_hyperparameters(
        self,
        params: Union[Dict[str, Any], Any],
        name: str = "General",
        prefix: str = "",
    ):
        """
        Log hyperparameters to ClearML.
        
        Args:
            params: Dictionary or dataclass of hyperparameters
            name: Section name in ClearML UI (e.g., "Training", "Model")
            prefix: Optional prefix for parameter names
        
        Example:
            tracker.log_hyperparameters({"lr": 1e-4, "epochs": 10})
            tracker.log_hyperparameters(training_config, name="Training")
        """
        if not self.is_active:
            return
        
        # Convert dataclass to dict
        if is_dataclass(params) and not isinstance(params, dict):
            params = asdict(params)
        elif hasattr(params, "to_dict"):
            params = params.to_dict()
        elif hasattr(params, "__dict__"):
            params = vars(params)
        
        # Flatten nested dicts with prefix
        flat_params = self._flatten_dict(params, prefix)
        
        self.task.connect(flat_params, name=name)
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        series: str = "train",
    ):
        """
        Log a single metric value.
        
        Args:
            name: Metric name (e.g., "loss", "accuracy")
            value: Metric value
            step: Training step/iteration (auto-incremented if None)
            series: Series name for grouping (e.g., "train", "val")
        
        Example:
            tracker.log_metric("loss", 0.5, step=100)
            tracker.log_metric("accuracy", 0.95, step=100, series="val")
        """
        if not self.is_active or self.logger is None:
            return
        
        # Auto-increment step if not provided
        if step is None:
            key = f"{series}/{name}"
            step = self._step_counters.get(key, 0)
            self._step_counters[key] = step + 1
        
        self.logger.report_scalar(
            title=name,
            series=series,
            value=value,
            iteration=step,
        )
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        series: str = "train",
    ):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step/iteration
            series: Series name for grouping
        
        Example:
            tracker.log_metrics({"loss": 0.5, "ppl": 50.0, "lr": 1e-4}, step=100)
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step, series)
    
    def log_text(self, title: str, text: str, step: int = 0):
        """Log text output (e.g., generated samples, debug info)."""
        if not self.is_active or self.logger is None:
            return
        self.logger.report_text(text, level=logging.INFO, print_console=False)
    
    def log_plot(
        self,
        title: str,
        series: str,
        x: List[float],
        y: List[float],
        step: int = 0,
        mode: str = "lines",
    ):
        """
        Log a custom plot.
        
        Args:
            title: Plot title
            series: Series name
            x: X-axis values
            y: Y-axis values
            step: Iteration
            mode: Plot mode ("lines", "markers", "lines+markers")
        """
        if not self.is_active or self.logger is None:
            return
        
        self.logger.report_line_plot(
            title=title,
            series=[series],
            xaxis="x",
            yaxis="y",
            iteration=step,
            mode=mode,
        )
    
    def log_histogram(
        self,
        title: str,
        values: List[float],
        step: int = 0,
        bins: int = 50,
    ):
        """Log histogram of values (e.g., gradient norms, weight distributions)."""
        if not self.is_active or self.logger is None:
            return
        
        import numpy as np
        hist, edges = np.histogram(values, bins=bins)
        
        self.logger.report_histogram(
            title=title,
            series="distribution",
            values=values,
            iteration=step,
            xaxis="Value",
            yaxis="Count",
        )
    
    def upload_artifact(
        self,
        name: str,
        artifact: Union[str, Path, Dict, Any],
        artifact_type: str = "file",
        metadata: Optional[Dict] = None,
    ):
        """
        Upload an artifact to ClearML.
        
        Args:
            name: Artifact name
            artifact: File path, directory, dict, or object to upload
            artifact_type: Type hint ("file", "folder", "dict", "object")
            metadata: Optional metadata dictionary
        
        Example:
            tracker.upload_artifact("config", config_dict)
            tracker.upload_artifact("checkpoint", "output/model.pt")
            tracker.upload_artifact("results", results_df)  # pandas DataFrame
        """
        if not self.is_active:
            return
        
        if isinstance(artifact, (str, Path)):
            path = Path(artifact)
            if path.is_file():
                self.task.upload_artifact(name=name, artifact_object=str(path))
            elif path.is_dir():
                self.task.upload_artifact(name=name, artifact_object=str(path))
            else:
                logger.warning(f"Artifact path does not exist: {path}")
        elif isinstance(artifact, dict):
            self.task.upload_artifact(name=name, artifact_object=artifact)
        else:
            # Try to upload as generic object
            self.task.upload_artifact(name=name, artifact_object=artifact)
    
    def upload_model(
        self,
        model_path: Union[str, Path],
        model_name: str = "model",
        framework: str = "PyTorch",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Upload a trained model to ClearML Model Registry.
        
        Args:
            model_path: Path to model file or directory
            model_name: Model name in registry
            framework: Framework name (PyTorch, TensorFlow, etc.)
            tags: Optional tags for the model
            metadata: Optional metadata dictionary
        
        Returns:
            Model ID if successful, None otherwise
        
        Example:
            model_id = tracker.upload_model(
                "output/checkpoints/final.safetensors",
                model_name="phi4-lora-math",
                tags=["lora", "math", "production"]
            )
        """
        if not self.is_active or OutputModel is None:
            return None
        
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return None
        
        output_model = OutputModel(
            task=self.task,
            name=model_name,
            framework=framework,
            tags=tags or [],
        )
        
        output_model.update_weights(
            weights_filename=str(model_path),
            auto_delete_file=False,
        )
        
        if metadata:
            output_model.update_design(config_dict=metadata)
        
        return output_model.id
    
    def set_status(self, status: str, message: str = ""):
        """
        Set task status with optional message.
        
        Args:
            status: One of "completed", "failed", "stopped"
            message: Optional status message
        """
        if not self.is_active:
            return
        
        if status == "completed":
            self.task.mark_completed()
        elif status == "failed":
            self.task.mark_failed(status_message=message)
        elif status == "stopped":
            self.task.mark_stopped(status_message=message)
    
    def add_tags(self, tags: List[str]):
        """Add tags to the current task."""
        if not self.is_active:
            return
        current_tags = self.task.get_tags() or []
        self.task.set_tags(list(set(current_tags + tags)))
    
    def set_comment(self, comment: str):
        """Set task comment/description."""
        if not self.is_active:
            return
        self.task.set_comment(comment)
    
    def get_task_id(self) -> Optional[str]:
        """Get the current task ID."""
        return self.task.id if self.is_active else None
    
    def get_task_url(self) -> Optional[str]:
        """Get the URL to view this task in ClearML UI."""
        if not self.is_active:
            return None
        return self.task.get_output_log_web_page()
    
    def close(self):
        """Close the tracker and finalize the task."""
        if self.is_active:
            # Upload any remaining buffered metrics
            self._flush_metrics()
            self.task.close()
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        prefix: str = "",
        sep: str = "/"
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, key, sep))
            else:
                items[key] = v
        return items
    
    def _flush_metrics(self):
        """Flush any buffered metrics."""
        # Currently metrics are logged immediately, but this provides
        # a hook for future batching optimization
        pass


@contextmanager
def track_experiment(
    project_name: str = "ONN Training",
    task_name: str = "experiment",
    task_type: str = "training",
    tags: Optional[List[str]] = None,
):
    """
    Context manager for experiment tracking.
    
    Automatically handles task initialization and cleanup.
    
    Example:
        with track_experiment("ONN", "my_experiment", tags=["test"]) as tracker:
            tracker.log_hyperparameters(config)
            for step in range(100):
                loss = train_step()
                tracker.log_metric("loss", loss, step)
    """
    tracker = ClearMLTracker(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        tags=tags,
    )
    try:
        yield tracker
    except Exception as e:
        if tracker.is_active:
            tracker.set_status("failed", str(e))
        raise
    finally:
        tracker.close()


# Convenience function for quick metric logging without full tracker setup
_global_task: Optional[Any] = None


def get_current_task() -> Optional[Any]:
    """Get the current ClearML task if one exists."""
    global _global_task
    if _global_task is not None:
        return _global_task
    if CLEARML_AVAILABLE:
        return Task.current_task()
    return None


def log_metric_simple(name: str, value: float, step: Optional[int] = None):
    """
    Simple metric logging using current task.
    
    Use this when you don't want to pass around a tracker object.
    
    Example:
        # After Task.init() has been called somewhere
        log_metric_simple("loss", 0.5, step=100)
    """
    task = get_current_task()
    if task is None:
        return
    
    logger = task.get_logger()
    logger.report_scalar(
        title=name,
        series="train",
        value=value,
        iteration=step or 0,
    )


# Docker/venv environment tracking helpers
def log_docker_config(
    task: Optional[Any],
    image: str,
    arguments: Optional[str] = None,
    setup_shell_script: Optional[str] = None,
):
    """
    Configure Docker environment for ClearML Agent execution.
    
    This enables reproducibility when running via ClearML Agent.
    
    Example:
        log_docker_config(
            task,
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            arguments="--gpus all --shm-size 8g"
        )
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    task.set_base_docker(
        docker_image=image,
        docker_arguments=arguments,
        docker_setup_bash_script=setup_shell_script,
    )


def log_packages(task: Optional[Any], packages: Optional[List[str]] = None):
    """
    Log required packages for reproducibility.
    
    If packages is None, automatically detects from current environment.
    
    Example:
        log_packages(task, ["torch>=2.0.0", "transformers>=4.30.0"])
    """
    if task is None or not CLEARML_AVAILABLE:
        return
    
    if packages is None:
        # Auto-detect from pip
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True
            )
            packages = result.stdout.strip().split("\n")
        except Exception:
            packages = []
    
    task.update_task({"script": {"requirements": {"pip": "\n".join(packages)}}})
