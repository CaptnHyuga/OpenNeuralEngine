"""Experiment tracking helpers for Aim/MLflow backends.

Defaults to Aim (dockerized via ``aimstack/aim``) but can fall back to
MLflow for backward compatibility. Both backends expose the same public API
so training/evaluation code does not need to know which tracker is active.
"""
from __future__ import annotations

import getpass
import logging
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Default tracking configuration (Aim first, fallback to legacy vars)
DEFAULT_AIM_REPO = os.environ.get("AIM_TRACKING_URI", "aim://localhost:53800")
DEFAULT_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
GLOBAL_EXPERIMENT_NAME = os.environ.get("SNN_TRACKING_EXPERIMENT") or os.environ.get("SNN_MLFLOW_EXPERIMENT", "snn")
DEFAULT_RUN_TYPE = os.environ.get("SNN_RUN_TYPE", "train")
DEFAULT_RUN_GROUP = os.environ.get("SNN_RUN_GROUP")
DEFAULT_TRACKING_MODE = os.environ.get("SNN_TRACKING_MODE") or os.environ.get("SNN_MLFLOW_MODE", "enabled")
DEFAULT_TRACKING_BACKEND = (os.environ.get("SNN_TRACKING_BACKEND", "aim").lower())

# Check Aim availability
try:
    from aim import File as AimFile
    from aim import Image as AimImage
    from aim import Run as AimRun
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    AimRun = AimImage = AimFile = None

# Check MLflow availability
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.debug("mlflow not installed. Run: pip install mlflow if needed")


@dataclass
class TrackerConfig:
    """Configuration for experiment tracking."""

    experiment: str = GLOBAL_EXPERIMENT_NAME
    tracking_uri: str = DEFAULT_AIM_REPO
    run_type: str = DEFAULT_RUN_TYPE
    run_group: Optional[str] = DEFAULT_RUN_GROUP
    tags: Dict[str, str] = field(default_factory=dict)
    autolog: bool = False
    mode: str = DEFAULT_TRACKING_MODE
    backend: str = DEFAULT_TRACKING_BACKEND

    def merged_tags(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        tags = {
            "run_type": self.run_type,
            **({"run_group": self.run_group} if self.run_group else {}),
        }
        tags.update(self.tags)
        if extra:
            tags.update(extra)
        return tags


class ExperimentTracker:
    """Backend-agnostic experiment tracker (Aim default)."""
    
    def __init__(
        self,
        experiment: str = GLOBAL_EXPERIMENT_NAME,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = DEFAULT_TRACKING_MODE,
        autolog: bool = False,
        run_type: str = DEFAULT_RUN_TYPE,
        run_group: Optional[str] = DEFAULT_RUN_GROUP,
        backend: Optional[str] = None,
    ):
        """Initialize tracker connecting to Aim/MLflow backend via Docker.
        
        Args:
            experiment: Experiment name
            run_name: Run name (auto-generated if not provided)
            tracking_uri: Tracking endpoint
            tags: Tags for the run
            config: Hyperparameters to log
            mode: "enabled" or "disabled"
            autolog: Enable PyTorch autolog
            backend: "aim" (default) or "mlflow"
        """
        self.experiment_name = experiment
        self.run_type = run_type or DEFAULT_RUN_TYPE
        self.run_group = run_group
        self.backend = (backend or DEFAULT_TRACKING_BACKEND).lower()
        base_tags = {
            "run_type": self.run_type,
            "user": getpass.getuser(),
            "host": platform.node(),
        }
        if self.run_group:
            base_tags["run_group"] = self.run_group
        if tags:
            base_tags.update(tags)
        self.tags = base_tags
        self.run_name = run_name or generate_run_name(
            self.run_type,
            self.tags.get("model") or self.tags.get("model_name"),
            self.tags.get("dataset") or self.tags.get("dataset_name"),
        )
        self.config = config or {}
        self.mode = mode
        self._run = None
        self._step = 0
        self.run_id: Optional[str] = None
        self.run_url: Optional[str] = None
        self.tracking_uri = tracking_uri or (DEFAULT_AIM_REPO if self.backend == "aim" else DEFAULT_MLFLOW_URI)
        
        if self.mode == "disabled":
            return
        
        if self.backend not in {"aim", "mlflow"}:
            logger.warning(f"Unknown tracking backend '{self.backend}'. Falling back to Aim.")
            self.backend = "aim"
        
        if self.backend == "aim":
            self._init_aim_backend()
        else:
            self._init_mlflow_backend(autolog=autolog)

    def _init_aim_backend(self) -> None:
        if not AIM_AVAILABLE:
            self.mode = "disabled"
            logger.warning("Aim not available. Install aim>=3.0 or disable tracking.")
            return
        try:
            self._run = AimRun(repo=self.tracking_uri, experiment=self.experiment_name)
            self.run_id = getattr(self._run, "hash", None)
            if self.tags:
                self._run["tags"] = self.tags
            if self.config:
                self.log_params(flatten_dict(_ensure_plain_dict(self.config)))
            self.run_url = _aim_repo_to_ui(self.tracking_uri, self.run_id)
            logger.info(f"Aim run: {self.run_name} → {self.run_url or self.tracking_uri}")
        except Exception as exc:
            logger.warning(f"Aim connection failed: {exc}")
            logger.info("Start Aim server: docker compose up aim-server -d")
            self.mode = "disabled"
            self._run = None

    def _init_mlflow_backend(self, autolog: bool) -> None:
        if not MLFLOW_AVAILABLE:
            self.mode = "disabled"
            logger.warning("MLflow not available. Install mlflow or switch backend to Aim.")
            return
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            if autolog:
                try:
                    mlflow.pytorch.autolog(log_models=True, log_every_n_step=100)
                except Exception as exc:  # pragma: no cover - best effort informational
                    logger.debug(f"Autolog: {exc}")
            self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
            self.run_id = self._run.info.run_id
            if self.config:
                self.log_params(flatten_dict(_ensure_plain_dict(self.config)))
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if exp:
                self.run_url = f"{self.tracking_uri}/#/experiments/{exp.experiment_id}/runs/{self.run_id}"
            logger.info(f"MLflow run: {self.run_name} → {self.run_url or self.tracking_uri}")
        except Exception as exc:
            logger.warning(f"MLflow connection failed: {exc}")
            logger.info("Start MLflow: docker compose up mlflow -d")
            self.mode = "disabled"
            self._run = None
    
    @property
    def is_active(self) -> bool:
        return self._run is not None and self.mode == "enabled"
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if not self.is_active:
            return
        try:
            flat = {k: _stringify_param(v) for k, v in flatten_dict(_ensure_plain_dict(params)).items()}
            if self.backend == "aim":
                for key, value in flat.items():
                    self._run[f"params/{key}"] = value
            else:
                mlflow.log_params(flat)
        except Exception as e:
            logger.debug(f"Param log failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if not self.is_active:
            return
        if step is None:
            step = self._step
            self._step += 1
        try:
            if self.backend == "aim":
                for key, value in metrics.items():
                    self._run.track(value, name=key, step=step)
            else:
                mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.debug(f"Metric log failed: {e}")
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics (alias)."""
        self.log_metrics(data, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log file/directory as artifact."""
        if not self.is_active:
            return
        try:
            if self.backend == "aim":
                if AimFile is None:
                    raise RuntimeError("Aim File type unavailable")
                name = artifact_path or Path(local_path).name
                self._run.track(AimFile(local_path), name=f"artifacts/{name}")
            else:
                mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Artifact log failed: {e}")
    
    def log_model(
        self,
        model: torch.nn.Module,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        registered_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Log PyTorch model."""
        if not self.is_active:
            return None
        try:
            if self.backend == "aim":
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    torch.save(model.state_dict(), f.name)
                    self.log_artifact(f.name, artifact_path=name)
                    os.unlink(f.name)
                if aliases:
                    for alias in aliases:
                        self.set_tag(f"checkpoint_alias.{alias}", name)
                logger.info(f"Logged model checkpoint: {name}")
                return None
            info = mlflow.pytorch.log_model(
                model, artifact_path=name, registered_model_name=registered_name
            )
            if aliases:
                for alias in aliases:
                    self.set_tag(f"checkpoint_alias.{alias}", name)
            logger.info(f"Logged model: {name}")
            return info.model_uri
        except Exception as e:
            logger.warning(f"Model log failed: {e}")
            if self.backend == "mlflow":
                # Fallback: save state dict
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                        torch.save(model.state_dict(), f.name)
                        mlflow.log_artifact(f.name, name)
                        os.unlink(f.name)
                except Exception:
                    pass
        return None
    
    def load_model(self, model_uri: str) -> torch.nn.Module:
        """Load model from MLflow."""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow not available")
        return mlflow.pytorch.load_model(model_uri)
    
    def log_table(self, name: str, data: Dict[str, List[Any]]) -> None:
        """Log tabular data as CSV."""
        if not self.is_active:
            return
        try:
            import csv
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                if data:
                    writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                    writer.writeheader()
                    for row in zip(*data.values(), strict=False):
                        row_dict = dict(zip(data.keys(), row, strict=False))
                        writer.writerow(row_dict)
                path = f.name
            self.log_artifact(path, f"tables/{name}")
            os.unlink(path)
        except Exception as e:
            logger.debug(f"Table log failed: {e}")
    
    def log_image(self, name: str, image: Any) -> None:
        """Log image."""
        if not self.is_active:
            return
        try:
            if self.backend == "aim":
                if AimImage is None:
                    raise RuntimeError("Aim image support unavailable")
                self._run.track(AimImage(image), name=f"images/{name}")
            else:
                mlflow.log_image(image, f"images/{name}.png")
        except Exception as e:
            logger.debug(f"Image log failed: {e}")
    
    def set_tag(self, key: str, value: str) -> None:
        """Set tag."""
        if not self.is_active:
            return
        try:
            if self.backend == "aim":
                self._run.set(f"tags/{key}", value)
            else:
                mlflow.set_tag(key, value)
        except Exception as e:
            logger.debug(f"Tag failed: {e}")
    
    def set_summary(self, metrics: Dict[str, Any]) -> None:
        """Set summary metrics as tags."""
        for k, v in metrics.items():
            self.set_tag(f"summary.{k}", str(v))

    def log_config_dicts(self, *configs: Any) -> None:
        """Flatten and log structured configs as params."""
        merged: Dict[str, Any] = {}
        for cfg in configs:
            merged.update(flatten_dict(_ensure_plain_dict(cfg)))
        if merged:
            self.log_params(merged)

    def log_composite_score(
        self,
        metrics: Dict[str, float],
        preferences: Optional[List["MetricPreference"]] = None,
    ) -> Optional[float]:
        """Compute and log the composite score for the SNN philosophy."""
        score = compute_composite_score(metrics, preferences)
        if score is None:
            return None
        self.log_metrics({"summary/composite_score": score})
        self.set_summary({"composite_score": score})
        return score

    def unwatch(self) -> None:  # pragma: no cover - compatibility shim
        """Compatibility method for legacy tracker interface."""
        return None
    
    def finish(self) -> None:
        """End run."""
        if self._run is not None:
            try:
                if self.backend == "aim":
                    self._run.close()
                else:
                    mlflow.end_run()
                logger.info(f"Run ended: {self.run_name}")
            except Exception:
                pass
            self._run = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.finish()
        return False


def create_tracker(
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    mode: Optional[str] = None,
    config: Optional[TrackerConfig] = None,
    **overrides,
) -> ExperimentTracker:
    """Create tracker with global defaults."""
    base: Dict[str, Any] = {}
    if config is not None:
        base.update(asdict(config))
    if experiment is not None:
        base["experiment"] = experiment
    if run_name is not None:
        base["run_name"] = run_name
    if mode is not None:
        base["mode"] = mode
    base.update(overrides)
    return ExperimentTracker(**base)


def _aim_repo_to_ui(repo_uri: str, run_id: Optional[str]) -> Optional[str]:
    """Convert Aim repo URI (aim://host:port) to dashboard URL."""
    if not repo_uri:
        return None
    if repo_uri.startswith("aim://"):
        base = "http://" + repo_uri[len("aim://") :]
    else:
        base = repo_uri
    if run_id:
        return f"{base}/runs/{run_id}"
    return base


# =============================================================================
# Inference Tracking
# =============================================================================

class InferenceTracker:
    """Track inference metrics."""
    
    def __init__(self, tracker: Optional[ExperimentTracker] = None, max_samples: int = 100):
        self.tracker = tracker
        self.max_samples = max_samples
        self._samples: List[Dict[str, Any]] = []
        self._latencies: List[float] = []
        self._tokens: int = 0
        self._requests: int = 0
    
    def log_request(self, prompt: str, response: str, latency_s: float, tokens: int, **extra) -> None:
        """Log inference request."""
        self._latencies.append(latency_s)
        self._tokens += tokens
        self._requests += 1
        
        if len(self._samples) < self.max_samples:
            self._samples.append({
                "prompt": prompt[:200], "response": response[:500],
                "latency_s": latency_s, "tokens": tokens, **extra
            })
        
        if self.tracker:
            self.tracker.log_metrics({
                "inference/latency_s": latency_s,
                "inference/tokens": tokens,
                "inference/tokens_per_s": tokens / latency_s if latency_s > 0 else 0,
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary stats."""
        if not self._latencies:
            return {}
        import statistics
        return {
            "total_requests": self._requests,
            "total_tokens": self._tokens,
            "avg_latency_s": statistics.mean(self._latencies),
            "p50_latency_s": statistics.median(self._latencies),
            "avg_tokens_per_s": self._tokens / sum(self._latencies) if sum(self._latencies) > 0 else 0,
        }
    
    def finish(self) -> None:
        """Finalize tracking."""
        if self.tracker:
            for k, v in self.get_summary().items():
                self.tracker.set_tag(f"summary/{k}", str(v))
            if self._samples:
                self.tracker.log_table("inference_samples", {
                    "prompt": [s["prompt"] for s in self._samples],
                    "response": [s["response"] for s in self._samples],
                    "latency_s": [s["latency_s"] for s in self._samples],
                    "tokens": [s["tokens"] for s in self._samples],
                })


# =============================================================================
# Utility Functions
# =============================================================================

def get_mlflow_url() -> str:
    """Backward-compatible helper returning tracking UI URL."""
    return get_tracking_url()


def get_tracking_url() -> str:
    """Return Aim/MLflow dashboard URL for user-facing UIs."""
    uri = os.environ.get("AIM_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if uri.startswith("aim://"):
        return "http://" + uri[len("aim://"):]
    return uri


def _aim_repo_to_ui(repo_uri: str, run_hash: Optional[str] = None) -> Optional[str]:
    """Convert Aim repo URI to web UI URL."""
    if not repo_uri:
        return None
    if repo_uri.startswith("aim://"):
        base_url = "http://" + repo_uri[len("aim://"):]
    else:
        base_url = repo_uri
    if run_hash:
        return f"{base_url}/runs/{run_hash}"
    return base_url


def list_experiments() -> List[Dict[str, Any]]:
    """List experiments from active backend."""
    backend = DEFAULT_TRACKING_BACKEND
    if backend == "aim" and AIM_AVAILABLE:
        # Aim experiments are implicit (experiment is a Run attribute)
        return []
    if backend == "mlflow" and MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(DEFAULT_MLFLOW_URI)
            client = MlflowClient()
            return [{"name": e.name, "id": e.experiment_id} for e in client.search_experiments()]
        except Exception:
            return []
    return []


def get_best_run(experiment: str, metric: str = "loss", mode: str = "min") -> Optional[Dict[str, Any]]:
    """Get best run from experiment (MLflow only for now)."""
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_URI)
        exp = mlflow.get_experiment_by_name(experiment)
        if not exp:
            return None
        order = "ASC" if mode == "min" else "DESC"
        runs = mlflow.search_runs([exp.experiment_id], order_by=[f"metrics.{metric} {order}"], max_results=1)
        if runs.empty:
            return None
        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "metrics": {k.replace("metrics.", ""): v for k, v in best.items() if k.startswith("metrics.")},
            "params": {k.replace("params.", ""): v for k, v in best.items() if k.startswith("params.")},
        }
    except Exception:
        return None


# =============================================================================
# Tracking Helpers
# =============================================================================


def generate_run_name(run_type: str, model_name: Optional[str], dataset_name: Optional[str]) -> str:
    parts = [run_type or DEFAULT_RUN_TYPE]
    if model_name:
        parts.append(model_name)
    if dataset_name:
        parts.append(dataset_name)
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "-".join(parts)


def _ensure_plain_dict(data: Any) -> Dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if is_dataclass(data):
        return asdict(data)
    return {}


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value
    return items


def _stringify_param(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class MetricPreference:
    key: str
    weight: float
    higher_is_better: bool = True


DEFAULT_METRIC_PREFERENCES: List[MetricPreference] = [
    MetricPreference("eval/accuracy", weight=0.6, higher_is_better=True),
    MetricPreference("val/accuracy", weight=0.5, higher_is_better=True),
    MetricPreference("perf/tokens_per_s", weight=0.2, higher_is_better=True),
    MetricPreference("perf/latency_ms", weight=0.15, higher_is_better=False),
    MetricPreference("model/size_mb", weight=0.05, higher_is_better=False),
]


def compute_composite_score(
    metrics: Dict[str, float],
    preferences: Optional[List[MetricPreference]] = None,
) -> Optional[float]:
    prefs = preferences or DEFAULT_METRIC_PREFERENCES
    if not metrics or not prefs:
        return None
    score = 0.0
    total_weight = 0.0
    for pref in prefs:
        value = metrics.get(pref.key)
        if value is None:
            continue
        normalized = _normalize_metric(value, pref.higher_is_better)
        score += pref.weight * normalized
        total_weight += abs(pref.weight)
    if total_weight == 0.0:
        return None
    return score / total_weight


def _normalize_metric(value: float, higher_is_better: bool) -> float:
    safe_value = max(value, 1e-9)
    transformed = math.log1p(safe_value)
    return transformed if higher_is_better else -transformed
