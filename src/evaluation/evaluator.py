"""ONN Evaluator - Intelligent model evaluation wrapper.

Wraps lm-evaluation-harness for production-grade evaluation.
Automatically configures batch size, precision, and device settings
based on hardware profiler.

Why lm-evaluation-harness?
- Battle-tested by thousands of researchers
- 200+ evaluation tasks supported
- Standardized, reproducible results
- Active maintenance by EleutherAI

ONN adds:
- Hardware-aware auto-configuration
- Simplified API
- Integration with ONN experiment tracking
- Quick-eval mode for sanity checks
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# Check for lm-evaluation-harness
try:
    import lm_eval
    from lm_eval import evaluator as lm_evaluator
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False

# ONN imports
try:
    from ..orchestration import get_hardware_profile, HardwareProfile
    ONN_ORCHESTRATION = True
except ImportError:
    ONN_ORCHESTRATION = False


# ============================================================================
# Quick Eval Task Sets (curated for different use cases)
# ============================================================================

QUICK_TASKS = ["hellaswag"]  # ~2 min, good sanity check

STANDARD_TASKS = [
    "hellaswag",      # Common sense reasoning
    "arc_easy",       # Science QA (easy)
    "arc_challenge",  # Science QA (hard)
    "winogrande",     # Coreference resolution
]

REASONING_TASKS = [
    "gsm8k",          # Grade school math
    "mathqa",         # Math reasoning
    "logiqa",         # Logical reasoning
]

KNOWLEDGE_TASKS = [
    "triviaqa",       # Trivia questions
    "naturalqs",      # Natural questions (Wikipedia)
]

SAFETY_TASKS = [
    "truthfulqa_mc1", # Truthfulness (single)
    "truthfulqa_mc2", # Truthfulness (multi)
]

TASK_PRESETS = {
    "quick": QUICK_TASKS,
    "standard": STANDARD_TASKS,
    "reasoning": REASONING_TASKS,
    "knowledge": KNOWLEDGE_TASKS,
    "safety": SAFETY_TASKS,
    "full": STANDARD_TASKS + REASONING_TASKS + SAFETY_TASKS,
}


@dataclass
class EvalResult:
    """Results from an evaluation run."""
    
    model_name: str
    tasks_evaluated: List[str]
    results: Dict[str, Dict[str, float]]  # task -> metric -> score
    
    # Performance metadata
    total_samples: int = 0
    eval_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    
    # Configuration used
    batch_size: int = 1
    device: str = "cpu"
    precision: str = "fp16"
    
    # Aggregate scores
    average_accuracy: float = 0.0
    
    # Errors if any
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute aggregate metrics."""
        if self.results:
            accuracies = []
            for task_results in self.results.values():
                # lm-eval-harness uses 'acc' or 'acc_norm' as key
                acc = task_results.get("acc_norm") or task_results.get("acc") or 0.0
                accuracies.append(acc)
            if accuracies:
                self.average_accuracy = sum(accuracies) / len(accuracies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "tasks_evaluated": self.tasks_evaluated,
            "results": self.results,
            "total_samples": self.total_samples,
            "eval_time_seconds": self.eval_time_seconds,
            "samples_per_second": self.samples_per_second,
            "batch_size": self.batch_size,
            "device": self.device,
            "precision": self.precision,
            "average_accuracy": self.average_accuracy,
            "errors": self.errors,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Evaluation Results for: {self.model_name}",
            f"=" * 60,
            f"Average Accuracy: {self.average_accuracy:.2%}",
            f"Tasks Evaluated: {len(self.tasks_evaluated)}",
            f"Time: {self.eval_time_seconds:.1f}s ({self.samples_per_second:.1f} samples/sec)",
            "",
            "Per-Task Results:",
        ]
        
        for task, metrics in self.results.items():
            acc = metrics.get("acc_norm") or metrics.get("acc") or 0.0
            lines.append(f"  {task}: {acc:.2%}")
        
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  ⚠️ {err}")
        
        return "\n".join(lines)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class OnnEvaluator:
    """Hardware-aware model evaluator wrapping lm-evaluation-harness.
    
    Automatically configures evaluation based on available hardware:
    - Batch size based on VRAM
    - Precision based on GPU capabilities
    - Device selection
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        device: str = "auto",
        batch_size: Union[int, str] = "auto",
        precision: str = "auto",
        max_samples: Optional[int] = None,
        trust_remote_code: bool = True,
    ):
        """Initialize evaluator.
        
        Args:
            model: Model name/path (HuggingFace ID or local path).
            device: Device to use ("auto", "cuda", "cpu").
            batch_size: Batch size ("auto" to determine from hardware).
            precision: Precision ("auto", "fp16", "bf16", "fp32").
            max_samples: Limit samples per task (for quick testing).
            trust_remote_code: Trust remote code for HF models.
        """
        if not LM_EVAL_AVAILABLE:
            raise ImportError(
                "lm-evaluation-harness not installed. "
                "Install with: pip install lm-eval"
            )
        
        self.model_name = model
        self.max_samples = max_samples
        self.trust_remote_code = trust_remote_code
        
        # Auto-configure from hardware
        self._configure(device, batch_size, precision)
    
    def _configure(self, device: str, batch_size: Union[int, str], precision: str):
        """Configure evaluator based on hardware."""
        # Get hardware profile if available
        profile = None
        if ONN_ORCHESTRATION:
            try:
                profile = get_hardware_profile()
            except Exception:
                pass
        
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Batch size auto-configuration
        if batch_size == "auto":
            if profile and profile.has_gpu:
                vram_gb = profile.available_vram_mb / 1024
                # Conservative estimate: larger VRAM = larger batch
                if vram_gb >= 24:
                    self.batch_size = 32
                elif vram_gb >= 16:
                    self.batch_size = 16
                elif vram_gb >= 8:
                    self.batch_size = 8
                elif vram_gb >= 4:
                    self.batch_size = 4
                else:
                    self.batch_size = 1
            else:
                self.batch_size = 1
        else:
            self.batch_size = int(batch_size)
        
        # Precision auto-configuration
        if precision == "auto":
            if profile and profile.supports_bf16:
                self.precision = "bf16"
            elif self.device == "cuda":
                self.precision = "fp16"
            else:
                self.precision = "fp32"
        else:
            self.precision = precision
    
    def evaluate(
        self,
        tasks: Union[str, List[str]] = "standard",
        model: Optional[str] = None,
        num_fewshot: int = 0,
        output_path: Optional[str] = None,
    ) -> EvalResult:
        """Run evaluation on specified tasks.
        
        Args:
            tasks: Task names or preset ("quick", "standard", "reasoning", etc.)
            model: Override model (uses instance default if None).
            num_fewshot: Number of few-shot examples.
            output_path: Save results to this path.
            
        Returns:
            EvalResult with scores and metadata.
        """
        model_name = model or self.model_name
        if not model_name:
            raise ValueError("No model specified. Pass model= or set in constructor.")
        
        # Resolve task preset
        if isinstance(tasks, str):
            if tasks in TASK_PRESETS:
                task_list = TASK_PRESETS[tasks]
            else:
                task_list = [tasks]
        else:
            task_list = tasks
        
        print(f"🔬 Evaluating: {model_name}")
        print(f"   Tasks: {', '.join(task_list)}")
        print(f"   Device: {self.device}, Batch Size: {self.batch_size}, Precision: {self.precision}")
        
        start_time = time.time()
        errors = []
        
        try:
            # Run evaluation using lm-evaluation-harness
            results = lm_evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_name},trust_remote_code={self.trust_remote_code}",
                tasks=task_list,
                num_fewshot=num_fewshot,
                batch_size=self.batch_size,
                device=self.device,
                limit=self.max_samples,
            )
            
            # Extract results
            task_results = {}
            total_samples = 0
            
            for task in task_list:
                if task in results.get("results", {}):
                    task_data = results["results"][task]
                    task_results[task] = {
                        k: v for k, v in task_data.items() 
                        if isinstance(v, (int, float))
                    }
                    # Count samples
                    if "samples" in results.get("samples", {}):
                        total_samples += len(results["samples"].get(task, []))
            
        except Exception as e:
            errors.append(str(e))
            task_results = {}
            total_samples = 0
        
        eval_time = time.time() - start_time
        
        result = EvalResult(
            model_name=model_name,
            tasks_evaluated=task_list,
            results=task_results,
            total_samples=total_samples,
            eval_time_seconds=eval_time,
            samples_per_second=total_samples / eval_time if eval_time > 0 else 0,
            batch_size=self.batch_size,
            device=self.device,
            precision=self.precision,
            errors=errors,
        )
        
        # Save if requested
        if output_path:
            result.save(output_path)
            print(f"💾 Results saved to: {output_path}")
        
        print(result.summary())
        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def evaluate_model(
    model: str,
    tasks: Union[str, List[str]] = "standard",
    batch_size: Union[int, str] = "auto",
    device: str = "auto",
    num_fewshot: int = 0,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
) -> EvalResult:
    """Evaluate a model on benchmark tasks.
    
    One-liner for model evaluation:
    
        results = evaluate_model("meta-llama/Llama-2-7b")
    
    Args:
        model: HuggingFace model ID or local path.
        tasks: Tasks to run ("quick", "standard", or list of task names).
        batch_size: Batch size ("auto" for hardware-optimized).
        device: Device ("auto", "cuda", "cpu").
        num_fewshot: Few-shot examples (0 for zero-shot).
        max_samples: Limit samples for quick testing.
        output_path: Save results JSON to this path.
        
    Returns:
        EvalResult with scores and metadata.
    """
    evaluator = OnnEvaluator(
        model=model,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    return evaluator.evaluate(tasks=tasks, num_fewshot=num_fewshot, output_path=output_path)


def quick_eval(model: str, max_samples: int = 100) -> EvalResult:
    """Quick sanity check evaluation (~2 minutes).
    
    Runs HellaSwag with limited samples for fast iteration.
    
        results = quick_eval("./my_checkpoint")
        print(f"Sanity check accuracy: {results.average_accuracy:.2%}")
    
    Args:
        model: Model path or HuggingFace ID.
        max_samples: Number of samples to evaluate.
        
    Returns:
        EvalResult with quick benchmark scores.
    """
    evaluator = OnnEvaluator(
        model=model,
        max_samples=max_samples,
    )
    return evaluator.evaluate(tasks="quick")


def list_tasks(preset: Optional[str] = None) -> List[str]:
    """List available evaluation tasks.
    
    Args:
        preset: If provided, lists tasks in that preset.
                Otherwise lists all preset names.
    
    Returns:
        List of task names or preset names.
    """
    if preset:
        return TASK_PRESETS.get(preset, [])
    return list(TASK_PRESETS.keys())
