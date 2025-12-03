#!/usr/bin/env python3
"""
SNN Model Evaluation CLI

Evaluate models with comprehensive metrics and automatic experiment tracking via Aim.

Usage:
    python evaluate.py                    # Evaluate default model
    python evaluate.py --model best       # Evaluate best checkpoint
    python evaluate.py --suite all        # Run all evaluation suites
    python evaluate.py --output eval.json # Save results to JSON

Features:
- Automatic Aim experiment tracking (dockerized)
- Comprehensive evaluation suites (basic, reasoning, knowledge)
- Platform-aware (works on Windows/Linux, CPU/GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


# =============================================================================
# Platform Detection
# =============================================================================

def get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


# =============================================================================
# Model Resolution
# =============================================================================

def resolve_model_path(model_spec: str) -> Path:
    """Resolve model specification to a local checkpoint path."""
    if model_spec in ("best", "latest", "champion"):
        print("ℹ️ Using default checkpoint directory: src/Core_Models/Save")
        model_spec = "src/Core_Models/Save"
    
    # Local path
    path = Path(model_spec)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    
    # Validate path contains checkpoint files
    if path.is_dir():
        checkpoint_files = list(path.glob("*.safetensors"))
        if not checkpoint_files:
            raise ValueError(
                f"Model directory has no .safetensors files: {path}\n"
                "Ensure the directory contains trained model checkpoints."
            )
    elif path.is_file():
        if not path.suffix == ".safetensors":
            raise ValueError(
                f"Model file must be .safetensors format: {path}\n"
                "Unsafe pickle-based checkpoints are not supported."
            )
    
    return path


# =============================================================================
# Experiment Tracking Integration
# =============================================================================

class EvalTracker:
    """Evaluation tracking using the configured backend (Aim default)."""
    
    def __init__(
        self,
        experiment: str = "snn",
        run_name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.tracker = None
        self._run_url = None
        
        if not enabled:
            return
        
        try:
            from src.Core_Models.experiment_tracking import ExperimentTracker
            
            self.tracker = ExperimentTracker(
                experiment=experiment,
                run_name=run_name or f"eval-{int(time.time())}",
                config={"run_type": "evaluation"},
            )
            self._run_url = self.tracker.run_url
            
        except Exception as e:
            print(f"⚠️ Tracking disabled: {e}")
            self.enabled = False
    
    @property
    def run_url(self) -> Optional[str]:
        return self._run_url
    
    def log_params(self, params: Dict[str, Any]):
        if self.tracker:
            try:
                self.tracker.log_params(params)
            except Exception:
                pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.tracker:
            try:
                self.tracker.log_metrics(metrics, step=step)
            except Exception:
                pass
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.tracker:
            try:
                self.tracker.log_artifact(local_path, artifact_path)
            except Exception:
                pass
    
    def finish(self):
        if self.tracker:
            try:
                self.tracker.unwatch()
            except Exception:
                pass


# =============================================================================
# Evaluation Runner
# =============================================================================

def run_evaluation(
    model_path: Path,
    device: torch.device,
    suites: List[str],
    max_tokens: int = 128,
    temperature: float = 0.7,
    tracker: Optional[EvalTracker] = None,
) -> Dict[str, Any]:
    """Run evaluation and return results."""
    from src.Core_Models.sota_evaluator import (
        SOTAEvaluator,
        get_basic_tasks,
        get_reasoning_tasks,
        get_knowledge_tasks,
        EvalSuiteResult,
    )
    
    print(f"📦 Loading model from {model_path}...")
    
    try:
        evaluator = SOTAEvaluator.from_checkpoint(
            model_path,
            device=str(device),
            tracker=tracker.tracker if tracker else None,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Model not found: {e}") from e
    
    # Map suite names to task lists
    suite_map = {
        "basic": ("Basic Tasks", get_basic_tasks()),
        "reasoning": ("Reasoning Tasks", get_reasoning_tasks()),
        "knowledge": ("Knowledge Tasks", get_knowledge_tasks()),
    }
    
    # Determine which suites to run
    if "all" in suites:
        suites_to_run = list(suite_map.keys())
    else:
        suites_to_run = [s for s in suites if s in suite_map]
    
    if not suites_to_run:
        suites_to_run = ["basic"]  # Default
    
    results: Dict[str, EvalSuiteResult] = {}
    all_metrics: Dict[str, float] = {}
    
    for suite_key in suites_to_run:
        suite_name, tasks = suite_map[suite_key]
        print(f"\n🔬 Running {suite_name}...")
        
        result = evaluator.evaluate_suite(
            tasks=tasks,
            suite_name=suite_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results[suite_key] = result
        
        # Log metrics
        prefix = f"eval/{suite_key}"
        suite_metrics = {
            f"{prefix}/accuracy": result.accuracy,
            f"{prefix}/tokens_per_second": result.tokens_per_second,
            f"{prefix}/avg_latency_s": result.avg_latency_s,
            f"{prefix}/passed": result.passed_tasks,
            f"{prefix}/total": result.total_tasks,
        }
        all_metrics.update(suite_metrics)
        
        if tracker:
            tracker.log_metrics(suite_metrics)
    
    # Compute aggregate metrics
    total_passed = sum(r.passed_tasks for r in results.values())
    total_tasks = sum(r.total_tasks for r in results.values())
    overall_accuracy = total_passed / total_tasks if total_tasks > 0 else 0.0
    
    all_metrics["eval/overall_accuracy"] = overall_accuracy
    all_metrics["eval/total_passed"] = total_passed
    all_metrics["eval/total_tasks"] = total_tasks
    
    if tracker:
        tracker.log_metrics({
            "eval/overall_accuracy": overall_accuracy,
            "eval/total_passed": total_passed,
            "eval/total_tasks": total_tasks,
        })
    
    return {
        "results": results,
        "metrics": all_metrics,
        "overall_accuracy": overall_accuracy,
    }


def print_results(eval_output: Dict[str, Any]):
    """Print evaluation results in a nice format."""
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    for suite_key, result in eval_output["results"].items():
        print(f"\n{suite_key.upper()}")
        print("-" * 40)
        print(f"  Accuracy:    {result.accuracy*100:.1f}%")
        print(f"  Passed:      {result.passed_tasks}/{result.total_tasks}")
        print(f"  Speed:       {result.tokens_per_second:.1f} tok/s")
        print(f"  Latency:     {result.avg_latency_s*1000:.1f}ms avg")
        
        # Show per-type breakdown
        if result.results_by_type:
            print("  By type:")
            for task_type, type_metrics in result.results_by_type.items():
                acc = type_metrics.get("accuracy", 0) * 100
                print(f"    {task_type}: {acc:.1f}%")
    
    print("\n" + "=" * 70)
    print(f"OVERALL ACCURACY: {eval_output['overall_accuracy']*100:.1f}%")
    print("=" * 70)


def save_results(eval_output: Dict[str, Any], output_path: Path):
    """Save results to JSON file."""
    serializable = {
        "overall_accuracy": eval_output["overall_accuracy"],
        "metrics": eval_output["metrics"],
        "suites": {},
    }
    
    for suite_key, result in eval_output["results"].items():
        serializable["suites"][suite_key] = {
            "accuracy": result.accuracy,
            "passed_tasks": result.passed_tasks,
            "total_tasks": result.total_tasks,
            "tokens_per_second": result.tokens_per_second,
            "avg_latency_s": result.avg_latency_s,
            "results_by_type": result.results_by_type,
            "individual_results": [
                {
                    "task_name": r.task_name,
                    "passed": r.passed,
                    "score": r.score,
                    "latency_s": r.latency_s,
                    "response": r.response[:200] if r.response else "",  # Truncate
                    "expected": r.expected,
                }
                for r in result.individual_results
            ],
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2))
    print(f"\n📁 Results saved to: {output_path}")


# =============================================================================
# Main CLI
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SNN Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                       # Evaluate default model
  python evaluate.py --model best          # Evaluate best checkpoint
  python evaluate.py --suite all           # Run all evaluation suites
  python evaluate.py --device cpu          # Force CPU evaluation
  python evaluate.py --output results.json # Save results
  python evaluate.py --no-tracking         # Disable experiment tracking
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="src/Core_Models/Save",
        help="Model checkpoint path or directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--suite",
        type=str,
        nargs="+",
        default=["basic"],
        choices=["basic", "reasoning", "knowledge", "all"],
        help="Evaluation suites to run",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable experiment tracking",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="snn",
        help="Experiment name for tracking",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (less output)",
    )
    
    args = parser.parse_args()
    
    # Banner
    if not args.quiet:
        print("=" * 70)
        print("SNN Model Evaluation")
        print("=" * 70)
    
    # Get device
    device = get_device(args.device)
    print(f"🔧 Device: {device}")
    
    # Resolve model path
    try:
        model_path = resolve_model_path(args.model)
        print(f"📦 Model: {model_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Setup tracking
    tracker = None
    if not args.no_tracking:
        tracker = EvalTracker(
            experiment=args.experiment,
            run_name=f"eval-{model_path.name}-{int(time.time())}",
            enabled=True,
        )
        if tracker.run_url:
            print(f"📊 Tracking: {tracker.run_url}")
        
        # Log evaluation config
        tracker.log_params({
            "model_path": str(model_path),
            "device": str(device),
            "suites": ",".join(args.suite),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        })
    
    # Run evaluation
    try:
        eval_output = run_evaluation(
            model_path=model_path,
            device=device,
            suites=args.suite,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tracker=tracker,
        )
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if tracker:
            tracker.finish()
        return 1
    
    # Print results
    if not args.quiet:
        print_results(eval_output)
    
    # Save results
    if args.output:
        save_results(eval_output, Path(args.output))
    
    # Cleanup
    if tracker:
        tracker.finish()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
