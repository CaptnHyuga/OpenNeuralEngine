"""ONN Evaluation Module - Universal model evaluation.

Wraps lm-evaluation-harness (EleutherAI) for comprehensive model benchmarking.
Provides a dead-simple interface: one command evaluates your model on standard benchmarks.

Usage:
    from src.evaluation import evaluate_model, quick_eval
    
    # Quick eval (fast sanity check)
    results = quick_eval("./my_checkpoint")
    
    # Full benchmark suite
    results = evaluate_model(
        model="./my_checkpoint",
        tasks=["hellaswag", "arc_easy", "truthfulqa"],
        batch_size="auto"
    )
"""

from .evaluator import (
    OnnEvaluator,
    EvalResult,
    evaluate_model,
    quick_eval,
    list_tasks,
)

from .metrics import (
    compute_perplexity,
    compute_accuracy,
    compute_f1,
)

__all__ = [
    # Main evaluator
    "OnnEvaluator",
    "EvalResult",
    "evaluate_model",
    "quick_eval",
    "list_tasks",
    # Metrics
    "compute_perplexity",
    "compute_accuracy",
    "compute_f1",
]
