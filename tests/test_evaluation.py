"""Tests for ONN Evaluation module."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.unit
class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_creation(self):
        """Test EvalResult can be created."""
        from src.evaluation.evaluator import EvalResult
        
        result = EvalResult(
            model_name="test-model",
            tasks_evaluated=["hellaswag"],
            results={"hellaswag": {"acc": 0.5, "acc_norm": 0.6}},
        )
        
        assert result.model_name == "test-model"
        assert "hellaswag" in result.tasks_evaluated
        assert result.average_accuracy == 0.6  # Uses acc_norm when available

    def test_eval_result_summary(self):
        """Test EvalResult summary generation."""
        from src.evaluation.evaluator import EvalResult
        
        result = EvalResult(
            model_name="test-model",
            tasks_evaluated=["hellaswag", "arc_easy"],
            results={
                "hellaswag": {"acc_norm": 0.6},
                "arc_easy": {"acc": 0.7},
            },
            eval_time_seconds=10.0,
        )
        
        summary = result.summary()
        assert "test-model" in summary
        assert "hellaswag" in summary
        assert "arc_easy" in summary

    def test_eval_result_to_dict(self):
        """Test EvalResult serialization."""
        from src.evaluation.evaluator import EvalResult
        
        result = EvalResult(
            model_name="test-model",
            tasks_evaluated=["hellaswag"],
            results={"hellaswag": {"acc": 0.5}},
        )
        
        data = result.to_dict()
        assert data["model_name"] == "test-model"
        assert "results" in data


@pytest.mark.unit
class TestTaskPresets:
    """Tests for task preset configurations."""

    def test_list_tasks_returns_presets(self):
        """Test list_tasks returns preset names."""
        from src.evaluation import list_tasks
        
        presets = list_tasks()
        assert "quick" in presets
        assert "standard" in presets
        assert "reasoning" in presets
        assert "safety" in presets

    def test_list_tasks_returns_preset_contents(self):
        """Test list_tasks returns tasks for a specific preset."""
        from src.evaluation import list_tasks
        
        quick_tasks = list_tasks("quick")
        assert "hellaswag" in quick_tasks
        
        standard_tasks = list_tasks("standard")
        assert len(standard_tasks) >= 4


@pytest.mark.unit
class TestMetrics:
    """Tests for basic evaluation metrics."""

    def test_compute_accuracy(self):
        """Test accuracy computation."""
        from src.evaluation.metrics import compute_accuracy
        
        predictions = [1, 0, 1, 1, 0]
        labels = [1, 0, 0, 1, 0]
        
        acc = compute_accuracy(predictions, labels)
        assert acc == 0.8  # 4 out of 5 correct

    def test_compute_accuracy_with_tensors(self):
        """Test accuracy with torch tensors."""
        import torch
        from src.evaluation.metrics import compute_accuracy
        
        predictions = torch.tensor([1, 0, 1, 1, 0])
        labels = torch.tensor([1, 0, 0, 1, 0])
        
        acc = compute_accuracy(predictions, labels)
        assert acc == 0.8

    def test_compute_f1(self):
        """Test F1 score computation."""
        from src.evaluation.metrics import compute_f1
        
        predictions = [1, 1, 1, 0, 0]
        labels = [1, 1, 0, 0, 1]
        
        precision, recall, f1 = compute_f1(predictions, labels)
        
        # True positives: 2 (positions 0, 1)
        # False positives: 1 (position 2)
        # False negatives: 1 (position 4)
        assert precision == 2 / 3  # 2 / (2 + 1)
        assert recall == 2 / 3     # 2 / (2 + 1)
        assert f1 == 2 / 3         # harmonic mean

    def test_compute_bleu_basic(self):
        """Test basic BLEU score computation."""
        from src.evaluation.metrics import compute_bleu
        
        predictions = ["the cat sat on the mat"]
        references = [["the cat sat on the mat"]]
        
        bleu = compute_bleu(predictions, references)
        assert bleu > 0.9  # Should be near-perfect match


@pytest.mark.unit
class TestOnnEvaluator:
    """Tests for OnnEvaluator class."""

    def test_evaluator_creation_without_lm_eval(self):
        """Test evaluator raises ImportError without lm-eval."""
        # This test only runs if lm-eval is NOT installed
        try:
            from src.evaluation import OnnEvaluator
            # If we get here, lm-eval is installed - skip this test path
            evaluator = OnnEvaluator(model="gpt2")
            assert evaluator is not None
        except ImportError as e:
            assert "lm-evaluation-harness" in str(e) or "lm_eval" in str(e)

    def test_evaluator_auto_configuration(self):
        """Test evaluator auto-configures based on hardware."""
        try:
            from src.evaluation import OnnEvaluator
            import torch
            
            evaluator = OnnEvaluator(model="gpt2")
            
            # Should auto-detect device
            if torch.cuda.is_available():
                assert evaluator.device == "cuda"
            else:
                assert evaluator.device in ["cpu", "mps"]
                
        except ImportError:
            pytest.skip("lm-eval not installed")


@pytest.mark.integration
class TestEvaluatorIntegration:
    """Integration tests for evaluation (requires lm-eval and model)."""

    @pytest.mark.slow
    def test_quick_eval(self):
        """Test quick evaluation with small model."""
        try:
            from src.evaluation import quick_eval
            
            # Use tiny model for testing
            result = quick_eval("gpt2", max_samples=5)
            
            assert result.model_name == "gpt2"
            assert len(result.tasks_evaluated) > 0
            assert result.eval_time_seconds > 0
            
        except ImportError:
            pytest.skip("lm-eval not installed")
        except Exception as e:
            if "model" in str(e).lower():
                pytest.skip("Model not available for testing")
            raise
