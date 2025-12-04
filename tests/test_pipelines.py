"""Tests for training pipelines v2 and v3."""

import json
import numpy as np
import pytest
import sys
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestPipelineV2Config:
    """Tests for pipeline v2 training config."""
    
    def test_training_config_defaults(self):
        """Test default config values."""
        from src.training.pipeline_v2 import TrainingConfig
        
        config = TrainingConfig()
        assert config.epochs == 1
        assert config.batch_size == 8
        assert config.max_seq_len == 128
        assert config.learning_rate == 1e-4
        assert config.lora_rank == 8
        assert config.val_split == 0.1
    
    def test_training_config_to_dict(self):
        """Test config serialization."""
        from src.training.pipeline_v2 import TrainingConfig
        
        config = TrainingConfig(epochs=5, batch_size=16)
        d = config.to_dict()
        
        assert d["epochs"] == 5
        assert d["batch_size"] == 16
        assert "model_path" in d


@pytest.mark.unit  
class TestPipelineV3Config:
    """Tests for pipeline v3 optimized config."""
    
    def test_config_defaults(self):
        """Test v3 default config."""
        from src.training.pipeline_v3 import TrainingConfigV3
        
        config = TrainingConfigV3()
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.use_amp == True
    
    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        from src.training.pipeline_v3 import TrainingConfigV3
        
        config = TrainingConfigV3(batch_size=4, gradient_accumulation_steps=8)
        assert config.effective_batch_size == 32


@pytest.mark.unit
class TestDatasetLoader:
    """Tests for dataset loading."""
    
    def test_load_jsonl(self, tmp_path):
        """Test loading JSONL datasets."""
        from src.training.pipeline_v2 import DatasetLoader
        
        # Create test file
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"text": "Hello world"}\n')
            f.write('{"text": "Test sample"}\n')
        
        data = DatasetLoader.load(str(jsonl_file), max_samples=10)
        assert len(data) == 2
        assert data[0]["text"] == "Hello world"
    
    def test_extract_text_simple(self):
        """Test text extraction from simple dict."""
        from src.training.pipeline_v2 import DatasetLoader
        
        item = {"text": "Hello world"}
        text = DatasetLoader.extract_text(item)
        assert text == "Hello world"
    
    def test_extract_text_prompt(self):
        """Test text extraction from prompt field."""
        from src.training.pipeline_v2 import DatasetLoader
        
        item = {"prompt": "What is 2+2?", "response": "4"}
        text = DatasetLoader.extract_text(item)
        assert "2+2" in text
    
    def test_extract_text_multimodal(self):
        """Test text extraction from multimodal prompt."""
        from src.training.pipeline_v2 import DatasetLoader
        
        item = {
            "prompt": [
                {"text": "Describe this image"},
                {"image": "base64..."},
                {"text": "in detail"}
            ]
        }
        text = DatasetLoader.extract_text(item)
        assert "Describe" in text
        assert "detail" in text


@pytest.mark.unit
class TestLoRALayer:
    """Tests for LoRA layer implementation."""
    
    def test_lora_layer_creation(self):
        """Test LoRA layer initialization."""
        import torch
        from src.training.pipeline_v2 import LoRALayer
        
        layer = LoRALayer(in_features=512, out_features=512, rank=8)
        assert layer.rank == 8
        assert layer.scale == 16 / 8  # alpha/rank
    
    def test_lora_layer_forward(self):
        """Test LoRA forward pass."""
        import torch
        from src.training.pipeline_v2 import LoRALayer
        
        layer = LoRALayer(in_features=128, out_features=128, rank=4)
        x = torch.randn(2, 10, 128)
        
        output = layer(x)
        assert output.shape == x.shape
    
    def test_lora_layer_zero_init(self):
        """Test that LoRA B is zero-initialized."""
        import torch
        from src.training.pipeline_v2 import LoRALayer
        
        layer = LoRALayer(in_features=64, out_features=64, rank=4)
        # B should be zeros, so initial output should equal input
        x = torch.randn(1, 5, 64)
        output = layer(x)
        
        # With dropout=0.1, there's some variance, but B=0 means lora_out ≈ 0
        # So output should be close to input
        torch.testing.assert_close(output.float(), x.float(), rtol=0.2, atol=0.5)


@pytest.mark.unit
class TestCosineScheduler:
    """Tests for learning rate scheduler."""
    
    def test_scheduler_warmup(self):
        """Test warmup phase."""
        import torch
        from src.training.pipeline_v2 import CosineScheduler
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineScheduler(optimizer, warmup_steps=10, total_steps=100)
        
        # At step 0, lr should be near 0
        assert optimizer.param_groups[0]['lr'] == 1e-3
        
        # After 5 steps, should be at ~50% of base LR
        for _ in range(5):
            scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        assert 0.4e-3 < lr < 0.6e-3
    
    def test_scheduler_cosine_decay(self):
        """Test cosine decay after warmup."""
        import torch
        from src.training.pipeline_v2 import CosineScheduler
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineScheduler(optimizer, warmup_steps=10, total_steps=100)
        
        # Complete warmup
        for _ in range(10):
            scheduler.step()
        
        # At end of warmup, should be at full LR
        assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3, rel=0.1)
        
        # At end of training, should be at min LR (10% of base)
        for _ in range(90):
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < 0.2e-3


@pytest.mark.unit
class TestCheckpointResume:
    """Test checkpoint save/load for resume training."""
    
    def test_save_checkpoint_creates_files(self, tmp_path):
        """Test that save_checkpoint creates necessary files."""
        # Create minimal mock components
        import torch.nn as nn
        from src.training.pipeline_v3 import TrainingConfigV3
        
        class MockLora(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
        
        class MockTrainer:
            def __init__(self, output_dir):
                self.lora = MockLora()
                self.output_dir = output_dir
                self.config = TrainingConfigV3()
                self.optimizer = torch.optim.AdamW(self.lora.parameters(), lr=1e-4)
                self.scheduler = None
                self.scaler = None
                self.metrics = {"train_loss": [1.0, 0.9, 0.8], "val_loss": [1.1, 0.95]}
                
                # Import the save method
                from src.training.pipeline_v3 import OptimizedTrainer
                self.save_checkpoint = OptimizedTrainer.save_checkpoint.__get__(self, MockTrainer)
        
        trainer = MockTrainer(tmp_path)
        trainer.save_checkpoint("test", epoch=1, global_step=50, optimizer_step=12)
        
        checkpoint_dir = tmp_path / "checkpoints"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "test.safetensors").exists()
        assert (checkpoint_dir / "test_config.json").exists()
        assert (checkpoint_dir / "test_state.pt").exists()
        
        # Verify state contents
        state = torch.load(checkpoint_dir / "test_state.pt")
        assert state["epoch"] == 1
        assert state["global_step"] == 50
        assert state["optimizer_step"] == 12
        assert "optimizer_state_dict" in state
        assert "metrics" in state
    
    def test_config_includes_resume(self):
        """Test TrainingConfigV3 includes resume_from field."""
        from src.training.pipeline_v3 import TrainingConfigV3
        
        config = TrainingConfigV3(resume_from="latest")
        assert config.resume_from == "latest"
        
        config_dict = config.to_dict()
        assert "resume_from" in config_dict
        assert config_dict["resume_from"] == "latest"


@pytest.mark.unit
class TestLRFinder:
    """Tests for Learning Rate Finder."""
    
    def test_lr_finder_import(self):
        """Test LRFinder can be imported."""
        from src.training.lr_finder import LRFinder
        assert LRFinder is not None
    
    def test_lr_finder_analyze_synthetic(self):
        """Test analysis with synthetic data."""
        from src.training.lr_finder import LRFinder
        
        # Create mock trainer
        class MockTrainer:
            pass
        
        finder = LRFinder(MockTrainer())
        
        # Create synthetic history (typical LR finder curve)
        lrs = np.logspace(-7, 0, 100)  # 1e-7 to 1.0
        # Loss typically decreases then increases
        losses = 10 * np.exp(-5 * lrs) + 0.5 * lrs + 0.5
        # Add noise
        losses += np.random.randn(100) * 0.1
        
        finder.history = {
            "lr": lrs.tolist(),
            "loss": losses.tolist(),
            "smoothed_loss": losses.tolist(),
        }
        
        result = finder._analyze_results()
        
        assert "optimal_lr" in result
        assert "min_loss" in result
        assert "suggested_min_lr" in result
        assert result["optimal_lr"] > 0
        assert result["suggested_min_lr"] < result["optimal_lr"]


@pytest.mark.unit
class TestEarlyStopping:
    """Tests for Early Stopping."""
    
    def test_early_stopping_import(self):
        """Test EarlyStopping can be imported."""
        from src.training.early_stopping import EarlyStopping, ReduceLROnPlateau
        assert EarlyStopping is not None
        assert ReduceLROnPlateau is not None
    
    def test_early_stopping_improvement(self):
        """Test that improvements reset counter."""
        from src.training.early_stopping import EarlyStopping
        
        stopper = EarlyStopping(patience=3, verbose=False)
        
        # Improving losses
        assert stopper(1.0) == False
        assert stopper(0.9) == False
        assert stopper(0.8) == False
        
        assert stopper.counter == 0
        assert stopper.best_loss == 0.8
    
    def test_early_stopping_trigger(self):
        """Test early stopping triggers after patience exhausted."""
        from src.training.early_stopping import EarlyStopping
        
        stopper = EarlyStopping(patience=3, verbose=False)
        
        # Initial improvement
        assert stopper(1.0) == False
        
        # No improvement for patience epochs
        assert stopper(1.1) == False
        assert stopper(1.2) == False
        assert stopper(1.3) == True  # Should trigger
        
        assert stopper.counter == 3
    
    def test_early_stopping_min_delta(self):
        """Test min_delta threshold."""
        from src.training.early_stopping import EarlyStopping
        
        stopper = EarlyStopping(patience=3, min_delta=0.1, verbose=False)
        
        assert stopper(1.0) == False
        # 0.95 is not 0.1 better than 1.0, so no improvement
        assert stopper(0.95) == False
        assert stopper.counter == 1
        
        # 0.85 IS 0.1+ better than 1.0
        stopper.counter = 0
        stopper.best_loss = 1.0
        assert stopper(0.85) == False
        assert stopper.counter == 0
    
    def test_reduce_lr_on_plateau(self):
        """Test LR reduction on plateau."""
        from src.training.early_stopping import ReduceLROnPlateau
        
        # Create mock optimizer
        class MockOptimizer:
            param_groups = [{"lr": 0.1}]
        
        opt = MockOptimizer()
        scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=False)
        
        # Improving
        scheduler(1.0)
        scheduler(0.9)
        assert opt.param_groups[0]["lr"] == 0.1  # No reduction yet
        
        # Plateau
        scheduler(0.95)  # counter=1
        scheduler(0.96)  # counter=2, trigger
        
        assert opt.param_groups[0]["lr"] == 0.05  # Reduced by factor


@pytest.mark.unit
class TestLoRAMerger:
    """Tests for LoRA weight merging."""
    
    def test_merge_lora_import(self):
        """Test LoRAMerger can be imported."""
        from src.training.merge_lora import LoRAMerger, export_lora_adapter
        assert LoRAMerger is not None
        assert export_lora_adapter is not None
    
    def test_export_lora_adapter(self, tmp_path):
        """Test exporting LoRA adapter."""
        from src.training.merge_lora import export_lora_adapter
        from safetensors.torch import save_file as save_safetensors
        
        # Create mock checkpoint
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Mock weights
        weights = {"layers.0.lora_A": torch.randn(8, 5120)}
        save_safetensors(weights, str(checkpoint_dir / "best.safetensors"))
        
        # Mock config
        config = {"lora_rank": 8, "lora_alpha": 16, "target_layers": [0]}
        with open(checkpoint_dir / "best_config.json", 'w') as f:
            json.dump(config, f)
        
        # Export
        output_path = tmp_path / "exported"
        export_lora_adapter(checkpoint_dir, output_path, "best")
        
        # Verify
        assert (output_path / "adapter_model.safetensors").exists()
        assert (output_path / "adapter_config.json").exists()
        
        with open(output_path / "adapter_config.json") as f:
            exported_config = json.load(f)
        assert exported_config["r"] == 8
        assert exported_config["lora_alpha"] == 16


@pytest.mark.integration
class TestTrainCLI:
    """Integration tests for train.py CLI."""
    
    def test_cli_help(self):
        """Test CLI help message."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "--dataset" in result.stdout
        assert "--pipeline" in result.stdout
        assert "--auto" in result.stdout
        assert "--resume" in result.stdout


@pytest.mark.integration
class TestInferenceCLI:
    """Integration tests for inference.py CLI."""
    
    def test_cli_help(self):
        """Test inference CLI help."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "inference.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--prompt" in result.stdout
        assert "--interactive" in result.stdout
