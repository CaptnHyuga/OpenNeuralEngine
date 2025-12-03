"""Tests for ONN 2.0 wrappers module."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wrappers.model_loader import UniversalModelLoader, ModelSource, ModelInfo
from src.wrappers.hf_trainer_wrapper import HFTrainerWrapper, TrainingResult
from src.wrappers.quantization_wrapper import QuantizationWrapper, QuantizationType, QuantizationConfig


@pytest.mark.unit
class TestUniversalModelLoader:
    """Tests for universal model loading functionality."""

    def test_loader_creation(self):
        """Test UniversalModelLoader can be instantiated."""
        loader = UniversalModelLoader()
        assert loader is not None

    def test_detect_source_local_weights(self, tmp_path):
        """Test detecting local weight files."""
        loader = UniversalModelLoader()
        
        # Create a dummy weight file
        weight_file = tmp_path / "model.safetensors"
        weight_file.touch()
        
        source = loader.detect_source(str(weight_file))
        assert source == ModelSource.LOCAL_WEIGHTS

    def test_detect_source_huggingface(self):
        """Test detecting HuggingFace model identifiers."""
        loader = UniversalModelLoader()
        
        # Common HF model patterns
        assert loader.detect_source("gpt2") == ModelSource.HUGGINGFACE
        assert loader.detect_source("facebook/opt-125m") == ModelSource.HUGGINGFACE
        assert loader.detect_source("microsoft/DialoGPT-small") == ModelSource.HUGGINGFACE

    def test_detect_source_timm_prefix(self):
        """Test detecting timm models with explicit prefix."""
        loader = UniversalModelLoader()
        
        source = loader.detect_source("timm:resnet50")
        assert source == ModelSource.TIMM

    def test_detect_source_torchvision_prefix(self):
        """Test detecting torchvision models with explicit prefix."""
        loader = UniversalModelLoader()
        
        source = loader.detect_source("torchvision:resnet18")
        assert source == ModelSource.TORCHVISION

    def test_looks_like_hf_model(self):
        """Test HuggingFace model name detection."""
        loader = UniversalModelLoader()
        
        # Should recognize common model families
        assert loader._looks_like_hf_model("gpt2") is True
        assert loader._looks_like_hf_model("bert-base") is True
        assert loader._looks_like_hf_model("llama-7b") is True
        assert loader._looks_like_hf_model("mistral-7b") is True
        
        # Should not match random names
        assert loader._looks_like_hf_model("my_random_model") is False

    def test_available_backends(self):
        """Test backend availability detection."""
        loader = UniversalModelLoader()
        
        # HuggingFace should be available (it's a dependency)
        assert loader._hf_available is True


@pytest.mark.unit
class TestHFTrainerWrapper:
    """Tests for HuggingFace trainer wrapper."""

    def test_wrapper_creation(self):
        """Test HFTrainerWrapper can be instantiated."""
        wrapper = HFTrainerWrapper()
        assert wrapper is not None

    def test_wrapper_has_output_dir(self):
        """Test wrapper has default output directory."""
        wrapper = HFTrainerWrapper()
        assert wrapper.output_dir is not None
        assert isinstance(wrapper.output_dir, Path)

    def test_wrapper_with_custom_output_dir(self, tmp_path):
        """Test wrapper with custom output directory."""
        wrapper = HFTrainerWrapper(output_dir=tmp_path)
        assert wrapper.output_dir == tmp_path

    def test_wrapper_has_monitor(self):
        """Test wrapper has resource monitor."""
        wrapper = HFTrainerWrapper()
        assert wrapper.monitor is not None

    def test_training_result_dataclass(self):
        """Test TrainingResult can be instantiated."""
        result = TrainingResult()
        assert result.success is True
        assert result.epochs_completed == 0
        assert result.final_loss == float("inf")
    
    def test_training_result_summary(self):
        """Test TrainingResult summary generation."""
        result = TrainingResult(
            success=True,
            epochs_completed=3,
            final_loss=0.5,
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Success" in summary


@pytest.mark.unit
class TestQuantizationWrapper:
    """Tests for quantization wrapper functionality."""

    def test_wrapper_creation(self):
        """Test QuantizationWrapper can be instantiated."""
        wrapper = QuantizationWrapper()
        assert wrapper is not None

    def test_available_methods(self):
        """Test listing available quantization methods."""
        wrapper = QuantizationWrapper()
        methods = wrapper.available_methods
        
        # NONE and DYNAMIC should always be available
        assert QuantizationType.NONE in methods
        assert QuantizationType.DYNAMIC in methods

    def test_dynamic_quantization(self):
        """Test dynamic quantization on a simple model."""
        wrapper = QuantizationWrapper()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        # Apply dynamic quantization
        quantized = wrapper.quantize_model_dynamic(model)
        
        assert quantized is not None
        # Model should still work
        x = torch.randn(1, 128)
        output = quantized(x)
        assert output.shape == (1, 32)

    def test_quantization_config_defaults(self):
        """Test quantization config default values."""
        config = QuantizationConfig()
        
        # Default is BNB_INT4 for bitsandbytes 4-bit quantization
        assert config.quant_type == QuantizationType.BNB_INT4
        assert config.bnb_4bit_compute_dtype == "float16"


@pytest.mark.unit 
class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test ModelInfo can be created with required fields."""
        info = ModelInfo(
            source=ModelSource.HUGGINGFACE,
            name="test-model",
            num_params=1000000,
            architecture_type="transformer",
            task_type="causal_lm",
            config={},
        )
        
        assert info.name == "test-model"
        assert info.source == ModelSource.HUGGINGFACE
        assert info.num_params == 1000000

    def test_model_info_num_params_human(self):
        """Test ModelInfo human-readable parameter count."""
        info = ModelInfo(
            source=ModelSource.HUGGINGFACE,
            name="test-model",
            num_params=7_000_000_000,  # 7B
            architecture_type="transformer",
            task_type="causal_lm",
            config={},
        )
        
        assert info.num_params_human == "7.0B"


@pytest.mark.integration
class TestWrapperIntegration:
    """Integration tests for wrapper components."""

    def test_loader_to_quantizer_flow(self):
        """Test flow from model loading to quantization."""
        loader = UniversalModelLoader()
        wrapper = QuantizationWrapper()
        
        # Create a dummy model (simulating a loaded model)
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Quantize
        quantized = wrapper.quantize_model_dynamic(model)
        
        # Should work end-to-end
        x = torch.randn(2, 256)
        output = quantized(x)
        assert output.shape == (2, 64)
