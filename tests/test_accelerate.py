"""Tests for ONN Accelerate wrapper (DeepSpeed/FSDP integration)."""
from __future__ import annotations

import pytest
import json
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.unit
class TestDistributedStrategy:
    """Tests for DistributedStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values exist."""
        from src.wrappers.accelerate_wrapper import DistributedStrategy
        
        strategies = [s.value for s in DistributedStrategy]
        
        assert "none" in strategies
        assert "ddp" in strategies
        assert "fsdp" in strategies
        assert "deepspeed_zero1" in strategies
        assert "deepspeed_zero2" in strategies
        assert "deepspeed_zero3" in strategies


@pytest.mark.unit
class TestAccelerateConfig:
    """Tests for AccelerateConfig dataclass."""

    def test_config_creation(self):
        """Test AccelerateConfig can be created."""
        from src.wrappers.accelerate_wrapper import AccelerateConfig, DistributedStrategy
        
        config = AccelerateConfig(
            strategy=DistributedStrategy.DEEPSPEED_ZERO2,
            mixed_precision="bf16",
            gradient_accumulation_steps=4,
        )
        
        assert config.strategy == DistributedStrategy.DEEPSPEED_ZERO2
        assert config.mixed_precision == "bf16"
        assert config.gradient_accumulation_steps == 4

    def test_config_defaults(self):
        """Test AccelerateConfig has sensible defaults."""
        from src.wrappers.accelerate_wrapper import AccelerateConfig, DistributedStrategy
        
        config = AccelerateConfig()
        
        assert config.strategy == DistributedStrategy.NONE
        assert config.mixed_precision == "fp16"
        assert config.gradient_accumulation_steps == 1
        assert config.zero_stage == 2


@pytest.mark.unit
class TestDeepSpeedZeroConfig:
    """Tests for DeepSpeed ZeRO configuration generation."""

    def test_zero2_config(self):
        """Test ZeRO-2 config generation."""
        from src.wrappers.accelerate_wrapper import DeepSpeedZeroConfig
        
        config = DeepSpeedZeroConfig(stage=2)
        config_dict = config.to_dict()
        
        assert config_dict["zero_optimization"]["stage"] == 2
        assert "offload_optimizer" not in config_dict["zero_optimization"]
        assert "train_batch_size" in config_dict

    def test_zero3_config(self):
        """Test ZeRO-3 config generation."""
        from src.wrappers.accelerate_wrapper import DeepSpeedZeroConfig
        
        config = DeepSpeedZeroConfig(stage=3)
        config_dict = config.to_dict()
        
        assert config_dict["zero_optimization"]["stage"] == 3
        assert "stage3_prefetch_bucket_size" in config_dict["zero_optimization"]

    def test_zero3_with_offloading(self):
        """Test ZeRO-3 config with CPU offloading."""
        from src.wrappers.accelerate_wrapper import DeepSpeedZeroConfig
        
        config = DeepSpeedZeroConfig(
            stage=3,
            offload_optimizer=True,
            offload_param=True,
        )
        config_dict = config.to_dict()
        
        assert config_dict["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
        assert config_dict["zero_optimization"]["offload_param"]["device"] == "cpu"

    def test_config_save(self, tmp_path):
        """Test saving config to JSON file."""
        from src.wrappers.accelerate_wrapper import DeepSpeedZeroConfig
        
        config = DeepSpeedZeroConfig(stage=2)
        save_path = tmp_path / "ds_config.json"
        
        result_path = config.save(save_path)
        
        assert Path(result_path).exists()
        with open(result_path) as f:
            loaded = json.load(f)
        assert loaded["zero_optimization"]["stage"] == 2


@pytest.mark.unit
class TestAccelerateWrapper:
    """Tests for AccelerateWrapper class."""

    def test_wrapper_creation(self):
        """Test AccelerateWrapper can be created."""
        try:
            from src.wrappers.accelerate_wrapper import AccelerateWrapper, AccelerateConfig
            
            config = AccelerateConfig()
            wrapper = AccelerateWrapper(config)
            
            assert wrapper.config is not None
            assert wrapper.config.strategy.value == "none"
        except ImportError:
            pytest.skip("accelerate not installed")

    def test_wrapper_summary(self):
        """Test wrapper generates summary."""
        try:
            from src.wrappers.accelerate_wrapper import AccelerateWrapper, AccelerateConfig, DistributedStrategy
            
            config = AccelerateConfig(
                strategy=DistributedStrategy.DEEPSPEED_ZERO2,
                mixed_precision="bf16",
            )
            wrapper = AccelerateWrapper(config)
            summary = wrapper.summary()
            
            assert "deepspeed_zero2" in summary
            assert "bf16" in summary
        except ImportError:
            pytest.skip("accelerate not installed")


@pytest.mark.unit
class TestStrategySelection:
    """Tests for automatic strategy selection."""

    def test_single_gpu_small_model(self):
        """Test strategy selection for small model on single GPU."""
        from src.wrappers.accelerate_wrapper import AccelerateWrapper, DistributedStrategy
        
        strategy = AccelerateWrapper._select_strategy(
            num_gpus=1,
            total_vram_gb=24,
            model_size_gb=5,  # Small model
        )
        
        # Small model on good GPU should be NONE (no sharding needed)
        assert strategy == DistributedStrategy.NONE

    def test_single_gpu_large_model(self):
        """Test strategy selection for large model on single GPU."""
        from src.wrappers.accelerate_wrapper import AccelerateWrapper, DistributedStrategy
        
        strategy = AccelerateWrapper._select_strategy(
            num_gpus=1,
            total_vram_gb=8,
            model_size_gb=30,  # Large model - needs offloading
        )
        
        # Large model should trigger ZeRO-3 for CPU offloading
        assert strategy == DistributedStrategy.DEEPSPEED_ZERO3

    def test_multi_gpu_moderate_model(self):
        """Test strategy selection for moderate model on multi-GPU."""
        from src.wrappers.accelerate_wrapper import AccelerateWrapper, DistributedStrategy
        
        strategy = AccelerateWrapper._select_strategy(
            num_gpus=4,
            total_vram_gb=96,  # 4x24GB
            model_size_gb=40,  # Moderate - fits in aggregate but needs sharding
        )
        
        # Should use ZeRO-2 or FSDP
        assert strategy in (
            DistributedStrategy.DEEPSPEED_ZERO2,
            DistributedStrategy.FSDP,
        )

    def test_multi_gpu_small_model(self):
        """Test strategy selection for small model on multi-GPU."""
        from src.wrappers.accelerate_wrapper import AccelerateWrapper, DistributedStrategy
        
        strategy = AccelerateWrapper._select_strategy(
            num_gpus=2,
            total_vram_gb=48,
            model_size_gb=5,  # Small model
        )
        
        # Small model on 2 GPUs should just use DDP
        assert strategy == DistributedStrategy.DDP


@pytest.mark.unit
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_deepspeed_config(self):
        """Test get_deepspeed_config function."""
        from src.wrappers.accelerate_wrapper import get_deepspeed_config
        
        config = get_deepspeed_config(stage=2)
        
        assert config["zero_optimization"]["stage"] == 2
        assert "train_batch_size" in config

    def test_get_deepspeed_config_with_offload(self):
        """Test get_deepspeed_config with offloading."""
        from src.wrappers.accelerate_wrapper import get_deepspeed_config
        
        config = get_deepspeed_config(
            stage=3,
            offload_optimizer=True,
            offload_param=True,
        )
        
        assert config["zero_optimization"]["stage"] == 3
        assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"


@pytest.mark.unit  
class TestAutoConfiguration:
    """Tests for auto-configuration functionality."""

    def test_auto_configure_small_model(self):
        """Test auto-configuration for small model."""
        try:
            from src.wrappers.accelerate_wrapper import AccelerateWrapper
            
            wrapper = AccelerateWrapper.auto_configure(model_size_gb=2)
            
            # Should have some configuration
            assert wrapper.config is not None
        except ImportError:
            pytest.skip("accelerate not installed")

    def test_auto_configure_from_params(self):
        """Test auto-configuration from parameter count."""
        try:
            from src.wrappers.accelerate_wrapper import AccelerateWrapper
            
            # 7B params ~ 14GB in fp16 + optimizer states
            wrapper = AccelerateWrapper.auto_configure(num_params=7_000_000_000)
            
            assert wrapper.config is not None
        except ImportError:
            pytest.skip("accelerate not installed")
