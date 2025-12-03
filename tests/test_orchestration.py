"""Tests for ONN 2.0 orchestration module."""
from __future__ import annotations

import pytest
import torch
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration import HardwareProfiler, ConfigOrchestrator, ResourceMonitor
from src.orchestration.hardware_profiler import (
    GPUInfo,
    CPUInfo,
    MemoryInfo,
    HardwareProfile,
)
from src.orchestration.config_orchestrator import (
    TrainingConfig,
    Precision,
    ParallelismStrategy,
)


@pytest.mark.unit
class TestHardwareProfiler:
    """Tests for hardware profiling functionality."""

    def test_profiler_creation(self):
        """Test HardwareProfiler can be instantiated."""
        profiler = HardwareProfiler()
        assert profiler is not None

    def test_profile_system(self):
        """Test system profiling returns valid HardwareProfile."""
        profiler = HardwareProfiler()
        profile = profiler.profile()
        
        assert isinstance(profile, HardwareProfile)
        assert profile.hostname is not None
        assert profile.os_name is not None
        assert profile.python_version is not None
        assert profile.torch_version is not None

    def test_profile_has_cpu_info(self):
        """Test profile includes CPU information."""
        profiler = HardwareProfiler()
        profile = profiler.profile()
        
        assert profile.cpu is not None
        assert isinstance(profile.cpu, CPUInfo)
        assert profile.cpu.physical_cores > 0
        assert profile.cpu.logical_cores >= profile.cpu.physical_cores

    def test_profile_has_memory_info(self):
        """Test profile includes memory information."""
        profiler = HardwareProfiler()
        profile = profiler.profile()
        
        assert profile.memory is not None
        assert isinstance(profile.memory, MemoryInfo)
        assert profile.memory.total_ram_mb > 0
        assert profile.memory.available_ram_mb > 0
        assert profile.memory.available_ram_mb <= profile.memory.total_ram_mb

    def test_profile_summary(self):
        """Test profile summary generation."""
        profiler = HardwareProfiler()
        profile = profiler.profile()
        summary = profile.summary()
        
        assert isinstance(summary, str)
        assert "Hardware Profile" in summary
        assert "CPU" in summary or "GPUs" in summary

    def test_profile_caches_result(self):
        """Test that profiler caches result and doesn't re-profile."""
        profiler = HardwareProfiler()
        profile1 = profiler.profile()
        profile2 = profiler.profile()
        # Second call should return cached result
        assert profile1 is profile2

    def test_profiler_get_vram_mb(self):
        """Test get_vram_mb() returns VRAM info."""
        profiler = HardwareProfiler()
        # Returns float for VRAM (0 if no GPU)
        vram = profiler.get_vram_mb(device_index=0)
        assert isinstance(vram, float)
        assert vram >= 0


@pytest.mark.unit
class TestConfigOrchestrator:
    """Tests for automatic configuration orchestration."""

    def test_orchestrator_creation(self):
        """Test ConfigOrchestrator can be instantiated."""
        orchestrator = ConfigOrchestrator()
        assert orchestrator is not None

    def test_orchestrate_basic(self):
        """Test config generation for training via orchestrate()."""
        orchestrator = ConfigOrchestrator()
        # orchestrate() is the main method
        config = orchestrator.orchestrate(
            model_name_or_path="gpt2",
        )
        
        assert isinstance(config, TrainingConfig)
        assert config.per_device_batch_size > 0

    def test_orchestrate_with_device(self):
        """Test orchestrate respects device preference."""
        orchestrator = ConfigOrchestrator()
        config = orchestrator.orchestrate(
            model_name_or_path="gpt2",
            prefer_device="cpu",
        )
        
        assert config.device == "cpu"

    def test_config_has_precision(self):
        """Test config includes precision setting."""
        orchestrator = ConfigOrchestrator()
        config = orchestrator.orchestrate(
            model_name_or_path="gpt2",
        )
        
        # Should have a precision setting
        assert config.precision is not None
        assert isinstance(config.precision, Precision)

    def test_config_summary(self):
        """Test config summary generation."""
        orchestrator = ConfigOrchestrator()
        config = orchestrator.orchestrate(
            model_name_or_path="gpt2",
        )
        
        summary = config.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_config_to_hf_training_args(self):
        """Test config can be converted to HF training args."""
        orchestrator = ConfigOrchestrator()
        config = orchestrator.orchestrate(
            model_name_or_path="gpt2",
        )
        
        hf_args = config.to_hf_training_args()
        assert isinstance(hf_args, dict)
        assert "per_device_train_batch_size" in hf_args


@pytest.mark.unit
class TestResourceMonitor:
    """Tests for resource monitoring functionality."""

    def test_monitor_creation(self):
        """Test ResourceMonitor can be instantiated."""
        monitor = ResourceMonitor()
        assert monitor is not None

    def test_monitor_creation_with_interval(self):
        """Test ResourceMonitor creation with custom interval."""
        monitor = ResourceMonitor(monitoring_interval=0.5)
        assert monitor is not None
        assert monitor.interval == 0.5

    def test_monitor_start_stop(self):
        """Test starting and stopping monitoring."""
        monitor = ResourceMonitor(monitoring_interval=0.1)
        
        # Should be able to start
        monitor.start()
        assert monitor._running is True
        
        # Should be able to stop
        monitor.stop()
        assert monitor._running is False

    def test_get_stats_without_monitoring(self):
        """Test getting stats without running monitor."""
        monitor = ResourceMonitor()
        stats = monitor.get_stats()
        
        # Should return empty stats
        assert stats is not None
        assert len(stats.snapshots) == 0


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests for orchestration components."""

    def test_full_profiling_to_config_flow(self):
        """Test complete flow from profiling to config generation."""
        # Profile hardware
        profiler = HardwareProfiler()
        profile = profiler.profile()
        
        # Generate config using orchestrate
        orchestrator = ConfigOrchestrator()
        config = orchestrator.orchestrate(
            model_name_or_path="gpt2",
        )
        
        # Verify coherent output
        assert config.per_device_batch_size > 0

    def test_orchestrator_with_profiler(self):
        """Test ConfigOrchestrator can use pre-computed profiler."""
        profiler = HardwareProfiler()
        profile = profiler.profile()
        
        # Orchestrator can be created with a profiler
        orchestrator = ConfigOrchestrator(profiler=profiler)
        config = orchestrator.orchestrate(
            model_name_or_path="gpt2",
        )
        
        assert config is not None
        assert config.per_device_batch_size > 0
