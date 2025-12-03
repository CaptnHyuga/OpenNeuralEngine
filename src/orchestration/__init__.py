"""ONN Orchestration Layer - Intelligent hardware-aware configuration.

This module provides the core intelligence of ONN 2.0:
- Hardware profiling (VRAM, RAM, CPU, disk detection)
- Configuration orchestration (auto-optimal settings)
- Resource monitoring (runtime adaptation)
"""
from .hardware_profiler import HardwareProfiler, HardwareProfile
from .config_orchestrator import ConfigOrchestrator, TrainingConfig, Precision, ParallelismStrategy
from .resource_monitor import ResourceMonitor

# Convenience function to get a singleton profiler
_profiler_instance = None

def get_profiler() -> HardwareProfiler:
    """Get or create a singleton HardwareProfiler instance."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = HardwareProfiler()
    return _profiler_instance

__all__ = [
    "HardwareProfiler",
    "HardwareProfile", 
    "ConfigOrchestrator",
    "TrainingConfig",
    "Precision",
    "ParallelismStrategy",
    "ResourceMonitor",
    "get_profiler",
]
