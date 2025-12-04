"""ONN Orchestration Layer - Intelligent hardware-aware configuration.

This module provides the core intelligence of ONN 2.0:
- Hardware profiling (VRAM, RAM, CPU, disk detection)
- Configuration orchestration (auto-optimal settings)
- Resource monitoring (runtime adaptation)
- Architecture detection (model analysis)
"""
from .hardware_profiler import (
    HardwareProfiler,
    HardwareProfile,
    get_hardware_profile,
    get_profiler,
)
from .config_orchestrator import (
    ConfigOrchestrator,
    TrainingConfig,
    Precision,
    ParallelismStrategy,
    auto_configure,
)
from .resource_monitor import ResourceMonitor, get_monitor
from .architecture_detector import (
    ArchitectureDetector,
    ArchitectureInfo,
    ArchitectureType,
    TaskType,
    analyze_model,
    detect_architecture,
    detect_task,
)

__all__ = [
    # Hardware Profiler
    "HardwareProfiler",
    "HardwareProfile",
    "get_hardware_profile",
    "get_profiler",
    # Config Orchestrator
    "ConfigOrchestrator",
    "TrainingConfig",
    "Precision",
    "ParallelismStrategy",
    "auto_configure",
    # Resource Monitor
    "ResourceMonitor",
    "get_monitor",
    # Architecture Detector
    "ArchitectureDetector",
    "ArchitectureInfo",
    "ArchitectureType",
    "TaskType",
    "analyze_model",
    "detect_architecture",
    "detect_task",
]
