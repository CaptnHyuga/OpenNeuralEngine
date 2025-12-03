"""Resource Monitor - Runtime monitoring and adaptation.

Monitors system resources during training and can trigger adaptations:
- VRAM usage monitoring
- OOM detection and recovery
- Dynamic batch size adjustment
- Training health metrics
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import warnings

import torch


@dataclass
class ResourceSnapshot:
    """Point-in-time resource measurement."""
    
    timestamp: float
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_utilization: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_utilization: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class MonitoringStats:
    """Aggregated monitoring statistics."""
    
    snapshots: List[ResourceSnapshot] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    avg_gpu_utilization: float = 0.0
    oom_events: int = 0
    
    @property
    def avg_vram_utilization(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s.vram_utilization for s in self.snapshots) / len(self.snapshots)
    
    def summary(self) -> str:
        return (
            f"Resource Stats:\n"
            f"  Peak VRAM: {self.peak_vram_mb:.0f} MB\n"
            f"  Peak RAM: {self.peak_ram_mb:.0f} MB\n"
            f"  Avg GPU Utilization: {self.avg_gpu_utilization:.1f}%\n"
            f"  OOM Events: {self.oom_events}"
        )


class ResourceMonitor:
    """Monitors and manages system resources during training."""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        vram_warning_threshold: float = 0.90,
        vram_critical_threshold: float = 0.95,
        enable_auto_recovery: bool = True,
    ):
        """Initialize resource monitor.
        
        Args:
            monitoring_interval: Seconds between resource checks.
            vram_warning_threshold: VRAM utilization % to trigger warning.
            vram_critical_threshold: VRAM utilization % to trigger critical action.
            enable_auto_recovery: Whether to attempt OOM recovery.
        """
        self.interval = monitoring_interval
        self.warning_threshold = vram_warning_threshold
        self.critical_threshold = vram_critical_threshold
        self.auto_recovery = enable_auto_recovery
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = MonitoringStats()
        self._callbacks: List[Callable[[ResourceSnapshot], None]] = []
        self._oom_callback: Optional[Callable[[], None]] = None
    
    def start(self):
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def get_stats(self) -> MonitoringStats:
        """Get current monitoring statistics."""
        return self._stats
    
    def add_callback(self, callback: Callable[[ResourceSnapshot], None]):
        """Add callback to be called on each snapshot."""
        self._callbacks.append(callback)
    
    def set_oom_callback(self, callback: Callable[[], None]):
        """Set callback for OOM events."""
        self._oom_callback = callback
    
    def take_snapshot(self) -> ResourceSnapshot:
        """Take a single resource snapshot."""
        snapshot = ResourceSnapshot(timestamp=time.time())
        
        # GPU/VRAM metrics
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**2)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**2)
            
            snapshot.vram_used_mb = allocated
            snapshot.vram_total_mb = total
            snapshot.vram_utilization = (allocated / total * 100) if total > 0 else 0
            
            # GPU utilization (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                snapshot.gpu_utilization = util.gpu
            except (ImportError, Exception):
                pass
        
        # RAM metrics
        try:
            import psutil
            vm = psutil.virtual_memory()
            snapshot.ram_used_mb = vm.used / (1024**2)
            snapshot.ram_total_mb = vm.total / (1024**2)
            snapshot.ram_utilization = vm.percent
            snapshot.cpu_utilization = psutil.cpu_percent()
        except ImportError:
            pass
        
        return snapshot
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                snapshot = self.take_snapshot()
                self._process_snapshot(snapshot)
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}", stacklevel=2)

            time.sleep(self.interval)
    
    def _process_snapshot(self, snapshot: ResourceSnapshot):
        """Process a resource snapshot."""
        self._stats.snapshots.append(snapshot)
        
        # Update peaks
        if snapshot.vram_used_mb > self._stats.peak_vram_mb:
            self._stats.peak_vram_mb = snapshot.vram_used_mb
        if snapshot.ram_used_mb > self._stats.peak_ram_mb:
            self._stats.peak_ram_mb = snapshot.ram_used_mb
        
        # Update averages
        if self._stats.snapshots:
            self._stats.avg_gpu_utilization = sum(
                s.gpu_utilization for s in self._stats.snapshots
            ) / len(self._stats.snapshots)
        
        # Check thresholds
        if snapshot.vram_utilization >= self.critical_threshold * 100:
            self._handle_critical_vram(snapshot)
        elif snapshot.vram_utilization >= self.warning_threshold * 100:
            warnings.warn(
                f"VRAM utilization high: {snapshot.vram_utilization:.1f}%",
                stacklevel=2,
            )
        
        # Call user callbacks
        for callback in self._callbacks:
            try:
                callback(snapshot)
            except Exception:
                pass
    
    def _handle_critical_vram(self, snapshot: ResourceSnapshot):
        """Handle critical VRAM situation."""
        warnings.warn(
            f"Critical VRAM utilization: {snapshot.vram_utilization:.1f}%",
            stacklevel=2,
        )
        
        if self.auto_recovery:
            # Try to free some memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def handle_oom(self):
        """Handle OOM event."""
        self._stats.oom_events += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self._oom_callback:
            self._oom_callback()
    
    def clear_cuda_cache(self):
        """Manually clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        if exc_type is RuntimeError and "out of memory" in str(exc_val):
            self.handle_oom()
        return False


# Global monitor instance
_global_monitor: Optional[ResourceMonitor] = None


def get_monitor() -> ResourceMonitor:
    """Get global resource monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
    return _global_monitor
