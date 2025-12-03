"""Hardware Profiler - Comprehensive system capability detection.

Detects and profiles available hardware resources:
- GPU: VRAM, compute capability, architecture (NVIDIA, AMD, Apple Silicon)
- CPU: Core count, architecture, cache sizes
- Memory: Total RAM, available RAM
- Storage: Disk speed, available space
- Network: Bandwidth (for distributed training)

Caches profiles for fast subsequent runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch


@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    
    index: int
    name: str
    vendor: str  # "nvidia", "amd", "apple", "intel"
    total_vram_mb: float
    available_vram_mb: float
    compute_capability: Optional[Tuple[int, int]] = None  # CUDA only
    architecture: Optional[str] = None
    driver_version: Optional[str] = None
    
    # Capabilities
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_int4: bool = False
    supports_flash_attention: bool = False
    
    def __post_init__(self):
        """Determine capabilities based on architecture."""
        if self.vendor == "nvidia" and self.compute_capability:
            major, minor = self.compute_capability
            # Ampere (8.x) and later support BF16 and better INT8
            self.supports_bf16 = major >= 8
            self.supports_int8 = major >= 7
            self.supports_int4 = major >= 8
            # Flash attention requires SM 8.0+
            self.supports_flash_attention = major >= 8
        elif self.vendor == "apple":
            # Apple Silicon supports most formats
            self.supports_bf16 = True
            self.supports_int8 = True


@dataclass 
class CPUInfo:
    """Information about CPU resources."""
    
    name: str
    architecture: str  # "x86_64", "arm64"
    physical_cores: int
    logical_cores: int
    frequency_mhz: Optional[float] = None
    cache_size_kb: Optional[int] = None
    
    # Capabilities
    supports_avx: bool = False
    supports_avx2: bool = False
    supports_avx512: bool = False


@dataclass
class MemoryInfo:
    """Information about system memory."""
    
    total_ram_mb: float
    available_ram_mb: float
    swap_total_mb: float = 0.0
    swap_available_mb: float = 0.0


@dataclass
class StorageInfo:
    """Information about storage capabilities."""
    
    available_space_gb: float
    is_ssd: bool = True  # Assume SSD by default
    read_speed_mbps: Optional[float] = None
    write_speed_mbps: Optional[float] = None


@dataclass
class HardwareProfile:
    """Complete hardware profile for a system."""
    
    # System info
    hostname: str
    os_name: str
    os_version: str
    python_version: str
    torch_version: str
    
    # Hardware
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu: Optional[CPUInfo] = None
    memory: Optional[MemoryInfo] = None
    storage: Optional[StorageInfo] = None
    
    # Computed properties
    total_vram_mb: float = 0.0
    available_vram_mb: float = 0.0
    has_gpu: bool = False
    
    # Profile metadata
    profile_time: float = 0.0
    profile_hash: str = ""
    
    def __post_init__(self):
        """Compute derived properties."""
        if self.gpus:
            self.has_gpu = True
            self.total_vram_mb = sum(g.total_vram_mb for g in self.gpus)
            self.available_vram_mb = sum(g.available_vram_mb for g in self.gpus)
    
    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        """Get the primary (first) GPU."""
        return self.gpus[0] if self.gpus else None
    
    @property
    def supports_bf16(self) -> bool:
        """Check if any GPU supports BF16."""
        return any(g.supports_bf16 for g in self.gpus)
    
    @property
    def supports_flash_attention(self) -> bool:
        """Check if any GPU supports Flash Attention."""
        return any(g.supports_flash_attention for g in self.gpus)
    
    @property
    def total_ram_mb(self) -> float:
        """Get total system RAM."""
        return self.memory.total_ram_mb if self.memory else 0.0
    
    @property 
    def available_ram_mb(self) -> float:
        """Get available system RAM."""
        return self.memory.available_ram_mb if self.memory else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize profile to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareProfile":
        """Create profile from dictionary."""
        # Convert nested dataclasses
        if "gpus" in data:
            data["gpus"] = [GPUInfo(**g) for g in data["gpus"]]
        if "cpu" in data and data["cpu"]:
            data["cpu"] = CPUInfo(**data["cpu"])
        if "memory" in data and data["memory"]:
            data["memory"] = MemoryInfo(**data["memory"])
        if "storage" in data and data["storage"]:
            data["storage"] = StorageInfo(**data["storage"])
        return cls(**data)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Hardware Profile ({self.hostname})",
            f"  OS: {self.os_name} {self.os_version}",
            f"  Python: {self.python_version}, PyTorch: {self.torch_version}",
        ]
        
        if self.gpus:
            lines.append(f"  GPUs: {len(self.gpus)}")
            for gpu in self.gpus:
                avail = gpu.available_vram_mb
                total = gpu.total_vram_mb
                lines.append(f"    [{gpu.index}] {gpu.name}: {avail:.0f}/{total:.0f} MB VRAM")
        else:
            lines.append("  GPUs: None detected")
        
        if self.cpu:
            lines.append(f"  CPU: {self.cpu.name} ({self.cpu.physical_cores}c/{self.cpu.logical_cores}t)")
        
        if self.memory:
            lines.append(f"  RAM: {self.memory.available_ram_mb:.0f}/{self.memory.total_ram_mb:.0f} MB")
        
        if self.storage:
            lines.append(f"  Storage: {self.storage.available_space_gb:.1f} GB available")
        
        return "\n".join(lines)


class HardwareProfiler:
    """Profiles system hardware and caches results."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl_seconds: float = 3600,  # 1 hour default
    ):
        """Initialize profiler.
        
        Args:
            cache_dir: Directory for caching profiles. Defaults to ~/.onn/hardware_cache/
            cache_ttl_seconds: How long cached profiles are valid.
        """
        self.cache_dir = cache_dir or Path.home() / ".onn" / "hardware_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_seconds
        self._profile: Optional[HardwareProfile] = None
    
    def profile(self, force_refresh: bool = False) -> HardwareProfile:
        """Get hardware profile, using cache if available.
        
        Args:
            force_refresh: Force re-profiling even if cache is valid.
        
        Returns:
            Complete hardware profile.
        """
        if not force_refresh and self._profile:
            return self._profile
        
        # Try loading from cache
        cache_file = self.cache_dir / "profile.json"
        if not force_refresh and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                if time.time() - cached.get("profile_time", 0) < self.cache_ttl:
                    self._profile = HardwareProfile.from_dict(cached)
                    return self._profile
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Cache invalid, re-profile
        
        # Profile fresh
        self._profile = self._do_profile()
        
        # Save to cache
        try:
            cache_file.write_text(self._profile.to_json())
        except OSError:
            pass  # Cache write failed, not critical
        
        return self._profile
    
    def _do_profile(self) -> HardwareProfile:
        """Perform full hardware profiling."""
        start_time = time.time()
        
        profile = HardwareProfile(
            hostname=platform.node(),
            os_name=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            profile_time=start_time,
        )
        
        # Profile GPUs
        profile.gpus = self._profile_gpus()
        
        # Profile CPU
        profile.cpu = self._profile_cpu()
        
        # Profile Memory
        profile.memory = self._profile_memory()
        
        # Profile Storage
        profile.storage = self._profile_storage()
        
        # Compute hash for change detection
        profile.profile_hash = self._compute_hash(profile)
        
        # Recompute derived properties
        profile.__post_init__()
        
        return profile
    
    def _profile_gpus(self) -> List[GPUInfo]:
        """Profile all available GPUs."""
        gpus = []
        
        # CUDA GPUs (NVIDIA)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Get current memory state
                total_mem = props.total_memory / (1024**2)
                allocated = torch.cuda.memory_allocated(i) / (1024**2)
                available = total_mem - allocated
                
                gpu = GPUInfo(
                    index=i,
                    name=props.name,
                    vendor="nvidia",
                    total_vram_mb=total_mem,
                    available_vram_mb=available,
                    compute_capability=(props.major, props.minor),
                    architecture=self._get_nvidia_arch(props.major),
                )
                gpus.append(gpu)
        
        # MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            # Apple Silicon shares memory with system
            # Estimate available as portion of system RAM
            try:
                import psutil
                total_ram = psutil.virtual_memory().total / (1024**2)
                # Estimate 50% of RAM available for GPU on Apple Silicon
                gpu_available = total_ram * 0.5
            except ImportError:
                gpu_available = 8192  # Default 8GB estimate
            
            gpu = GPUInfo(
                index=0,
                name="Apple Silicon",
                vendor="apple",
                total_vram_mb=gpu_available,
                available_vram_mb=gpu_available,
                architecture="apple_silicon",
            )
            gpus.append(gpu)
        
        return gpus
    
    def _get_nvidia_arch(self, major: int) -> str:
        """Map CUDA major version to architecture name."""
        arch_map = {
            3: "Kepler",
            5: "Maxwell", 
            6: "Pascal",
            7: "Volta/Turing",
            8: "Ampere",
            9: "Hopper",
            10: "Blackwell",
        }
        return arch_map.get(major, f"SM_{major}x")
    
    def _profile_cpu(self) -> CPUInfo:
        """Profile CPU capabilities."""
        try:
            import psutil
            
            freq = psutil.cpu_freq()
            
            return CPUInfo(
                name=platform.processor() or "Unknown CPU",
                architecture=platform.machine(),
                physical_cores=psutil.cpu_count(logical=False) or 1,
                logical_cores=psutil.cpu_count(logical=True) or 1,
                frequency_mhz=freq.max if freq else None,
            )
        except ImportError:
            # Fallback without psutil
            return CPUInfo(
                name=platform.processor() or "Unknown CPU",
                architecture=platform.machine(),
                physical_cores=os.cpu_count() or 1,
                logical_cores=os.cpu_count() or 1,
            )
    
    def _profile_memory(self) -> MemoryInfo:
        """Profile system memory."""
        try:
            import psutil
            
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return MemoryInfo(
                total_ram_mb=vm.total / (1024**2),
                available_ram_mb=vm.available / (1024**2),
                swap_total_mb=swap.total / (1024**2),
                swap_available_mb=swap.free / (1024**2),
            )
        except ImportError:
            # Very rough fallback
            return MemoryInfo(
                total_ram_mb=8192,  # Assume 8GB
                available_ram_mb=4096,  # Assume half available
            )
    
    def _profile_storage(self) -> StorageInfo:
        """Profile storage capabilities."""
        try:
            import psutil
            
            # Check current working directory's disk
            disk = psutil.disk_usage(os.getcwd())
            
            return StorageInfo(
                available_space_gb=disk.free / (1024**3),
                is_ssd=True,  # Assume SSD (hard to detect reliably)
            )
        except (ImportError, OSError):
            return StorageInfo(
                available_space_gb=100,  # Assume 100GB available
            )
    
    def _compute_hash(self, profile: HardwareProfile) -> str:
        """Compute stable hash for profile to detect changes."""
        # Hash key hardware characteristics
        key_data = json.dumps({
            "gpus": [(g.name, g.total_vram_mb) for g in profile.gpus],
            "cpu": profile.cpu.name if profile.cpu else "",
            "ram": profile.memory.total_ram_mb if profile.memory else 0,
        }, sort_keys=True)
        # MD5 used for cache key generation, not security  # nosec B324
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()[:12]
    
    def get_vram_mb(self, device_index: int = 0) -> float:
        """Get available VRAM for a specific device.
        
        Args:
            device_index: GPU index.
        
        Returns:
            Available VRAM in MB, or inf for CPU.
        """
        profile = self.profile()
        
        if not profile.gpus:
            return float("inf")  # CPU has "unlimited" VRAM
        
        if device_index >= len(profile.gpus):
            return 0.0
        
        return profile.gpus[device_index].available_vram_mb
    
    def get_optimal_device(
        self,
        vram_required_mb: Optional[float] = None,
        prefer_device: Optional[str] = None,
    ) -> str:
        """Select optimal device based on requirements.
        
        Args:
            vram_required_mb: Minimum VRAM needed (None = any).
            prefer_device: User preference ("cuda", "mps", "cpu").
        
        Returns:
            Device string ("cuda", "cuda:0", "mps", "cpu").
        """
        profile = self.profile()
        
        # Honor explicit preference
        if prefer_device == "cpu":
            return "cpu"
        
        if prefer_device and prefer_device.startswith("cuda"):
            if profile.gpus:
                # Check if preferred device has enough VRAM
                idx = int(prefer_device.split(":")[-1]) if ":" in prefer_device else 0
                if idx < len(profile.gpus):
                    gpu = profile.gpus[idx]
                    if vram_required_mb is None or gpu.available_vram_mb >= vram_required_mb:
                        return prefer_device
                    warnings.warn(
                        f"Requested {prefer_device} has {gpu.available_vram_mb:.0f}MB VRAM, "
                        f"but {vram_required_mb:.0f}MB required. Falling back.",
                        stacklevel=2,
                    )
        
        if prefer_device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
        
        # Auto-select: CUDA > MPS > CPU
        if profile.gpus:
            for gpu in profile.gpus:
                if vram_required_mb is None or gpu.available_vram_mb >= vram_required_mb:
                    return f"cuda:{gpu.index}" if len(profile.gpus) > 1 else "cuda"
        
        if torch.backends.mps.is_available():
            return "mps"
        
        return "cpu"
    
    def estimate_model_vram_mb(
        self,
        num_params: int,
        precision: str = "fp32",
        include_optimizer: bool = True,
        include_gradients: bool = True,
        batch_size: int = 1,
        seq_len: int = 512,
    ) -> float:
        """Estimate VRAM requirements for a model.
        
        Args:
            num_params: Total model parameters.
            precision: "fp32", "fp16", "bf16", "int8", "int4".
            include_optimizer: Include optimizer states (AdamW = 2x params).
            include_gradients: Include gradient storage.
            batch_size: Training batch size.
            seq_len: Sequence length for activation memory.
        
        Returns:
            Estimated VRAM in MB.
        """
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
        }.get(precision, 4)
        
        # Model weights
        model_mb = (num_params * bytes_per_param) / (1024**2)
        
        # Gradients (same size as model in training precision)
        grad_mb = model_mb if include_gradients else 0
        
        # Optimizer states (AdamW: 2x for momentum and variance, in fp32)
        if include_optimizer:
            optimizer_mb = (num_params * 4 * 2) / (1024**2)  # Always fp32
        else:
            optimizer_mb = 0
        
        # Activations (rough estimate based on batch size and seq len)
        # This is highly model-dependent, but ~4 bytes per element is typical
        hidden_dim_estimate = int((num_params / 12) ** 0.5)  # Very rough
        activations_mb = (batch_size * seq_len * hidden_dim_estimate * 4) / (1024**2)
        
        # Add safety margin (20%)
        total = (model_mb + grad_mb + optimizer_mb + activations_mb) * 1.2
        
        return total


# Global profiler instance
_global_profiler: Optional[HardwareProfiler] = None


def get_profiler() -> HardwareProfiler:
    """Get global hardware profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = HardwareProfiler()
    return _global_profiler


def get_hardware_profile(force_refresh: bool = False) -> HardwareProfile:
    """Get current hardware profile using global profiler."""
    return get_profiler().profile(force_refresh)
