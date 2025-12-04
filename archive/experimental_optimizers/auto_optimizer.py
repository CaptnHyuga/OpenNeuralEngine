"""
Universal Auto-Optimizer for Low-VRAM Training & Inference

This module automatically finds the optimal configuration for ANY hardware
by profiling the system and testing different configurations.

Features:
1. Hardware Profiler - Detects VRAM, PCIe bandwidth, compute capability
2. Auto-Tuner - Finds optimal chunk_size and batch_size
3. Benchmark Suite - Compares optimized vs baseline performance
4. Universal Compatibility - Works with any transformer architecture

Usage:
    from auto_optimizer import AutoOptimizer
    
    optimizer = AutoOptimizer(model_path="models/phi-4")
    config = optimizer.find_optimal_config()
    print(f"Optimal config: {config}")
    
    # Get improvement percentages
    results = optimizer.benchmark_vs_baseline()
    print(f"Training speedup: {results['training_speedup']:.1f}x")
    print(f"Inference speedup: {results['inference_speedup']:.1f}x")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from collections import defaultdict
from pathlib import Path
import time
import json
import gc
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import math


@dataclass
class HardwareProfile:
    """Hardware capabilities detected by profiler."""
    device_name: str
    vram_total_gb: float
    vram_available_gb: float
    compute_capability: Tuple[int, int]
    cuda_cores: int  # Estimated
    memory_bandwidth_gbps: float  # Measured
    pcie_bandwidth_gbps: float  # Measured (CPU->GPU)
    cpu_to_gpu_pinned_bandwidth_gbps: float  # With pinned memory
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class ModelProfile:
    """Model characteristics."""
    name: str
    total_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    layer_size_mb: float
    total_size_gb: float
    safetensor_files: List[str]
    layer_to_file: Dict[int, str]


@dataclass
class OptimalConfig:
    """Optimal configuration found by auto-tuner."""
    chunk_size: int  # Layers to load at once
    batch_size: int  # Samples to process per weight load
    sparse_layers: List[int]  # Which layers to train
    use_pinned_memory: bool
    estimated_vram_mb: int
    estimated_time_per_sample_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class HardwareProfiler:
    """Profile hardware capabilities."""
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    
    def profile(self) -> HardwareProfile:
        """Run all hardware profiling tests."""
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        # Basic info
        device_name = props.name
        vram_total = props.total_memory / 1e9
        vram_available = (props.total_memory - torch.cuda.memory_allocated()) / 1e9
        compute_cap = (props.major, props.minor)
        
        # Estimate CUDA cores (varies by architecture)
        cuda_cores = self._estimate_cuda_cores(props)
        
        # Measure bandwidths
        print("Profiling hardware...")
        mem_bw = self._measure_memory_bandwidth()
        pcie_bw, pinned_bw = self._measure_pcie_bandwidth()
        
        return HardwareProfile(
            device_name=device_name,
            vram_total_gb=vram_total,
            vram_available_gb=vram_available,
            compute_capability=compute_cap,
            cuda_cores=cuda_cores,
            memory_bandwidth_gbps=mem_bw,
            pcie_bandwidth_gbps=pcie_bw,
            cpu_to_gpu_pinned_bandwidth_gbps=pinned_bw,
        )
    
    def _estimate_cuda_cores(self, props) -> int:
        """Estimate CUDA cores based on SM count and architecture."""
        sm_count = props.multi_processor_count
        # Cores per SM varies by compute capability
        cores_per_sm = {
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere GA100
            (8, 6): 128,  # Ampere GA102
            (8, 9): 128,  # Ada Lovelace
            (9, 0): 128,  # Hopper
        }
        major, minor = props.major, props.minor
        per_sm = cores_per_sm.get((major, minor), 64)
        return sm_count * per_sm
    
    def _measure_memory_bandwidth(self) -> float:
        """Measure GPU memory bandwidth (GB/s)."""
        torch.cuda.synchronize()
        size = 256 * 1024 * 1024  # 256MB
        a = torch.randn(size // 4, device='cuda', dtype=torch.float32)
        b = torch.empty_like(a)
        
        # Warmup
        for _ in range(3):
            b.copy_(a)
        torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            b.copy_(a)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        bytes_transferred = size * 2 * iterations  # Read + write
        bandwidth_gbps = bytes_transferred / elapsed / 1e9
        
        del a, b
        torch.cuda.empty_cache()
        return bandwidth_gbps
    
    def _measure_pcie_bandwidth(self) -> Tuple[float, float]:
        """Measure CPU->GPU transfer bandwidth (regular and pinned)."""
        size = 100 * 1024 * 1024  # 100MB
        
        # Regular transfer
        cpu_tensor = torch.randn(size // 4, dtype=torch.float32)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            gpu_tensor = cpu_tensor.cuda()
            torch.cuda.synchronize()
            del gpu_tensor
        elapsed = time.perf_counter() - start
        regular_bw = size * iterations / elapsed / 1e9
        
        # Pinned memory transfer
        cpu_pinned = cpu_tensor.pin_memory()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            gpu_tensor = cpu_pinned.cuda(non_blocking=True)
            torch.cuda.synchronize()
            del gpu_tensor
        elapsed = time.perf_counter() - start
        pinned_bw = size * iterations / elapsed / 1e9
        
        del cpu_tensor, cpu_pinned
        torch.cuda.empty_cache()
        return regular_bw, pinned_bw


class ModelProfiler:
    """Profile model characteristics."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
    def profile(self) -> ModelProfile:
        """Analyze model structure."""
        safetensor_files = sorted(self.model_path.glob("model-*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensor files in {self.model_path}")
        
        # Map layers to files
        layer_to_file = {}
        layer_sizes = {}
        config = {}
        
        for fpath in safetensor_files:
            with safe_open(str(fpath), framework='pt') as f:
                for key in f.keys():
                    # Detect layer index
                    if '.layers.' in key:
                        parts = key.split('.')
                        layer_idx = int(parts[parts.index('layers') + 1])
                        layer_to_file[layer_idx] = str(fpath)
                        
                        # Sum layer size
                        tensor = f.get_tensor(key)
                        if layer_idx not in layer_sizes:
                            layer_sizes[layer_idx] = 0
                        layer_sizes[layer_idx] += tensor.numel() * tensor.element_size()
                        
                        # Extract config from tensor shapes
                        if 'qkv_proj' in key or 'q_proj' in key:
                            config['hidden_size'] = tensor.shape[1]
                        if 'gate_up_proj' in key or 'gate_proj' in key:
                            if 'gate_up' in key:
                                config['intermediate_size'] = tensor.shape[0] // 2
                            else:
                                config['intermediate_size'] = tensor.shape[0]
        
        total_layers = len(layer_to_file)
        avg_layer_size = sum(layer_sizes.values()) / len(layer_sizes) if layer_sizes else 0
        total_size = sum(layer_sizes.values())
        
        # Try to load config.json for accurate values
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
                config['hidden_size'] = model_config.get('hidden_size', config.get('hidden_size', 4096))
                config['intermediate_size'] = model_config.get('intermediate_size', config.get('intermediate_size', 11008))
                config['num_attention_heads'] = model_config.get('num_attention_heads', 32)
                config['num_kv_heads'] = model_config.get('num_key_value_heads', model_config.get('num_attention_heads', 32))
        
        return ModelProfile(
            name=self.model_path.name,
            total_layers=total_layers,
            hidden_size=config.get('hidden_size', 4096),
            intermediate_size=config.get('intermediate_size', 11008),
            num_attention_heads=config.get('num_attention_heads', 32),
            num_kv_heads=config.get('num_kv_heads', 32),
            layer_size_mb=avg_layer_size / 1e6,
            total_size_gb=total_size / 1e9,
            safetensor_files=[str(f) for f in safetensor_files],
            layer_to_file=layer_to_file,
        )


class AutoTuner:
    """Find optimal configuration for given hardware and model."""
    
    def __init__(self, hardware: HardwareProfile, model: ModelProfile):
        self.hardware = hardware
        self.model = model
    
    def find_optimal_config(
        self,
        target_layers: Optional[List[int]] = None,
        max_vram_usage: float = 0.85,  # Use up to 85% of VRAM
        test_batch_size: int = 8,  # Batch size for empirical testing
    ) -> OptimalConfig:
        """
        Find optimal chunk_size and batch_size for this hardware/model combo
        using EMPIRICAL testing (not just theoretical estimates).
        
        Args:
            target_layers: Which layers to train (None = auto-select sparse)
            max_vram_usage: Maximum fraction of VRAM to use
            test_batch_size: Batch size to use during testing
        """
        # Auto-select sparse layers if not specified
        if target_layers is None:
            # Default: first 10% + last 10% of layers
            n = self.model.total_layers
            first_n = max(1, n // 10)
            last_n = max(1, n // 10)
            target_layers = list(range(first_n)) + list(range(n - last_n, n))
        
        # Calculate available VRAM budget
        vram_budget_mb = self.hardware.vram_available_gb * 1000 * max_vram_usage
        
        # Reserve space for activations, gradients, optimizer states
        reserved_mb = 500
        weight_budget_mb = vram_budget_mb - reserved_mb
        
        # Find max chunk_size that fits in VRAM
        layer_size_mb = self.model.layer_size_mb / 2  # FP16
        max_chunk = max(1, int(weight_budget_mb / layer_size_mb))
        
        print(f"\nEMPIRICAL testing of chunk sizes (max feasible: {max_chunk})...")
        print(f"Test batch size: {test_batch_size}")
        
        best_config = None
        best_time = float('inf')
        
        # Group layers by file
        file_to_layers = defaultdict(list)
        for layer in target_layers:
            if layer in self.model.layer_to_file:
                file_to_layers[self.model.layer_to_file[layer]].append(layer)
        
        for chunk_size in range(1, min(max_chunk + 1, len(target_layers) + 1)):
            # Create chunks respecting file boundaries
            chunks = []
            for fpath, layers in file_to_layers.items():
                for i in range(0, len(layers), chunk_size):
                    chunk = layers[i:i+chunk_size]
                    chunks.append((fpath, chunk))
            
            n_chunks = len(chunks)
            
            # Actually run the benchmark
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                elapsed = self._benchmark_chunk_config(chunks, test_batch_size)
                per_sample_ms = elapsed / test_batch_size * 1000
                peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
                
                print(f"  chunk={chunk_size}: {n_chunks} loads, {per_sample_ms:.0f}ms/sample, "
                      f"{peak_vram_mb:.0f}MB VRAM")
                
                if per_sample_ms < best_time and peak_vram_mb < vram_budget_mb:
                    best_time = per_sample_ms
                    best_config = OptimalConfig(
                        chunk_size=chunk_size,
                        batch_size=self._find_optimal_batch_size(chunks, per_sample_ms),
                        sparse_layers=target_layers,
                        use_pinned_memory=True,
                        estimated_vram_mb=int(peak_vram_mb),
                        estimated_time_per_sample_ms=per_sample_ms,
                    )
                    
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"  chunk={chunk_size}: OOM")
                    torch.cuda.empty_cache()
                    break  # Larger chunks will also OOM
                raise
        
        return best_config
    
    def _benchmark_chunk_config(self, chunks: List[Tuple[str, List[int]]], n_samples: int) -> float:
        """Actually benchmark a chunk configuration."""
        import torch.nn.functional as F
        
        start = time.time()
        
        for fpath, chunk in chunks:
            with safe_open(fpath, framework='pt') as f:
                weights = {}
                for layer in chunk:
                    # Try to load weights - handle different architectures
                    weights[layer] = {}
                    for name in ['self_attn.qkv_proj', 'self_attn.q_proj', 'self_attn.o_proj',
                                 'mlp.gate_up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
                        key = f'model.layers.{layer}.{name}.weight'
                        try:
                            w = f.get_tensor(key)
                            weights[layer][name] = w.pin_memory().cuda(non_blocking=True).half()
                        except:
                            pass
            torch.cuda.synchronize()
            
            # Forward pass for all samples
            for _ in range(n_samples):
                for layer, layer_weights in weights.items():
                    for name, w in layer_weights.items():
                        x = torch.randn(1, 32, w.shape[1], device='cuda', dtype=torch.float16)
                        y = F.linear(x, w)
            torch.cuda.synchronize()
            
            del weights
            torch.cuda.empty_cache()
        
        return time.time() - start
    
    def _find_optimal_batch_size(self, chunks: List, base_time_ms: float) -> int:
        """Find optimal batch size based on amortization curve."""
        # Test a few batch sizes and extrapolate
        # Optimal is where marginal improvement becomes small
        
        # Simple heuristic: batch_size where load time is well amortized
        # If base_time at batch=8 is X ms/sample, optimal is ~32-64
        if base_time_ms < 500:
            return 64
        elif base_time_ms < 1000:
            return 32
        elif base_time_ms < 2000:
            return 16
        else:
            return 8
    
    def validate_config(self, config: OptimalConfig) -> Tuple[bool, str]:
        """Actually test if config works without OOM."""
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Simulate loading chunk_size layers
            dummy_weights = []
            for i in range(config.chunk_size):
                # Approximate layer size
                w = torch.randn(
                    self.model.hidden_size, self.model.hidden_size,
                    device='cuda', dtype=torch.float16
                )
                dummy_weights.append(w)
            
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            
            del dummy_weights
            torch.cuda.empty_cache()
            
            if peak_mb > self.hardware.vram_total_gb * 1000 * 0.95:
                return False, f"Would use {peak_mb:.0f}MB, too close to limit"
            
            return True, f"Validated: {peak_mb:.0f}MB peak"
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                return False, "OOM during validation"
            raise


class BenchmarkSuite:
    """Compare optimized vs baseline performance."""
    
    def __init__(self, model_path: str, optimal_config: OptimalConfig):
        self.model_path = Path(model_path)
        self.config = optimal_config
        
        # Map layers to files
        self.layer_to_file = {}
        safetensor_files = sorted(self.model_path.glob("model-*.safetensors"))
        for fpath in safetensor_files:
            with safe_open(str(fpath), framework='pt') as f:
                for key in f.keys():
                    if 'mlp.gate_up_proj' in key or 'mlp.gate_proj' in key:
                        parts = key.split('.')
                        layer_idx = int(parts[parts.index('layers') + 1])
                        self.layer_to_file[layer_idx] = str(fpath)
    
    def benchmark_inference(self, n_iterations: int = 10) -> Dict[str, float]:
        """Benchmark inference: baseline vs optimized."""
        print("\n" + "=" * 60)
        print("INFERENCE BENCHMARK")
        print("=" * 60)
        
        layers = self.config.sparse_layers[:4]  # Use subset for speed
        
        # Baseline: Load one layer at a time, no pinned memory
        print("\nBaseline (per-layer, no pinned memory)...")
        torch.cuda.empty_cache()
        
        baseline_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            for layer in layers:
                fpath = self.layer_to_file[layer]
                with safe_open(fpath, framework='pt') as f:
                    for name in ['qkv_proj', 'o_proj']:
                        key = f'model.layers.{layer}.self_attn.{name}.weight'
                        try:
                            w = f.get_tensor(key).cuda().half()
                        except:
                            continue
                        x = torch.randn(1, 32, w.shape[1], device='cuda', dtype=torch.float16)
                        y = F.linear(x, w)
                        del w
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline_times.append(time.perf_counter() - start)
        
        baseline_ms = sum(baseline_times) / len(baseline_times) * 1000
        print(f"  Baseline: {baseline_ms:.0f}ms for {len(layers)} layers")
        
        # Optimized: Batched loading with pinned memory
        print("\nOptimized (chunked, pinned memory)...")
        torch.cuda.empty_cache()
        
        # Group layers by file
        file_to_layers = defaultdict(list)
        for layer in layers:
            file_to_layers[self.layer_to_file[layer]].append(layer)
        
        optimized_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            for fpath, chunk_layers in file_to_layers.items():
                with safe_open(fpath, framework='pt') as f:
                    weights = {}
                    for layer in chunk_layers:
                        for name in ['qkv_proj', 'o_proj']:
                            key = f'model.layers.{layer}.self_attn.{name}.weight'
                            try:
                                w = f.get_tensor(key)
                                weights[(layer, name)] = w.pin_memory().cuda(non_blocking=True).half()
                            except:
                                continue
                    torch.cuda.synchronize()
                    
                    # Compute
                    for (layer, name), w in weights.items():
                        x = torch.randn(1, 32, w.shape[1], device='cuda', dtype=torch.float16)
                        y = F.linear(x, w)
                    
                    del weights
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            optimized_times.append(time.perf_counter() - start)
        
        optimized_ms = sum(optimized_times) / len(optimized_times) * 1000
        print(f"  Optimized: {optimized_ms:.0f}ms for {len(layers)} layers")
        
        speedup = baseline_ms / optimized_ms
        improvement_pct = (baseline_ms - optimized_ms) / baseline_ms * 100
        
        print(f"\n  Speedup: {speedup:.2f}x ({improvement_pct:.1f}% faster)")
        
        return {
            'baseline_ms': baseline_ms,
            'optimized_ms': optimized_ms,
            'speedup': speedup,
            'improvement_pct': improvement_pct,
        }
    
    def benchmark_training(self, n_samples: int = 8) -> Dict[str, float]:
        """Benchmark training: baseline vs optimized (with batching)."""
        print("\n" + "=" * 60)
        print("TRAINING BENCHMARK")
        print("=" * 60)
        
        layers = self.config.sparse_layers[:4]  # Use subset for speed
        hidden_size = 5120  # Will be auto-detected in full impl
        
        # Baseline: Process one sample at a time, load weights each time
        print(f"\nBaseline (no batching, {n_samples} samples)...")
        torch.cuda.empty_cache()
        
        start = time.perf_counter()
        for sample_idx in range(n_samples):
            for layer in layers:
                fpath = self.layer_to_file[layer]
                with safe_open(fpath, framework='pt') as f:
                    for name in ['qkv_proj', 'o_proj']:
                        key = f'model.layers.{layer}.self_attn.{name}.weight'
                        try:
                            w = f.get_tensor(key).cuda().half()
                        except:
                            continue
                        x = torch.randn(1, 32, w.shape[1], device='cuda', dtype=torch.float16, requires_grad=True)
                        y = F.linear(x, w)
                        loss = y.sum()
                        loss.backward()
                        del w
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - start) * 1000
        baseline_per_sample = baseline_ms / n_samples
        print(f"  Baseline: {baseline_ms:.0f}ms total, {baseline_per_sample:.0f}ms/sample")
        
        # Optimized: Batch samples, load weights once per chunk
        print(f"\nOptimized (batched, {n_samples} samples)...")
        torch.cuda.empty_cache()
        
        file_to_layers = defaultdict(list)
        for layer in layers:
            file_to_layers[self.layer_to_file[layer]].append(layer)
        
        start = time.perf_counter()
        for fpath, chunk_layers in file_to_layers.items():
            with safe_open(fpath, framework='pt') as f:
                weights = {}
                for layer in chunk_layers:
                    for name in ['qkv_proj', 'o_proj']:
                        key = f'model.layers.{layer}.self_attn.{name}.weight'
                        try:
                            w = f.get_tensor(key)
                            weights[(layer, name)] = w.pin_memory().cuda(non_blocking=True).half()
                        except:
                            continue
                torch.cuda.synchronize()
                
                # Process ALL samples with these weights
                for sample_idx in range(n_samples):
                    for (layer, name), w in weights.items():
                        x = torch.randn(1, 32, w.shape[1], device='cuda', dtype=torch.float16, requires_grad=True)
                        y = F.linear(x, w)
                        loss = y.sum() / n_samples
                        loss.backward()
                
                del weights
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        optimized_ms = (time.perf_counter() - start) * 1000
        optimized_per_sample = optimized_ms / n_samples
        print(f"  Optimized: {optimized_ms:.0f}ms total, {optimized_per_sample:.0f}ms/sample")
        
        speedup = baseline_per_sample / optimized_per_sample
        improvement_pct = (baseline_per_sample - optimized_per_sample) / baseline_per_sample * 100
        
        print(f"\n  Speedup: {speedup:.2f}x ({improvement_pct:.1f}% faster)")
        
        return {
            'baseline_total_ms': baseline_ms,
            'baseline_per_sample_ms': baseline_per_sample,
            'optimized_total_ms': optimized_ms,
            'optimized_per_sample_ms': optimized_per_sample,
            'speedup': speedup,
            'improvement_pct': improvement_pct,
        }


class AutoOptimizer:
    """
    Main interface for automatic optimization.
    
    Usage:
        optimizer = AutoOptimizer("models/phi-4")
        config = optimizer.find_optimal_config()
        results = optimizer.benchmark_all()
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Profile hardware
        print("=" * 60)
        print("AUTO-OPTIMIZER: Universal Training & Inference Optimizer")
        print("=" * 60)
        
        print("\n[1/4] Profiling Hardware...")
        self.hardware_profiler = HardwareProfiler()
        self.hardware = self.hardware_profiler.profile()
        self._print_hardware()
        
        # Profile model
        print("\n[2/4] Analyzing Model...")
        self.model_profiler = ModelProfiler(model_path)
        self.model = self.model_profiler.profile()
        self._print_model()
        
        # Auto-tuner
        self.tuner = AutoTuner(self.hardware, self.model)
        self.config = None
    
    def _print_hardware(self):
        h = self.hardware
        print(f"  Device: {h.device_name}")
        print(f"  VRAM: {h.vram_total_gb:.1f}GB total, {h.vram_available_gb:.1f}GB available")
        print(f"  Compute: SM {h.compute_capability[0]}.{h.compute_capability[1]}, ~{h.cuda_cores} cores")
        print(f"  Memory BW: {h.memory_bandwidth_gbps:.0f} GB/s")
        print(f"  PCIe BW: {h.pcie_bandwidth_gbps:.2f} GB/s (regular), {h.cpu_to_gpu_pinned_bandwidth_gbps:.2f} GB/s (pinned)")
    
    def _print_model(self):
        m = self.model
        print(f"  Model: {m.name}")
        print(f"  Layers: {m.total_layers}")
        print(f"  Hidden: {m.hidden_size}, Intermediate: {m.intermediate_size}")
        print(f"  Attention: {m.num_attention_heads} heads, {m.num_kv_heads} KV heads")
        print(f"  Layer size: {m.layer_size_mb:.0f}MB, Total: {m.total_size_gb:.1f}GB")
    
    def find_optimal_config(
        self,
        target_layers: Optional[List[int]] = None,
        max_vram_usage: float = 0.85,
    ) -> OptimalConfig:
        """Find optimal configuration for this hardware/model."""
        print("\n[3/4] Finding Optimal Configuration...")
        
        self.config = self.tuner.find_optimal_config(target_layers, max_vram_usage)
        
        print(f"\n  ✓ Optimal Config Found:")
        print(f"    chunk_size: {self.config.chunk_size}")
        print(f"    batch_size: {self.config.batch_size}")
        print(f"    sparse_layers: {self.config.sparse_layers}")
        print(f"    pinned_memory: {self.config.use_pinned_memory}")
        print(f"    estimated_vram: {self.config.estimated_vram_mb}MB")
        print(f"    estimated_time: {self.config.estimated_time_per_sample_ms:.0f}ms/sample")
        
        return self.config
    
    def benchmark_all(self) -> Dict[str, Any]:
        """Run full benchmark suite comparing optimized vs baseline."""
        if self.config is None:
            self.find_optimal_config()
        
        print("\n[4/4] Running Benchmarks...")
        
        benchmark = BenchmarkSuite(self.model_path, self.config)
        
        inference_results = benchmark.benchmark_inference()
        training_results = benchmark.benchmark_training()
        
        results = {
            'hardware': self.hardware.to_dict(),
            'model': {
                'name': self.model.name,
                'total_layers': self.model.total_layers,
                'total_size_gb': self.model.total_size_gb,
            },
            'config': self.config.to_dict(),
            'inference': inference_results,
            'training': training_results,
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"\nHardware: {self.hardware.device_name}")
        print(f"Model: {self.model.name} ({self.model.total_size_gb:.1f}GB)")
        print(f"\nOptimal Configuration:")
        print(f"  chunk_size={self.config.chunk_size}, batch_size={self.config.batch_size}")
        print(f"  VRAM usage: ~{self.config.estimated_vram_mb}MB")
        print(f"\nPerformance Improvements:")
        print(f"  Inference: {inference_results['speedup']:.2f}x faster ({inference_results['improvement_pct']:.1f}% improvement)")
        print(f"  Training:  {training_results['speedup']:.2f}x faster ({training_results['improvement_pct']:.1f}% improvement)")
        print("=" * 60)
        
        return results
    
    def save_config(self, path: str):
        """Save optimal config to JSON."""
        if self.config is None:
            raise ValueError("No config to save. Run find_optimal_config() first.")
        
        config_dict = {
            'hardware': self.hardware.to_dict(),
            'model': {
                'name': self.model.name,
                'path': str(self.model_path),
                'total_layers': self.model.total_layers,
            },
            'optimal_config': self.config.to_dict(),
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved config to {path}")
    
    @classmethod
    def load_config(cls, path: str) -> OptimalConfig:
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        return OptimalConfig(**data['optimal_config'])


def main():
    """Run auto-optimizer on phi-4 model."""
    import sys
    
    model_path = "models/phi-4"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Usage: python auto_optimizer.py [model_path]")
        return
    
    # Run auto-optimizer
    optimizer = AutoOptimizer(model_path)
    config = optimizer.find_optimal_config()
    results = optimizer.benchmark_all()
    
    # Save config
    config_path = Path(model_path) / "optimal_config.json"
    optimizer.save_config(str(config_path))


if __name__ == "__main__":
    main()
