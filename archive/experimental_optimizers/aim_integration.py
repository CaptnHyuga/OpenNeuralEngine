"""
AIM Integration for ONN Training & Benchmarks

Provides experiment tracking via AIM Docker container:
- Real-time metric visualization
- Hardware utilization tracking
- Training/inference comparison
- Model architecture inspection

Usage:
    # Start AIM server
    python -m src.benchmarks.aim_integration --start
    
    # Run benchmark with tracking
    python -m src.benchmarks.aim_integration --benchmark
    
    # Stop server
    python -m src.benchmarks.aim_integration --stop
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
import time
import json
import os
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Check AIM availability
try:
    from aim import Run, Image, Figure
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False


AIM_CONTAINER_NAME = "onn-aim-tracker"
AIM_PORT = 53800
AIM_REPO_PATH = Path("./aim_data").absolute()


class AIMServer:
    """Manage AIM Docker container."""
    
    @staticmethod
    def is_running() -> bool:
        """Check if AIM container is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={AIM_CONTAINER_NAME}"],
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except:
            return False
    
    @staticmethod
    def start() -> bool:
        """Start AIM server in Docker."""
        if AIMServer.is_running():
            print(f"✓ AIM server already running on port {AIM_PORT}")
            return True
        
        print(f"Starting AIM server...")
        
        # Create data directory
        AIM_REPO_PATH.mkdir(exist_ok=True)
        
        # Start container
        cmd = [
            "docker", "run", "-d",
            "--name", AIM_CONTAINER_NAME,
            "-p", f"{AIM_PORT}:{AIM_PORT}",
            "-v", f"{AIM_REPO_PATH}:/aim",
            "aimstack/aim:latest",
            "up", "--host", "0.0.0.0", "--port", str(AIM_PORT)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ AIM server started on http://localhost:{AIM_PORT}")
                print(f"  Container: {AIM_CONTAINER_NAME}")
                print(f"  Data: {AIM_REPO_PATH}")
                time.sleep(3)  # Wait for server to initialize
                return True
            else:
                print(f"✗ Failed to start AIM server: {result.stderr}")
                return False
        except Exception as e:
            print(f"✗ Error starting AIM: {e}")
            return False
    
    @staticmethod
    def stop() -> bool:
        """Stop AIM server."""
        if not AIMServer.is_running():
            print("AIM server not running")
            return True
        
        print("Stopping AIM server...")
        try:
            subprocess.run(["docker", "stop", AIM_CONTAINER_NAME], capture_output=True)
            subprocess.run(["docker", "rm", AIM_CONTAINER_NAME], capture_output=True)
            print("✓ AIM server stopped")
            return True
        except Exception as e:
            print(f"✗ Error stopping AIM: {e}")
            return False
    
    @staticmethod
    def get_url() -> str:
        """Get AIM dashboard URL."""
        return f"http://localhost:{AIM_PORT}"


class ONNExperimentTracker:
    """
    Experiment tracker for ONN training and benchmarks.
    
    Tracks:
    - Training metrics (loss, throughput, memory)
    - Hardware utilization
    - Model architecture info
    - Optimization comparisons
    """
    
    def __init__(
        self,
        experiment: str = "onn-training",
        run_name: Optional[str] = None,
        mode: str = "optimized",  # "optimized", "baseline", "comparison"
        auto_start_server: bool = True,
    ):
        self.experiment = experiment
        self.mode = mode
        self.run = None
        
        # Auto-start AIM server if needed
        if auto_start_server and not AIMServer.is_running():
            AIMServer.start()
        
        if not AIM_AVAILABLE:
            print("⚠️ AIM not installed. Run: pip install aim")
            return
        
        # Initialize AIM run
        try:
            self.run = Run(
                repo=str(AIM_REPO_PATH),
                experiment=experiment,
            )
            
            # Set run name
            if run_name:
                self.run.name = run_name
            else:
                self.run.name = f"{mode}-{int(time.time())}"
            
            # Log basic info
            self.run["mode"] = mode
            self.run["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"✓ Tracking experiment: {experiment}")
            print(f"  Run: {self.run.name}")
            print(f"  Dashboard: {AIMServer.get_url()}")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize AIM: {e}")
            self.run = None
    
    def log_hardware(self, info: Dict[str, Any]):
        """Log hardware information."""
        if self.run:
            for key, value in info.items():
                self.run[f"hardware/{key}"] = value
    
    def log_model_config(self, config: Dict[str, Any]):
        """Log model configuration."""
        if self.run:
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    self.run[f"model/{key}"] = json.dumps(value)
                else:
                    self.run[f"model/{key}"] = value
    
    def log_optimizer_config(self, config: Dict[str, Any]):
        """Log optimizer configuration."""
        if self.run:
            for key, value in config.items():
                self.run[f"optimizer/{key}"] = value
    
    def track_training(
        self,
        step: int,
        loss: float,
        throughput: Optional[float] = None,
        memory_mb: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ):
        """Track training metrics."""
        if self.run:
            self.run.track(loss, name="loss", step=step)
            if throughput is not None:
                self.run.track(throughput, name="throughput", step=step)
            if memory_mb is not None:
                self.run.track(memory_mb, name="memory_mb", step=step)
            if latency_ms is not None:
                self.run.track(latency_ms, name="latency_ms", step=step)
    
    def track_batch(
        self,
        batch_idx: int,
        loss: float,
        batch_time_ms: float,
        samples_in_batch: int,
    ):
        """Track per-batch metrics."""
        if self.run:
            self.run.track(loss, name="batch_loss", step=batch_idx, context={"subset": "train"})
            self.run.track(batch_time_ms, name="batch_time_ms", step=batch_idx)
            self.run.track(
                samples_in_batch / (batch_time_ms / 1000),
                name="samples_per_sec",
                step=batch_idx,
            )
    
    def track_epoch(
        self,
        epoch: int,
        avg_loss: float,
        total_time_s: float,
        total_samples: int,
    ):
        """Track per-epoch metrics."""
        if self.run:
            self.run.track(avg_loss, name="epoch_loss", step=epoch)
            self.run.track(total_time_s, name="epoch_time_s", step=epoch)
            self.run.track(total_samples / total_time_s, name="epoch_throughput", step=epoch)
    
    def track_memory(self, step: int):
        """Track GPU memory usage."""
        import torch
        if torch.cuda.is_available() and self.run:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            
            self.run.track(allocated, name="gpu_memory_allocated_mb", step=step)
            self.run.track(reserved, name="gpu_memory_reserved_mb", step=step)
            self.run.track(peak, name="gpu_memory_peak_mb", step=step)
    
    def log_benchmark_results(self, results: Dict[str, Any]):
        """Log final benchmark results."""
        if self.run:
            # Flatten and log
            for key, value in results.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (int, float, str, bool)):
                            self.run[f"results/{key}/{k}"] = v
                elif isinstance(value, (int, float, str, bool)):
                    self.run[f"results/{key}"] = value
    
    def log_comparison(
        self,
        optimized: Dict[str, Any],
        baseline: Dict[str, Any],
    ):
        """Log comparison between optimized and baseline."""
        if self.run:
            speedup = baseline.get("time_per_sample_ms", 1) / max(optimized.get("time_per_sample_ms", 1), 1)
            throughput_ratio = optimized.get("samples_per_second", 1) / max(baseline.get("samples_per_second", 0.001), 0.001)
            
            self.run["comparison/speedup_x"] = speedup
            self.run["comparison/throughput_ratio"] = throughput_ratio
            self.run["comparison/optimized_time_ms"] = optimized.get("time_per_sample_ms")
            self.run["comparison/baseline_time_ms"] = baseline.get("time_per_sample_ms")
    
    def close(self):
        """Close the run."""
        if self.run:
            self.run.close()
            print(f"✓ Run completed. View at: {AIMServer.get_url()}")


def run_tracked_benchmark(model_path: str = "models/phi-4"):
    """Run benchmark with full AIM tracking."""
    import torch
    from src.benchmarks.training_benchmark import TrainingBenchmark
    
    print("=" * 70)
    print("ONN TRACKED BENCHMARK")
    print("=" * 70)
    
    # Initialize tracker
    tracker = ONNExperimentTracker(
        experiment="onn-benchmark",
        mode="comparison",
    )
    
    # Log hardware info
    if torch.cuda.is_available():
        tracker.log_hardware({
            "device": torch.cuda.get_device_name(),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "cuda_version": torch.version.cuda,
        })
    
    # Load model config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        tracker.log_model_config({
            "name": Path(model_path).name,
            "hidden_size": model_config.get("hidden_size"),
            "num_layers": model_config.get("num_hidden_layers"),
            "num_params": model_config.get("num_parameters", "unknown"),
        })
    
    # Load optimizer config
    opt_config_path = Path(model_path) / "universal_config.json"
    if opt_config_path.exists():
        with open(opt_config_path) as f:
            opt_config = json.load(f)
        tracker.log_optimizer_config({
            "chunk_size": opt_config.get("config", {}).get("chunk_size"),
            "batch_size": opt_config.get("config", {}).get("batch_size"),
            "sparse_layers": str(opt_config.get("config", {}).get("active_layers")),
        })
    
    # Run benchmark
    benchmark = TrainingBenchmark(model_path)
    
    # Optimized run
    print("\n--- OPTIMIZED RUN ---")
    opt_tracker = ONNExperimentTracker(
        experiment="onn-benchmark",
        run_name="optimized-run",
        mode="optimized",
        auto_start_server=False,
    )
    
    from src.wrappers.batched_sparse import BatchedSparseTrainer
    
    if opt_config_path.exists():
        with open(opt_config_path) as f:
            config = json.load(f)
        chunk_size = config.get("config", {}).get("chunk_size", 3)
        sparse_layers = config.get("config", {}).get("active_layers", [0,1,2,3,36,37,38,39])
    else:
        chunk_size = 3
        sparse_layers = [0, 1, 2, 3, 36, 37, 38, 39]
    
    trainer = BatchedSparseTrainer(
        model_path=model_path,
        chunk_size=chunk_size,
        sparse_layers=sparse_layers,
    )
    
    data = benchmark.training_data[:64]
    batch_size = 32
    
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    total_start = time.time()
    
    for batch_idx, i in enumerate(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        batch_start = time.time()
        loss = trainer.train_step(batch)
        batch_time = (time.time() - batch_start) * 1000
        
        opt_tracker.track_batch(batch_idx, loss, batch_time, len(batch))
        opt_tracker.track_memory(batch_idx)
        print(f"  Batch {batch_idx}: loss={loss:.4f}, time={batch_time:.0f}ms")
    
    opt_total_time = time.time() - total_start
    opt_peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    opt_results = {
        "time_per_sample_ms": (opt_total_time * 1000) / len(data),
        "samples_per_second": len(data) / opt_total_time,
        "peak_memory_mb": opt_peak_memory,
    }
    
    opt_tracker.log_benchmark_results({"optimized": opt_results})
    opt_tracker.close()
    
    del trainer
    torch.cuda.empty_cache()
    
    # Baseline run (fewer samples)
    print("\n--- BASELINE RUN ---")
    base_result = benchmark.benchmark_baseline(4)
    
    base_results = {
        "time_per_sample_ms": base_result.time_per_sample_ms,
        "samples_per_second": base_result.samples_per_second,
        "peak_memory_mb": base_result.peak_memory_mb,
    }
    
    # Log comparison
    tracker.log_comparison(opt_results, base_results)
    tracker.log_benchmark_results({
        "optimized": opt_results,
        "baseline": base_results,
    })
    
    # Print summary
    speedup = base_results["time_per_sample_ms"] / opt_results["time_per_sample_ms"]
    
    print("\n" + "=" * 70)
    print("TRACKED BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\n⚡ SPEEDUP: {speedup:.1f}x faster")
    print(f"📊 View dashboard: {AIMServer.get_url()}")
    
    tracker.close()
    
    return {
        "optimized": opt_results,
        "baseline": base_results,
        "speedup": speedup,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ONN AIM Integration")
    parser.add_argument("--start", action="store_true", help="Start AIM server")
    parser.add_argument("--stop", action="store_true", help="Stop AIM server")
    parser.add_argument("--status", action="store_true", help="Check server status")
    parser.add_argument("--benchmark", action="store_true", help="Run tracked benchmark")
    parser.add_argument("--model", default="models/phi-4", help="Model path")
    
    args = parser.parse_args()
    
    if args.start:
        AIMServer.start()
    elif args.stop:
        AIMServer.stop()
    elif args.status:
        if AIMServer.is_running():
            print(f"✓ AIM server running at {AIMServer.get_url()}")
        else:
            print("✗ AIM server not running")
    elif args.benchmark:
        run_tracked_benchmark(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
