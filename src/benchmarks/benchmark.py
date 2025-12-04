"""
ONN Benchmark Suite
===================

Unified benchmark suite to validate training pipeline performance.

Usage:
    python -m src.benchmarks.benchmark --quick    # Fast test (~1min)
    python -m src.benchmarks.benchmark --full     # Full benchmark (~10min)
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class BenchmarkSuite:
    """Unified benchmark suite for ONN."""
    
    def __init__(self, model_path: str = "models/phi-4"):
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}
        
        # Load model config
        with open(self.model_path / "config.json") as f:
            config = json.load(f)
        self.hidden_size = config.get("hidden_size", 5120)
        self.intermediate_size = config.get("intermediate_size", 17920)
        self.num_layers = config.get("num_hidden_layers", 40)
        
        print(f"Model: {self.model_path.name}")
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def benchmark_speed(self, batch_sizes: List[int] = [1, 8, 32]) -> Dict:
        """Benchmark training speed at different batch sizes."""
        print("\n" + "="*60)
        print("SPEED BENCHMARK")
        print("="*60)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Create simple LoRA layer
            A = nn.Parameter(torch.randn(8, self.hidden_size, device=self.device, dtype=torch.float32) * 0.01)
            B = nn.Parameter(torch.zeros(self.hidden_size, 8, device=self.device, dtype=torch.float32))
            optimizer = torch.optim.AdamW([A, B], lr=1e-4)
            
            # Load one layer weight
            safetensor_file = list(self.model_path.glob("model-*.safetensors"))[0]
            with safe_open(str(safetensor_file), framework='pt') as f:
                for key in f.keys():
                    if 'o_proj.weight' in key:
                        base_weight = f.get_tensor(key).cuda().float()
                        break
            
            # Warmup
            for _ in range(3):
                x = torch.randn(batch_size, 32, self.hidden_size, device=self.device, dtype=torch.float32)
                optimizer.zero_grad()
                with torch.no_grad():
                    base_out = F.linear(x, base_weight)
                lora_out = F.linear(F.linear(x, A), B) * 2.0
                out = base_out + lora_out
                loss = F.mse_loss(out, torch.zeros_like(out))
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            losses = []
            n_iters = 10
            
            for _ in range(n_iters):
                x = torch.randn(batch_size, 32, self.hidden_size, device=self.device, dtype=torch.float32)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                optimizer.zero_grad()
                with torch.no_grad():
                    base_out = F.linear(x, base_weight)
                lora_out = F.linear(F.linear(x, A), B) * 2.0
                out = base_out + lora_out
                loss = F.mse_loss(out, torch.zeros_like(out))
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                times.append(elapsed * 1000)  # ms
                losses.append(loss.item())
            
            avg_time = np.mean(times)
            time_per_sample = avg_time / batch_size
            
            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "avg_time_ms": avg_time,
                "time_per_sample_ms": time_per_sample,
                "samples_per_sec": 1000 / time_per_sample,
                "final_loss": losses[-1],
            }
            
            print(f"  Time: {avg_time:.1f}ms/batch, {time_per_sample:.1f}ms/sample")
            print(f"  Throughput: {1000/time_per_sample:.2f} samples/sec")
            
            del A, B, base_weight, optimizer
            torch.cuda.empty_cache()
        
        self.results["speed"] = results
        return results
    
    def benchmark_learning(self, epochs: int = 10, samples: int = 20) -> Dict:
        """Validate that training actually learns (loss decreases)."""
        print("\n" + "="*60)
        print("LEARNING VALIDATION")
        print("="*60)
        
        # Create LoRA layer
        A = nn.Parameter(torch.randn(8, self.hidden_size, device=self.device, dtype=torch.float32) * 0.01)
        B = nn.Parameter(torch.zeros(self.hidden_size, 8, device=self.device, dtype=torch.float32))
        optimizer = torch.optim.AdamW([A, B], lr=1e-3)
        
        # Load weight
        safetensor_file = list(self.model_path.glob("model-*.safetensors"))[0]
        with safe_open(str(safetensor_file), framework='pt') as f:
            for key in f.keys():
                if 'o_proj.weight' in key:
                    base_weight = f.get_tensor(key).cuda().float()
                    break
        
        # Create training data
        train_data = [
            (torch.randn(8, 32, self.hidden_size, device=self.device, dtype=torch.float32),
             torch.randn(8, 32, self.hidden_size, device=self.device, dtype=torch.float32) * 0.1)
            for _ in range(samples)
        ]
        
        # Train
        loss_history = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for x, target in train_data:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    base_out = F.linear(x, base_weight)
                lora_out = F.linear(F.linear(x, A), B) * 2.0
                out = base_out + lora_out
                
                loss = F.mse_loss(out, target)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        # Analyze
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        is_learning = final_loss < initial_loss
        
        results = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_pct": loss_reduction,
            "is_learning": is_learning,
            "loss_history": loss_history,
        }
        
        print(f"\n{'='*40}")
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Reduction: {loss_reduction:.1f}%")
        print(f"IS LEARNING: {'✅ YES' if is_learning else '❌ NO'}")
        print(f"{'='*40}")
        
        self.results["learning"] = results
        
        del A, B, base_weight, optimizer
        torch.cuda.empty_cache()
        
        return results
    
    def benchmark_memory(self) -> Dict:
        """Benchmark VRAM usage."""
        print("\n" + "="*60)
        print("MEMORY BENCHMARK")
        print("="*60)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        baseline_mem = torch.cuda.memory_allocated() / 1e6
        
        # Load one layer weight
        safetensor_file = list(self.model_path.glob("model-*.safetensors"))[0]
        with safe_open(str(safetensor_file), framework='pt') as f:
            for key in f.keys():
                if 'o_proj.weight' in key:
                    base_weight = f.get_tensor(key).cuda().float()
                    break
        
        after_weight_mem = torch.cuda.memory_allocated() / 1e6
        
        # Create LoRA
        A = nn.Parameter(torch.randn(8, self.hidden_size, device=self.device, dtype=torch.float32) * 0.01)
        B = nn.Parameter(torch.zeros(self.hidden_size, 8, device=self.device, dtype=torch.float32))
        
        after_lora_mem = torch.cuda.memory_allocated() / 1e6
        
        # Forward pass
        x = torch.randn(32, 32, self.hidden_size, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            base_out = F.linear(x, base_weight)
        lora_out = F.linear(F.linear(x, A), B) * 2.0
        out = base_out + lora_out
        loss = F.mse_loss(out, torch.zeros_like(out))
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        
        results = {
            "baseline_mb": baseline_mem,
            "after_weight_mb": after_weight_mem,
            "after_lora_mb": after_lora_mem,
            "peak_mb": peak_mem,
            "weight_size_mb": after_weight_mem - baseline_mem,
            "lora_size_mb": after_lora_mem - after_weight_mem,
        }
        
        print(f"Weight memory: {results['weight_size_mb']:.1f}MB")
        print(f"LoRA memory: {results['lora_size_mb']:.1f}MB")
        print(f"Peak memory: {peak_mem:.1f}MB")
        
        self.results["memory"] = results
        
        del A, B, base_weight, x, out, loss
        torch.cuda.empty_cache()
        
        return results
    
    def run_all(self, quick: bool = False) -> Dict:
        """Run all benchmarks."""
        print("="*60)
        print("ONN BENCHMARK SUITE")
        print("="*60)
        print(f"Mode: {'Quick' if quick else 'Full'}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if quick:
            self.benchmark_speed([1, 32])
            self.benchmark_learning(epochs=5, samples=10)
        else:
            self.benchmark_speed([1, 8, 32, 64])
            self.benchmark_learning(epochs=10, samples=20)
        
        self.benchmark_memory()
        
        # Summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        if "speed" in self.results:
            batch_1 = self.results["speed"].get("batch_1", {})
            batch_32 = self.results["speed"].get("batch_32", {})
            if batch_1 and batch_32:
                speedup = batch_1.get("time_per_sample_ms", 1) / batch_32.get("time_per_sample_ms", 1)
                print(f"Speed: {speedup:.1f}x speedup (batch=32 vs batch=1)")
        
        if "learning" in self.results:
            is_learning = self.results["learning"].get("is_learning", False)
            reduction = self.results["learning"].get("loss_reduction_pct", 0)
            print(f"Learning: {'✅ YES' if is_learning else '❌ NO'} ({reduction:.1f}% reduction)")
        
        if "memory" in self.results:
            peak = self.results["memory"].get("peak_mb", 0)
            print(f"Memory: {peak:.0f}MB peak VRAM")
        
        return self.results
    
    def save_results(self, path: str = None):
        """Save benchmark results."""
        if path is None:
            path = f"benchmark_results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Make serializable
        def make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, list):
                return [make_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            return obj
        
        with open(path, 'w') as f:
            json.dump(make_serializable(self.results), f, indent=2)
        
        print(f"\n💾 Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="ONN Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (~1min)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (~10min)")
    parser.add_argument("--model", default="models/phi-4", help="Model path")
    
    args = parser.parse_args()
    
    suite = BenchmarkSuite(args.model)
    
    quick = args.quick or not args.full
    suite.run_all(quick=quick)
    suite.save_results()


if __name__ == "__main__":
    main()
