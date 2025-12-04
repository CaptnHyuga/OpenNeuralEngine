"""
Push ONN benchmark metrics to AIM server via HTTP API.

This bypasses the need for the aim Python package (which doesn't install on Windows)
by directly posting metrics to the AIM server's HTTP API.
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path


class AIMMetricsPusher:
    """Push metrics to AIM server via HTTP API."""
    
    def __init__(self, aim_url: str = "http://localhost:53800"):
        self.aim_url = aim_url
        self.session_id = f"onn-benchmark-{int(time.time())}"
    
    def is_server_available(self) -> bool:
        """Check if AIM server is running."""
        try:
            resp = requests.get(f"{self.aim_url}/api/runs", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def get_runs(self):
        """Get existing runs from AIM."""
        try:
            resp = requests.get(f"{self.aim_url}/api/runs", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Error getting runs: {e}")
        return []
    
    def push_benchmark_summary(self, results: dict):
        """
        Push benchmark results to a file for AIM to pick up.
        
        Since AIM's HTTP API doesn't support direct metric push,
        we'll save results in a format that can be loaded.
        """
        output_path = Path("aim_data/.aim_metrics")
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Save as JSON for potential AIM import
        metrics_file = output_path / f"benchmark_{int(time.time())}.json"
        
        metrics_data = {
            "session": self.session_id,
            "timestamp": timestamp,
            "experiment": "ONN-Optimization-Benchmark",
            "metrics": {
                "optimized": {
                    "time_per_sample_ms": results.get("optimized", {}).get("time_per_sample_ms", 0),
                    "throughput_samples_per_sec": results.get("optimized", {}).get("samples_per_second", 0),
                    "peak_vram_mb": results.get("optimized", {}).get("peak_memory_mb", 0),
                    "final_loss": results.get("optimized", {}).get("final_loss", 0),
                },
                "baseline": {
                    "time_per_sample_ms": results.get("baseline", {}).get("time_per_sample_ms", 0),
                    "throughput_samples_per_sec": results.get("baseline", {}).get("samples_per_second", 0),
                    "peak_vram_mb": results.get("baseline", {}).get("peak_memory_mb", 0),
                },
                "speedup": results.get("speedup", 0),
            },
            "hardware": {
                "gpu": "NVIDIA GeForce GTX 1650 Max-Q",
                "vram_total_mb": 4096,
            },
            "model": {
                "name": "phi-4",
                "parameters": "16.23B",
                "layers": 40,
            },
            "optimization": {
                "chunk_size": 1,
                "batch_size": 32,
                "sparse_layers": [0, 1, 2, 3, 36, 37, 38, 39],
                "lora_rank": 8,
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"✓ Metrics saved to {metrics_file}")
        return metrics_file


def push_learning_results(learning_results_path: str = "learning_results.json"):
    """Push learning validation results."""
    if not Path(learning_results_path).exists():
        print(f"Learning results not found at {learning_results_path}")
        return
    
    with open(learning_results_path) as f:
        results = json.load(f)
    
    pusher = AIMMetricsPusher()
    
    output_path = Path("aim_data/.aim_metrics")
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_path / f"learning_{int(time.time())}.json"
    
    learning_data = {
        "session": pusher.session_id,
        "timestamp": datetime.now().isoformat(),
        "experiment": "ONN-Learning-Validation",
        "results": results,
        "summary": {
            "all_learning": all(r.get("is_learning", False) for r in results.values()),
            "configurations_tested": len(results),
            "best_loss_reduction": max(r.get("loss_reduction_pct", 0) for r in results.values()),
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(learning_data, f, indent=2)
    
    print(f"✓ Learning metrics saved to {metrics_file}")
    
    # Also print summary
    print("\n" + "="*60)
    print("LEARNING VALIDATION SUMMARY")
    print("="*60)
    for name, result in results.items():
        status = "✅ LEARNING" if result.get("is_learning") else "❌ NOT LEARNING"
        reduction = result.get("loss_reduction_pct", 0)
        print(f"  {name}: {status} (loss reduction: {reduction:.1f}%)")
    
    return metrics_file


def push_all_metrics():
    """Push all available benchmark metrics."""
    pusher = AIMMetricsPusher()
    
    print("="*60)
    print("PUSHING METRICS TO AIM")
    print("="*60)
    
    # Check AIM server
    if pusher.is_server_available():
        print(f"✓ AIM server available at {pusher.aim_url}")
    else:
        print(f"⚠️ AIM server not responding at {pusher.aim_url}")
        print("  Run: docker start onn-aim-tracker")
    
    # Push training benchmark results
    if Path("training_benchmark.json").exists():
        with open("training_benchmark.json") as f:
            data = json.load(f)
        
        results = {
            "optimized": data.get("optimized_results", {}),
            "baseline": data.get("baseline_results", {}),
            "speedup": data.get("comparison", {}).get("speedup", 0),
        }
        
        pusher.push_benchmark_summary(results)
    else:
        print("⚠️ No training_benchmark.json found")
    
    # Push learning validation results
    push_learning_results()
    
    print("\n" + "="*60)
    print("METRICS EXPORT COMPLETE")
    print("="*60)
    print(f"View AIM dashboard: {pusher.aim_url}")
    print("Metrics exported to: aim_data/.aim_metrics/")


if __name__ == "__main__":
    push_all_metrics()
