"""
AIM Metrics Logger via Docker
============================

Logs metrics to AIM by running aim commands inside the Docker container.
This bypasses the need for the aim Python package on Windows.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


class AIMDockerLogger:
    """Log metrics to AIM via Docker container."""
    
    def __init__(self, container_name: str = "onn-aim-tracker"):
        self.container = container_name
        self.run_hash = None
    
    def _exec(self, cmd: str) -> str:
        """Execute command in AIM container."""
        full_cmd = f'docker exec {self.container} {cmd}'
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    
    def start_run(self, experiment: str = "onn-training") -> str:
        """Start a new AIM run."""
        # Create a simple Python script to start run and return hash
        script = f'''
import aim
run = aim.Run(experiment="{experiment}")
print(run.hash)
'''
        # Write script to mounted volume
        script_path = Path("aim_data/_start_run.py")
        script_path.write_text(script)
        
        # Execute in container
        result = self._exec("python /aim/_start_run.py")
        self.run_hash = result.strip().split('\n')[-1]
        
        # Cleanup
        script_path.unlink(missing_ok=True)
        
        print(f"Started AIM run: {self.run_hash}")
        return self.run_hash
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric to the current run."""
        if not self.run_hash:
            self.start_run()
        
        step_str = f", step={step}" if step else ""
        script = f'''
import aim
run = aim.Run(run_hash="{self.run_hash}")
run.track({value}, name="{name}"{step_str})
'''
        script_path = Path("aim_data/_log_metric.py")
        script_path.write_text(script)
        self._exec("python /aim/_log_metric.py")
        script_path.unlink(missing_ok=True)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def close(self):
        """Alias for finish_run."""
        self.finish_run()
    
    def log_params(self, params: dict):
        """Log parameters to the current run."""
        if not self.run_hash:
            self.start_run()
        
        params_json = json.dumps(params)
        script = f'''
import aim
import json
run = aim.Run(run_hash="{self.run_hash}")
params = json.loads('{params_json}')
for k, v in params.items():
    run[k] = v
'''
        script_path = Path("aim_data/_log_params.py")
        script_path.write_text(script)
        self._exec("python /aim/_log_params.py")
        script_path.unlink(missing_ok=True)
    
    def finish_run(self):
        """Close the current run."""
        if not self.run_hash:
            return
        
        script = f'''
import aim
run = aim.Run(run_hash="{self.run_hash}")
run.close()
'''
        script_path = Path("aim_data/_finish_run.py")
        script_path.write_text(script)
        self._exec("python /aim/_finish_run.py")
        script_path.unlink(missing_ok=True)
        
        print(f"Finished AIM run: {self.run_hash}")
        self.run_hash = None


def log_benchmark_to_aim(results: dict, experiment: str = "onn-benchmark"):
    """Log benchmark results to AIM."""
    logger = AIMDockerLogger()
    logger.start_run(experiment)
    
    # Log speed metrics
    if "speed" in results:
        for batch_name, metrics in results["speed"].items():
            logger.log_metric(f"speed/{batch_name}/time_per_sample_ms", metrics["time_per_sample_ms"])
            logger.log_metric(f"speed/{batch_name}/samples_per_sec", metrics["samples_per_sec"])
    
    # Log learning metrics  
    if "learning" in results:
        learning = results["learning"]
        logger.log_metric("learning/initial_loss", learning["initial_loss"])
        logger.log_metric("learning/final_loss", learning["final_loss"])
        logger.log_metric("learning/reduction_pct", learning["loss_reduction_pct"])
        
        # Log loss history
        for step, loss in enumerate(learning.get("loss_history", [])):
            logger.log_metric("learning/loss", loss, step=step)
    
    # Log memory metrics
    if "memory" in results:
        memory = results["memory"]
        logger.log_metric("memory/peak_mb", memory["peak_mb"])
        logger.log_metric("memory/weight_mb", memory["weight_size_mb"])
        logger.log_metric("memory/lora_mb", memory["lora_size_mb"])
    
    # Log parameters
    logger.log_params({
        "model": "phi-4",
        "timestamp": datetime.now().isoformat(),
        "hardware": "GTX 1650 Max-Q",
    })
    
    logger.finish_run()
    print(f"\n✅ Results logged to AIM")
    print(f"   View at: http://localhost:53800")


if __name__ == "__main__":
    # Test logging
    print("Testing AIM Docker Logger...")
    
    # Load latest benchmark results
    results_dir = Path("benchmark_results")
    latest = sorted(results_dir.glob("benchmark_*.json"))[-1] if list(results_dir.glob("benchmark_*.json")) else None
    
    if latest:
        print(f"Loading: {latest}")
        with open(latest) as f:
            results = json.load(f)
        log_benchmark_to_aim(results)
    else:
        print("No benchmark results found. Run benchmark first.")
