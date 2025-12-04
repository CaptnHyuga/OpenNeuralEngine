"""
Metrics Tracker for ONN Benchmarks

File-based tracking that generates HTML reports for visualization.
Works without external dependencies.

Usage:
    tracker = MetricsTracker("benchmark-run")
    tracker.track("loss", 0.5, step=1)
    tracker.track("throughput", 100, step=1)
    tracker.save_report()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class MetricSeries:
    """A series of metric values over time."""
    name: str
    values: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class MetricsTracker:
    """
    Simple file-based metrics tracker with HTML report generation.
    """
    
    def __init__(
        self,
        experiment: str = "onn-benchmark",
        run_name: Optional[str] = None,
        output_dir: str = "benchmark_results",
    ):
        self.experiment = experiment
        self.run_name = run_name or f"run-{int(time.time())}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics: Dict[str, MetricSeries] = {}
        self.params: Dict[str, Any] = {}
        self.artifacts: List[str] = []
        self.start_time = time.time()
        
        print(f"📊 Tracking experiment: {experiment}")
        print(f"   Run: {self.run_name}")
    
    def log_param(self, name: str, value: Any):
        """Log a parameter."""
        self.params[name] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        self.params.update(params)
    
    def track(self, name: str, value: float, step: Optional[int] = None):
        """Track a metric value."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name)
        
        series = self.metrics[name]
        series.values.append(value)
        series.steps.append(step if step is not None else len(series.values) - 1)
        series.timestamps.append(time.time() - self.start_time)
    
    def track_batch(
        self,
        batch_idx: int,
        loss: float,
        batch_time_ms: float,
        samples: int,
    ):
        """Track batch-level metrics."""
        self.track("batch_loss", loss, batch_idx)
        self.track("batch_time_ms", batch_time_ms, batch_idx)
        self.track("throughput", samples / (batch_time_ms / 1000), batch_idx)
    
    def track_memory(self, step: int):
        """Track GPU memory."""
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            self.track("gpu_memory_mb", allocated, step)
            self.track("gpu_peak_memory_mb", peak, step)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            "experiment": self.experiment,
            "run_name": self.run_name,
            "duration_s": time.time() - self.start_time,
            "params": self.params,
            "metrics": {},
        }
        
        for name, series in self.metrics.items():
            if series.values:
                summary["metrics"][name] = {
                    "final": series.values[-1],
                    "min": min(series.values),
                    "max": max(series.values),
                    "mean": sum(series.values) / len(series.values),
                    "count": len(series.values),
                }
        
        return summary
    
    def save_json(self, filename: Optional[str] = None) -> str:
        """Save metrics to JSON."""
        filename = filename or f"{self.run_name}.json"
        filepath = self.output_dir / filename
        
        data = {
            "experiment": self.experiment,
            "run_name": self.run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": time.time() - self.start_time,
            "params": self.params,
            "metrics": {
                name: asdict(series) for name, series in self.metrics.items()
            },
            "summary": self.get_summary()["metrics"],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved: {filepath}")
        return str(filepath)
    
    def save_html_report(self, filename: Optional[str] = None) -> str:
        """Generate and save HTML report with charts."""
        filename = filename or f"{self.run_name}.html"
        filepath = self.output_dir / filename
        
        summary = self.get_summary()
        
        # Generate chart data
        chart_data = {}
        for name, series in self.metrics.items():
            chart_data[name] = {
                "labels": series.steps,
                "values": series.values,
            }
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ONN Benchmark Report - {self.run_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #0d1117;
            color: #c9d1d9;
        }}
        h1, h2, h3 {{ color: #58a6ff; }}
        .header {{
            background: linear-gradient(135deg, #238636 0%, #1f6feb 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{ color: white; margin: 0; }}
        .header p {{ color: rgba(255,255,255,0.8); margin: 10px 0 0 0; }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric-box {{
            background: #21262d;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #58a6ff;
        }}
        .metric-label {{
            color: #8b949e;
            font-size: 0.9em;
        }}
        .chart-container {{
            background: #21262d;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }}
        th {{ color: #58a6ff; }}
        .highlight {{ color: #3fb950; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 ONN Benchmark Report</h1>
        <p>Experiment: {self.experiment} | Run: {self.run_name}</p>
        <p>Duration: {summary['duration_s']:.1f}s | Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="card">
        <h2>📊 Summary Metrics</h2>
        <div class="metrics-grid">
"""
        
        # Add summary metrics
        for name, stats in summary["metrics"].items():
            html += f"""
            <div class="metric-box">
                <div class="metric-value">{stats['final']:.2f}</div>
                <div class="metric-label">{name} (final)</div>
            </div>
"""
        
        html += """
        </div>
    </div>
    
    <div class="card">
        <h2>⚙️ Parameters</h2>
        <table>
"""
        
        for key, value in self.params.items():
            html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += """
        </table>
    </div>
"""
        
        # Add charts
        for name, series in self.metrics.items():
            if len(series.values) > 1:
                html += f"""
    <div class="card">
        <h3>{name}</h3>
        <div class="chart-container">
            <canvas id="chart-{name.replace('/', '-')}"></canvas>
        </div>
    </div>
"""
        
        # Chart scripts
        html += """
    <script>
"""
        
        for name, series in self.metrics.items():
            if len(series.values) > 1:
                chart_id = name.replace("/", "-")
                html += f"""
        new Chart(document.getElementById('chart-{chart_id}'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(series.steps)},
                datasets: [{{
                    label: '{name}',
                    data: {json.dumps(series.values)},
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: true,
                    tension: 0.3,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#c9d1d9' }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
                    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }}
                }}
            }}
        }});
"""
        
        html += """
    </script>
</body>
</html>
"""
        
        with open(filepath, "w") as f:
            f.write(html)
        
        print(f"📈 Report saved: {filepath}")
        return str(filepath)
    
    def save_report(self):
        """Save both JSON and HTML reports."""
        self.save_json()
        return self.save_html_report()


class ComparisonReport:
    """Generate comparison report between optimized and baseline."""
    
    @staticmethod
    def generate(
        optimized: Dict[str, Any],
        baseline: Dict[str, Any],
        output_path: str = "benchmark_results/comparison.html",
    ) -> str:
        """Generate HTML comparison report."""
        
        # Calculate comparison metrics
        speedup = baseline.get("time_per_sample_ms", 1) / max(optimized.get("time_per_sample_ms", 1), 0.001)
        throughput_ratio = optimized.get("samples_per_second", 1) / max(baseline.get("samples_per_second", 0.001), 0.001)
        memory_diff = optimized.get("peak_memory_mb", 0) - baseline.get("peak_memory_mb", 0)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ONN Optimization Comparison Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #0d1117;
            color: #c9d1d9;
        }}
        h1, h2 {{ color: #58a6ff; }}
        .header {{
            background: linear-gradient(135deg, #238636 0%, #da3633 100%);
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{ color: white; margin: 0; font-size: 2.5em; }}
        .speedup {{
            font-size: 4em;
            font-weight: bold;
            color: #3fb950;
            margin: 20px 0;
        }}
        .card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .optimized {{ border-left: 4px solid #3fb950; }}
        .baseline {{ border-left: 4px solid #da3633; }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #30363d;
        }}
        .metric-name {{ color: #8b949e; }}
        .metric-value {{ font-weight: bold; }}
        .chart-container {{
            height: 300px;
        }}
        .winner {{ color: #3fb950; }}
        .loser {{ color: #da3633; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ ONN Optimization Results</h1>
        <div class="speedup">{speedup:.1f}x FASTER</div>
        <p style="color: rgba(255,255,255,0.8);">Optimized vs Baseline Training Performance</p>
    </div>
    
    <div class="comparison-grid">
        <div class="card optimized">
            <h2>✅ Optimized</h2>
            <div class="metric">
                <span class="metric-name">Time per sample</span>
                <span class="metric-value winner">{optimized.get('time_per_sample_ms', 0):.0f}ms</span>
            </div>
            <div class="metric">
                <span class="metric-name">Throughput</span>
                <span class="metric-value winner">{optimized.get('samples_per_second', 0):.2f}/sec</span>
            </div>
            <div class="metric">
                <span class="metric-name">Peak Memory</span>
                <span class="metric-value">{optimized.get('peak_memory_mb', 0):.0f}MB</span>
            </div>
        </div>
        
        <div class="card baseline">
            <h2>❌ Baseline</h2>
            <div class="metric">
                <span class="metric-name">Time per sample</span>
                <span class="metric-value loser">{baseline.get('time_per_sample_ms', 0):.0f}ms</span>
            </div>
            <div class="metric">
                <span class="metric-name">Throughput</span>
                <span class="metric-value loser">{baseline.get('samples_per_second', 0):.3f}/sec</span>
            </div>
            <div class="metric">
                <span class="metric-name">Peak Memory</span>
                <span class="metric-value">{baseline.get('peak_memory_mb', 0):.0f}MB</span>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>📊 Performance Comparison</h2>
        <div class="chart-container">
            <canvas id="comparison-chart"></canvas>
        </div>
    </div>
    
    <div class="card">
        <h2>🎯 Key Insights</h2>
        <ul style="font-size: 1.1em; line-height: 1.8;">
            <li>Training is <strong class="winner">{speedup:.1f}x faster</strong> with optimizations</li>
            <li>Throughput improved by <strong class="winner">{throughput_ratio:.1f}x</strong></li>
            <li>Memory usage: {optimized.get('peak_memory_mb', 0):.0f}MB (optimized) vs {baseline.get('peak_memory_mb', 0):.0f}MB (baseline)</li>
            <li>Optimizations: Batched sparse training, LoRA adapters, chunk-based weight loading</li>
        </ul>
    </div>
    
    <script>
        new Chart(document.getElementById('comparison-chart'), {{
            type: 'bar',
            data: {{
                labels: ['Time/Sample (ms)', 'Throughput (samples/sec × 100)', 'Peak Memory (MB ÷ 10)'],
                datasets: [
                    {{
                        label: 'Optimized',
                        data: [{optimized.get('time_per_sample_ms', 0)}, {optimized.get('samples_per_second', 0) * 100}, {optimized.get('peak_memory_mb', 0) / 10}],
                        backgroundColor: 'rgba(63, 185, 80, 0.8)',
                        borderColor: '#3fb950',
                        borderWidth: 2
                    }},
                    {{
                        label: 'Baseline',
                        data: [{baseline.get('time_per_sample_ms', 0)}, {baseline.get('samples_per_second', 0) * 100}, {baseline.get('peak_memory_mb', 0) / 10}],
                        backgroundColor: 'rgba(218, 54, 51, 0.8)',
                        borderColor: '#da3633',
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#c9d1d9' }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
                    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"Report saved: {output_path}")
        return output_path
