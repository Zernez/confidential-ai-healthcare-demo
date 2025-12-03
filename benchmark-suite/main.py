#!/usr/bin/env python3
"""
Benchmark Suite for Confidential AI Healthcare Demo

Runs multiple iterations of benchmarks for:
- Python Native Baseline (cuML/RAPIDS)
- C++ WASM Module
- Rust WASM Module

Collects timing metrics, computes statistics, and generates
publication-ready plots and LaTeX tables.

Usage:
    python main.py                    # Run all benchmarks (20 iterations)
    python main.py --runs 10          # Custom number of runs
    python main.py --skip-run         # Skip benchmarks, use existing data
    python main.py --plot-only        # Only generate plots from JSON
"""

import os
import sys
import json
import argparse
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import statistics

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    num_runs: int = 20
    warmup_runs: int = 1
    project_root: str = ""
    output_dir: str = ""
    
    # Benchmark commands (relative to project root)
    python_cmd: str = "python-baseline/run.sh"
    cpp_cmd: str = "./run_with_attestation.sh --cpp"
    rust_cmd: str = "./run_with_attestation.sh --rust"
    
    # WASM module paths (relative to project root)
    rust_wasm_path: str = "wasm-ml/target/wasm32-wasip1/release/wasm-ml-benchmark.wasm"
    cpp_wasm_path: str = "wasmwebgpu-ml/build/wasmwebgpu-ml-benchmark.wasm"
    
    def __post_init__(self):
        if not self.project_root:
            # Auto-detect project root
            self.project_root = str(Path(__file__).parent.parent.absolute())
        if not self.output_dir:
            self.output_dir = str(Path(__file__).parent / "results")


@dataclass
class SingleRunResult:
    """Result from a single benchmark run"""
    language: str = ""
    gpu_device: str = ""
    gpu_backend: str = ""
    tee_type: str = ""
    gpu_available: bool = False
    tee_available: bool = False
    attestation_ms: float = 0.0
    training_ms: float = 0.0
    inference_ms: float = 0.0
    mse: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    run_index: int = 0
    timestamp: str = ""
    success: bool = True
    error: str = ""


@dataclass 
class StatisticalMetrics:
    """Statistical metrics for a series of measurements"""
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    p5: float = 0.0   # 5th percentile
    p95: float = 0.0  # 95th percentile
    cv: float = 0.0   # Coefficient of variation
    n: int = 0
    values: List[float] = field(default_factory=list)
    
    @classmethod
    def from_values(cls, values: List[float]) -> 'StatisticalMetrics':
        if not values:
            return cls()
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        
        return cls(
            mean=mean,
            std=std,
            min=min(values),
            max=max(values),
            median=statistics.median(values),
            p5=sorted_vals[int(n * 0.05)] if n >= 20 else sorted_vals[0],
            p95=sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
            cv=(std / mean * 100) if mean > 0 else 0.0,
            n=n,
            values=values
        )


@dataclass
class BenchmarkStats:
    """Aggregated statistics for a benchmark type"""
    name: str = ""
    display_name: str = ""
    num_runs: int = 0
    successful_runs: int = 0
    
    # Timing statistics
    attestation: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    training: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    inference: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    total_time: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    
    # Model quality
    mse: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    
    # Module size (WASM only)
    module_size_bytes: int = 0
    module_size_kb: float = 0.0
    module_size_mb: float = 0.0
    
    # Metadata
    gpu_device: str = ""
    gpu_backend: str = ""
    tee_type: str = ""
    
    # Raw results
    raw_results: List[SingleRunResult] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Complete benchmark report"""
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, BenchmarkStats] = field(default_factory=dict)
    system_info: Dict[str, str] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Runs benchmarks and collects results"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, List[SingleRunResult]] = {
            "python": [],
            "cpp": [],
            "rust": []
        }
        
    def _extract_json_from_output(self, output: str) -> Optional[dict]:
        """Extract JSON benchmark data from command output"""
        # Look for JSON between markers
        pattern = r'### BENCHMARK_JSON ###\s*(\{.*?\})\s*### END_BENCHMARK_JSON ###'
        match = re.search(pattern, output, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                print(f"    [WARN] Failed to parse JSON: {e}")
                return None
        return None
    
    def _run_single_benchmark(self, name: str, cmd: str, run_index: int) -> SingleRunResult:
        """Run a single benchmark and return result"""
        result = SingleRunResult(
            run_index=run_index,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # Run command from project root
            full_cmd = f"cd {self.config.project_root} && {cmd}"
            
            process = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            output = process.stdout + process.stderr
            
            # Extract JSON result
            json_data = self._extract_json_from_output(output)
            
            if json_data:
                result.language = json_data.get("language", name)
                result.gpu_device = json_data.get("gpu_device", "")
                result.gpu_backend = json_data.get("gpu_backend", "")
                result.tee_type = json_data.get("tee_type", "")
                result.gpu_available = json_data.get("gpu_available", False)
                result.tee_available = json_data.get("tee_available", False)
                result.attestation_ms = json_data.get("attestation_ms", 0.0)
                result.training_ms = json_data.get("training_ms", 0.0)
                result.inference_ms = json_data.get("inference_ms", 0.0)
                result.mse = json_data.get("mse", 0.0)
                result.train_samples = json_data.get("train_samples", 0)
                result.test_samples = json_data.get("test_samples", 0)
                result.success = True
            else:
                result.success = False
                result.error = "Failed to extract JSON from output"
                # Save output for debugging
                print(f"    [DEBUG] Output excerpt: {output[:500]}...")
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error = "Timeout (600s)"
        except Exception as e:
            result.success = False
            result.error = str(e)
            
        return result
    
    def run_all(self) -> Dict[str, List[SingleRunResult]]:
        """Run all benchmarks"""
        benchmarks = [
            ("python", "Python Native", self.config.python_cmd),
            ("cpp", "C++ WASM", self.config.cpp_cmd),
            ("rust", "Rust WASM", self.config.rust_cmd),
        ]
        
        total_runs = self.config.num_runs + self.config.warmup_runs
        
        print("\n" + "═" * 70)
        print("  BENCHMARK SUITE - Confidential AI Healthcare Demo")
        print("═" * 70)
        print(f"  Runs per benchmark: {self.config.num_runs} (+{self.config.warmup_runs} warmup)")
        print(f"  Total runs: {len(benchmarks) * total_runs}")
        print(f"  Project root: {self.config.project_root}")
        print("═" * 70 + "\n")
        
        for bench_name, display_name, cmd in benchmarks:
            print(f"\n{'─' * 70}")
            print(f"  Running: {display_name}")
            print(f"  Command: {cmd}")
            print(f"{'─' * 70}")
            
            for i in range(total_runs):
                is_warmup = i < self.config.warmup_runs
                run_type = "WARMUP" if is_warmup else f"RUN {i - self.config.warmup_runs + 1}/{self.config.num_runs}"
                
                print(f"\n  [{run_type}] {display_name}...", end=" ", flush=True)
                
                result = self._run_single_benchmark(bench_name, cmd, i)
                
                if result.success:
                    print(f"✓ (attest: {result.attestation_ms:.1f}ms, "
                          f"train: {result.training_ms:.1f}ms, "
                          f"infer: {result.inference_ms:.2f}ms)")
                else:
                    print(f"✗ ({result.error})")
                
                # Only save non-warmup runs
                if not is_warmup:
                    self.results[bench_name].append(result)
                    
                # Small delay between runs
                time.sleep(1)
        
        return self.results


# ═══════════════════════════════════════════════════════════════════════════
# Statistics Calculator
# ═══════════════════════════════════════════════════════════════════════════

class StatisticsCalculator:
    """Calculate statistics from benchmark results"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
    def _get_wasm_size(self, path: str) -> int:
        """Get WASM module size in bytes"""
        full_path = Path(self.config.project_root) / path
        if full_path.exists():
            return full_path.stat().st_size
        return 0
    
    def calculate(self, results: Dict[str, List[SingleRunResult]]) -> BenchmarkReport:
        """Calculate statistics for all benchmarks"""
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            config=asdict(self.config)
        )
        
        # System info
        report.system_info = self._get_system_info()
        
        # Calculate stats for each benchmark
        benchmark_info = {
            "python": ("Python Native", None),
            "cpp": ("C++ WASM", self.config.cpp_wasm_path),
            "rust": ("Rust WASM", self.config.rust_wasm_path),
        }
        
        for name, (display_name, wasm_path) in benchmark_info.items():
            runs = results.get(name, [])
            successful_runs = [r for r in runs if r.success]
            
            stats = BenchmarkStats(
                name=name,
                display_name=display_name,
                num_runs=len(runs),
                successful_runs=len(successful_runs)
            )
            
            if successful_runs:
                # Timing statistics
                stats.attestation = StatisticalMetrics.from_values(
                    [r.attestation_ms for r in successful_runs]
                )
                stats.training = StatisticalMetrics.from_values(
                    [r.training_ms for r in successful_runs]
                )
                stats.inference = StatisticalMetrics.from_values(
                    [r.inference_ms for r in successful_runs]
                )
                stats.total_time = StatisticalMetrics.from_values(
                    [r.attestation_ms + r.training_ms + r.inference_ms for r in successful_runs]
                )
                stats.mse = StatisticalMetrics.from_values(
                    [r.mse for r in successful_runs]
                )
                
                # Metadata from first successful run
                first = successful_runs[0]
                stats.gpu_device = first.gpu_device
                stats.gpu_backend = first.gpu_backend
                stats.tee_type = first.tee_type
                
            # WASM module size
            if wasm_path:
                size_bytes = self._get_wasm_size(wasm_path)
                stats.module_size_bytes = size_bytes
                stats.module_size_kb = size_bytes / 1024
                stats.module_size_mb = size_bytes / (1024 * 1024)
                
            stats.raw_results = runs
            report.benchmarks[name] = stats
            
        return report
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        info = {}
        
        try:
            # GPU info
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                info["gpu_name"] = parts[0] if parts else "Unknown"
                info["driver_version"] = parts[1] if len(parts) > 1 else "Unknown"
                info["gpu_memory"] = parts[2] if len(parts) > 2 else "Unknown"
        except:
            pass
            
        try:
            # CPU info
            result = subprocess.run(["lscpu"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Model name" in line:
                        info["cpu_name"] = line.split(":")[1].strip()
                    elif "CPU(s):" in line and "NUMA" not in line:
                        info["cpu_cores"] = line.split(":")[1].strip()
        except:
            pass
            
        try:
            # Memory info
            result = subprocess.run(["free", "-h"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split("\n")
                if len(lines) > 1:
                    parts = lines[1].split()
                    info["total_memory"] = parts[1] if len(parts) > 1 else "Unknown"
        except:
            pass
            
        return info


# ═══════════════════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generate JSON reports and visualizations"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_json(self, report: BenchmarkReport, filename: str = None) -> str:
        """Save report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        # Convert dataclasses to dict
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_dict(v) for v in obj]
            return obj
        
        data = to_dict(report)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n📄 Report saved: {filepath}")
        return str(filepath)
    
    def load_json(self, filepath: str) -> BenchmarkReport:
        """Load report from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Reconstruct report object
        report = BenchmarkReport(
            timestamp=data.get("timestamp", ""),
            config=data.get("config", {}),
            system_info=data.get("system_info", {})
        )
        
        # Reconstruct benchmark stats
        for name, stats_data in data.get("benchmarks", {}).items():
            stats = BenchmarkStats(
                name=stats_data.get("name", ""),
                display_name=stats_data.get("display_name", ""),
                num_runs=stats_data.get("num_runs", 0),
                successful_runs=stats_data.get("successful_runs", 0),
                module_size_bytes=stats_data.get("module_size_bytes", 0),
                module_size_kb=stats_data.get("module_size_kb", 0.0),
                module_size_mb=stats_data.get("module_size_mb", 0.0),
                gpu_device=stats_data.get("gpu_device", ""),
                gpu_backend=stats_data.get("gpu_backend", ""),
                tee_type=stats_data.get("tee_type", ""),
            )
            
            # Reconstruct statistical metrics
            for metric_name in ["attestation", "training", "inference", "total_time", "mse"]:
                if metric_name in stats_data:
                    m = stats_data[metric_name]
                    setattr(stats, metric_name, StatisticalMetrics(
                        mean=m.get("mean", 0.0),
                        std=m.get("std", 0.0),
                        min=m.get("min", 0.0),
                        max=m.get("max", 0.0),
                        median=m.get("median", 0.0),
                        p5=m.get("p5", 0.0),
                        p95=m.get("p95", 0.0),
                        cv=m.get("cv", 0.0),
                        n=m.get("n", 0),
                        values=m.get("values", [])
                    ))
                    
            report.benchmarks[name] = stats
            
        return report
    
    def print_summary(self, report: BenchmarkReport):
        """Print summary to console"""
        print("\n" + "═" * 70)
        print("  BENCHMARK SUMMARY")
        print("═" * 70)
        
        if report.system_info:
            print(f"\n  System: {report.system_info.get('gpu_name', 'Unknown GPU')}")
            print(f"  Driver: {report.system_info.get('driver_version', 'Unknown')}")
        
        print(f"\n  {'Benchmark':<20} {'Attest (ms)':<15} {'Train (ms)':<15} {'Infer (ms)':<15} {'MSE':<12}")
        print("  " + "─" * 77)
        
        for name in ["python", "cpp", "rust"]:
            if name in report.benchmarks:
                b = report.benchmarks[name]
                print(f"  {b.display_name:<20} "
                      f"{b.attestation.mean:>7.2f}±{b.attestation.std:<5.2f} "
                      f"{b.training.mean:>7.2f}±{b.training.std:<5.2f} "
                      f"{b.inference.mean:>7.2f}±{b.inference.std:<5.2f} "
                      f"{b.mse.mean:>10.2f}")
        
        print("\n  WASM Module Sizes:")
        print("  " + "─" * 40)
        for name in ["cpp", "rust"]:
            if name in report.benchmarks:
                b = report.benchmarks[name]
                if b.module_size_bytes > 0:
                    print(f"  {b.display_name:<20} {b.module_size_kb:>10.2f} KB ({b.module_size_mb:.2f} MB)")
        
        print("\n" + "═" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Suite for Confidential AI Healthcare Demo"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=20,
        help="Number of benchmark runs (default: 20)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running benchmarks, use existing data"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing JSON"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSON file for --plot-only mode"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = BenchmarkConfig(
        num_runs=args.runs,
        warmup_runs=args.warmup
    )
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Initialize components
    generator = ReportGenerator(config)
    
    if args.plot_only or args.skip_run:
        # Load existing report
        if args.input:
            report = generator.load_json(args.input)
        else:
            # Find latest report
            reports = sorted(Path(config.output_dir).glob("benchmark_report_*.json"))
            if reports:
                report = generator.load_json(str(reports[-1]))
                print(f"📂 Loaded: {reports[-1]}")
            else:
                print("❌ No existing reports found. Run benchmarks first.")
                return 1
    else:
        # Run benchmarks
        runner = BenchmarkRunner(config)
        results = runner.run_all()
        
        # Calculate statistics
        calculator = StatisticsCalculator(config)
        report = calculator.calculate(results)
        
        # Save JSON
        json_path = generator.save_json(report)
    
    # Print summary
    generator.print_summary(report)
    
    # Generate plots
    try:
        from plot_generator import PlotGenerator
        plotter = PlotGenerator(config)
        plotter.generate_all(report)
    except ImportError:
        print("\n⚠️  Plot generation skipped (matplotlib not available)")
        print("   Install with: pip install matplotlib seaborn")
    
    # Generate LaTeX table
    try:
        from latex_generator import LatexGenerator
        latex_gen = LatexGenerator(config)
        latex_gen.generate_all(report)
    except ImportError:
        print("\n⚠️  LaTeX generation skipped (module not found)")
    
    print("\n✅ Benchmark suite completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
