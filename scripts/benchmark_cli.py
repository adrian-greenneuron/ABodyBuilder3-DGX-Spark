#!/usr/bin/env python3
"""
Comprehensive CLI Benchmark Script

Benchmarks the predict.py CLI across:
- Models: base, plddt, language
- Batch sizes: 1, 10, 100
- Relaxation: on/off

Tracks: time, CPU%, GPU%, system memory (unified memory)
"""

import subprocess
import time
import csv
import json
import os
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import threading

# Sample antibody sequences for benchmarking
ANTIBODIES = [
    ("trastuzumab", "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS", "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"),
    ("adalimumab", "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS", "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK"),
    ("pembrolizumab", "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS", "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK"),
    ("bevacizumab", "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS", "DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"),
    ("rituximab", "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA", "QIVLSQSPAILSASPGEKVTMTCRASSSVSYIHWFQQKPGSSPKPWIYATSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWTSNPPTFGGGTKLEIK"),
]


@dataclass
class BenchmarkResult:
    model: str
    batch_size: int
    relaxation: bool
    total_time: float
    time_per_antibody: float
    throughput_per_min: float
    peak_memory_gb: float
    memory_delta_gb: float
    avg_gpu_util: float
    avg_cpu_util: float
    success: bool
    error: Optional[str] = None


def create_test_csv(num_antibodies: int, output_path: Path):
    """Create test CSV with specified number of antibodies."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'heavy', 'light'])
        writer.writeheader()
        for i in range(num_antibodies):
            ab = ANTIBODIES[i % len(ANTIBODIES)]
            writer.writerow({
                'name': f"{ab[0]}_{i}",
                'heavy': ab[1],
                'light': ab[2],
            })


def get_system_memory_gb():
    """Get current system memory usage in GB."""
    try:
        with open('/proc/meminfo') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value = int(parts[1]) / 1024 / 1024  # KB to GB
                    meminfo[key] = value
        total = meminfo.get('MemTotal', 0)
        available = meminfo.get('MemAvailable', 0)
        used = total - available
        return used, total
    except Exception:
        return 0, 0


def monitor_resources(stop_event: threading.Event, results: dict):
    """Monitor CPU, GPU, and system memory in background."""
    gpu_utils = []
    memory_samples = []
    cpu_samples = []
    
    # Get baseline memory
    baseline_mem, _ = get_system_memory_gb()
    results['baseline_memory'] = baseline_mem
    
    while not stop_event.is_set():
        # GPU utilization
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_utils.append(float(result.stdout.strip()))
        except Exception:
            pass
        
        # System memory
        used_mem, _ = get_system_memory_gb()
        memory_samples.append(used_mem)
        
        # CPU - read /proc/stat
        try:
            with open('/proc/stat') as f:
                line = f.readline()
                parts = line.split()
                if len(parts) >= 5:
                    user = int(parts[1])
                    nice = int(parts[2])
                    system = int(parts[3])
                    idle = int(parts[4])
                    cpu_samples.append((user + nice + system, user + nice + system + idle))
        except Exception:
            pass
        
        time.sleep(0.5)
    
    results['gpu_utils'] = gpu_utils
    results['memory_samples'] = memory_samples
    results['cpu_samples'] = cpu_samples


def run_benchmark(
    model: str,
    batch_size: int,
    relaxation: bool,
    scripts_dir: Path,
    csv_path: Path,
    output_dir: Path,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup monitoring
    stop_event = threading.Event()
    monitor_results = {}
    
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, monitor_results))
    
    # Build command
    cmd = [
        'docker', 'run', '--rm', '--gpus', 'all', '--ipc=host',
        '-v', f'{scripts_dir}:/scripts',
        '-v', f'{csv_path}:/input/test.csv',
        '-v', f'{output_dir}:/output',
        'abodybuilder3-spark:latest',
        'python3', '/scripts/predict.py', 'predict',
        '--csv', '/input/test.csv',
        '-o', '/output/',
        '--model', model,
        '--workers', 'auto',
    ]
    
    if relaxation:
        cmd.append('--relaxation')
    
    # Start monitoring
    monitor_thread.start()
    time.sleep(0.5)  # Let monitoring baseline settle
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        success = result.returncode == 0
        if not success:
            # Get last meaningful error lines
            stderr_lines = [l for l in result.stderr.split('\n') if l.strip() and 'UserWarning' not in l]
            error = '\n'.join(stderr_lines[-5:]) if stderr_lines else "Unknown error"
        else:
            error = None
    except subprocess.TimeoutExpired:
        success = False
        error = "Timeout (1 hour)"
    except Exception as e:
        success = False
        error = str(e)
    
    elapsed = time.time() - start_time
    
    # Stop monitoring
    stop_event.set()
    monitor_thread.join(timeout=2)
    
    # Calculate metrics
    gpu_utils = monitor_results.get('gpu_utils', [])
    memory_samples = monitor_results.get('memory_samples', [])
    cpu_samples = monitor_results.get('cpu_samples', [])
    baseline_mem = monitor_results.get('baseline_memory', 0)
    
    avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
    peak_memory = max(memory_samples) if memory_samples else 0
    memory_delta = peak_memory - baseline_mem
    
    # Calculate CPU utilization from deltas
    cpu_utils = []
    for i in range(1, len(cpu_samples)):
        busy_delta = cpu_samples[i][0] - cpu_samples[i-1][0]
        total_delta = cpu_samples[i][1] - cpu_samples[i-1][1]
        if total_delta > 0:
            cpu_utils.append(100 * busy_delta / total_delta)
    avg_cpu_util = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0
    
    # Check output files exist
    if success:
        output_files = list(output_dir.glob('*.pdb'))
        if len(output_files) != batch_size:
            success = False
            error = f"Expected {batch_size} PDB files, got {len(output_files)}"
    
    return BenchmarkResult(
        model=model,
        batch_size=batch_size,
        relaxation=relaxation,
        total_time=elapsed,
        time_per_antibody=elapsed / batch_size if batch_size > 0 else 0,
        throughput_per_min=60 * batch_size / elapsed if elapsed > 0 else 0,
        peak_memory_gb=peak_memory,
        memory_delta_gb=memory_delta,
        avg_gpu_util=avg_gpu_util,
        avg_cpu_util=avg_cpu_util,
        success=success,
        error=error,
    )


def main():
    print("=" * 80)
    print("ABodyBuilder3 CLI Benchmark (Unified Memory Architecture)")
    print("=" * 80)
    
    # Configuration
    models = ['base', 'plddt']
    batch_sizes = [1, 25]
    relaxation_options = [False]
    
    scripts_dir = Path('/home/adrian/data/abodybuilder3/scripts')
    benchmark_dir = Path('/tmp/cli_benchmark')
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)
    benchmark_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Count total tests
    total_tests = len(models) * len(batch_sizes) * len(relaxation_options)
    
    test_num = 0
    
    for model in models:
        for batch_size in batch_sizes:
            for relaxation in relaxation_options:
                test_num += 1
                relax_str = "w/ relax" if relaxation else "no relax"
                
                print(f"\n[{test_num}/{total_tests}] {model}, {batch_size} abs, {relax_str}")
                
                # Create test CSV
                csv_path = benchmark_dir / f'test_{batch_size}.csv'
                create_test_csv(batch_size, csv_path)
                
                # Create output directory
                output_dir = benchmark_dir / f'output_{model}_{batch_size}_{relaxation}'
                
                # Run benchmark
                result = run_benchmark(
                    model=model,
                    batch_size=batch_size,
                    relaxation=relaxation,
                    scripts_dir=scripts_dir,
                    csv_path=csv_path,
                    output_dir=output_dir,
                )
                
                results.append(result)
                
                if result.success:
                    print(f"  ✓ {result.total_time:.1f}s ({result.time_per_antibody:.2f}s/ab, {result.throughput_per_min:.0f}/min)")
                    print(f"    Mem: {result.memory_delta_gb:.2f}GB delta | GPU: {result.avg_gpu_util:.0f}% | CPU: {result.avg_cpu_util:.0f}%")
                else:
                    print(f"  ✗ Failed: {result.error}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<10} {'Batch':<6} {'Relax':<6} {'Time(s)':<9} {'Per Ab':<8} {'Rate':<10} {'MemΔ(GB)':<10} {'GPU%':<6} {'CPU%':<6} {'OK'}")
    print("-" * 80)
    
    for r in results:
        relax = "Yes" if r.relaxation else "No"
        status = "✓" if r.success else "✗"
        print(f"{r.model:<10} {r.batch_size:<6} {relax:<6} {r.total_time:<9.1f} {r.time_per_antibody:<8.2f} {r.throughput_per_min:<10.1f} {r.memory_delta_gb:<10.2f} {r.avg_gpu_util:<6.0f} {r.avg_cpu_util:<6.0f} {status}")
    
    # Save results to JSON
    results_path = benchmark_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
