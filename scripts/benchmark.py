#!/usr/bin/env python3
"""
ABodyBuilder3 Benchmark Script

Benchmarks inference performance across:
- Different model types (base, language, plddt)
- Different batch sizes (1, 10, 100, 1000)
- Memory tracking (absolute and incremental)
- GPU utilization
"""

import gc
import time
import torch
import ml_collections
import subprocess
from pathlib import Path
from typing import List, Tuple
import random

# Register safe globals for PyTorch 2.10+
torch.serialization.add_safe_globals([ml_collections.ConfigDict])

from abodybuilder3.utils import string_to_input, output_to_pdb, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3


# Example antibody sequences (varying lengths for realistic benchmarks)
EXAMPLE_ANTIBODIES = [
    # 6yio_H0-L0 (122 + 107 = 229 residues)
    ("QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS",
     "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"),
    # Trastuzumab-like (121 + 107 = 228 residues)
    ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"),
    # Adalimumab-like (121 + 107 = 228 residues)
    ("EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK"),
    # Pembrolizumab-like (120 + 108 = 228 residues)
    ("QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS",
     "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK"),
    # Bevacizumab-like (119 + 108 = 227 residues)
    ("EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"),
]


def get_gpu_utilization() -> float:
    """Get current GPU utilization percentage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return float(result.stdout.strip().split('\n')[0])
    except:
        return 0.0


def get_memory_stats() -> Tuple[float, float]:
    """Get current and max GPU memory in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        current = torch.cuda.memory_allocated() / 1024 / 1024
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        return current, peak
    return 0.0, 0.0


def reset_memory():
    """Reset memory tracking."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def create_batch(batch_size: int) -> List[Tuple[str, str]]:
    """Create a batch of antibodies by cycling through examples."""
    batch = []
    for i in range(batch_size):
        heavy, light = EXAMPLE_ANTIBODIES[i % len(EXAMPLE_ANTIBODIES)]
        batch.append((heavy, light))
    return batch


def benchmark_model(
    model_name: str,
    checkpoint_path: str,
    batch_sizes: List[int],
    device: torch.device,
    warmup_runs: int = 2,
) -> dict:
    """Benchmark a single model across different batch sizes."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Reset and measure baseline memory
    reset_memory()
    baseline_mem = get_memory_stats()[0]
    
    # Load model
    print("Loading model...")
    load_start = time.perf_counter()
    module = LitABB3.load_from_checkpoint(checkpoint_path)
    model = module.model
    model.to(device)
    model.eval()
    load_time = time.perf_counter() - load_start
    
    model_mem_current, model_mem_peak = get_memory_stats()
    model_memory = model_mem_peak - baseline_mem
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Model memory: {model_memory:.1f} MB ({model_memory/1024:.3f} GB)")
    
    results = {
        "model": model_name,
        "load_time_s": load_time,
        "model_memory_mb": model_memory,
        "batch_results": []
    }
    
    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        
        # Create batch
        batch = create_batch(batch_size)
        
        # Warmup runs
        print(f"    Warmup ({warmup_runs} runs)...", end="", flush=True)
        for _ in range(warmup_runs):
            heavy, light = batch[0]
            ab_input = string_to_input(heavy=heavy, light=light)
            ab_input_batch = {
                key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
                for key, value in ab_input.items()
            }
            with torch.no_grad():
                _ = model(ab_input_batch, ab_input_batch["aatype"])
        print(" done")
        
        # Reset memory for accurate measurement (but model stays loaded)
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark run
        times = []
        gpu_utils = []
        
        for i, (heavy, light) in enumerate(batch):
            # Sample GPU utilization periodically
            if i % max(1, batch_size // 20) == 0:
                gpu_utils.append(get_gpu_utilization())
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            ab_input = string_to_input(heavy=heavy, light=light)
            ab_input_batch = {
                key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
                for key, value in ab_input.items()
            }
            
            with torch.no_grad():
                output = model(ab_input_batch, ab_input_batch["aatype"])
                output = add_atom37_to_output(output, ab_input["aatype"].to(device))
            
            # Generate PDB to include full pipeline
            pdb_string = output_to_pdb(output, ab_input)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if batch_size >= 100 and (i + 1) % (batch_size // 10) == 0:
                print(f"    Progress: {i+1}/{batch_size}", end="\r", flush=True)
        
        if batch_size >= 100:
            print()  # New line after progress
        
        # Get memory stats
        inference_mem_current, inference_mem_peak = get_memory_stats()
        total_peak_memory = model_memory + (inference_mem_peak - model_mem_current)
        
        total_time = sum(times)
        avg_time = total_time / len(times)
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
        
        result = {
            "batch_size": batch_size,
            "total_time_s": total_time,
            "avg_time_per_antibody_s": avg_time,
            "throughput_per_min": 60.0 / avg_time,
            "inference_memory_mb": inference_mem_peak - model_mem_current,
            "total_peak_memory_mb": model_memory + (inference_mem_peak - model_mem_current),
            "gpu_utilization_pct": avg_gpu_util,
            "min_time_s": min(times),
            "max_time_s": max(times),
        }
        results["batch_results"].append(result)
        
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Avg per antibody: {avg_time:.3f}s ({60/avg_time:.1f}/min)")
        print(f"    Peak memory (total): {result['total_peak_memory_mb']:.1f} MB ({result['total_peak_memory_mb']/1024:.2f} GB)")
        print(f"    GPU utilization: {avg_gpu_util:.0f}%")
    
    # Cleanup
    del model
    del module
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def print_summary_table(all_results: List[dict]):
    """Print a summary table of all benchmark results."""
    
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Model':<12} {'Batch':<8} {'Total(s)':<10} {'Per Ab(s)':<11} {'Throughput':<14} {'Memory(GB)':<12} {'GPU %':<8}")
    print("-"*100)
    
    for result in all_results:
        model_name = result["model"]
        for batch in result["batch_results"]:
            print(f"{model_name:<12} {batch['batch_size']:<8} "
                  f"{batch['total_time_s']:<10.2f} "
                  f"{batch['avg_time_per_antibody_s']:<11.3f} "
                  f"{batch['throughput_per_min']:<14.1f}/min "
                  f"{batch['total_peak_memory_mb']/1024:<12.2f} "
                  f"{batch['gpu_utilization_pct']:<8.0f}")


def main():
    print("="*60)
    print("ABodyBuilder3 Comprehensive Benchmark")
    print("="*60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model configurations
    base_path = Path("/opt/abodybuilder3/output")
    models = [
        ("pLDDT", str(base_path / "plddt-loss" / "best_second_stage.ckpt")),
        ("Base", str(base_path / "base-loss" / "best_second_stage.ckpt")),
        ("Language", str(base_path / "language-loss" / "best_second_stage.ckpt")),
    ]
    
    # Batch sizes to test (including 1000)
    batch_sizes = [1, 10, 100, 1000]
    
    # Run benchmarks
    all_results = []
    for model_name, checkpoint_path in models:
        if Path(checkpoint_path).exists():
            result = benchmark_model(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                batch_sizes=batch_sizes,
                device=device,
            )
            all_results.append(result)
        else:
            print(f"\nSkipping {model_name}: checkpoint not found at {checkpoint_path}")
    
    # Print summary
    print_summary_table(all_results)
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()
