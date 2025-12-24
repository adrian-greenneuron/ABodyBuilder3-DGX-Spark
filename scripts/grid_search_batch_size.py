#!/usr/bin/env python3
"""
Grid Search: Workers × Batch Size

Tests all combinations of workers and batch sizes to find optimal configuration.
Monitors ACTUAL system memory (not just PyTorch allocations) on unified memory systems.
"""

import time
import gc
import os
import torch
import ml_collections
from typing import List, Tuple, Dict
from pathlib import Path

torch.serialization.add_safe_globals([ml_collections.ConfigDict])

from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.batched_inference import chunked_predict
from abodybuilder3.dgx_spark import (
    get_memory_usage_gb,
    get_available_memory_gb,
    get_total_memory_gb,
    is_unified_memory_architecture,
)


# Sample antibodies
ANTIBODIES = [
    ('QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS',
     'DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK'),
    ('EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
     'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK'),
]


def get_antibody_list(n: int) -> List[Tuple[str, str]]:
    return [ANTIBODIES[i % len(ANTIBODIES)] for i in range(n)]


def run_benchmark(
    model,
    antibodies: List[Tuple[str, str]],
    batch_size: int,
    device: str = "cuda",
) -> Dict:
    """Run single benchmark configuration and measure actual memory."""
    gc.collect()
    torch.cuda.empty_cache()
    
    # Measure memory BEFORE
    mem_before = get_memory_usage_gb()
    mem_available_before = get_available_memory_gb()
    
    # Run inference
    try:
        start = time.perf_counter()
        pdbs = chunked_predict(model, antibodies, device, batch_size=batch_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Measure memory AFTER
        mem_after = get_memory_usage_gb()
        mem_available_after = get_available_memory_gb()
        
        return {
            "success": True,
            "batch_size": batch_size,
            "count": len(pdbs),
            "time_s": elapsed,
            "rate_per_min": 60 * len(pdbs) / elapsed,
            "mem_before_gb": mem_before,
            "mem_after_gb": mem_after,
            "mem_used_gb": mem_after - mem_before,
            "mem_available_gb": mem_available_after,
            "mem_peak_estimate_gb": get_total_memory_gb() - mem_available_after,
        }
    except Exception as e:
        return {
            "success": False,
            "batch_size": batch_size,
            "error": str(e)[:50],
        }


def main():
    N = 1000  # Number of antibodies to test
    
    print("=" * 80)
    print("GRID SEARCH: Batch Size Optimization")
    print("=" * 80)
    
    # System info
    print(f"\nSystem: {'Unified Memory (DGX Spark)' if is_unified_memory_architecture() else 'Discrete GPU'}")
    print(f"Total Memory: {get_total_memory_gb():.1f} GB")
    print(f"Available Memory: {get_available_memory_gb():.1f} GB")
    print(f"Current Usage: {get_memory_usage_gb():.1f} GB")
    
    device = "cuda"
    
    # Load model
    print("\nLoading model...")
    model_path = Path("/opt/abodybuilder3/output/plddt-loss/best_second_stage.ckpt")
    module = LitABB3.load_from_checkpoint(str(model_path))
    model = module.model.to(device).eval()
    
    mem_with_model = get_memory_usage_gb()
    print(f"Memory after model load: {mem_with_model:.1f} GB")
    
    # Test data
    antibodies = get_antibody_list(N)
    
    # Warmup
    print("\nWarming up...")
    _ = chunked_predict(model, antibodies[:5], device, batch_size=16)
    
    # Grid search batch sizes
    batch_sizes = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
    
    results = []
    
    print(f"\nTesting {len(batch_sizes)} batch sizes on {N} antibodies...")
    print()
    print(f"{'BS':<6} {'Time(s)':<10} {'Rate/min':<12} {'MemUsed(GB)':<14} {'MemAvail(GB)':<14} {'Status'}")
    print("-" * 80)
    
    for bs in batch_sizes:
        # Check if we have enough memory (rough estimate)
        available = get_available_memory_gb()
        if available < 10:  # Less than 10GB free, skip
            print(f"{bs:<6} {'SKIP':<10} {'---':<12} {'---':<14} {available:<14.1f} LOW MEMORY")
            continue
        
        result = run_benchmark(model, antibodies, bs, device)
        results.append(result)
        
        if result["success"]:
            print(f"{bs:<6} {result['time_s']:<10.2f} {result['rate_per_min']:<12.0f} "
                  f"{result['mem_used_gb']:<14.1f} {result['mem_available_gb']:<14.1f} OK")
        else:
            print(f"{bs:<6} {'ERROR':<10} {'---':<12} {'---':<14} {'---':<14} {result.get('error', 'Unknown')}")
            break  # Stop on error (likely OOM)
    
    # Find best
    successful = [r for r in results if r["success"]]
    if successful:
        best = max(successful, key=lambda x: x["rate_per_min"])
        
        print()
        print("=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"Batch Size: {best['batch_size']}")
        print(f"Throughput: {best['rate_per_min']:.0f} antibodies/min")
        print(f"Time for {N}: {best['time_s']:.1f}s")
        print(f"Memory Used: {best['mem_used_gb']:.1f} GB")
        print("=" * 80)
    
    # Save results
    import json
    results_path = Path("/tmp/grid_search_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
