#!/usr/bin/env python3
"""
Benchmark: Batched Inference vs Multi-Worker (1000 antibodies)

Compares:
- Single-sample sequential (baseline)
- Batched inference (batch_size=16, 32, 64)
- Note: Multi-worker comparison requires running predict.py CLI separately
"""

import time
import gc
import torch
import ml_collections
from typing import List, Tuple

torch.serialization.add_safe_globals([ml_collections.ConfigDict])

from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.utils import string_to_input, add_atom37_to_output, output_to_pdb
from abodybuilder3.batched_inference import predict_batch, chunked_predict
from abodybuilder3.dgx_spark import print_system_info, get_optimal_batch_size

# Sample antibodies (cycling through 5 examples)
EXAMPLE_ANTIBODIES = [
    ("QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS",
     "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"),
    ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"),
    ("EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK"),
    ("QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS",
     "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK"),
    ("EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"),
]

def create_antibody_list(n: int) -> List[Tuple[str, str]]:
    """Create a list of n antibodies by cycling through examples."""
    return [EXAMPLE_ANTIBODIES[i % len(EXAMPLE_ANTIBODIES)] for i in range(n)]

def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def benchmark_sequential(model, antibodies: List[Tuple[str, str]], device) -> dict:
    """Benchmark sequential single-sample inference."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    count = 0
    
    for heavy, light in antibodies:
        ab_input = string_to_input(heavy=heavy, light=light, device=str(device))
        ab_input_batch = {
            key: (value.unsqueeze(0) if key not in ["single", "pair"] else value)
            for key, value in ab_input.items()
        }
        with torch.no_grad():
            output = model(ab_input_batch, ab_input_batch["aatype"])
            output = add_atom37_to_output(output, ab_input["aatype"])
        pdb_str = output_to_pdb(output, ab_input)
        count += 1
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        "method": "Sequential",
        "count": count,
        "total_time": elapsed,
        "per_antibody": elapsed / count,
        "throughput_per_min": 60 * count / elapsed,
        "peak_memory_mb": peak_memory,
    }

def benchmark_batched(model, antibodies: List[Tuple[str, str]], device, batch_size: int) -> dict:
    """Benchmark batched inference."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    pdbs = chunked_predict(model, antibodies, device, batch_size=batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        "method": f"Batched (bs={batch_size})",
        "count": len(pdbs),
        "total_time": elapsed,
        "per_antibody": elapsed / len(pdbs),
        "throughput_per_min": 60 * len(pdbs) / elapsed,
        "peak_memory_mb": peak_memory,
    }

def main():
    N = 1000
    
    print("=" * 70)
    print(f"BENCHMARK: 1000 Antibodies - Batched vs Sequential")
    print("=" * 70)
    
    # Print system info
    print_system_info()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model
    print("\nLoading model...")
    from pathlib import Path
    model_path = Path("/opt/abodybuilder3/output/plddt-loss/best_second_stage.ckpt")
    module = LitABB3.load_from_checkpoint(str(model_path))
    model = module.model.to(device).eval()
    print("Model loaded.")
    
    # Create test data
    antibodies = create_antibody_list(N)
    print(f"Created {len(antibodies)} antibodies for testing.\n")
    
    # Warmup
    print("Warming up GPU...")
    _ = predict_batch(model, antibodies[:5], device)
    print("Warmup complete.\n")
    
    results = []
    
    # Test batched inference with different batch sizes
    for batch_size in [16, 32, 64, 128]:
        print(f"Testing batched (batch_size={batch_size})...")
        result = benchmark_batched(model, antibodies, device, batch_size)
        results.append(result)
        print(f"  Time: {result['total_time']:.2f}s | {result['throughput_per_min']:.0f}/min | Memory: {result['peak_memory_mb']:.0f} MB")
    
    # Test sequential (sample of 100 to save time)
    print(f"\nTesting sequential (100 antibodies, extrapolating to 1000)...")
    result = benchmark_sequential(model, antibodies[:100], device)
    # Extrapolate to 1000
    result["count"] = N
    result["total_time"] *= 10
    result["method"] = "Sequential (extrapolated)"
    results.append(result)
    print(f"  Extrapolated: {result['total_time']:.2f}s | {result['throughput_per_min']:.0f}/min | Memory: {result['peak_memory_mb']:.0f} MB")
    
    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Time(s)':<10} {'Rate/min':<12} {'Memory(MB)':<12} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = results[-1]["total_time"]  # Sequential
    for r in results:
        speedup = baseline_time / r["total_time"]
        print(f"{r['method']:<30} {r['total_time']:<10.2f} {r['throughput_per_min']:<12.0f} {r['peak_memory_mb']:<12.0f} {speedup:<10.2f}x")
    
    print("=" * 70)
    print("\nNote: To compare with 8 workers, run:")
    print("  python predict.py predict --csv test.csv -o output/ --workers 8")
    print("=" * 70)

if __name__ == "__main__":
    main()
