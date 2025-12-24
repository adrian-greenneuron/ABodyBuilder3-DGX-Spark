#!/usr/bin/env python3
"""
Test script for batched inference.

Verifies that batched inference produces identical results to single-sample inference.
"""

import time
import torch
import numpy as np
import ml_collections
from pathlib import Path

# Register safe globals for PyTorch 2.10+
torch.serialization.add_safe_globals([ml_collections.ConfigDict])

print("=" * 60)
print("ABodyBuilder3 Batched Inference Test")
print("=" * 60)

# Test antibodies
ANTIBODIES = [
    ("QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS",
     "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"),
    ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"),
    ("EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
     "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK"),
]

# Import modules
from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.utils import string_to_input, add_atom37_to_output, output_to_pdb
from abodybuilder3.batched_inference import predict_batch, strings_to_batch_input
from abodybuilder3.dgx_spark import print_system_info, is_unified_memory_architecture, get_optimal_batch_size

# Print system info
print("\n--- System Information ---")
print_system_info()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Load model
print("\nLoading model...")
model_path = Path("/opt/abodybuilder3/output/plddt-loss/best_second_stage.ckpt")
if not model_path.exists():
    model_path = Path("output/plddt-loss/best_second_stage.ckpt")

module = LitABB3.load_from_checkpoint(str(model_path))
model = module.model
model.to(device)
model.eval()
print(f"Model loaded from: {model_path}")

# ============================================
# Test 1: Single-sample inference (baseline)
# ============================================
print("\n" + "=" * 60)
print("Test 1: Single-sample inference (baseline)")
print("=" * 60)

single_pdbs = []
single_start = time.perf_counter()

for i, (heavy, light) in enumerate(ANTIBODIES):
    ab_input = string_to_input(heavy=heavy, light=light, device=str(device))
    ab_input_batch = {
        key: (value.unsqueeze(0) if key not in ["single", "pair"] else value)
        for key, value in ab_input.items()
    }
    
    with torch.no_grad():
        output = model(ab_input_batch, ab_input_batch["aatype"])
        output = add_atom37_to_output(output, ab_input["aatype"])
    
    pdb_str = output_to_pdb(output, ab_input)
    single_pdbs.append(pdb_str)
    print(f"  Antibody {i+1}: {len(pdb_str)} bytes")

single_time = time.perf_counter() - single_start
print(f"\nSingle-sample total time: {single_time:.3f}s ({single_time/len(ANTIBODIES):.3f}s per antibody)")

# ============================================
# Test 2: Batched inference
# ============================================
print("\n" + "=" * 60)
print("Test 2: Batched inference")
print("=" * 60)

batch_start = time.perf_counter()
batch_pdbs = predict_batch(model, ANTIBODIES, device)
batch_time = time.perf_counter() - batch_start

for i, pdb_str in enumerate(batch_pdbs):
    print(f"  Antibody {i+1}: {len(pdb_str)} bytes")

print(f"\nBatched total time: {batch_time:.3f}s ({batch_time/len(ANTIBODIES):.3f}s per antibody)")
print(f"Speedup: {single_time/batch_time:.2f}x")

# ============================================
# Test 3: Verify outputs match
# ============================================
print("\n" + "=" * 60)
print("Test 3: Verifying outputs match")
print("=" * 60)

def extract_atom_coords(pdb_str: str) -> np.ndarray:
    """Extract CA atom coordinates from PDB string."""
    coords = []
    for line in pdb_str.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords)

all_match = True
for i in range(len(ANTIBODIES)):
    single_coords = extract_atom_coords(single_pdbs[i])
    batch_coords = extract_atom_coords(batch_pdbs[i])
    
    if single_coords.shape != batch_coords.shape:
        print(f"  Antibody {i+1}: SHAPE MISMATCH - single {single_coords.shape} vs batch {batch_coords.shape}")
        all_match = False
        continue
    
    # Compare coordinates
    max_diff = np.max(np.abs(single_coords - batch_coords))
    rmsd = np.sqrt(np.mean((single_coords - batch_coords)**2))
    
    if max_diff < 0.01:  # 0.01 Angstrom tolerance
        print(f"  Antibody {i+1}: ✓ MATCH (max diff: {max_diff:.6f} Å, RMSD: {rmsd:.6f} Å)")
    else:
        print(f"  Antibody {i+1}: ✗ MISMATCH (max diff: {max_diff:.6f} Å, RMSD: {rmsd:.6f} Å)")
        all_match = False

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if all_match:
    print("✓ All tests PASSED!")
    print(f"  - Single-sample: {single_time:.3f}s")
    print(f"  - Batched: {batch_time:.3f}s")
    print(f"  - Speedup: {single_time/batch_time:.2f}x")
    print(f"  - Unified Memory: {is_unified_memory_architecture()}")
    print(f"  - Recommended batch size: {get_optimal_batch_size()}")
else:
    print("✗ Some tests FAILED!")
    print("  Please investigate the mismatches above.")

print("=" * 60)
