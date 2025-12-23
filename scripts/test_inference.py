#!/usr/bin/env python3
"""
Test script for ABodyBuilder3 antibody structure prediction.
Uses example sequences from PDB structure 6yio_H0-L0.
"""

import torch
from pathlib import Path

print("=" * 60)
print("ABodyBuilder3 Structure Prediction Test")
print("=" * 60)

# Example antibody sequences (from 6yio_H0-L0)
heavy = "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSLAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGSVSGTLVDFDIWGQGTMVTVSS"
light = "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNIYPITFGGGTKVEIK"

print(f"\nHeavy chain ({len(heavy)} residues):")
print(f"  {heavy[:50]}...")
print(f"\nLight chain ({len(light)} residues):")
print(f"  {light[:50]}...")

# Import ABodyBuilder3 modules
from abodybuilder3.utils import string_to_input, output_to_pdb, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Load model (pLDDT model includes confidence scores)
print("\nLoading model...")
model_path = Path("/opt/abodybuilder3/output/plddt-loss/best_second_stage.ckpt")
if not model_path.exists():
    # Try alternative paths
    model_path = Path("output/plddt-loss/best_second_stage.ckpt")

# Register safe globals for PyTorch 2.10+ (weights_only=True by default)
import ml_collections
torch.serialization.add_safe_globals([ml_collections.ConfigDict])
    
module = LitABB3.load_from_checkpoint(str(model_path))
model = module.model
model.to(device)
model.eval()
print(f"  Model loaded from: {model_path}")

# Prepare input
print("\nPreparing input...")
ab_input = string_to_input(heavy=heavy, light=light)
ab_input_batch = {
    key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
    for key, value in ab_input.items()
}

# Run inference
print("\nRunning inference...")
with torch.no_grad():
    output = model(ab_input_batch, ab_input_batch["aatype"])
    output = add_atom37_to_output(output, ab_input["aatype"].to(device))

# Convert to PDB
print("\nGenerating PDB structure...")
pdb_string = output_to_pdb(output, ab_input)

# Save output
output_path = Path("/tmp/test_antibody.pdb")
with open(output_path, "w") as f:
    f.write(pdb_string)

# Print summary
lines = pdb_string.split("\n")
atom_count = sum(1 for line in lines if line.startswith("ATOM"))
print(f"\n✅ Structure prediction successful!")
print(f"  Output file: {output_path}")
print(f"  Total atoms: {atom_count}")
print(f"  Total residues: {len(heavy) + len(light)}")

# Show first few lines of PDB
print("\nFirst 10 lines of PDB:")
for line in lines[:10]:
    print(f"  {line}")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
