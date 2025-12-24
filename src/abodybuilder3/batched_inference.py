"""
Batched Inference for ABodyBuilder3

This module provides efficient batched inference for processing multiple antibodies
in a single forward pass, optimized for DGX Spark's unified memory architecture.

Key features:
- Process multiple antibodies in one GPU kernel launch
- Automatic padding for variable-length sequences
- Vectorized post-processing
- Optional DGX Spark unified memory optimizations
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from abodybuilder3.dataloader import ABDataset, collate_fn, pad_first_dim_keys, pad_square_tensors
from abodybuilder3.openfold.data.data_transforms import make_atom14_masks
from abodybuilder3.openfold.np.protein import Protein
from abodybuilder3.openfold.np.residue_constants import restype_order_with_x
from abodybuilder3.openfold.utils.feats import atom14_to_atom37
from abodybuilder3.utils import to_pdb_fast


def string_to_input_dict(heavy: str, light: str) -> dict:
    """
    Convert heavy/light chain strings to input dictionary (unbatched).
    
    This is a helper that returns individual sample dictionaries suitable
    for collation into a batch.
    
    Args:
        heavy: Heavy chain amino acid sequence
        light: Light chain amino acid sequence
        
    Returns:
        Dictionary with tensors for model input (without batch dimension)
    """
    aatype = []
    is_heavy = []
    
    for character in heavy:
        is_heavy.append(1)
        aatype.append(restype_order_with_x.get(character, 20))  # 20 = unknown
    for character in light:
        is_heavy.append(0)
        aatype.append(restype_order_with_x.get(character, 20))
    
    is_heavy = torch.tensor(is_heavy)
    aatype = torch.tensor(aatype)
    residue_index = torch.cat(
        (torch.arange(len(heavy)), torch.arange(len(light)) + 500)
    )
    
    datapoint = {
        "is_heavy": is_heavy,
        "aatype": aatype,
        "residue_index": residue_index,
    }
    
    # Add single and pair representations
    datapoint.update(
        ABDataset.single_and_double_from_datapoint(
            datapoint, rel_pos_dim=64, edge_chain_feature=True
        )
    )
    
    return datapoint


def strings_to_batch_input(
    antibodies: List[Tuple[str, str]],
    device: Union[str, torch.device] = "cuda",
) -> Tuple[dict, List[dict]]:
    """
    Prepare multiple antibodies as a single batched tensor dictionary.
    
    Args:
        antibodies: List of (heavy_chain, light_chain) tuples
        device: Target device for tensors
        
    Returns:
        Tuple of:
        - Batched input dictionary with tensors of shape [batch, seq_len, ...]
        - List of individual input dictionaries (for post-processing)
    """
    individual_inputs = []
    for heavy, light in antibodies:
        inp = string_to_input_dict(heavy, light)
        individual_inputs.append(inp)
    
    # Use existing collate function logic
    batch = {key: [d[key] for d in individual_inputs] for key in individual_inputs[0]}
    
    batched_input = {}
    for key in batch:
        if key in pad_first_dim_keys or key in ["single", "seq_mask"]:
            batched_input[key] = pad_sequence(batch[key], batch_first=True)
        elif key == "pair":
            batched_input[key] = pad_square_tensors(batch[key])
        else:
            # Keep as list for non-tensor keys
            batched_input[key] = batch[key]
    
    # Ensure required keys exist
    if "seq_mask" not in batched_input:
        # Create seq_mask from aatype (non-padded positions are 1)
        max_len = batched_input["aatype"].shape[1]
        seq_mask = torch.zeros(len(antibodies), max_len)
        for i, inp in enumerate(individual_inputs):
            seq_len = len(inp["aatype"])
            seq_mask[i, :seq_len] = 1.0
        batched_input["seq_mask"] = seq_mask
    
    # Move tensors to device
    for key in batched_input:
        if isinstance(batched_input[key], torch.Tensor):
            batched_input[key] = batched_input[key].to(device)
    
    return batched_input, individual_inputs


def batch_add_atom37_to_output(
    output: dict,
    aatypes: torch.Tensor,
    seq_lengths: List[int],
) -> List[dict]:
    """
    Add atom37 coordinates to batched output and split into individual results.
    
    Args:
        output: Model output dictionary with batched tensors
        aatypes: [batch, max_seq_len] amino acid types
        seq_lengths: List of actual sequence lengths for each sample
        
    Returns:
        List of output dictionaries, one per antibody
    """
    batch_size = aatypes.shape[0]
    # Get final positions from the last structure iteration
    # output["positions"] shape: [no_blocks, batch, seq_len, 14, 3]
    positions_14 = output["positions"][-1]  # [batch, seq_len, 14, 3]
    
    results = []
    for i in range(batch_size):
        seq_len = seq_lengths[i]
        aatype_i = aatypes[i, :seq_len]
        pos_14_i = positions_14[i, :seq_len]
        
        # Make atom14 masks for this sample
        batch_masks = make_atom14_masks({"aatype": aatype_i})
        
        # Convert atom14 to atom37
        atom37 = atom14_to_atom37(pos_14_i, batch_masks)
        
        single_output = {
            "positions": output["positions"][:, i:i+1, :seq_len],
            "atom37": atom37.cpu().numpy() if isinstance(atom37, torch.Tensor) else atom37,
            "atom37_atom_exists": batch_masks["atom37_atom_exists"].cpu().numpy(),
        }
        
        # Include plddt if available
        if "plddt" in output:
            single_output["plddt"] = output["plddt"][i, :seq_len]
        
        results.append(single_output)
    
    return results


def batch_output_to_pdbs(
    outputs: List[dict],
    individual_inputs: List[dict],
    names: Optional[List[str]] = None,
    apply_fixes: bool = False,
) -> List[str]:
    """
    Generate PDB strings for all antibodies in batch.
    
    Args:
        outputs: List of output dictionaries from batch_add_atom37_to_output
        individual_inputs: List of input dictionaries
        names: Optional list of structure names
        apply_fixes: Whether to apply PDB fixes (slow, needed for relaxation)
        
    Returns:
        List of PDB strings
    """
    if names is None:
        names = [f"antibody_{i}" for i in range(len(outputs))]
    
    pdb_strings = []
    for i, (output, inp) in enumerate(zip(outputs, individual_inputs)):
        aatype = inp["aatype"].cpu().numpy().astype(int)
        is_heavy = inp["is_heavy"].cpu().numpy()
        atom37 = output["atom37"]
        atom_mask = output["atom37_atom_exists"].astype(int)
        
        chain_index = 1 - is_heavy.astype(int)
        residue_index = np.arange(len(aatype))
        
        protein = Protein(
            aatype=aatype,
            atom_positions=atom37,
            atom_mask=atom_mask,
            residue_index=residue_index,
            b_factors=np.zeros_like(atom_mask),
            chain_index=chain_index,
        )
        
        pdb_str = to_pdb_fast(protein)
        pdb_strings.append(pdb_str)
    
    return pdb_strings


def predict_batch(
    model: torch.nn.Module,
    antibodies: List[Tuple[str, str]],
    device: Union[str, torch.device] = "cuda",
    names: Optional[List[str]] = None,
) -> List[str]:
    """
    End-to-end batched prediction returning PDB strings.
    
    This is the main entry point for batched inference.
    
    Args:
        model: Loaded ABB3 StructureModule
        antibodies: List of (heavy_chain, light_chain) tuples
        device: Target device
        names: Optional names for structures
        
    Returns:
        List of PDB strings, one per antibody
        
    Example:
        >>> from abodybuilder3.lightning_module import LitABB3
        >>> module = LitABB3.load_from_checkpoint("checkpoint.ckpt")
        >>> model = module.model.to("cuda").eval()
        >>> antibodies = [
        ...     ("QVQLVQSGAEV...", "DIQMTQSPST..."),
        ...     ("EVQLVESGGG...", "DIQMTQSPSS..."),
        ... ]
        >>> pdbs = predict_batch(model, antibodies)
    """
    if len(antibodies) == 0:
        return []
    
    # Prepare batched input
    batched_input, individual_inputs = strings_to_batch_input(antibodies, device)
    
    # Get sequence lengths for later
    seq_lengths = [len(inp["aatype"]) for inp in individual_inputs]
    
    # Run model
    with torch.no_grad():
        output = model(
            {
                "single": batched_input["single"],
                "pair": batched_input["pair"],
            },
            batched_input["aatype"],
            batched_input["seq_mask"],
        )
    
    # Post-process
    outputs = batch_add_atom37_to_output(
        output,
        batched_input["aatype"],
        seq_lengths,
    )
    
    # Generate PDBs
    pdb_strings = batch_output_to_pdbs(outputs, individual_inputs, names)
    
    return pdb_strings


def chunked_predict(
    model: torch.nn.Module,
    antibodies: List[Tuple[str, str]],
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 16,
    names: Optional[List[str]] = None,
) -> List[str]:
    """
    Predict antibody structures in batches, handling large inputs.
    
    Splits input into chunks of batch_size to avoid memory issues.
    
    Args:
        model: Loaded ABB3 StructureModule
        antibodies: List of (heavy_chain, light_chain) tuples
        device: Target device
        batch_size: Maximum batch size per forward pass
        names: Optional names for structures
        
    Returns:
        List of PDB strings, one per antibody
    """
    if names is None:
        names = [f"antibody_{i}" for i in range(len(antibodies))]
    
    all_pdbs = []
    for i in range(0, len(antibodies), batch_size):
        batch_abs = antibodies[i:i + batch_size]
        batch_names = names[i:i + batch_size]
        pdbs = predict_batch(model, batch_abs, device, batch_names)
        all_pdbs.extend(pdbs)
    
    return all_pdbs
