"""FP32 batched structure inference using the unchanged ABodyBuilder3 model.

CPU feature construction avoids per-item dense one-hot temporaries. All raw
coordinates are transferred to CPU before PDB formatting. No prediction cache,
precision change, model compilation or automatic alternative model is used.
"""
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from abodybuilder3.openfold.data.data_transforms import make_atom14_masks
from abodybuilder3.openfold.np.protein import Protein
from abodybuilder3.openfold.np.residue_constants import restype_order_with_x
from abodybuilder3.openfold.utils.feats import atom14_to_atom37
from abodybuilder3.utils import to_pdb_fast


def prepare_cpu_batch(antibodies):
    if not antibodies:
        raise ValueError('Cannot prepare an empty batch.')
    lengths = [len(h) + len(l) for h, l in antibodies]
    if any(not h or not l for h, l in antibodies):
        raise ValueError('Each antibody requires nonempty heavy and light chains.')
    size, longest = len(antibodies), max(lengths)
    aatype = torch.zeros((size, longest), dtype=torch.long)
    heavy = torch.zeros_like(aatype)
    index = torch.zeros_like(aatype)
    valid = torch.zeros((size, longest), dtype=torch.float32)
    inputs = []
    for i, ((h, l), length) in enumerate(zip(antibodies, lengths)):
        aatype[i, :length] = torch.tensor([restype_order_with_x[x] for x in h + l])
        heavy[i, :len(h)] = 1
        index[i, :len(h)] = torch.arange(len(h))
        index[i, len(h):length] = torch.arange(len(l)) + 500
        valid[i, :length] = 1
        inputs.append({'aatype': aatype[i, :length], 'is_heavy': heavy[i, :length],
                       'residue_index': index[i, :length]})
    single = torch.zeros((size, longest, 23), dtype=torch.float32)
    single.scatter_(-1, aatype.unsqueeze(-1), 1.)
    single.scatter_(-1, (21 + heavy).unsqueeze(-1), 1.)
    single.mul_(valid.unsqueeze(-1))
    relative = (index[:, None, :] - index[:, :, None]).clamp(-64, 64) + 64
    pair = torch.zeros((size, longest, longest, 132), dtype=torch.float32)
    pair.scatter_(-1, (3 + relative).unsqueeze(-1), 1.)
    chain = 2 * heavy[:, :, None] * heavy[:, None, :] + (1-heavy[:, :, None]) * (1-heavy[:, None, :])
    pair.scatter_(-1, chain.unsqueeze(-1), 1.)
    pair.mul_((valid[:, :, None] * valid[:, None, :]).unsqueeze(-1))
    return {'single': single, 'pair': pair, 'aatype': aatype, 'seq_mask': valid}, inputs


def string_to_input_dict(heavy, light):
    batch, inputs = prepare_cpu_batch([(heavy, light)])
    return {**inputs[0], 'single': batch['single'][0], 'pair': batch['pair'][0]}


def strings_to_batch_input(antibodies, device='cuda'):
    batch, inputs = prepare_cpu_batch(antibodies)
    return {k: v.to(device) for k, v in batch.items()}, inputs


def batch_add_atom37_to_output(output, aatypes, seq_lengths):
    """Vectorised atom conversion followed by one CPU transfer per output array."""
    masks = make_atom14_masks({'aatype': aatypes})
    coords = atom14_to_atom37(output['positions'][-1], masks).detach().cpu().numpy()
    exists = masks['atom37_atom_exists'].detach().cpu().numpy()
    confidence = output.get('plddt')
    if confidence is not None:
        confidence = confidence.detach().cpu().numpy()
    results = []
    for i, length in enumerate(seq_lengths):
        result = {'atom37': coords[i, :length], 'atom37_atom_exists': exists[i, :length]}
        if confidence is not None:
            result['plddt'] = confidence[i, :length]
        if not np.isfinite(result['atom37'][result['atom37_atom_exists'].astype(bool)]).all():
            raise ValueError(f'Nonfinite predicted coordinates in batch item {i}.')
        results.append(result)
    return results


def infer_prepared_batch(model, prepared, device='cuda'):
    cpu_batch, inputs = prepared
    batch = {k: v.to(device) for k, v in cpu_batch.items()}
    with torch.no_grad():
        output = model({'single': batch['single'], 'pair': batch['pair']},
                       batch['aatype'], batch['seq_mask'])
        results = batch_add_atom37_to_output(output, batch['aatype'], [len(x['aatype']) for x in inputs])
    return results, inputs


def batch_output_to_pdbs(outputs, individual_inputs, names=None, apply_fixes=False):
    if apply_fixes:
        raise ValueError('Use the explicit preparation pipeline instead of implicit PDB repair.')
    pdbs = []
    for output, inp in zip(outputs, individual_inputs):
        aatype = inp['aatype'].cpu().numpy().astype(int)
        mask = output['atom37_atom_exists'].astype(int)
        protein = Protein(aatype=aatype, atom_positions=output['atom37'], atom_mask=mask,
                          residue_index=np.arange(len(aatype)), b_factors=np.zeros_like(mask),
                          chain_index=1-inp['is_heavy'].cpu().numpy().astype(int))
        pdbs.append(to_pdb_fast(protein))
    return pdbs


def predict_batch(model, antibodies, device='cuda', names=None):
    if not antibodies:
        return []
    results, inputs = infer_prepared_batch(model, prepare_cpu_batch(antibodies), device)
    return batch_output_to_pdbs(results, inputs, names)


def chunked_predict(model, antibodies, device='cuda', batch_size=16, names=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    return [pdb for i in range(0, len(antibodies), batch_size)
            for pdb in predict_batch(model, antibodies[i:i+batch_size], device)]
