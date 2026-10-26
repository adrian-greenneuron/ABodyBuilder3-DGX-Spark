"""Isolated bounded asynchronous transfer adapter; unchanged model arithmetic.

The installed scheduler still bounds prepared and output batches. The model
process enqueues conversion and copies; an output worker waits for that batch's
event only when it needs CPU arrays. No GPU inference occurs in output workers.
"""
from collections.abc import Mapping
import threading

import numpy as np
import torch
from abodybuilder3.openfold.data.data_transforms import make_atom14_masks
from abodybuilder3.openfold.utils.feats import atom14_to_atom37


class PendingBatch:
    def __init__(self, buffers, event, lengths):
        self.buffers = buffers
        self.event = event
        self.lengths = lengths
        self.lock = threading.Lock()
        self.arrays = None

    def resolve(self):
        with self.lock:
            if self.arrays is None:
                self.event.synchronize()
                arrays = {key: value.numpy() for key, value in self.buffers.items()}
                # Match the installed eager/normal-AOT finite-coordinate check.
                for i, length in enumerate(self.lengths):
                    coords = arrays['atom37'][i, :length]
                    mask = arrays['atom37_atom_exists'][i, :length].astype(bool)
                    if not np.isfinite(coords[mask]).all():
                        raise ValueError(f'Nonfinite predicted coordinates in batch item {i}.')
                self.arrays = arrays
            return self.arrays


class PendingResult(Mapping):
    def __init__(self, batch, index):
        self.batch, self.index = batch, index

    def __getitem__(self, key):
        if key not in self.batch.buffers:
            raise KeyError(key)
        return self.batch.resolve()[key][self.index, :self.batch.lengths[self.index]]

    def __iter__(self):
        return iter(self.batch.buffers)

    def __len__(self):
        return len(self.batch.buffers)


def pin_prepared(prepared):
    cpu, inputs = prepared
    return {key: tensor.pin_memory() for key, tensor in cpu.items()}, inputs


def infer_prepared_batch(model, prepared, device='cuda', *, final_only=True, reuse_pair_buffer=True):
    if torch.device(device).type != 'cuda':
        raise ValueError('Deferred transfer experiment requires CUDA.')
    cpu, inputs = prepared
    if not all(value.is_pinned() for value in cpu.values()):
        raise ValueError('Deferred H2D transfer requires pinned, immutable CPU features.')
    with torch.no_grad():
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu.items()}
        options = {'_inference_final_only': final_only,
                   '_inference_pair_buffer': reuse_pair_buffer}
        if 'pair_code' in batch:
            from abodybuilder3.inference_features import PairProjection
            projector = getattr(model, '_inference_pair_projection', None)
            if projector is None:
                projector = PairProjection(model.linear_in_edge)
                model._inference_pair_projection = projector
            batch['pair'] = projector(batch.pop('pair_code'))
            options['_inference_pair_projected'] = True
        output = model({'single': batch['single'], 'pair': batch['pair']},
                       batch['aatype'], batch['seq_mask'], **options)
        masks = make_atom14_masks({'aatype': batch['aatype']})
        tensors = {'atom37': atom14_to_atom37(output['positions'][-1], masks),
                   'atom37_atom_exists': masks['atom37_atom_exists']}
        if 'plddt' in output:
            tensors['plddt'] = output['plddt']
        buffers = {key: torch.empty_like(value, device='cpu', pin_memory=True)
                   for key, value in tensors.items()}
        for key, value in tensors.items():
            buffers[key].copy_(value, non_blocking=True)
        # Copies use the compute stream in this first candidate: allocator
        # lifetimes remain ordered on that stream, and inference stays serial.
        event = torch.cuda.Event()
        event.record()
    pending = PendingBatch(buffers, event, [len(item['aatype']) for item in inputs])
    return [PendingResult(pending, i) for i in range(len(inputs))], inputs
