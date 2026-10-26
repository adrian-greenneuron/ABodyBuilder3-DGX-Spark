"""Isolated fixed-length routing with explicit padding and dynamic fallback."""
from pathlib import Path

import torch.nn.functional as F
from abodybuilder3.aot_runtime import AOTForward as OriginalForward


def pad_prepared(prepared):
    cpu, inputs = prepared
    n = cpu['aatype'].shape[1]
    target = 240 if n <= 240 else (248 if n <= 248 else n)
    if target == n:
        return prepared
    result = {}
    for key, value in cpu.items():
        if key == 'pair_code':
            result[key] = F.pad(value, (0, target-n, 0, target-n), value=387)
        elif key == 'single':
            result[key] = F.pad(value, (0, 0, 0, target-n))
        else:
            result[key] = F.pad(value, (0, target-n))
    return result, inputs


class ShapeBankForward:
    def __init__(self, bank_root, dynamic_root, model_name, checkpoint, fallback):
        self.routes = {'dynamic': OriginalForward(dynamic_root, model_name, checkpoint, fallback)}
        for length in (240, 248):
            self.routes[length] = OriginalForward(Path(bank_root) / f'fixed{length}', model_name, checkpoint, fallback)
        self.stats = {'execution': 'experimental_shape_banks', 'precision': 'strict_fp32',
                      'aot_batches': 0, 'eager_shape_fallback_batches': 0,
                      'route_batches': {'dynamic': 0, '240': 0, '248': 0}}

    def __call__(self, features, aatype, mask, **options):
        length = aatype.shape[1]
        key = length if length in (240, 248) else 'dynamic'
        active = self.routes[key]
        before = active.stats['aot_batches']
        result = active(features, aatype, mask, **options)
        if active.stats['aot_batches'] > before:
            self.stats['aot_batches'] += 1
        else:
            self.stats['eager_shape_fallback_batches'] += 1
        self.stats['route_batches'][str(key)] += 1
        return result
