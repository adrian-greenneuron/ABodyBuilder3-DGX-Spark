"""Isolated ahead-of-time adapter; never installed on main or dev.

Preserves checkpoint weights and all eight learned structure updates. The
wrapper exposes tensor inputs to export and retains the existing inference
shortcuts and broadcast point-attention reduction.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch


@contextmanager
def symbolic_shape_identities():
    """Bypass only memoization of the original identity-tensor functions.

    Their LRU keys cannot hash SymInt dimensions. Use the exact undecorated
    functions while exporting, then restore all process-local bindings.
    This does not replace any quaternion/rotation arithmetic.
    """
    from abodybuilder3.openfold.utils import rigid_utils
    names = ('identity_rot_mats', 'identity_trans', 'identity_quats')
    original = {name:getattr(rigid_utils,name) for name in names}
    try:
        for name, function in original.items():
            setattr(rigid_utils,name,function.__wrapped__)
        yield
    finally:
        for name, function in original.items():
            setattr(rigid_utils,name,function)


class ModelExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, single, pair, aatype, mask):
        return self.model(
            {'single': single, 'pair': pair}, aatype, mask,
            _inference_pair_projected=True,
            _inference_final_only=True,
            _inference_pair_buffer=True,
        )


class AOTForward:
    """Load the two exported batch domains; fail closed outside their bounds.

    Export bounds are broader than the 219--242-residue test cohort; accepting
    a shape here is not a claim of experimental qualification at every length.
    """
    def __init__(self, root, model_name):
        import torch._inductor.codecache  # Pinned NGC package-loader prerequisite.
        root = Path(root)
        manifest = json.loads((root/'aot_packages_261026.json').read_text())
        self.events = []
        self.models = {}
        self.bounds = {}
        for domain in ('multi','single'):
            package = root/manifest[model_name][domain]
            metadata = json.loads((package.parent/'preparation_261026.json').read_text())
            assert metadata['model'] == model_name
            assert metadata['precision'] == 'strict_fp32' and metadata['dynamic']
            assert metadata['attention'] == 'original_broadcast'
            assert metadata['identity_cache_bypass_eager_exact']
            start = time.perf_counter()
            self.models[domain] = torch._inductor.aoti_load_package(str(package))
            self.bounds[domain] = metadata
            self.events.append({'domain':domain,'package':str(package),
                                'package_load_seconds':time.perf_counter()-start})

    def __call__(self, features, aatype, mask, **options):
        if torch.is_grad_enabled():
            raise ValueError('AOT deployment is inference-only.')
        if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != 'highest':
            raise ValueError('This AOT package requires the explicit strict-FP32 profile.')
        if not all(options.get(k,False) for k in ('_inference_final_only','_inference_pair_buffer','_inference_pair_projected')):
            raise ValueError('AOT package requires the qualified inference feature path.')
        if options.get('_inference_point_matmul') or options.get('_inference_point_ieee') or options.get('_offload_inference'):
            raise ValueError('AOT package uses the original point reduction and no offload.')
        size, length = aatype.shape
        domain = 'single' if size == 1 else 'multi'
        bounds = self.bounds[domain]
        if not 1 <= size <= bounds['batch_size']:
            raise ValueError(f'Batch size {size} outside the exported AOT domain.')
        if not bounds['min_length'] <= length <= bounds['max_length']:
            raise ValueError(f'Length {length} outside the exported AOT domain.')
        return self.models[domain](features['single'],features['pair'],aatype,mask)
