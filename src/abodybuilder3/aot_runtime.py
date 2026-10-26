"""Saved strict-FP32 execution with explicit identity checks and shape fallback.

Packages are built from the installed model; loading never invokes torch.compile.
The normal model remains available for inputs outside the packaged shape domain.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch


SCHEMA_VERSION = 1
SOURCE_FILES = (
    'aot_runtime.py', 'inference_features.py',
    'openfold/model/structure_module.py', 'openfold/utils/rigid_utils.py',
    'openfold/np/residue_constants.py',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in SOURCE_FILES}


def runtime_identity():
    return {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'python': platform.python_version(), 'machine': platform.machine(),
            'capability': list(torch.cuda.get_device_capability())}


def configure_precision(precision):
    if precision not in ('strict', 'tf32'):
        raise ValueError('Precision must be strict or tf32.')
    strict = precision == 'strict'
    torch.set_float32_matmul_precision('highest' if strict else 'high')
    torch.backends.cuda.matmul.allow_tf32 = not strict
    torch.backends.cudnn.allow_tf32 = not strict


def require_strict_fp32():
    if (torch.get_float32_matmul_precision() != 'highest'
            or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32):
        raise ValueError('Saved AOT execution requires strict FP32.')


@contextmanager
def symbolic_shape_identities():
    """Export the exact identity functions without their SymInt-incompatible LRU."""
    from abodybuilder3.openfold.utils import rigid_utils
    names = ('identity_rot_mats', 'identity_trans', 'identity_quats')
    original = {name: getattr(rigid_utils, name) for name in names}
    try:
        for name, function in original.items():
            setattr(rigid_utils, name, function.__wrapped__)
        yield
    finally:
        for name, function in original.items():
            setattr(rigid_utils, name, function)


class ModelExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, single, pair, aatype, mask):
        return self.model({'single': single, 'pair': pair}, aatype, mask,
                          _inference_pair_projected=True,
                          _inference_final_only=True,
                          _inference_pair_buffer=True)


def safe_package_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or invalid AOT package path: {relative}')
    return path


class AOTForward:
    """Use the saved executable in-domain and report strict-eager shape fallbacks.

    Missing, corrupt or incompatible packages are errors. They never trigger a
    silent rebuild or fallback. Shape fallback preserves the selected model and
    strict precision, and is recorded even when normal CLI progress is quiet.
    """
    def __init__(self, root, model_name, checkpoint_sha256, fallback):
        require_strict_fp32()
        root = Path(root)
        manifest_path = root / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError(f'AOT manifest missing: {manifest_path}. '
                             'Build the packaged dev image or request --execution eager.')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Incompatible AOT manifest schema.')
        if manifest.get('runtime') != runtime_identity():
            raise ValueError('AOT runtime/GPU identity differs; rebuild the packages for this runtime.')
        if manifest.get('sources') != source_identity():
            raise ValueError('AOT model/adapter source identity differs; rebuild the packages.')
        entry = manifest.get('models', {}).get(model_name)
        if entry is None or entry.get('checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('AOT checkpoint identity differs; use matching packages or --execution eager.')
        if entry.get('precision') != 'strict_fp32' or entry.get('attention') != 'original_broadcast':
            raise ValueError('AOT precision/attention protocol differs.')
        self.fallback = fallback
        self.stats = {'execution': 'aot', 'precision': 'strict_fp32',
                      'checkpoint_sha256': checkpoint_sha256,
                      'manifest_sha256': sha256(manifest_path), 'packages': {},
                      'aot_batches': 0, 'eager_shape_fallback_batches': 0,
                      'fallback_reasons': {}}
        self.models = {}
        self.bounds = {}
        # The pinned NGC runtime requires the loader registration in codecache.
        import torch._inductor.codecache
        for domain in ('single', 'multi'):
            record = entry.get('domains', {}).get(domain, {})
            low, high = record.get('min_batch'), record.get('max_batch')
            min_length, max_length = record.get('min_length'), record.get('max_length')
            if (domain == 'single' and (low, high) != (1, 1)) or (
                    domain == 'multi' and (low != 2 or not isinstance(high, int) or high < 2)):
                raise ValueError(f'Invalid AOT batch domain: {domain}')
            if not isinstance(min_length, int) or not isinstance(max_length, int) or not 1 <= min_length <= max_length:
                raise ValueError(f'Invalid AOT length domain: {domain}')
            package = safe_package_path(root, record.get('path', ''))
            if sha256(package) != record.get('sha256'):
                raise ValueError(f'AOT package checksum failed: {domain}')
            started = time.perf_counter()
            self.models[domain] = torch._inductor.aoti_load_package(str(package))
            self.bounds[domain] = record
            self.stats['packages'][domain] = {
                'sha256': record['sha256'],
                'load_seconds': time.perf_counter() - started}

    def __call__(self, features, aatype, mask, **options):
        require_strict_fp32()
        if torch.is_grad_enabled():
            raise ValueError('Saved AOT execution is inference-only.')
        required = ('_inference_pair_projected', '_inference_final_only', '_inference_pair_buffer')
        if set(options) != set(required) or not all(options[name] is True for name in required):
            raise ValueError('Saved AOT execution requires the qualified compact inference path.')
        size, length = aatype.shape
        domain = 'single' if size == 1 else 'multi'
        bound = self.bounds[domain]
        if (not bound['min_batch'] <= size <= bound['max_batch']
                or not bound['min_length'] <= length <= bound['max_length']):
            reason = f'{domain}:batch={size},length={length}'
            reasons = self.stats['fallback_reasons']
            if reason not in reasons:
                print(f'AOT shape outside packaged bounds ({reason}); using the same model in strict FP32 eager execution.',
                      file=sys.stderr, flush=True)
            reasons[reason] = reasons.get(reason, 0) + 1
            self.stats['eager_shape_fallback_batches'] += 1
            return self.fallback(features, aatype, mask, **options)
        values = (features['single'], features['pair'], aatype, mask)
        if (values[0].dtype != torch.float32 or values[1].dtype != torch.float32
                or aatype.dtype != torch.int64 or mask.dtype != torch.float32
                or any(value.device.type != 'cuda' for value in values)):
            raise ValueError('AOT inputs require CUDA FP32 features/masks and int64 residue types.')
        result = self.models[domain](*values)
        self.stats['aot_batches'] += 1
        return result
