"""Explicit opt-in activation inside the isolated experimental image only."""
import atexit
import json
import os
from pathlib import Path

if os.environ.get('ABB3_STRICT_FP32')=='1':
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False

native=None
if os.environ.get('ABB3_NATIVE_FORMATTER')=='1':
    import abodybuilder3.utils as utils
    from candidate.format_pdb import to_pdb_fast
    from candidate import _pdb_native as native
    utils.to_pdb_fast=to_pdb_fast


def finish():
    if os.environ.get('ABB3_FORMATTER_USAGE'):
        Path(os.environ['ABB3_FORMATTER_USAGE']).write_text(json.dumps({
            'native_enabled':native is not None,'native_calls':native.calls() if native else 0,
            'strict_fp32':os.environ.get('ABB3_STRICT_FP32')=='1'})+'\n')


atexit.register(finish)
