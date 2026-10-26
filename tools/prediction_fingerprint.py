"""Compact exact-output hashes for controlled inference comparisons."""
import hashlib
import numpy as np


def array_hash(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    value=np.ascontiguousarray(value)
    h=hashlib.sha256()
    h.update(repr((value.shape,value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def fingerprint(output,inputs):
    return {**{k:array_hash(inputs[k]) for k in ('aatype','is_heavy')},
            **{k:array_hash(output[k]) for k in ('atom37','atom37_atom_exists','plddt') if k in output}}
