"""Experimental compact features using the released edge-projection weights.

No learned parameter is changed. Numerical identity with the original dense
projection is tested, not assumed: a different GEMM shape can change rounding.
"""
import torch
from abodybuilder3.openfold.np.residue_constants import restype_order_with_x


def prepare_compact_batch(antibodies):
    if not antibodies or any(not h or not l for h, l in antibodies):
        raise ValueError('Each batch and both chains must be nonempty.')
    lengths = [len(h) + len(l) for h, l in antibodies]
    size, longest = len(lengths), max(lengths)
    aatype = torch.zeros((size, longest), dtype=torch.long)
    heavy = torch.zeros_like(aatype)
    index = torch.zeros_like(aatype)
    valid = torch.zeros((size, longest), dtype=torch.float32)
    inputs = []
    for i, ((h, l), length) in enumerate(zip(antibodies, lengths)):
        aatype[i, :length] = torch.tensor([restype_order_with_x[x] for x in h+l])
        heavy[i, :len(h)] = 1
        index[i, :len(h)] = torch.arange(len(h))
        index[i, len(h):length] = torch.arange(len(l)) + 500
        valid[i, :length] = 1
        inputs.append({'aatype': aatype[i, :length], 'is_heavy': heavy[i, :length],
                       'residue_index': index[i, :length]})
    single = torch.zeros((size, longest, 23), dtype=torch.float32)
    single.scatter_(-1, aatype.unsqueeze(-1), 1.)
    single.scatter_(-1, (21+heavy).unsqueeze(-1), 1.)
    single.mul_(valid.unsqueeze(-1))
    relative = (index[:, None, :] - index[:, :, None]).clamp(-64, 64) + 64
    chain = 2*heavy[:, :, None]*heavy[:, None, :] + (1-heavy[:, :, None])*(1-heavy[:, None, :])
    code = chain*129 + relative
    code.masked_fill_(~(valid[:, :, None].bool() & valid[:, None, :].bool()), 387)
    return {'single': single, 'pair_code': code, 'aatype': aatype, 'seq_mask': valid}, inputs


class PairProjection:
    """Cache 3*129 categorical combinations plus the all-zero padded input."""
    def __init__(self, linear):
        if linear.weight.shape[1] != 132:
            raise ValueError('Expected three chain and 129 relative-position channels.')
        self.linear = linear
        self.table = None
        self.identity = None

    def __call__(self, codes):
        linear = self.linear
        if linear.training or torch.is_grad_enabled():
            raise ValueError('Projection lookup is inference-only.')
        identity = (linear.weight.data_ptr(), linear.weight._version,
                    linear.weight.dtype, linear.weight.device,
                    None if linear.bias is None else (linear.bias.data_ptr(), linear.bias._version))
        if identity != self.identity:
            rows = torch.arange(387, device=linear.weight.device)
            features = linear.weight.new_zeros((388, 132))
            features[rows, rows//129] = 1
            features[rows, 3+rows%129] = 1
            self.table = linear(features)
            self.identity = identity
        return self.table[codes]
