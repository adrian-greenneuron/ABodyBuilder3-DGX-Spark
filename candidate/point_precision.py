"""Experimental IEEE-FP32 control for the weighted-point sum only.

The surrounding model keeps its original precision settings. This is a
single-GPU, single-inference-thread diagnostic; the backend setting is global.
It is never installed in the deployed predictor.
"""
from contextlib import contextmanager
import torch


@contextmanager
def ieee_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def weighted_points(attention, points):
    with ieee_matmul():
        result = [torch.matmul(attention, value.to(attention.dtype))
                  for value in torch.unbind(points, dim=-3)]
        return torch.stack(result, dim=-3)
