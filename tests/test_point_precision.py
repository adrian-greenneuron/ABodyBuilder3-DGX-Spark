import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

spec=importlib.util.spec_from_file_location('point_precision',Path(__file__).resolve().parents[1]/'candidate/point_precision.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PrecisionTests(unittest.TestCase):
    def test_restores_flag_including_exception(self):
        before=torch.backends.cuda.matmul.allow_tf32
        with m.ieee_matmul():
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,before)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with m.ieee_matmul():raise RuntimeError('test')
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,before)

    def test_weighted_sum_matches_float64_equation(self):
        torch.manual_seed(8)
        a=torch.randn(2,3,5,5,dtype=torch.float64)
        points=torch.randn(2,3,3,5,2,dtype=torch.float64)
        reference=(a[...,None,:,:,None]*points[...,None,:,:]).sum(-2)
        self.assertTrue(torch.allclose(m.weighted_points(a,points),reference,atol=1e-14,rtol=1e-14))


if __name__=='__main__':unittest.main()
