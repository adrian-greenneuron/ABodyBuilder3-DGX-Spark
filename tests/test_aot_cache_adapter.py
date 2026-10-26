"""Focused checks for the export-only identity-cache adapter."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from aot_model import AOTForward, symbolic_shape_identities
from abodybuilder3.openfold.utils import rigid_utils


class TestIdentityCacheAdapter(unittest.TestCase):
    def test_exact_original_function_and_values(self):
        for dims in [(2,4),(3,5),(1,7)]:
            functions = {name:getattr(rigid_utils,name) for name in
                         ('identity_rot_mats','identity_trans','identity_quats')}
            expected = {name:fn(dims,torch.float32,torch.device('cpu'),False).clone()
                        for name,fn in functions.items()}
            with symbolic_shape_identities():
                for name,fn in functions.items():
                    self.assertIs(getattr(rigid_utils,name),fn.__wrapped__)
                    actual = getattr(rigid_utils,name)(dims,torch.float32,torch.device('cpu'),False)
                    self.assertTrue(torch.equal(actual,expected[name]))
            for name,fn in functions.items():
                self.assertIs(getattr(rigid_utils,name),fn)

    def test_restores_bindings_on_failure(self):
        original = rigid_utils.identity_quats
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with symbolic_shape_identities():
                raise RuntimeError('probe')
        self.assertIs(rigid_utils.identity_quats,original)


class TestDeploymentGuards(unittest.TestCase):
    def setUp(self):
        self.precision=torch.get_float32_matmul_precision()
        self.tf32=torch.backends.cuda.matmul.allow_tf32
        self.adapter=AOTForward.__new__(AOTForward)
        self.adapter.bounds={'multi':{'batch_size':32,'min_length':200,'max_length':272},
                             'single':{'batch_size':1,'min_length':200,'max_length':272}}
        self.options={'_inference_final_only':True,'_inference_pair_buffer':True,
                      '_inference_pair_projected':True}

    def tearDown(self):
        torch.set_float32_matmul_precision(self.precision)
        torch.backends.cuda.matmul.allow_tf32=self.tf32

    def test_rejects_tf32_profile(self):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32=True
        with torch.no_grad(), self.assertRaisesRegex(ValueError,'strict-FP32'):
            self.adapter({},torch.zeros(2,225,dtype=torch.long),None,**self.options)

    def test_rejects_different_inference_path(self):
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.no_grad(), self.assertRaisesRegex(ValueError,'feature path'):
            self.adapter({},torch.zeros(2,225,dtype=torch.long),None)

    def test_rejects_unsupported_shapes_before_dispatch(self):
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        for shape in [(33,225),(2,273),(1,199)]:
            with torch.no_grad(), self.assertRaisesRegex(ValueError,'exported AOT domain'):
                self.adapter({},torch.zeros(*shape,dtype=torch.long),None,**self.options)


if __name__ == '__main__':
    unittest.main()
