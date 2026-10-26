"""Focused deployment checks: precision, immutable identities and shape fallback."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from abodybuilder3 import aot_runtime as runtime


class AOTRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.precision = torch.get_float32_matmul_precision()
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        self.cudnn = torch.backends.cudnn.allow_tf32
        runtime.configure_precision('strict')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'model.pt2').write_bytes(b'package-test')
        self.identity = {'test_runtime': 'fixed'}
        self.source = {'test_source': 'fixed'}
        domain = {'path': 'model.pt2', 'sha256': runtime.sha256(self.root / 'model.pt2'),
                  'min_length': 200, 'max_length': 272}
        self.manifest = {'schema_version': 1, 'runtime': self.identity, 'sources': self.source,
                         'models': {'base': {'checkpoint_sha256': 'checkpoint',
                             'precision': 'strict_fp32', 'attention': 'original_broadcast',
                             'domains': {'single': {**domain, 'min_batch': 1, 'max_batch': 1},
                                         'multi': {**domain, 'min_batch': 2, 'max_batch': 32}}}}}
        self.options = dict(_inference_pair_projected=True, _inference_final_only=True,
                            _inference_pair_buffer=True)

    def tearDown(self):
        torch.set_float32_matmul_precision(self.precision)
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cudnn.allow_tf32 = self.cudnn

    def load(self, manifest=None):
        (self.root / 'manifest.json').write_text(json.dumps(manifest or self.manifest))
        with patch.object(runtime, 'runtime_identity', return_value=self.identity), \
                patch.object(runtime, 'source_identity', return_value=self.source), \
                patch('torch._inductor.aoti_load_package', return_value=Mock()):
            return runtime.AOTForward(self.root, 'base', 'checkpoint', Mock(return_value='eager'))

    def test_rejects_runtime_source_checkpoint_and_protocol_mismatch(self):
        for field in ('runtime', 'sources', 'checkpoint', 'precision', 'attention'):
            record = copy.deepcopy(self.manifest)
            if field in ('runtime', 'sources'):
                record[field] = {'changed': True}
            else:
                key = 'checkpoint_sha256' if field == 'checkpoint' else field
                record['models']['base'][key] = 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(record)

    def test_rejects_missing_manifest(self):
        with self.assertRaisesRegex(ValueError, 'manifest missing'):
            runtime.AOTForward(self.root, 'base', 'checkpoint', Mock())

    def test_rejects_corrupt_or_escaping_package(self):
        for change in ({'sha256': 'wrong'}, {'path': '../outside.pt2'}):
            record = copy.deepcopy(self.manifest)
            record['models']['base']['domains']['single'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(record)

    def test_rejects_invalid_domain(self):
        for change in ({'min_batch': 2}, {'max_length': 199}, {'min_length': None}):
            record = copy.deepcopy(self.manifest)
            record['models']['base']['domains']['single'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(record)

    def test_fallback_is_recorded_and_stays_strict(self):
        adapter = self.load()
        with torch.no_grad():
            for shape in ((1, 199), (2, 273), (33, 225)):
                with self.subTest(shape=shape):
                    result = adapter({}, torch.zeros(shape, dtype=torch.int64), None, **self.options)
                    self.assertEqual(result, 'eager')
        self.assertEqual(adapter.stats['aot_batches'], 0)
        self.assertEqual(adapter.stats['eager_shape_fallback_batches'], 3)
        self.assertEqual(adapter.fallback.call_count, 3)
        runtime.require_strict_fp32()

    def test_rejects_precision_change_and_unqualified_options(self):
        adapter = self.load()
        runtime.configure_precision('tf32')
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'strict FP32'):
            adapter({}, torch.zeros(1, 225), None, **self.options)
        runtime.configure_precision('strict')
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'compact inference path'):
            adapter({}, torch.zeros(1, 225), None)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'CUDA FP32'):
            adapter({'single': torch.zeros(1, 225, 23), 'pair': torch.zeros(1)},
                    torch.zeros(1, 225, dtype=torch.int64), torch.ones(1, 225), **self.options)

    def test_restores_exact_identity_functions_after_export_failure(self):
        from abodybuilder3.openfold.utils import rigid_utils
        original = rigid_utils.identity_quats
        expected = original((2, 4), torch.float32, torch.device('cpu'), False)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with runtime.symbolic_shape_identities():
                self.assertIs(rigid_utils.identity_quats, original.__wrapped__)
                self.assertTrue(torch.equal(expected, rigid_utils.identity_quats(
                    (2, 4), torch.float32, torch.device('cpu'), False)))
                raise RuntimeError('probe')
        self.assertIs(rigid_utils.identity_quats, original)


if __name__ == '__main__':
    unittest.main()
