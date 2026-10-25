"""Compact-input boundary and cache invalidation regression tests."""
import importlib.util
from pathlib import Path
import unittest

import torch
from abodybuilder3.batched_inference import prepare_cpu_batch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('features', ROOT/'candidate/features.py')
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


class FeaturesTests(unittest.TestCase):
    def test_compact_codes_reconstruct_original_features_including_padding(self):
        pairs = [('ACDG','LK'),('W','GAS')]
        compact, inputs = features.prepare_compact_batch(pairs)
        original, expected_inputs = prepare_cpu_batch(pairs)
        table = torch.zeros(388,132)
        for code in range(387):
            table[code,code//129] = 1
            table[code,3+code%129] = 1
        self.assertTrue(torch.equal(table[compact['pair_code']], original['pair']))
        for key in ('single','aatype','seq_mask'):
            self.assertTrue(torch.equal(compact[key],original[key]),key)
        for actual,expected in zip(inputs,expected_inputs):
            for key in expected:
                self.assertTrue(torch.equal(actual[key],expected[key]))

    def test_cache_refreshes_after_weight_and_bias_updates(self):
        linear = torch.nn.Linear(132,7).eval()
        projector = features.PairProjection(linear)
        codes = torch.tensor([0,64,130,386,387])
        with torch.no_grad():
            before = projector(codes).clone()
            linear.weight.add_(1)
            after = projector(codes).clone()
            torch.testing.assert_close(after[:-1]-before[:-1],torch.full((4,7),2.))
            self.assertTrue(torch.equal(after[-1],linear.bias))
            linear.bias.add_(3)
            torch.testing.assert_close(projector(codes)-after,torch.full((5,7),3.))

    def test_training_and_gradient_use_are_rejected(self):
        linear = torch.nn.Linear(132,7)
        projector = features.PairProjection(linear)
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'inference-only'):
            projector(torch.tensor([0]))
        linear.eval()
        with self.assertRaisesRegex(ValueError,'inference-only'):
            projector(torch.tensor([0]))

    def test_empty_chains_and_incompatible_weights_are_rejected(self):
        for pairs in ([],[('','L')],[('H','')]):
            with self.assertRaises(ValueError):features.prepare_compact_batch(pairs)
        with self.assertRaises(ValueError):features.PairProjection(torch.nn.Linear(131,7))


if __name__ == '__main__':
    unittest.main()
