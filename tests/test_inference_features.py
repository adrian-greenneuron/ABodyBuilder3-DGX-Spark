"""Regression checks for the deployed compact-input and projection API."""
import unittest
import torch
from abodybuilder3.batched_inference import prepare_cpu_batch, infer_prepared_batch
from abodybuilder3.inference_features import PairProjection


class InferenceFeatureTests(unittest.TestCase):
    def test_dense_api_retained_and_compact_padding_maps_to_zero_input(self):
        pairs = [('ACDG','LK'),('W','GAS')]
        dense, expected = prepare_cpu_batch(pairs)
        compact, actual = prepare_cpu_batch(pairs,compact=True)
        self.assertIn('pair',dense)
        self.assertNotIn('pair',compact)
        self.assertEqual(compact['pair_code'][1,-1,-1].item(),387)
        for key in ('single','aatype','seq_mask'):
            self.assertTrue(torch.equal(dense[key],compact[key]))
        for a,b in zip(actual,expected):
            for key in b:self.assertTrue(torch.equal(a[key],b[key]))

    def test_compact_projection_matches_dense_and_invalidates_cache(self):
        dense,_ = prepare_cpu_batch([('ACDG','LK'),('W','GAS')])
        compact,_ = prepare_cpu_batch([('ACDG','LK'),('W','GAS')],compact=True)
        linear=torch.nn.Linear(132,7).eval()
        # Integer-valued parameters make the numerical assertion exact across
        # CPU GEMM implementations; real checkpoint equality is a GPU check.
        with torch.no_grad():
            linear.weight.copy_(torch.arange(132*7).reshape(7,132)%5)
            linear.bias.fill_(3)
            project=PairProjection(linear)
            self.assertTrue(torch.equal(project(compact['pair_code']),linear(dense['pair'])))
            linear.bias.add_(2)
            self.assertTrue(torch.equal(project(compact['pair_code']),linear(dense['pair'])))
            linear.weight.add_(1)
            self.assertTrue(torch.equal(project(compact['pair_code']),linear(dense['pair'])))

    def test_unsupported_use_fails_explicitly(self):
        with self.assertRaises(ValueError):PairProjection(torch.nn.Linear(131,7))
        project=PairProjection(torch.nn.Linear(132,7))
        with torch.no_grad(),self.assertRaises(ValueError):project(torch.tensor([0]))
        project.linear.eval()
        with self.assertRaises(ValueError):project(torch.tensor([0]))


if __name__=='__main__':unittest.main()
