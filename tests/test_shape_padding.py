"""Verify padding preserves inputs/masks and never truncates long sequences."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'candidate'))
from shape_banks import pad_prepared
from abodybuilder3.inference_features import prepare_compact_batch


class ShapePaddingTests(unittest.TestCase):
    def test_original_values_and_individual_metadata_are_preserved(self):
        batch, inputs = prepare_compact_batch([('A' * 120, 'L' * 107), ('A' * 110, 'L' * 107)])
        actual, returned_inputs = pad_prepared((batch, inputs))
        self.assertIs(inputs, returned_inputs)
        self.assertEqual(actual['aatype'].shape, (2, 240))
        for key in batch:
            n = batch['aatype'].shape[1]
            cropped = actual[key][:, :n, :n] if key == 'pair_code' else actual[key][:, :n]
            self.assertTrue(torch.equal(cropped, batch[key]), key)
        self.assertTrue((actual['seq_mask'][:, 227:] == 0).all())
        self.assertTrue((actual['pair_code'][:, 227:, :] == 387).all())
        self.assertTrue((actual['pair_code'][:, :, 227:] == 387).all())

    def test_second_bucket_and_outside_domain_are_explicit(self):
        for length, expected in ((241, 248), (248, 248), (249, 249), (280, 280)):
            batch, inputs = prepare_compact_batch([('A' * (length - 100), 'L' * 100)])
            actual, _ = pad_prepared((batch, inputs))
            self.assertEqual(actual['aatype'].shape[1], expected)
            self.assertEqual(int(actual['seq_mask'].sum()), length)


if __name__ == '__main__':
    unittest.main()
