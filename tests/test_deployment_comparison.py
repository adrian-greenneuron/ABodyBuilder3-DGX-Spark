"""Guard against incorrect agreement or concurrent-memory claims."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from compare_predictions import compare
from benchmark_cli import cuda_summary


class ComparisonTests(unittest.TestCase):
    def make(self, folder, name, shift=0., mask_change=False):
        positions = np.arange(4 * 37 * 3, dtype=float).reshape(4, 37, 3) / 100 + shift
        masks = np.ones((4, 37))
        if mask_change:
            masks[0, 1] = 0
        path = folder / (name + '.npz')
        np.savez(path, positions=positions, masks=masks, aatype=np.arange(4),
                 is_heavy=np.array([1, 1, 0, 0]), confidence=np.zeros((4, 50)))
        path.with_suffix('.pdb').write_text('test')
        return path

    def test_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self.make(Path(folder), 'a')
            r = compare(p, p)
            self.assertTrue(r['raw_coordinates_exact'])
            self.assertEqual(r['ca_lddt_agreement'], 1.)

    def test_alignment_cannot_hide_coordinate_change(self):
        with tempfile.TemporaryDirectory() as folder:
            a = self.make(Path(folder), 'a')
            b = self.make(Path(folder), 'b', shift=5.)
            r = compare(a, b)
            self.assertFalse(r['raw_coordinates_exact'])
            self.assertAlmostEqual(r['max_ca_coordinate_difference_angstrom'], 5.)
            self.assertLess(r['ca_aligned_rmsd_angstrom'], 1e-10)

    def test_mask_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            a = self.make(Path(folder), 'a')
            b = self.make(Path(folder), 'b', mask_change=True)
            with self.assertRaises(ValueError):
                compare(a, b)

    def test_nonoverlapping_peaks_are_not_added(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '1.csv').write_text('monotonic_seconds,cuda_allocated_bytes,cuda_reserved_bytes\n1,10,20\n2,0,0\n')
            (root / '2.csv').write_text('monotonic_seconds,cuda_allocated_bytes,cuda_reserved_bytes\n3,10,20\n4,0,0\n')
            r = cuda_summary(root)
            self.assertEqual(r['sampled_concurrent_cuda_allocated_peak_bytes'], 10)

    def test_confidence_logits_are_converted_to_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            a = self.make(Path(folder), 'a')
            b = self.make(Path(folder), 'b')
            with np.load(b) as data:
                values = {k: data[k] for k in data.files}
            values['confidence'][:, -1] = 100.
            np.savez(b, **values)
            # Uniform bins average 50; the final bin is centred on 99.
            self.assertAlmostEqual(compare(a, b)['max_plddt_score_difference'], 49.)


if __name__ == '__main__':
    unittest.main()
