"""CPU-only lazy-output semantics and validation before expensive GPU tests."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'candidate'))
from deferred_inference import PendingBatch, PendingResult


class PendingTests(unittest.TestCase):
    def fixture(self):
        coords = torch.arange(2 * 5 * 37 * 3, dtype=torch.float32).reshape(2, 5, 37, 3)
        masks = torch.ones(2, 5, 37)
        event = Mock()
        batch = PendingBatch({'atom37': coords, 'atom37_atom_exists': masks}, event, [3, 5])
        return batch, event

    def test_shapes_and_standard_mapping_operations(self):
        batch, event = self.fixture()
        item = PendingResult(batch, 0)
        self.assertEqual(len(item), 2)
        self.assertIsNone(item.get('plddt'))
        self.assertEqual(item['atom37'].shape, (3, 37, 3))
        self.assertEqual(dict(item)['atom37_atom_exists'].shape, (3, 37))
        event.synchronize.assert_called_once()

    def test_concurrent_resolution_synchronises_once(self):
        batch, event = self.fixture()
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(lambda _: PendingResult(batch, 1)['atom37'].copy(), range(16)))
        self.assertTrue(all(np.array_equal(values[0], value) for value in values))
        event.synchronize.assert_called_once()

    def test_nonfinite_valid_atoms_propagate(self):
        batch, event = self.fixture()
        batch.buffers['atom37'][0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            PendingResult(batch, 0)['atom37']

    def test_nonfinite_masked_atoms_are_ignored(self):
        batch, event = self.fixture()
        batch.buffers['atom37'][0, 0, 0, 0] = float('nan')
        batch.buffers['atom37_atom_exists'][0, 0, 0] = 0
        self.assertEqual(PendingResult(batch, 0)['atom37'].shape, (3, 37, 3))


if __name__ == '__main__':
    unittest.main()
