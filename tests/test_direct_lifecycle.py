"""CPU contract test using the unchanged inherited predict-step interface."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidate.model import ProtT5


class LifecycleTests(unittest.TestCase):
    def test_original_predict_step_is_used_without_trainer_predict(self):
        item = ProtT5.__new__(ProtT5)
        item.batch_size = 1; item.inference_device = torch.device('cpu')
        item.collate_fn = lambda values: {'input_ids': torch.tensor([[len(values[0])]])}
        item.embedding_module = SimpleNamespace(predict_step=Mock(side_effect=lambda data, index: data['input_ids'].float().unsqueeze(-1)))
        item.trainer = SimpleNamespace(predict=Mock(side_effect=AssertionError('Trainer must not run')))
        outputs = item._embed_sequences(['AAA', 'LL'])
        self.assertEqual([float(value.item()) for value in outputs], [3., 2.])
        self.assertEqual(item.embedding_module.predict_step.call_count, 2)
        item.trainer.predict.assert_not_called()

    def test_does_not_silently_enable_embedding_batching(self):
        item = ProtT5.__new__(ProtT5); item.batch_size = 2
        with self.assertRaisesRegex(ValueError, 'batch-size-1'):
            item._embed_sequences(['AAA'])


if __name__ == '__main__':
    unittest.main()
