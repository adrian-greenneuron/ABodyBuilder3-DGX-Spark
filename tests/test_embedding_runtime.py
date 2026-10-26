"""Language backend dispatch/cache isolation and direct-loop call semantics."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

if not Path('/opt/abodybuilder3/scripts/predict.py').is_file():
    raise unittest.SkipTest('Run inside the deployment image.')
spec=importlib.util.spec_from_file_location('embedding_cli','/opt/abodybuilder3/scripts/predict.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
from abodybuilder3.language.direct_inference import ProtT5


class EmbeddingRuntimeTests(unittest.TestCase):
    def test_backend_change_cannot_reuse_the_wrong_encoder(self):
        reference, direct = object(), object()
        with patch.object(cli,'_prott5_model',None), patch.object(cli,'_prott5_device',None), \
                patch.object(cli,'_prott5_backend',None), \
                patch('abodybuilder3.language.model.ProtT5',return_value=reference) as old, \
                patch('abodybuilder3.language.direct_inference.ProtT5',return_value=direct) as new:
            self.assertIs(cli.load_prott5(torch.device('cuda'),'trainer'),reference)
            self.assertIs(cli.load_prott5(torch.device('cuda'),'auto'),direct)
            self.assertIs(cli.load_prott5(torch.device('cuda'),'direct'),direct)
            old.assert_called_once();new.assert_called_once()

    def test_explicit_direct_cpu_fails_without_loading_weights(self):
        with patch('abodybuilder3.language.direct_inference.ProtT5') as model:
            with self.assertRaisesRegex(ValueError,'requires CUDA'):
                cli.load_prott5(torch.device('cpu'),'direct')
            model.assert_not_called()

    def test_direct_loop_uses_original_step_without_trainer(self):
        item=ProtT5.__new__(ProtT5);item.batch_size=1;item.inference_device=torch.device('cpu')
        item.collate_fn=lambda values:{'input_ids':torch.tensor([[len(values[0])]])}
        item.embedding_module=SimpleNamespace(predict_step=Mock(side_effect=lambda data,index:data['input_ids'].float().unsqueeze(-1)))
        item.trainer=SimpleNamespace(predict=Mock(side_effect=AssertionError('Trainer must not run')))
        output=item._embed_sequences(['AAA','LL'])
        self.assertEqual([float(value.item()) for value in output],[3.,2.])
        self.assertEqual(item.embedding_module.predict_step.call_count,2)
        item.trainer.predict.assert_not_called()


if __name__=='__main__':unittest.main()
