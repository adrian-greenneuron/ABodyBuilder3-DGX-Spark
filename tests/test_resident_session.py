"""Prevent request-state leakage and stacked fallback adapters in a hot worker."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'candidate'))
from resident_session import ResidentSession


class FakeAdapter:
    def __init__(self, *args):
        self.models, self.bounds = {'shared': object()}, {'shape': object()}
        self.stats = {'aot_batches': 0, 'eager_shape_fallback_batches': 0, 'fallback_reasons': {}}


class SessionTests(unittest.TestCase):
    def test_reuses_model_but_restores_original_forward(self):
        forward = Mock()
        model = SimpleNamespace(forward=forward)
        cli = SimpleNamespace(load_model=Mock(return_value=model))
        session = ResidentSession(cli, SimpleNamespace(AOTForward=FakeAdapter))
        first = session.load_model('base', 'cuda')
        first.forward = Mock()
        second = session.load_model('base', 'cuda')
        self.assertIs(first, second); self.assertIs(second.forward, forward)
        cli.load_model.assert_called_once()

    def test_packages_share_executables_but_not_request_counters(self):
        session = ResidentSession(SimpleNamespace(load_model=Mock()), SimpleNamespace(AOTForward=FakeAdapter))
        first = session.load_packages('/packages', 'base', 'hash', Mock())
        first.stats['aot_batches'] = 9
        first.stats['fallback_reasons']['request_one'] = 2
        second = session.load_packages('/packages', 'base', 'hash', Mock())
        self.assertIs(first.models, second.models)
        self.assertEqual(second.stats['aot_batches'], 0)
        self.assertEqual(second.stats['fallback_reasons'], {})

    def test_rejects_custom_or_language_model_without_fallback(self):
        session = ResidentSession(SimpleNamespace(load_model=Mock()), SimpleNamespace(AOTForward=FakeAdapter))
        for model, checkpoint in (('language', None), ('base', Path('/custom'))):
            with self.assertRaises(ValueError):
                session.load_model(model, 'cuda', checkpoint)


if __name__ == '__main__':
    unittest.main()
