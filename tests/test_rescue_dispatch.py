"""Successful preparation must bypass rescue; unrelated failures must propagate."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rescue_policy',ROOT/'candidate/rescue_policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


class PrepError(Exception):
    def __init__(self,message,report):super().__init__(message);self.report=report


class RescueDispatchTests(unittest.TestCase):
    def setUp(self):self.options=SimpleNamespace(relax='none',ph=7.)

    def test_r8_function_body_is_retained_exactly(self):
        original=(ROOT/'reference/structure_preparation_r8.py').read_text()
        self.assertTrue((ROOT/'candidate/structure_preparation.py').read_text().startswith(original))

    def test_compatible_result_is_returned_without_rescue(self):
        value=('prepared',None,{'status':'completed'});original=Mock(return_value=value);rescue=Mock()
        with patch.dict(os.environ,AB3_EXPERIMENTAL_RESCUE='local',AB3_EXPERIMENTAL_GEOMETRY='robust_planar'):
            self.assertIs(policy.dispatch('raw',self.options,original,PrepError,runner=rescue),value)
        original.assert_called_once();rescue.assert_not_called()

    def test_unrelated_failure_never_runs_rescue(self):
        error=PrepError('parameterisation failed',{'status':'failed'});rescue=Mock()
        with patch.dict(os.environ,AB3_EXPERIMENTAL_RESCUE='restrained',AB3_EXPERIMENTAL_GEOMETRY='robust_planar'):
            with self.assertRaises(PrepError):policy.dispatch('raw',self.options,Mock(side_effect=error),PrepError,runner=rescue)
        rescue.assert_not_called()

    def test_explicit_frozen_failure_has_an_audited_rescue(self):
        cause=ValueError(policy.FROZEN_REJECTION);error=PrepError('wrapped',{'status':'failed','seconds':1.2});error.__cause__=cause
        rescue=Mock(return_value=('rescued',None,{'rescue':{},'status':'completed'}))
        with patch.dict(os.environ,AB3_EXPERIMENTAL_RESCUE='local',AB3_EXPERIMENTAL_GEOMETRY='robust_planar'):
            value=policy.dispatch('raw',self.options,Mock(side_effect=error),PrepError,runner=rescue)
        rescue.assert_called_once_with('raw',self.options,'local')
        self.assertEqual(value[2]['rescue']['trigger_failure'],error.report);self.assertTrue(value[2]['rescue']['applied'])

    def test_failed_rescue_stays_failed_without_alternate_policy(self):
        error=PrepError('wrapped',{'status':'failed'});error.__cause__=ValueError(policy.FROZEN_REJECTION)
        rescue=Mock(side_effect=ValueError('geometry failed'))
        with patch.dict(os.environ,AB3_EXPERIMENTAL_RESCUE='local',AB3_EXPERIMENTAL_GEOMETRY='robust_planar'):
            with self.assertRaises(PrepError) as failure:policy.dispatch('raw',self.options,Mock(side_effect=error),PrepError,runner=rescue)
        self.assertFalse(failure.exception.report['rescue']['applied']);rescue.assert_called_once()


if __name__=='__main__':unittest.main()
