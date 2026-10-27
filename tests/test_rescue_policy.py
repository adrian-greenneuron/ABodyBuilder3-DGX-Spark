"""Rejection-only rescue, explicit options and failure propagation."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from structure_preparation import Options, PreparationError
from rescue_policy import FROZEN_REJECTION, dispatch, is_frozen_rejection


def rejected():
    error = PreparationError("geometry rejected", {"status": "failed", "seconds": 1.2})
    error.__cause__ = ValueError(FROZEN_REJECTION)
    return error


class RescuePolicyTests(unittest.TestCase):
    def test_defaults_and_separate_whole_structure_relaxation(self):
        self.assertEqual(Options().rescue_policy, "local")
        self.assertEqual(Options(relax="restrained").rescue_policy, "off")
        self.assertEqual(Options(rescue="restrained").rescue_policy, "restrained")
        Options(ph=5, rescue="off").validate()
        for options in (Options(rescue="unknown"), Options(ph=5),
                        Options(relax="restrained", rescue="local")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                options.validate()

    def test_compatible_preparation_never_calls_rescue(self):
        value = ("prepared", None, {"status": "completed"})
        original, runner = Mock(return_value=value), Mock()
        result = dispatch("raw", Options(), original, PreparationError, runner)
        self.assertIs(result, value)
        self.assertFalse(result[2]["rescue"]["applied"])
        original.assert_called_once()
        runner.assert_not_called()

    def test_off_preserves_failure_and_never_repairs(self):
        error, runner = rejected(), Mock()
        with self.assertRaises(PreparationError) as caught:
            dispatch("raw", Options(rescue="off"), Mock(side_effect=error), PreparationError, runner)
        self.assertIs(caught.exception, error)
        runner.assert_not_called()

    def test_only_exact_chained_frozen_rejection_qualifies(self):
        self.assertTrue(is_frozen_rejection(rejected()))
        for cause in (RuntimeError(FROZEN_REJECTION), ValueError(FROZEN_REJECTION + " extra"),
                      ValueError("No compatible CUDA device is available")):
            error = PreparationError(str(cause), {"status": "failed"})
            error.__cause__ = cause
            runner = Mock()
            with self.subTest(cause=cause), self.assertRaises(PreparationError):
                dispatch("raw", Options(), Mock(side_effect=error), PreparationError, runner)
            runner.assert_not_called()
        cycle = ValueError("unrelated")
        cycle.__cause__ = cycle
        self.assertFalse(is_frozen_rejection(cycle))

    def test_selected_policy_and_trigger_are_audited(self):
        for policy in ("local", "restrained"):
            options, error = Options(rescue=policy), rejected()
            runner = Mock(return_value=("repaired", None, {"rescue": {}, "status": "completed"}))
            value = dispatch("raw", options, Mock(side_effect=error), PreparationError, runner)
            runner.assert_called_once_with("raw", options, policy)
            self.assertTrue(value[2]["rescue"]["applied"])
            self.assertEqual(value[2]["rescue"]["trigger_failure"], error.report)
            self.assertEqual(value[2]["rescue"]["policy"], policy)

    def test_failed_local_repair_never_escalates_to_restrained(self):
        options, runner = Options(), Mock(side_effect=ValueError("local geometry failed"))
        with self.assertRaises(PreparationError) as caught:
            dispatch("raw", options, Mock(side_effect=rejected()), PreparationError, runner)
        runner.assert_called_once_with("raw", options, "local")
        self.assertEqual(caught.exception.report["status"], "failed")
        self.assertFalse(caught.exception.report["rescue"]["applied"])

    def test_failed_rescue_retains_its_partial_attempt_audit(self):
        partial = {"status": "failed", "rescue": {"heavy_attempts": [{"radius": 1}]}}
        runner = Mock(side_effect=PreparationError("repair failed", partial))
        with self.assertRaises(PreparationError) as caught:
            dispatch("raw", Options(), Mock(side_effect=rejected()), PreparationError, runner)
        self.assertEqual(caught.exception.report["rescue"]["rescue_failure_audit"], partial)


if __name__ == "__main__":
    unittest.main()
