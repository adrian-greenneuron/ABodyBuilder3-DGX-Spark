"""Parallel preparation policies and bounded dispatch; no model/chemistry imports."""
from concurrent.futures import Future
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "candidate"
sys.path.insert(0, str(SCRIPTS))
import preparation_pool as pool


class PreparationPoolTests(unittest.TestCase):
    def test_auto_preserves_serial_relaxation_and_explicit_thread_setting(self):
        with patch.dict(os.environ, {"OPENMM_CPU_THREADS": "1"}), patch.object(os, "cpu_count", return_value=20):
            self.assertEqual(pool.worker_count("auto", 100, "none"), 8)
            self.assertEqual(pool.worker_count("auto", 2, "none"), 2)
            self.assertEqual(pool.worker_count("auto", 100, "restrained"), 1)
        with patch.dict(os.environ, {"OPENMM_CPU_THREADS": "4"}):
            self.assertEqual(pool.worker_count("auto", 100, "none"), 1)
            with self.assertRaisesRegex(ValueError, "OPENMM_CPU_THREADS"):
                pool.worker_count("4", 100, "none")
        for value, relax in (("0", "none"), ("9", "none"), ("bad", "none"), ("2", "restrained")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pool.worker_count(value, 8, relax)

    def test_bounded_dispatch_and_input_order(self):
        class Executor:
            def __init__(self):
                self.pending = []
                self.submitted = []

            def submit(self, function, job):
                future = Future()
                self.pending.append(future)
                self.submitted.append(job)
                return future

        executor = Executor()
        def complete(pending, return_when):
            self.assertLessEqual(len(pending), 4)
            # Complete out of order, retaining a backlog while filling the queue.
            future = list(pending)[-1]
            index = executor.pending.index(future)
            future.set_result(executor.submitted[index] * 2)
            return {future}, set(pending) - {future}

        with patch.object(pool, "wait", side_effect=complete):
            self.assertEqual(pool._bounded_map(executor, range(17), 4), [i * 2 for i in range(17)])

    def test_duplicate_paths_fail_before_launch(self):
        with patch.object(pool.subprocess, "run") as launch, self.assertRaisesRegex(ValueError, "Duplicate"):
            pool.prepare_paths(["same.pdb", "same.pdb"], None, 4)
        launch.assert_not_called()

    def test_import_has_no_runtime_side_effects(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import preparation_pool; "
            "assert not {'torch','openmm','pdbfixer','numpy'} & sys.modules.keys()"],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
