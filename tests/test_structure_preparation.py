"""Small CPU checks for the new interface and preparation policy."""
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from structure_preparation import Options


class PreparationPolicyTests(unittest.TestCase):
    def test_supported_modes(self):
        Options().validate()
        Options(relax="restrained", ph=6).validate()
        Options(hydrogen_method="reduce", sidechain_flips=True).validate()

    def test_rejects_unsupported_or_silent_settings(self):
        for options in (Options(ph=float("nan")), Options(ph=15),
                        Options(hydrogen_method="pdbfixer"),
                        Options(sidechain_flips=True), Options(max_iterations=0),
                        Options(restraint_k=-1), Options(tolerance=float("inf"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                options.validate()

    def test_retired_flags_fail_before_runtime_import(self):
        for argument in ("--relaxation", "-r", "--relaxation=true"):
            result = subprocess.run([sys.executable, "-S", str(SCRIPTS / "predict.py"), "predict", argument],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has been removed", result.stderr)
            self.assertIn("--relax restrained", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_import_has_no_heavy_runtime_side_effects(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import structure_preparation; "
            "assert not {'torch','openmm','pdbfixer','numpy'} & sys.modules.keys()"],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
