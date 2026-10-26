"""Policy isolation and the narrowly scoped third-party API extension."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repair_policy", ROOT / "candidate/repair_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class RepairPolicyTests(unittest.TestCase):
    def test_reference_has_no_properties(self):
        self.assertEqual(policy.repair_properties("reference"), (None, None))

    def test_cuda_keeps_single_precision(self):
        self.assertEqual(policy.repair_properties("cuda_deterministic"),
                         ("CUDA", {"Precision": "single", "DeterministicForces": "true"}))

    def test_unknown_policy_fails(self):
        with self.assertRaises(ValueError):
            policy.repair_properties("automatic-fallback")

    def test_cpu_is_explicit_single_thread(self):
        self.assertEqual(policy.repair_properties("cpu1"),
                         ("CPU", {"Threads": "1", "DeterministicForces": "true"}))

    def test_double_is_a_separate_precision_setting(self):
        self.assertEqual(policy.repair_properties("cuda_double"),
                         ("CUDA", {"Precision": "double", "DeterministicForces": "true"}))

    def test_only_missing_atom_method_changes(self):
        trees = [ast.parse((ROOT / folder / "pdbfixer.py").read_text()) for folder in ("reference", "candidate")]
        methods = [next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PDBFixer") for tree in trees]
        for cls in methods:
            cls.body = [node for node in cls.body if not (isinstance(node, ast.FunctionDef) and node.name == "addMissingAtoms")]
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))
        self.assertNotIn("setPropertyDefaultValue", (ROOT / "candidate/pdbfixer.py").read_text())


if __name__ == "__main__":
    unittest.main()
