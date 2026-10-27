import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hydrogen_policy", ROOT/"candidate/hydrogen_policy.py")
policy = importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


class HydrogenPolicyTests(unittest.TestCase):
    def test_cpu_retains_original_context_properties(self):
        self.assertEqual(policy.hydrogen_properties("cpu1"), ("CPU", None))

    def test_gpu_precisions_are_explicit(self):
        for name,value in [("gpu_single","single"),("gpu_mixed","mixed"),("gpu_double","double")]:
            self.assertEqual(policy.hydrogen_properties(name), ("CUDA", {"Precision":value,"DeterministicForces":"true"}))
        with self.assertRaises(ValueError):policy.hydrogen_properties("auto")

    def test_only_hydrogen_method_changes(self):
        trees=[ast.parse((ROOT/folder/"openmm_modeller.py").read_text()) for folder in ("reference","candidate")]
        for tree in trees:
            cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="Modeller")
            cls.body=[node for node in cls.body if not (isinstance(node,ast.FunctionDef) and node.name=="addHydrogens")]
        self.assertEqual(ast.dump(trees[0]),ast.dump(trees[1]))
        self.assertNotIn("setPropertyDefaultValue",(ROOT/"candidate/openmm_modeller.py").read_text())


if __name__=="__main__":unittest.main()
