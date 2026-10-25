"""Chemical checks in the Docker runtime using a saved public prediction.

Set ABB3_TEST_PDB to a heavy-atom PDB; tests need no prediction model or GPU.
"""
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from structure_preparation import Options, PreparationError, prepare_structure


@unittest.skipUnless(os.environ.get("ABB3_TEST_PDB"), "Set ABB3_TEST_PDB in the verified Docker runtime")
class ChemistryTests(unittest.TestCase):
    def setUp(self):
        from openmm import app
        self.app = app
        self.raw = Path(os.environ["ABB3_TEST_PDB"]).read_text()

    def test_openmm_preserves_prediction_and_disulfides(self):
        prepared, relaxed, report = prepare_structure(self.raw, Options(seed=42))
        self.assertIsNone(relaxed)
        self.assertEqual(report["preparation_changes"]["max_original_heavy_displacement_angstrom"], 0)
        pdb = self.app.PDBFile(io.StringIO(prepared))
        self.assertGreater(report["prepared_geometry"]["hydrogens"], 0)
        self.assertEqual(report["raw_geometry"]["residues"], report["prepared_geometry"]["residues"])
        self.assertEqual(len(report["raw_geometry"]["disulfides"]), len(report["prepared_geometry"]["disulfides"]))
        bonds = list(pdb.topology.bonds())
        for a,b in bonds:
            if a.name == b.name == "SG":
                self.assertFalse(any((x in (a,b) and y.element.symbol == "H") or
                                     (y in (a,b) and x.element.symbol == "H") for x,y in bonds))

    def test_ph_changes_protonation(self):
        # Use an input with internal histidines. Extreme-pH charged terminal
        # combinations are not all parameterised by the bundled Amber force field.
        path = Path(os.environ["ABB3_TEST_PDB"]).with_name("trastuzumab_like.pdb")
        raw = path.read_text()
        low = prepare_structure(raw, Options(ph=5, seed=42))[2]
        high = prepare_structure(raw, Options(ph=7, seed=42))[2]
        self.assertGreater(low["prepared_geometry"]["hydrogens"], high["prepared_geometry"]["hydrogens"])

    def test_missing_heavy_atom_is_repaired_without_moving_others(self):
        lines = self.raw.splitlines(keepends=True)
        removed = next(i for i,l in enumerate(lines) if l.startswith("ATOM") and l[12:16].strip() == "CB")
        raw = "".join(lines[:removed] + lines[removed+1:])
        report = prepare_structure(raw, Options(seed=42))[2]
        self.assertTrue(report["repair"]["missing_atoms"])
        self.assertEqual(report["preparation_changes"]["max_original_heavy_displacement_angstrom"], 0)

    def test_reduce_failure_never_falls_back(self):
        with patch("structure_preparation.subprocess.run", side_effect=FileNotFoundError("missing Reduce")):
            with self.assertRaises(PreparationError) as caught:
                prepare_structure(self.raw, Options(hydrogen_method="reduce"))
        self.assertEqual(caught.exception.report["status"], "failed")
        self.assertEqual(caught.exception.report["fallback"], "disabled")


if __name__ == "__main__":
    unittest.main()
