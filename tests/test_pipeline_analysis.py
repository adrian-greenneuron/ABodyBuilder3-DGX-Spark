import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_analysis", ROOT/"tools/analyse_hydrogen_pipeline.py")
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


class PipelineAnalysisTests(unittest.TestCase):
    def test_only_openmm_date_header_is_ignored(self):
        a = b"REMARK   1 CREATED WITH OPENMM 8.6.1, 2026-10-26\nEND\n"
        b = a.replace(b"2026-10-26", b"2026-10-27")
        self.assertEqual(analysis.without_date(a), analysis.without_date(b))
        self.assertNotEqual(analysis.without_date(a), analysis.without_date(b.replace(b"8.6.1", b"8.6.2")))
        self.assertNotEqual(analysis.without_date(a), analysis.without_date(b+b"REMARK 2026-10-27\n"))

    def test_date_only_change_does_not_hide_coordinate_change(self):
        line = "ATOM      1  HA  ALA H   1       1.000   2.000   3.000  1.00  0.00           H  \n"
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a.pdb", Path(tmp)/"b.pdb"
            a.write_text("REMARK   1 CREATED WITH OPENMM 8.6.1, 2026-10-26\n"+line+"END\n")
            b.write_text(a.read_text().replace("2026-10-26", "2026-10-27"))
            same = analysis.compare(a, b)
            self.assertTrue(same["date_only_difference"] and same["coordinate_identical"])
            b.write_text(b.read_text().replace("1.000   2.000", "1.001   2.000"))
            changed = analysis.compare(a, b)
            self.assertFalse(changed["date_only_difference"] or changed["coordinate_identical"])
            self.assertAlmostEqual(changed["hydrogens"]["max_A"], .001)


if __name__ == "__main__": unittest.main()
