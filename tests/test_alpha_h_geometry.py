import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("chemical_geometry",ROOT/"tools/measure_chemical_geometry.py")
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)


class AlphaHydrogenTests(unittest.TestCase):
    def test_ideal_tetrahedron_and_wrong_side(self):
        xyz={"CA":[0,0,0],"N":[1,1,1],"C":[1,-1,-1],"CB":[-1,1,-1],"HA":[-1,-1,1]}
        self.assertFalse(geometry.alpha_h_geometry(xyz)["wrong_side"])
        xyz["HA"]=[1,1,-1]
        self.assertTrue(geometry.alpha_h_geometry(xyz)["wrong_side"])

    def test_flat_heavy_geometry_is_flagged(self):
        xyz={"CA":[0,0,0],"N":[1,0,0],"C":[0,1,0],"CB":[1,1,0],"HA":[0,0,1]}
        self.assertFalse(geometry.alpha_h_geometry(xyz)["assessable"])


if __name__=="__main__":unittest.main()
