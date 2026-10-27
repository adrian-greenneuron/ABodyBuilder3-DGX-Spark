import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("planes", Path(__file__).resolve().parents[1] / "tools/score_hydrogen_planes.py")
planes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planes)


class StoredHydrogenPlaneTests(unittest.TestCase):
    def test_signed_volume_detects_a_four_substituent_inversion(self):
        heavy=np.array([[1.,1,1],[1,-1,-1],[-1,1,-1]])
        self.assertFalse(planes.signed_volume_inversion(heavy,np.array([-1.,-1,1])))
        self.assertTrue(planes.signed_volume_inversion(heavy,np.array([2.,2,2])))

    def test_distinguishes_planar_from_out_of_plane(self):
        a, b = np.array([1., 0, 0]), np.array([0., 1, 0])
        self.assertAlmostEqual(planes.plane_deviation(np.array([-1., -1, 0]), a, b), 0)
        self.assertAlmostEqual(planes.plane_deviation(np.array([1., 0, 1]), a, b), 45)
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            planes.plane_deviation(a, a, a)

    def test_inward_hydrogen_can_be_planar_but_is_not_trigonal(self):
        rays = np.array([[1., 0, 0], [-.5, np.sqrt(3)/2, 0], [-.5, -np.sqrt(3)/2, 0]])
        self.assertAlmostEqual(planes.trigonal_gap(rays), 0)
        inward = rays.copy()
        inward[2] *= -1
        self.assertAlmostEqual(planes.plane_deviation(inward[2], *inward[:2]), 0)
        self.assertGreater(planes.trigonal_gap(inward), 1)


if __name__ == "__main__":
    unittest.main()
