import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import assess_sidechains as m


class SidechainTests(unittest.TestCase):
    def test_dihedral_and_wrapping(self):
        p=np.array([[1.,0,0],[0.,0,0],[0.,1,0],[0.,1,1]])
        self.assertAlmostEqual(abs(m.dihedral(p)),90.)
        self.assertAlmostEqual(m.dihedral(p),m.dihedral(p+5.))
        self.assertEqual(m.angle_difference(179.,-179.),2.)

    def test_equivalent_sidechain_naming(self):
        original=np.array([[[1.,0,0],[-1.,0,0]]]);alternate=original[:,::-1].copy()
        mask=np.ones((1,2),bool)
        self.assertEqual(float(m.symmetry_error(alternate,original,alternate,mask)[0]),0.)
        self.assertGreater(float(m.symmetry_error(alternate,original,original,mask)[0]),0.)


if __name__=='__main__':unittest.main()
