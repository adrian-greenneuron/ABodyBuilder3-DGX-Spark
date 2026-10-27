import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bounds',ROOT/'tools/bound_fixed_heavy_contacts.py');bounds=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounds)


class ContactBoundsTests(unittest.TestCase):
    def test_unconstrained_sphere(self):
        self.assertAlmostEqual(bounds.minimum_linear_on_sphere([3,4,0],[],[]),-5)

    def test_single_plane_optimum(self):
        self.assertAlmostEqual(bounds.minimum_linear_on_sphere([-1,0,0],[[1,0,0]],[.5]),-.5)

    def test_two_active_planes(self):
        v=bounds.minimum_linear_on_sphere([-1,-1,0],[[1,0,0],[0,1,0]],[.2,.3])
        self.assertAlmostEqual(v,-.5)

    def test_bound_dominates_feasible_random_samples(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(50000,3));x/=np.linalg.norm(x,axis=1)[:,None]
        A=np.array([[1,0,0],[0,1,0],[0,0,-1]]);b=np.array([.4,.6,.8]);c=np.array([1,-2,.3])
        value=bounds.minimum_linear_on_sphere(c,A,b);feasible=x[np.all(x@A.T<=b,axis=1)]
        self.assertLessEqual(value,min(feasible@c)+1e-10)


if __name__=='__main__':unittest.main()
