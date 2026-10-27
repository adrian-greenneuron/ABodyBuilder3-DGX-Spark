import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import os
import numpy as np
import openmm as mm
from openmm import app,unit

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hydrogen_geometry',ROOT/'candidate/hydrogen_geometry.py')
hg=importlib.util.module_from_spec(spec);spec.loader.exec_module(hg)


class HydrogenGeometryTests(unittest.TestCase):
    def test_bound_roundoff_is_not_a_chemical_failure(self):
        result={'max_H_bond_error_A':0.001,'max_H_angle_error_deg':29.500000000000014,'wrong_side_tetrahedral_H':0}
        self.assertTrue(hg.geometry_passes(result,True))
        result['max_H_angle_error_deg']=29.50001
        self.assertFalse(hg.geometry_passes(result,True))
        result['max_H_angle_error_deg']=20
        self.assertFalse(hg.geometry_passes(result,False))

    def test_steric_backoff_is_explicit_bounded_and_restores_configuration(self):
        seen=[]
        def attempt(*args):
            value=float(os.environ['AB3_H_STERIC_K']);seen.append(value)
            hg.optimise.last_report={'quality':{'passed':value==200000}}
            if value==500000:raise ValueError('Hydrogen geometry did not meet the fixed-heavy preparation checks')
            return 'coordinates'
        with patch.dict(os.environ,{'AB3_EXPERIMENTAL_GEOMETRY':'robust_planar','AB3_H_STERIC_SCHEDULE':'500000,200000,100000','AB3_H_STERIC_K':'123'}):
            with patch.object(hg,'_optimise_once',side_effect=attempt):
                self.assertEqual(hg.optimise(None,None,None,None,None,None),'coordinates')
            self.assertEqual(os.environ['AB3_H_STERIC_K'],'123')
        self.assertEqual(seen,[500000,200000])
        self.assertEqual([a['accepted'] for a in hg.optimise.last_report['steric_attempts']],[False,True])

    def test_steric_backoff_does_not_hide_incompatible_heavy_atoms(self):
        with patch.dict(os.environ,{'AB3_EXPERIMENTAL_GEOMETRY':'robust_planar','AB3_H_STERIC_SCHEDULE':'500000,200000,100000'}):
            with patch.object(hg,'_optimise_once',side_effect=ValueError('Frozen heavy-atom geometry is incompatible')) as mocked:
                with self.assertRaisesRegex(ValueError,'heavy-atom geometry'):hg.optimise(None,None,None,None,None,None)
                self.assertEqual(mocked.call_count,1)

    def test_planar_sector_rejects_incompatible_heavy_angles(self):
        lo,hi=hg.planar_sector(np.deg2rad(120),np.deg2rad(120),np.deg2rad(120))
        self.assertLessEqual(lo,np.deg2rad(240));self.assertGreaterEqual(hi,np.deg2rad(240))
        with self.assertRaisesRegex(ValueError,'heavy-atom repair'):
            hg.planar_sector(np.deg2rad(56),np.deg2rad(120),np.deg2rad(120))

    def test_planar_coordinate_map_stays_in_plane_with_correct_jacobian(self):
        origins=np.array([[1.,2.,3.]]);lengths=np.array([.109]);frames={0:(np.array([1.,0,0]),np.array([0.,1.,0]))};q=np.array([.7,2.])
        xyz,dt,dp=hg.coordinate_map(q,origins,lengths,frames)
        self.assertAlmostEqual(xyz[0,2],3.);np.testing.assert_equal(dp,np.zeros((1,3)))
        for i in [0,1]:
            a=q.copy();b=q.copy();a[i]+=1e-6;b[i]-=1e-6
            numeric=(hg.coordinate_map(a,origins,lengths,frames)[0]-hg.coordinate_map(b,origins,lengths,frames)[0])/2e-6
            np.testing.assert_allclose(numeric,dt if i==0 else dp,atol=1e-9)

    def test_spherical_jacobian_and_fixed_lengths(self):
        rng=np.random.default_rng(3);q=rng.normal(size=12);origins=rng.normal(size=(6,3));lengths=np.full(6,.109)
        xyz,dt,dp=hg.spherical_map(q,origins,lengths)
        np.testing.assert_allclose(np.linalg.norm(xyz-origins,axis=1),lengths,atol=1e-14)
        for i in range(12):
            a=q.copy();b=q.copy();a[i]+=1e-6;b[i]-=1e-6
            numeric=(hg.spherical_map(a,origins,lengths)[0]-hg.spherical_map(b,origins,lengths)[0])/2e-6
            expected=np.zeros_like(numeric);expected[i//2]=(dt if i%2==0 else dp)[i//2]
            np.testing.assert_allclose(numeric,expected,atol=1e-9)

    def test_geometry_validation_catches_bond_angle_and_chirality(self):
        top=app.Topology();r=top.addResidue('ALA',top.addChain('A'))
        atoms=[top.addAtom(n,app.element.hydrogen if n=='HA' else app.element.carbon,r) for n in ['CA','N','C','CB','HA']]
        for a in atoms[1:]:top.addBond(atoms[0],a)
        xyz=np.array([[0,0,0],[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)*.109/np.sqrt(3)
        system=mm.System()
        for _ in atoms:system.addParticle(1)
        bond=mm.HarmonicBondForce();bond.addBond(0,4,.109,100);system.addForce(bond)
        angles=mm.HarmonicAngleForce()
        for i in [1,2,3]:angles.addAngle(i,0,4,np.arccos(-1/3),100)
        system.addForce(angles)
        self.assertTrue(hg.validate_geometry(system,top,xyz*unit.nanometer,[4])['passed'])
        bad=xyz.copy();bad[4]*=2
        self.assertFalse(hg.validate_geometry(system,top,bad*unit.nanometer,[4])['passed'])
        bad=xyz.copy();bad[4]=xyz[3]
        result=hg.validate_geometry(system,top,bad*unit.nanometer,[4])
        self.assertFalse(result['passed']);self.assertEqual(result['wrong_side_tetrahedral_H'],1)


if __name__=='__main__':unittest.main()
