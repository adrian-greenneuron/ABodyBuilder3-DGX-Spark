import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('metrics',Path(__file__).resolve().parents[1]/'tools/structural_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MetricsTests(unittest.TestCase):
    def test_rigid_motion_and_reflection(self):
        rng=np.random.default_rng(123);x=rng.normal(size=(20,3))
        rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        y=x @ rot+np.array([4.,7.,2.]);r,t=m.fit(x,y)
        self.assertLess(m.rmsd(x @ r+t,y),1e-12)
        y=x.copy();y[:,0]*=-1;r,t=m.fit(x,y)
        self.assertGreater(m.rmsd(x @ r+t,y),.1)
    def test_lddt_identical_and_mask(self):
        x=np.array([[0.,0.,0.],[3.,0.,0.],[6.,0.,0.],[20.,0.,0.]])
        result=m.ca_lddt(x,x,np.array([True,True,True,False]))
        self.assertEqual(result['score'],100.)
        self.assertEqual(result['directed_pair_count'],6)
        self.assertIsNone(result['per_residue'][-1])
        y=x.copy();y[1,0]+=1.5
        self.assertAlmostEqual(m.ca_lddt(y,x,np.array([True,True,True,False]))['score'],200/3)
    def test_region_fits_and_missing_atoms(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(12,37,3))
        mask=np.ones((12,37),bool);mask[3,1]=False
        heavy=np.array([1]*6+[0]*6,bool)
        regions=np.array(['fwh1']*3+['cdrh1']+['cdrh2']+['cdrh3']+['fwl1']*3+['cdrl1']+['cdrl2']+['cdrl3'])
        result=m.compare(x+5.,x,mask,mask,heavy,regions,False)
        self.assertEqual(result['matched_ca_count'],11)
        self.assertLess(result['aligned_ca_rmsd'],1e-12)
        self.assertLess(result['regions']['cdrh3']['framework_aligned_backbone_rmsd'],1e-12)
        changed=x.copy();changed[5]+=2
        result=m.compare(changed,x,mask,mask,heavy,regions,False)
        self.assertGreater(result['regions']['cdrh3']['framework_aligned_backbone_rmsd'],3.)


if __name__=='__main__':unittest.main()
