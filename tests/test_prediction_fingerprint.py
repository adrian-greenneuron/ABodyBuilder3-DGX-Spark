"""Exact comparisons must detect changes hidden by PDB coordinate rounding."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prediction_fingerprint import array_hash,fingerprint


class FingerprintTests(unittest.TestCase):
    def test_sub_pdb_precision_changes_are_detected(self):
        a=np.zeros((2,37,3),dtype=np.float32);b=a.copy();b[0,1,0]=.00001
        self.assertNotEqual(array_hash(a),array_hash(b))
        self.assertEqual(array_hash(a),array_hash(a.copy()))

    def test_shape_dtype_mask_and_confidence_are_part_of_comparison(self):
        a=np.arange(6,dtype=np.float32)
        self.assertNotEqual(array_hash(a),array_hash(a.reshape(2,3)))
        self.assertNotEqual(array_hash(a),array_hash(a.astype(np.float64)))
        inp={'aatype':np.array([1,2]),'is_heavy':np.array([1,0])}
        out={'atom37':np.zeros((2,37,3)),'atom37_atom_exists':np.ones((2,37)),'plddt':np.zeros((2,50))}
        before=fingerprint(out,inp);out['plddt'][0,0]=1
        self.assertNotEqual(before,fingerprint(out,inp))
        out['plddt'][0,0]=0;out['atom37_atom_exists'][0,0]=0
        self.assertNotEqual(before,fingerprint(out,inp))


if __name__=='__main__':unittest.main()
