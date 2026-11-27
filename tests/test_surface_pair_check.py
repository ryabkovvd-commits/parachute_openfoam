"""Check that diagnosis separates missing faces, offsets, and ambiguous matches."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'coupled'))
from check_surface_pairs import diagnose


def surfaces(centres):
    c=np.asarray(centres,dtype=float)
    a=np.ones(len(c));n=np.tile([0.,0.,1.],(len(c),1));p=np.zeros(len(c))
    return c,a,n,p


class PairDiagnosisTests(unittest.TestCase):
    def opposite(self,master):
        return master[0].copy(),master[1].copy(),-master[2],master[3].copy()

    def test_permuted_valid_pair(self):
        m=surfaces([[0,0,0],[1,0,0],[2,0,0]])
        s=tuple(x[[2,0,1]] for x in self.opposite(m))
        r=diagnose(m,s)
        self.assertTrue(r['pair_geometry_pass'])
        self.assertTrue(r['paired_normals_opposite'])
        self.assertEqual(r['master_unpaired'],0)

    def test_missing_face(self):
        m=surfaces([[0,0,0],[1,0,0]])
        s=tuple(x[:1] for x in self.opposite(m))
        r=diagnose(m,s)
        self.assertFalse(r['same_face_count'])
        self.assertFalse(r['pair_geometry_pass'])
        self.assertEqual(r['master_unpaired'],1)

    def test_offset_beyond_tolerance(self):
        m=surfaces([[0,0,0],[1,0,0]])
        s=self.opposite(m);s[0][:,2]=.0002
        r=diagnose(m,s)
        self.assertEqual(r['master_too_far'],2)
        self.assertFalse(r['pair_geometry_pass'])

    def test_duplicate_centres_are_not_accepted_as_pairs(self):
        m=surfaces([[0,0,0],[0,0,0]])
        r=diagnose(m,self.opposite(m))
        self.assertGreater(r['non_mutual_master'],0)
        self.assertFalse(r['pair_geometry_pass'])

    def test_same_normals_are_reported_separately(self):
        m=surfaces([[0,0,0],[1,0,0]])
        r=diagnose(m,m)
        self.assertTrue(r['pair_geometry_pass'])
        self.assertFalse(r['paired_normals_opposite'])


if __name__=='__main__':unittest.main()
