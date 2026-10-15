from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from uav_metric_research.fusion import global_local_fusion


class FusionTests(unittest.TestCase):
    def test_constant_context_offset_does_not_change_global_depth(self):
        baseline=np.full((50,60),20,np.float32)
        tiles=np.full((50,60),5,np.float32)
        np.testing.assert_array_equal(global_local_fusion(baseline,tiles,radius=7),baseline)

    def test_zero_alpha_and_identical_inputs(self):
        values=np.random.default_rng(42).uniform(15,25,(50,60)).astype(np.float32)
        np.testing.assert_array_equal(global_local_fusion(values,values+10,radius=7,alpha=0),values)
        np.testing.assert_array_equal(global_local_fusion(values,values,radius=7),values)

    def test_local_detail_is_added_without_gt(self):
        b=np.full((50,60),20,np.float32)
        q=np.full((50,60),5,np.float32)
        q[25,30]+=1
        value=global_local_fusion(b,q,radius=7)
        self.assertAlmostEqual(float(value[25,30]),20+1-1/225,places=5)
        self.assertLess(value[25,31],20)


if __name__=='__main__':unittest.main()
