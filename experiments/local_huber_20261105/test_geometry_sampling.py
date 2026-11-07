import unittest
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from local_loss import Geometry
from sampling import sample_pairs

class GeometryAndSampling(unittest.TestCase):
    def test_opencv_centers_and_flip(self):
        # An affine image is exactly interpolated by bilinear interpolation
        # away from resize boundaries. Compare original samples with mapped grid.
        h,w=12,16
        y,x=np.mgrid[:h,:w]
        original=(x+2*y+5).astype(np.float32)
        resized=cv2.resize(original,(24,18),interpolation=cv2.INTER_LINEAR)
        crop=resized[2:14,3:19][:,::-1].copy()
        geometry=Geometry((h,w),(18,24),(2,3),(12,16),True)
        xy=torch.tensor([[4.,4.],[8.,6.]])
        mapped=geometry.map_xy(xy)
        grid=torch.stack((2*mapped[:,0]/15-1,2*mapped[:,1]/11-1),-1)[None,:,None]
        actual=F.grid_sample(torch.from_numpy(crop)[None,None],grid,align_corners=True)[0,0,:,0]
        torch.testing.assert_close(actual,torch.tensor([17.,25.]),atol=1e-5,rtol=0)
    def test_sampling_reproducible_bounded(self):
        a=sample_pairs('image',2,(100,120),14,100,42)
        self.assertTrue(torch.equal(a,sample_pairs('image',2,(100,120),14,100,42)))
        self.assertFalse(torch.equal(a,sample_pairs('image',3,(100,120),14,100,42)))
        self.assertTrue(((a[:,:,0]>=0)&(a[:,:,0]<120)&(a[:,:,1]>=0)&(a[:,:,1]<100)).all())
        delta=(a[:,1]-a[:,0]).abs().sum(dim=1)
        self.assertTrue((delta==14).all())

if __name__ == '__main__': unittest.main()
