import unittest
import torch
from local_loss import Geometry, local_loss

class TestLocalLoss(unittest.TestCase):
    def setUp(self):
        self.geom = Geometry((4, 4), (4, 4), (0, 0), (4, 4))
        self.gt = torch.arange(1., 17.).reshape(4,4)
        self.pairs = torch.tensor([[[0,0],[1,0]], [[1,1],[1,2]]])
    def loss(self, p, gt=None, pairs=None, geom=None):
        return local_loss(p[None,None], self.gt if gt is None else gt,
                          self.pairs if pairs is None else pairs, geom or self.geom)
    def test_exact_and_translation(self):
        self.assertAlmostEqual(self.loss(self.gt)[0].item(), 0)
        self.assertAlmostEqual(self.loss(self.gt+10)[0].item(), 0)
    def test_constant_and_sign(self):
        constant = torch.full((4,4), 3.)
        self.assertEqual(self.loss(constant, torch.ones(4,4))[0].item(), 0)
        self.assertGreater(self.loss(-self.gt)[0].item(), self.loss(constant)[0].item())
    def test_invalid_and_empty(self):
        gt = self.gt.clone(); gt[0,0] = float('nan'); gt[1,1] = 0
        p = self.gt.clone().requires_grad_()
        loss, n = self.loss(p, gt)
        self.assertEqual(n, 0); self.assertEqual(loss.item(), 0)
        loss.backward(); self.assertTrue(torch.isfinite(p.grad).all())
    def test_boundary_and_gradient(self):
        geom = Geometry((4,4),(4,4),(0,1),(3,3))
        p = torch.ones(3,3,requires_grad=True)
        loss, n = self.loss(p, pairs=torch.tensor([[[0,0],[1,0]],[[1,0],[2,0]]]),geom=geom)
        self.assertEqual(n,1); loss.backward(); self.assertGreater(p.grad.abs().sum().item(),0)
    def test_mapping_and_flip(self):
        g = Geometry((4,8),(8,16),(2,3),(4,6),True)
        actual = g.map_xy(torch.tensor([[2.,1.]]))
        torch.testing.assert_close(actual,torch.tensor([[3.5,.5]]))

if __name__ == '__main__': unittest.main()

class MultiScaleTests(unittest.TestCase):
    def test_nonempty_scales_equal_weight_and_empty_connected(self):
        from local_loss import multiscale_loss
        g=Geometry((4,4),(4,4),(0,0),(4,4))
        gt=torch.arange(1.,17.).reshape(4,4)
        pred=torch.ones(1,1,4,4,requires_grad=True)
        a=torch.tensor([[[0,0],[1,0]]]); b=torch.tensor([[[0,0],[0,1]]])
        empty=torch.empty((0,2,2),dtype=torch.long)
        result,counts=multiscale_loss(pred,gt,{14:a,28:b,56:empty},g)
        expected=(local_loss(pred,gt,a,g)[0]+local_loss(pred,gt,b,g)[0])/2
        torch.testing.assert_close(result,expected)
        self.assertEqual(counts,{14:1,28:1,56:0})
        result.backward(); self.assertTrue(torch.isfinite(pred.grad).all())
        z,_=multiscale_loss(pred,gt,{14:empty,28:empty,56:empty},g)
        self.assertEqual(z.item(),0)

class TrainMaskTests(unittest.TestCase):
    def test_both_endpoints_follow_original_train_threshold(self):
        geom=Geometry((4,4),(4,4),(0,0),(4,4))
        gt=torch.ones(4,4); gt[0,0]=0.0005; gt[1,0]=0.001
        pred=torch.ones(1,1,4,4,requires_grad=True)
        bad=torch.tensor([[[0,0],[1,0]]]); good=torch.tensor([[[0,1],[1,1]]])
        self.assertEqual(local_loss(pred,gt,bad,geom)[1],0)
        self.assertEqual(local_loss(pred,gt,good,geom)[1],1)
