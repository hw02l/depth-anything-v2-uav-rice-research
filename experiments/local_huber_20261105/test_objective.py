import importlib.util
from pathlib import Path
import unittest
import torch
from local_loss import Geometry
from objective import objective

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('frozen_loss',ROOT/'experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/source_full_v1/losses.py')
frozen=importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)

class ObjectiveTests(unittest.TestCase):
    def test_e1_exact_and_e2_grad(self):
        gt=torch.linspace(1,5,80*80).reshape(80,80)
        pred=(gt[None]*.8).clone().requires_grad_()
        target=gt[None]; valid=torch.ones_like(target,dtype=torch.bool)
        args=dict(original_gt=gt,geometry=Geometry((80,80),(80,80),(0,0),(80,80)),image_id='sample',epoch=1,candidates_per_axis=100)
        total,s,local,counts=objective(pred,target,valid,frozen.silog_loss,local_weight=0.,**args)
        self.assertTrue(torch.equal(total,frozen.silog_loss(pred,target,valid)))
        self.assertEqual(counts,{14:0,28:0,56:0})
        self.assertEqual(float(local.detach()),0.0)
        total2,_,_,counts2=objective(pred,target,valid,frozen.silog_loss,local_weight=.05,**args)
        self.assertTrue(all(v==0 for v in counts.values()))
        self.assertTrue(all(v>0 for v in counts2.values()))
        total2.backward(); self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(),0)

if __name__=='__main__': unittest.main()
