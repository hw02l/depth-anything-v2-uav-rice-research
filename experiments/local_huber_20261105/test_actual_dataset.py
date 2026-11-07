import importlib.util
from pathlib import Path
import random
import unittest
import numpy as np
import torch
from data_local import UAVDepthDataset, read_manifest

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'experiments/metric_outdoor_full_20261014_s42_518_cap80_gt65/source_full_v1/data.py'
spec = importlib.util.spec_from_file_location('frozen_data_for_comparison', FROZEN)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

class ActualDatasetTests(unittest.TestCase):
    def test_actual_train_transform_identical(self):
        row=read_manifest(ROOT/'configs/uav3dcrop_verified/train.csv')[0]
        rgb=ROOT/'data/uav3dcrop_rgb'; depth=ROOT/'data/uav3dcrop_depth'
        if not (rgb/row['image_relpath']).is_file() or not (depth/row['depth_relpath']).is_file():
            self.skipTest('actual train RGB/GT not available locally')
        original=frozen.UAVDepthDataset([row],rgb,depth,image_size=518,training=True)
        derived=UAVDepthDataset([row],rgb,depth,image_size=518,training=True)
        for seed in (7, 42):
            random.seed(seed); before=original[0]
            random.seed(seed); after=derived[0]
            torch.testing.assert_close(before['pixel_values'],after['pixel_values'],rtol=0,atol=0)
            torch.testing.assert_close(before['depth'],after['depth'],rtol=0,atol=0,equal_nan=True)
            self.assertEqual(tuple(after['original_depth'].shape),after['geometry']['original_hw'])
            self.assertEqual(after['geometry']['crop_hw'],(518,518))

if __name__ == '__main__': unittest.main()

class ActualObjectiveTests(unittest.TestCase):
    def test_actual_gt_sparse_objective(self):
        from local_loss import Geometry
        from objective import objective
        row=read_manifest(ROOT/'configs/uav3dcrop_verified/train.csv')[0]
        rgb=ROOT/'data/uav3dcrop_rgb'; depth=ROOT/'data/uav3dcrop_depth'
        if not (rgb/row['image_relpath']).is_file() or not (depth/row['depth_relpath']).is_file():
            self.skipTest('actual train RGB/GT not available locally')
        random.seed(42)
        sample=UAVDepthDataset([row],rgb,depth,image_size=518,training=True)[0]
        geom=Geometry(**sample['geometry'])
        spec=importlib.util.spec_from_file_location('frozen_loss_actual',FROZEN.parent/'losses.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        pred=torch.full((1,518,518),20.,requires_grad=True)
        target=sample['depth'][None]
        valid=torch.isfinite(target)&(target>0)&(target<=65)
        total,s,l,counts=objective(pred,target,valid,mod.silog_loss,original_gt=sample['original_depth'],geometry=geom,image_id=row['image_relpath'],epoch=1,local_weight=.05,candidates_per_axis=8192)
        self.assertTrue(all(v>0 for v in counts.values()),str(counts))
        self.assertTrue(torch.isfinite(total)); self.assertTrue(torch.isfinite(l))
        grad=torch.autograd.grad(l,pred)[0]
        self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(grad.abs().sum().item(),0)
