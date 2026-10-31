import hashlib
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from uav_metric_research import multiscale as m


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = m.ROOT / 'work/multiscale_tests'
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_resize_original_rule(self):
        self.assertEqual(m.resized_shape(3956, 5280), (518, 686))
        self.assertEqual(m.resized_shape(3956, 5280, 1036), (1036, 1386))
        for shape in [(50, 80), (123, 97), (3956, 5280)]:
            h, w = m.resized_shape(*shape)
            scale = 518 / min(shape)
            self.assertLessEqual(abs(h - shape[0] * scale), 7)
            self.assertLessEqual(abs(w - shape[1] * scale), 7)

    def test_preprocess_matches_frozen_dataset(self):
        import cv2
        import tifffile
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            path = Path(temp)
            rgb = np.random.default_rng(42).integers(0, 256, (80, 110, 3), dtype=np.uint8)
            cv2.imwrite(str(path / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            tifffile.imwrite(path / 'gt.tif', np.ones((80, 110), np.float32))
            row = dict(image_relpath='rgb.png', depth_relpath='gt.tif', scene='s', view='nadir', crop='corn', day='d')
            dataset = m.frozen('data').UAVDepthDataset([row], path, path, training=False)
            self.assertTrue(torch.equal(dataset[0]['pixel_values'], m.preprocess(rgb)))

    def test_restore_corners_and_linear_coordinates(self):
        array = np.array([[1, 3], [4, 6]], np.float32)
        restored = m.restore(array, (7, 9))
        expected = 1 + np.linspace(0, 3, 7)[:, None] + np.linspace(0, 2, 9)[None, :]
        np.testing.assert_allclose(restored, expected, atol=1e-6)

    def test_tiles_coordinate_restore_and_positive_weights(self):
        for shape, tile, overlap in [((101, 137), 50, .1), ((101, 137), 50, .25), ((12, 9), 50, .25)]:
            yy, xx = np.indices(shape)
            surface = (1 + yy * .01 + xx * .02).astype(np.float32)
            merge = m.TileMerger(shape)
            for box in m.tiles(shape, tile, overlap):
                y, x, bottom, right = box
                merge.add(box, surface[y:bottom, x:right])
            self.assertTrue((merge.weight > 0).all())
            np.testing.assert_allclose(merge.result(), surface, rtol=5e-7, atol=1e-6)
            self.assertEqual(m.starts(shape[1], tile, overlap)[-1], max(0, shape[1] - tile))

    def test_uncovered_rejected(self):
        with self.assertRaises(ValueError):
            m.TileMerger((20, 20)).result()

    def test_gt_mask_and_invalid_prediction_failure(self):
        gt = np.array([[0, -1, np.nan, np.inf, .001, 65, 66]], np.float32)
        np.testing.assert_array_equal(m.valid_gt(gt), [[False, False, False, False, True, True, False]])
        pred = np.ones_like(gt)
        m.validate_prediction(pred, gt)
        for value in (0, -1, np.nan, np.inf, 81):
            pred[0, 0] = value
            with self.assertRaises(ValueError):
                m.validate_prediction(pred, gt)

    def test_pair_coordinates_legacy_exact_no_wrap(self):
        shape, stride = (31, 47), 14
        mask = np.ones(shape, bool)
        mask[0, 0] = False
        rng = np.random.default_rng(42 + 3 * 10000 + stride)
        for axis, y, x, yy, xx in m.sample_pairs(shape, mask, 3, stride, 100):
            hh, ww = shape[0] - (stride if axis == 0 else 0), shape[1] - (stride if axis == 1 else 0)
            flat = rng.choice(hh * ww, size=100, replace=False)
            ey, ex = flat // ww, flat % ww
            ey1, ex1 = ey + (stride if axis == 0 else 0), ex + (stride if axis == 1 else 0)
            keep = mask[ey, ex] & mask[ey1, ex1]
            np.testing.assert_array_equal(np.stack([y, x, yy, xx]), np.stack([ey[keep], ex[keep], ey1[keep], ex1[keep]]))
            self.assertTrue((yy < shape[0]).all() and (xx < shape[1]).all())
            self.assertTrue((np.abs(yy-y) + np.abs(xx-x) == stride).all())

    def test_bins_no_double_count_and_sign(self):
        gt = np.array([[1, 1.05, 1.1, 1.2, 1.5, 1.8]], float)
        y = np.zeros(5, int)
        x = np.zeros(5, int)
        yy, xx = y, np.arange(1, 6)
        delta, groups = m.pair_groups(gt, y, x, yy, xx)
        bins = [v for k, v in groups.items() if k.startswith('gt_delta_')]
        np.testing.assert_array_equal(np.sum(bins, axis=0), np.ones(5))
        self.assertTrue((delta < 0).all())

    def test_edge_both_endpoints_definition(self):
        gt = np.ones((1200, 1400), np.float32)
        y, x, yy, xx = map(np.array, ([512, 127, 128], [512, 600, 600], [512, 128, 129], [726, 600, 600]))
        _, groups = m.pair_groups(gt, y, x, yy, xx)
        np.testing.assert_array_equal(groups['interior_ge512'], [True, False, False])
        np.testing.assert_array_equal(groups['border_lt128'], [False, True, False])

    def test_stats_zero_baseline_amplitude_and_merge(self):
        gt = np.array([.1, -.3, 0, .5])
        stats = m.DeltaStats()
        stats.update(gt, np.zeros(4))
        result = stats.result()
        self.assertAlmostEqual(result['delta_mae_m'], np.abs(gt).mean())
        self.assertEqual(result['amplitude_ratio'], 0)
        self.assertEqual(result['sign_match_rate'], 0)
        first, second = m.DeltaStats(), m.DeltaStats()
        first.update(gt[:2], gt[:2] / 3)
        second.update(gt[2:], gt[2:] / 3)
        first.merge(second)
        self.assertAlmostEqual(first.result()['amplitude_ratio'], 1/3)

    def test_baseline_gate_rejects_count_and_precision(self):
        expected = dict(rmse_m=.8, mae_m=.5, abs_rel=.03, pearson_r=.99, local_depth_delta_mae_m=.06, valid_pixels=100, local_pair_count=200)
        self.assertTrue(m.check_baseline(expected, expected)['passed'])
        self.assertFalse(m.check_baseline(dict(expected, local_pair_count=199), expected)['passed'])
        self.assertFalse(m.check_baseline(dict(expected, rmse_m=.81), expected)['passed'])

    def test_bootstrap_direction(self):
        result = m.paired_bootstrap([-1, -2, -3], repeats=100)
        self.assertEqual(result['improved'], 3)
        self.assertLess(result['ci95'][1], 0)

    def test_cache_reuse_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            path = Path(temp)/'prediction.npy'
            np.save(path, np.ones((3, 5), np.float32))
            marker = dict(weights_sha256='abc', original_shape=[3, 5], prediction_dtype='float32', prediction_sha256=m.sha256(path))
            m.write_json(path.with_suffix('.json'), marker)
            self.assertEqual(m.load_prediction_cache(path, dict(weights_sha256='abc')).shape, (3, 5))
            with self.assertRaises(ValueError):
                m.load_prediction_cache(path, dict(weights_sha256='different'))
            np.save(path, np.full((3, 5), 2, np.float32))
            with self.assertRaises(ValueError):
                m.load_prediction_cache(path, dict(weights_sha256='abc'))

    def test_depth_accumulator_merge_matches_full(self):
        metrics = m.frozen('metrics')
        first, second, merged = [metrics.MetricAccumulator() for _ in range(3)]
        gt = torch.ones((20, 30)) * 20
        first.update(gt+.5, gt)
        second.update(gt-.25, gt)
        m.merge_depth(merged, first)
        m.merge_depth(merged, second)
        self.assertEqual(merged.result()['valid_pixels'], 1200)
        self.assertAlmostEqual(merged.result()['mae_m'], .375)
        self.assertAlmostEqual(merged.result()['rmse_m'], np.sqrt((.5**2+.25**2)/2))

    def test_strict_serialization_adapter_bijection_and_rejection(self):
        target, stored = {}, {}
        for layer in range(12):
            for old, new in [('attention.query','q_proj'),('attention.key','k_proj'),('attention.value','v_proj'),('output.dense','o_proj')]:
                for suffix in ('weight','bias'):
                    prefix = f'backbone.encoder.layer.{layer}.attention.'
                    target[prefix+new+'.'+suffix] = torch.zeros(2)
                    stored[prefix+old+'.'+suffix] = torch.ones(2)*layer
        class Mock:
            def __init__(self): self.state = dict(target)
            def state_dict(self): return self.state
            def load_state_dict(self, values, strict):
                assert strict and set(values) == set(self.state)
                self.state = {k:v.clone() for k,v in values.items()}
        audit = m.strict_attention_state(Mock(), stored)
        self.assertEqual(audit['renamed_tensors'], 96)
        self.assertTrue(audit['loaded_values_exact'])
        self.assertEqual(m.strict_attention_state(Mock(), target)['renamed_tensors'], 0)
        bad = dict(stored)
        bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError): m.strict_attention_state(Mock(), bad)
        bad = dict(stored, unapproved=torch.ones(2))
        with self.assertRaises(ValueError): m.strict_attention_state(Mock(), bad)

    def test_saved_stats_reconstruction(self):
        value=m.DeltaStats()
        value.update(np.array([.1,.3,-.5]),np.array([.2,.1,-.2]))
        result=value.result()
        reconstructed=m.stats_from_result(result).result()
        for name in result:
            if result[name] is not None:self.assertAlmostEqual(result[name],reconstructed[name],places=14)

    def test_common_seam_reference(self):
        method=dict(tile_size=50,overlap=.2)
        # Internal crop starts at x=40 and old tile ends at x=50.
        groups=m.seam_groups((100,100),method,np.array([10,10]),np.array([39,10]),np.array([10,10]),np.array([41,12]))
        np.testing.assert_array_equal(groups['cross_tile_boundary'],[True,False])

    def test_image_cluster_bootstrap_keeps_zero_pair_images(self):
        baseline=[.4]*8+[None]
        candidate=[.3]*8+[None]
        result=m.paired_weighted_bootstrap(baseline,candidate,[100]*8+[0],repeats=100)
        self.assertEqual(result['resampled_images'],9)
        self.assertEqual(result['nonempty_images'],8)
        self.assertAlmostEqual(result['pooled_mean_difference'],-.1)
        self.assertAlmostEqual(result['pooled_ci95_low'],-.1)

    def test_foreign_manifest_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            path = Path(temp) / 'manifest.csv'
            path.write_text('split,scene,crop,day,view,image_relpath,depth_relpath\nindependent_test,s,c,d,v,i,g\n')
            with self.assertRaises(ValueError):
                m.validation_rows(path)


if __name__ == '__main__':
    unittest.main()
