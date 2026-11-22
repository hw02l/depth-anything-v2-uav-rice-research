from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "make_fixed8_comparison.py"
SPEC = importlib.util.spec_from_file_location("fixed8_visualization", SCRIPT)
fixed8 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fixed8)


def test_prediction_cache_requires_fixed_schema_and_shape(tmp_path):
    path = tmp_path / "prediction.npz"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez_compressed(path, prediction_m=expected)
    assert np.array_equal(fixed8.load_prediction(path, (3, 4)), expected)
    with pytest.raises(ValueError, match="shape or values"):
        fixed8.load_prediction(path, (4, 3))


def test_prediction_cache_rejects_nonfinite_and_extra_arrays(tmp_path):
    nonfinite = tmp_path / "nonfinite.npz"
    np.savez_compressed(nonfinite, prediction_m=np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="shape or values"):
        fixed8.load_prediction(nonfinite, (1, 1))
    extra = tmp_path / "extra.npz"
    np.savez_compressed(extra, prediction_m=np.ones((1, 1), dtype=np.float32), rgb=np.zeros((1, 1, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="schema"):
        fixed8.load_prediction(extra, (1, 1))


def test_depth_and_error_figures_use_fixed_shared_ranges(tmp_path):
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    gt = np.full((32, 48), 20.0, dtype=np.float32)
    valid = np.ones((32, 48), dtype=bool)
    valid[:, :4] = False
    predictions = {key: gt.copy() for key in fixed8.DEPTH_CONDITIONS}
    predictions["C0-P0"][:, 8:12] += 12.0
    depth_path = tmp_path / "depth.png"
    error_path = tmp_path / "error.png"
    fixed8.render_depth_figure(rgb, gt, predictions, valid, 7, "corn", "nadir", depth_path)
    over_range = fixed8.render_error_figure(gt, predictions, valid, 7, "corn", "nadir", error_path)
    assert depth_path.is_file() and depth_path.stat().st_size > 1000
    assert error_path.is_file() and error_path.stat().st_size > 1000
    assert fixed8.DEPTH_MAX_M == 65.0
    assert fixed8.ERROR_MAX_M == 10.0
    assert over_range["C0-P0"] == 32 * 4
    assert over_range["C0-B3"] == 0
