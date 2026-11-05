import math

import numpy as np
import pytest

from cwfc.metrics import (
    compute_all_metrics,
    interval_coverage,
    mae,
    mase,
    normalized_mae,
    peak_mae,
    pinball_loss,
    rmse,
    smape,
)


def test_metrics_basic():
    y = [1, 2, 3]
    p = [1, 3, 1]
    assert mae(y, p) == 1.0
    assert round(rmse(y, p), 8) == round((5 / 3) ** 0.5, 8)
    assert smape(y, p) > 0
    assert normalized_mae(y, p, [1, 1, 1]) > 0


# ----------------------------------------------------------------------------- mae / rmse
def test_mae_hand_values():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert mae([0, 0], [0, 0]) == 0.0
    assert mae(np.array([1.5]), np.array([-0.5])) == pytest.approx(2.0)


def test_rmse_hand_values_and_ordering():
    assert rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(math.sqrt(5 / 3))
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    y, p = [1, 2, 3, 4], [2, 1, 5, 4]
    assert rmse(y, p) >= mae(y, p)


@pytest.mark.parametrize("fn", [mae, rmse, smape])
def test_shape_mismatch_raises(fn):
    with pytest.raises(ValueError):
        fn([1, 2, 3], [1, 2])


# ----------------------------------------------------------------------------------- smape
def test_smape_hand_values():
    assert smape([100], [50]) == pytest.approx(200.0 * 50 / 150)
    assert smape([10, 20], [10, 20]) == 0.0
    assert smape([0], [0]) == 0.0  # eps guard: no division by zero
    assert smape([0], [1]) == pytest.approx(200.0)  # maximum possible value
    assert smape([1, 2], [2, 4]) == pytest.approx(np.mean([2 / 3, 2 / 3]) * 100)


def test_smape_is_symmetric_and_bounded():
    a, b = np.array([5.0, 10.0, 0.0]), np.array([7.0, 2.0, 3.0])
    assert smape(a, b) == pytest.approx(smape(b, a))
    assert 0.0 <= smape(a, b) <= 200.0


# --------------------------------------------------------------------------- normalized_mae
def test_normalized_mae():
    assert normalized_mae([1, 2, 3], [1, 2, 3], [0, 0, 0]) == 0.0
    assert normalized_mae([1, 2, 3], [2, 2, 5], [3, 3, 3]) == pytest.approx(1.0 / 1.0)
    assert normalized_mae([10, 10], [9, 9], [5, 5]) == pytest.approx(1.0 / 5.0)
    assert normalized_mae([10, 10], [20, 20], [5, 5]) == pytest.approx(2.0)  # worse than baseline


def test_normalized_mae_zero_baseline_error():
    assert normalized_mae([1, 2], [1, 2], [1, 2]) == 0.0
    assert normalized_mae([1, 2], [2, 3], [1, 2]) == float("inf")


# ------------------------------------------------------------------------------------ mase
def test_mase_hand_values():
    # scale = mean(|diff([1, 3, 2, 4])|) = (2 + 1 + 2) / 3
    assert mase([10, 10], [11, 9], [1, 3, 2, 4]) == pytest.approx(1.0 / (5 / 3))
    assert mase([1, 2], [1, 2], [1, 3, 2, 4]) == 0.0


def test_mase_degenerate_training_series():
    assert math.isnan(mase([1], [1], [5.0]))
    assert mase([1, 2], [1, 2], [3, 3, 3]) == 0.0
    assert mase([1, 2], [2, 2], [3, 3, 3]) == float("inf")


# --------------------------------------------------------------------------------- pinball
def test_pinball_loss_hand_values():
    assert pinball_loss([10.0], [8.0], 0.9) == pytest.approx(0.9 * 2.0)  # under-prediction
    assert pinball_loss([6.0], [8.0], 0.9) == pytest.approx(0.1 * 2.0)  # over-prediction
    assert pinball_loss([10.0, 6.0], [8.0, 8.0], 0.9) == pytest.approx((1.8 + 0.2) / 2)
    assert pinball_loss([1.0, 2.0], [1.0, 2.0], 0.5) == 0.0


def test_pinball_median_is_half_mae_and_asymmetry():
    y, p = np.array([1.0, 5.0, 9.0]), np.array([2.0, 4.0, 12.0])
    assert pinball_loss(y, p, 0.5) == pytest.approx(0.5 * mae(y, p))
    # high quantile penalises under-prediction more than over-prediction
    assert pinball_loss([10.0], [5.0], 0.95) > pinball_loss([10.0], [15.0], 0.95)


def test_pinball_shape_mismatch():
    with pytest.raises(ValueError):
        pinball_loss([1, 2], [1], 0.5)


# ------------------------------------------------------------------------ interval_coverage
def test_interval_coverage():
    y = [1, 2, 3, 4]
    assert interval_coverage(y, [0, 0, 0, 0], [10, 10, 10, 10]) == 1.0
    assert interval_coverage(y, [2, 2, 2, 2], [3, 3, 3, 3]) == pytest.approx(0.5)
    assert interval_coverage(y, [5, 5, 5, 5], [6, 6, 6, 6]) == 0.0
    # bounds are inclusive
    assert interval_coverage([1, 3], [1, 3], [1, 3]) == 1.0


# ---------------------------------------------------------------------------------- peak_mae
def test_peak_mae():
    y, p = [10, 50, 60, 20], [12, 40, 70, 20]
    assert peak_mae(y, p, 50) == pytest.approx((10 + 10) / 2)  # >= threshold is inclusive
    assert peak_mae(y, p, 0) == pytest.approx(mae(y, p))
    assert math.isnan(peak_mae(y, p, 1000))


def test_peak_mae_shape_mismatch():
    with pytest.raises(ValueError):
        peak_mae([1, 2], [1], 0.0)


# ---------------------------------------------------------------------- compute_all_metrics
def test_compute_all_metrics_minimal_keys():
    y, p, b = [1, 2, 3], [1, 3, 1], [1, 1, 1]
    out = compute_all_metrics(y, p, b)
    assert set(out) == {"mae", "rmse", "smape", "normalized_mae", "baseline_mae"}
    assert out["mae"] == pytest.approx(mae(y, p))
    assert out["rmse"] == pytest.approx(rmse(y, p))
    assert out["smape"] == pytest.approx(smape(y, p))
    assert out["baseline_mae"] == pytest.approx(mae(y, b))
    assert out["normalized_mae"] == pytest.approx(mae(y, p) / mae(y, b))


def test_compute_all_metrics_optional_keys():
    y, p, b = [10, 50, 60, 20], [12, 40, 70, 20], [30, 30, 30, 30]
    out = compute_all_metrics(y, p, b, y_train=[1, 3, 2, 4], peak_threshold=50)
    assert out["mase"] == pytest.approx(mase(y, p, [1, 3, 2, 4]))
    assert out["peak_mae"] == pytest.approx(10.0)
    assert set(out) == {"mae", "rmse", "smape", "normalized_mae", "baseline_mae", "mase", "peak_mae"}
    assert all(isinstance(v, float) for v in out.values())
