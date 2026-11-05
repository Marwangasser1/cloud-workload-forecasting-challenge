"""Leakage guards: nothing in the holdout window may influence training-window outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import (
    drift_forecast,
    ewma_forecast,
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
)
from cwfc.features import build_feature_frame, build_origin_feature_vector
from cwfc.gates import verify_predictions
from cwfc.models import (
    active_vm_count_forecast,
    damped_ets_forecast,
    direct_hgb_forecast,
    direct_ridge_forecast,
    global_vm_panel_forecast,
    hybrid_ensemble_forecast,
    segment_panel_forecast,
    vm_shrinkage_forecast,
)
from cwfc.targets import (
    build_aggregate_targets,
    split_raw_readings_train_holdout,
    split_train_holdout,
)

CPU = "aggregate_cpu_utilization"
COUNT = "active_vm_count"
N_HOLDOUT = 24
ATOL = 1e-12


# ------------------------------------------------------------------------------ helpers
@pytest.fixture(scope="module")
def aggregate_series(synthetic_readings_df) -> pd.DataFrame:
    """80-row aggregate target frame (56 train + 24 holdout) with a *varying* VM count.

    ``build_aggregate_targets`` of the synthetic panel has a constant VM count, which
    would short-circuit the count model to persistence; vary it so Ridge is exercised.
    """
    df = build_aggregate_targets(synthetic_readings_df).copy()
    assert len(df) == 80
    t = np.arange(len(df))
    df[COUNT] = (50 + np.round(3 * np.sin(2 * np.pi * t / 20)) + (t // 10)).astype("int64")
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "timestamp"]


def _perturb_tail(df: pd.DataFrame, n: int, how: str) -> pd.DataFrame:
    out = df.copy()
    cols = _numeric_cols(out)
    int_cols = [c for c in cols if pd.api.types.is_integer_dtype(out[c].dtype)]
    out[cols] = out[cols].astype(float)
    if how == "x10":
        out.loc[out.index[-n:], cols] = out.loc[out.index[-n:], cols] * 10.0
    elif how == "const":
        out.loc[out.index[-n:], cols] = 99.0
    else:  # pragma: no cover
        raise ValueError(how)
    for c in int_cols:
        out[c] = out[c].round().astype("int64")
    return out


# --------------------------------------------------------------- Test 1: feature causality
@pytest.mark.parametrize("how", ["x10", "const"])
@pytest.mark.parametrize("t0", [0, 5, 23, 40, 55, 70])
def test_feature_frame_is_causal(aggregate_series, t0, how):
    perturbed = aggregate_series.copy()
    cols = _numeric_cols(perturbed)
    perturbed[cols] = perturbed[cols].astype(float)
    future = perturbed.index > t0
    if how == "x10":
        perturbed.loc[future, cols] = perturbed.loc[future, cols] * 10.0
    else:
        perturbed.loc[future, cols] = 99.9
    if t0 < len(aggregate_series) - 1:
        assert not np.allclose(perturbed[CPU].to_numpy(), aggregate_series[CPU].to_numpy())

    base = build_feature_frame(aggregate_series)
    pert = build_feature_frame(perturbed)
    pd.testing.assert_frame_equal(
        base.iloc[: t0 + 1], pert.iloc[: t0 + 1], check_exact=True, check_dtype=False
    )

    # the origin vector at t0 only reads rows 0..t0, so it is identical for either history
    v_clean = build_origin_feature_vector(aggregate_series.iloc[: t0 + 1])
    v_pert = build_origin_feature_vector(perturbed.iloc[: t0 + 1])
    assert v_clean == v_pert
    # ... and equals the vectorised row t0
    np.testing.assert_allclose(
        np.array([v_clean[k] for k in base.columns]), base.iloc[t0].to_numpy(), rtol=0, atol=ATOL
    )


def test_vectorised_frame_equals_prefix_origin_vectors(aggregate_series):
    """Row t of the vectorised frame equals the features of the prefix 0..t (no look-ahead)."""
    frame = build_feature_frame(aggregate_series)
    for t in range(0, 80, 7):
        vec = build_origin_feature_vector(aggregate_series.iloc[: t + 1])
        assert set(vec) == set(frame.columns)
        np.testing.assert_allclose(
            np.array([vec[k] for k in frame.columns]), frame.iloc[t].to_numpy(), rtol=0, atol=1e-10
        )


def test_origin_features_do_not_read_future_target_column():
    """Changing y[t] for t > t0 must not change any lag/rolling/ewma feature at t0."""
    y = np.linspace(10, 30, 40)
    df = pd.DataFrame({CPU: y})
    changed = df.copy()
    changed.loc[20:, CPU] = 1e6
    a = build_feature_frame(df).iloc[19]
    b = build_feature_frame(changed).iloc[19]
    pd.testing.assert_series_equal(a, b, check_exact=True)


# ------------------------------------------- Test 2: aggregate holdout-perturbation test
def _seasonal24(train, template, target):
    return seasonal_naive_forecast(train, template, target, season_length=24)


def _seasonal12(train, template, target):
    return seasonal_naive_forecast(train, template, target, season_length=12)


def _ma(train, template, target):
    return moving_average_forecast(train, template, target, window=12)


def _drift(train, template, target):
    return drift_forecast(train, template, target)


def _drift_damped(train, template, target):
    return drift_forecast(train, template, target, lookback=24, damping=0.9, clip=(0.0, 100.0))


def _ewma(train, template, target):
    return ewma_forecast(train, template, target, span=6)


BASELINES = {
    "persistence": persistence_forecast,
    "seasonal_naive_24": _seasonal24,
    "seasonal_naive_12": _seasonal12,
    "moving_average": _ma,
    "drift": _drift,
    "drift_damped": _drift_damped,
    "ewma": _ewma,
}

MODELS = {
    "damped_ets": (damped_ets_forecast, CPU),
    "direct_ridge": (direct_ridge_forecast, CPU),
    "direct_hgb": (direct_hgb_forecast, CPU),
    "hybrid_ensemble": (hybrid_ensemble_forecast, CPU),
    "active_vm_count": (active_vm_count_forecast, COUNT),
    "direct_ridge_count": (direct_ridge_forecast, COUNT),
}


@pytest.mark.parametrize("how", ["x10", "const"])
def test_perturbation_actually_changes_holdout(aggregate_series, how):
    """Sanity guard so the perturbation tests below cannot pass vacuously."""
    perturbed = _perturb_tail(aggregate_series, N_HOLDOUT, how)
    _, hold = split_train_holdout(aggregate_series, N_HOLDOUT)
    _, hold_p = split_train_holdout(perturbed, N_HOLDOUT)
    assert not np.allclose(hold[CPU].to_numpy(), hold_p[CPU].to_numpy())
    assert not np.allclose(hold[COUNT].to_numpy(), hold_p[COUNT].to_numpy())


@pytest.mark.parametrize("name", list(BASELINES))
@pytest.mark.parametrize("target", [CPU, COUNT])
def test_baselines_ignore_holdout(aggregate_series, name, target):
    fn = BASELINES[name]
    perturbed = _perturb_tail(aggregate_series, N_HOLDOUT, "x10")
    train, hold = split_train_holdout(aggregate_series, N_HOLDOUT)
    train_p, hold_p = split_train_holdout(perturbed, N_HOLDOUT)
    pd.testing.assert_frame_equal(train, train_p, check_exact=True)

    fc = fn(train, hold, target)
    fc_p = fn(train_p, hold_p, target)  # template carries 10x targets: must not be read
    assert len(fc) == N_HOLDOUT
    np.testing.assert_allclose(fc.to_numpy(), fc_p.to_numpy(), rtol=0, atol=ATOL)
    np.testing.assert_array_equal(fc.to_numpy(), fc_p.to_numpy())


@pytest.mark.parametrize("how", ["x10", "const"])
@pytest.mark.parametrize("name", list(MODELS))
def test_models_ignore_holdout(aggregate_series, name, how):
    fn, target = MODELS[name]
    perturbed = _perturb_tail(aggregate_series, N_HOLDOUT, how)
    train, hold = split_train_holdout(aggregate_series, N_HOLDOUT)
    train_p, hold_p = split_train_holdout(perturbed, N_HOLDOUT)
    assert len(train) == 56
    pd.testing.assert_frame_equal(train, train_p, check_exact=True)

    fc = fn(train, hold, target)
    fc_p = fn(train_p, hold_p, target)
    assert len(fc) == N_HOLDOUT and np.isfinite(fc.to_numpy()).all()
    np.testing.assert_allclose(fc.to_numpy(), fc_p.to_numpy(), rtol=0, atol=ATOL)
    np.testing.assert_array_equal(fc.to_numpy(), fc_p.to_numpy())


@pytest.mark.parametrize("name", ["damped_ets", "direct_ridge", "direct_hgb", "hybrid_ensemble"])
def test_models_are_deterministic(aggregate_series, name):
    fn, target = MODELS[name]
    train, hold = split_train_holdout(aggregate_series, N_HOLDOUT)
    a = fn(train, hold, target).to_numpy()
    b = fn(train.copy(), hold.copy(), target).to_numpy()
    np.testing.assert_array_equal(a, b)


def test_models_do_not_read_template_target_values(aggregate_series):
    """Replacing the template's target column with garbage must not change forecasts."""
    train, hold = split_train_holdout(aggregate_series, N_HOLDOUT)
    garbage = hold.copy()
    garbage[[CPU, COUNT]] = -12345.0
    for fn, target in MODELS.values():
        np.testing.assert_array_equal(
            fn(train, hold, target).to_numpy(), fn(train, garbage, target).to_numpy()
        )


# ------------------------------------------- Test 3: raw VM readings holdout-perturbation
@pytest.fixture(scope="module")
def perturbed_readings(synthetic_readings_df) -> pd.DataFrame:
    df = synthetic_readings_df.copy()
    last24 = np.sort(df["timestamp"].unique())[-N_HOLDOUT:]
    mask = df["timestamp"].isin(last24)
    for c in ("min_cpu", "avg_cpu", "max_cpu"):
        df.loc[mask, c] = np.clip(df.loc[mask, c] * 10.0, 0.0, 100.0)
    return df


def test_raw_perturbation_is_effective_and_confined_to_holdout(synthetic_readings_df, perturbed_readings):
    tr, ho, ts = split_raw_readings_train_holdout(synthetic_readings_df, N_HOLDOUT)
    tr_p, ho_p, ts_p = split_raw_readings_train_holdout(perturbed_readings, N_HOLDOUT)
    pd.testing.assert_frame_equal(tr, tr_p, check_exact=True)
    np.testing.assert_array_equal(ts, ts_p)
    assert not np.allclose(ho["avg_cpu"].to_numpy(), ho_p["avg_cpu"].to_numpy())
    assert (ho_p[["min_cpu", "avg_cpu", "max_cpu"]].to_numpy() <= 100.0).all()


def test_vm_shrinkage_ignores_holdout_readings(synthetic_readings_df, perturbed_readings):
    tr, _, ts = split_raw_readings_train_holdout(synthetic_readings_df, N_HOLDOUT)
    tr_p, _, ts_p = split_raw_readings_train_holdout(perturbed_readings, N_HOLDOUT)
    template = pd.DataFrame({"timestamp": ts})
    fc = vm_shrinkage_forecast(tr, template)
    fc_p = vm_shrinkage_forecast(tr_p, pd.DataFrame({"timestamp": ts_p}))
    assert len(fc) == N_HOLDOUT
    np.testing.assert_allclose(fc.to_numpy(), fc_p.to_numpy(), rtol=0, atol=ATOL)


def test_segment_and_global_panel_ignore_holdout_readings(
    synthetic_readings_df, perturbed_readings, synthetic_metadata_df
):
    tr, _, ts = split_raw_readings_train_holdout(synthetic_readings_df, N_HOLDOUT)
    tr_p, _, _ = split_raw_readings_train_holdout(perturbed_readings, N_HOLDOUT)
    template = pd.DataFrame({"timestamp": ts})
    for meta in (None, synthetic_metadata_df):
        a = segment_panel_forecast(tr, template, meta)
        b = segment_panel_forecast(tr_p, template, meta)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=ATOL)
    a = global_vm_panel_forecast(tr, template, synthetic_metadata_df, method="blend")
    b = global_vm_panel_forecast(tr_p, template, synthetic_metadata_df, method="blend")
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=ATOL)


def test_aggregate_pipeline_from_raw_ignores_holdout_readings(synthetic_readings_df, perturbed_readings):
    """Build targets from the *full* raw panel (as the pipeline does), then split."""
    agg = build_aggregate_targets(synthetic_readings_df)
    agg_p = build_aggregate_targets(perturbed_readings)
    train, hold = split_train_holdout(agg, N_HOLDOUT)
    train_p, hold_p = split_train_holdout(agg_p, N_HOLDOUT)
    pd.testing.assert_frame_equal(train, train_p, check_exact=True)
    assert not np.allclose(hold[CPU].to_numpy(), hold_p[CPU].to_numpy())

    fc = hybrid_ensemble_forecast(train, hold, CPU)
    fc_p = hybrid_ensemble_forecast(train_p, hold_p, CPU)
    np.testing.assert_allclose(fc.to_numpy(), fc_p.to_numpy(), rtol=0, atol=ATOL)

    for fn, target in MODELS.values():
        np.testing.assert_array_equal(
            fn(train, hold, target).to_numpy(), fn(train_p, hold_p, target).to_numpy()
        )


# ------------------------------------------------- end-to-end CLI holdout perturbation
def test_cli_predictions_identical_when_holdout_readings_perturbed(
    tmp_path, run_predict, synthetic_readings_df, perturbed_readings
):
    """``python -m cwfc.predict`` must emit byte-identical output when the final 24
    timestamps' readings are mutated (skipped until ``cwfc.predict`` exists)."""
    outputs = []
    for label, readings in (("clean", synthetic_readings_df), ("perturbed", perturbed_readings)):
        data_dir = tmp_path / label
        data_dir.mkdir()
        readings.to_csv(data_dir / "vm_cpu_readings.csv.gz", index=False)
        out = tmp_path / f"preds_{label}.csv"
        proc = run_predict(data_dir, out)
        assert proc.returncode == 0, proc.stderr[-2000:]
        outputs.append(out)

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    preds = pd.read_csv(outputs[0])
    expected_ts = np.sort(synthetic_readings_df["timestamp"].unique())[-N_HOLDOUT:]
    assert verify_predictions(preds, expected_timestamps=expected_ts) == []
