"""Smoke tests: intervals, policy, data-quality report, I/O and the ``cwfc.predict`` CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cwfc.gates import verify_predictions
from cwfc.intervals import compute_empirical_intervals, interval_coverage, residuals_by_horizon
from cwfc.io import load_cpu_readings, load_vm_metadata
from cwfc.policy import (
    RECOMMENDED_POLICY,
    capacity_policy_trajectories,
    compute_fleet_core_capacity,
    evaluate_capacity_policies,
    vm_core_weights,
)
from cwfc.quality import compute_data_quality_report, save_data_quality_report


# ----------------------------------------------------------------------------- io
def test_load_with_and_without_metadata(synthetic_data_dir, synthetic_data_dir_no_metadata, synthetic_readings_df):
    cpu = load_cpu_readings(synthetic_data_dir)
    meta = load_vm_metadata(synthetic_data_dir)
    assert len(cpu) == len(synthetic_readings_df)
    assert cpu["timestamp"].dtype == np.int64
    assert meta is not None and set(meta["virtual_core_bucket"]) == {"2", "4", "8", "24", ">24"}

    cpu2 = load_cpu_readings(synthetic_data_dir_no_metadata)
    assert len(cpu2) == len(cpu)
    assert load_vm_metadata(synthetic_data_dir_no_metadata) is None  # optional file


def test_load_cpu_readings_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cpu_readings(tmp_path)


def test_load_cpu_readings_missing_columns_raises(tmp_path):
    pd.DataFrame({"timestamp": [0], "vm_id": ["a"]}).to_csv(tmp_path / "vm_cpu_readings.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_cpu_readings(tmp_path)


# ------------------------------------------------------------------------- intervals
def test_residuals_by_horizon_hand_values():
    y = [[10.0, 20.0, 30.0], [12.0, 18.0, 33.0]]
    p = [[9.0, 21.0, 30.0], [10.0, 20.0, 30.0]]
    res = residuals_by_horizon(y, p)
    assert sorted(res) == [1, 2, 3]
    np.testing.assert_allclose(res[1], [1.0, 2.0])
    np.testing.assert_allclose(res[2], [-1.0, -2.0])
    np.testing.assert_allclose(res[3], [0.0, 3.0])


def test_residuals_by_horizon_ragged_nan_and_validation():
    res = residuals_by_horizon([[1.0, 2.0, 3.0], [1.0, 2.0]], [[0.0, 0.0, 0.0], [1.0, np.nan]])
    assert [len(res[h]) for h in (1, 2, 3)] == [2, 1, 1]  # NaN residual dropped
    with pytest.raises(ValueError):
        residuals_by_horizon([[1.0]], [[1.0], [2.0]])
    with pytest.raises(ValueError):
        residuals_by_horizon([[1.0, 2.0]], [[1.0]])


def _residual_book(n_steps=24, n_folds=9):
    rng = np.random.default_rng(0)
    return {h: rng.normal(0.0, 0.5 + 0.1 * h, n_folds) for h in range(1, n_steps + 1)}


def test_compute_empirical_intervals_hand_values():
    e = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    point = np.array([50.0, 50.0])
    out = compute_empirical_intervals(point, {1: e, 2: e * 2})
    assert list(out.columns) == [
        "horizon", "point_forecast", "p80_upper", "p90_upper", "p95_upper",
        "pi80_lower", "pi80_upper", "pi90_lower", "pi90_upper", "n_residuals",
    ]
    assert out["horizon"].tolist() == [1, 2]
    assert out["n_residuals"].tolist() == [5, 5]
    assert out["p90_upper"].iloc[0] == pytest.approx(50.0 + np.quantile(e, 0.90))
    assert out["p95_upper"].iloc[1] == pytest.approx(50.0 + np.quantile(2 * e, 0.95))
    assert out["pi80_lower"].iloc[0] == pytest.approx(50.0 + np.quantile(e, 0.10))
    assert out["pi80_upper"].iloc[0] == pytest.approx(50.0 + np.quantile(e, 0.90))
    assert out["pi90_lower"].iloc[0] == pytest.approx(50.0 + np.quantile(e, 0.05))


def test_compute_empirical_intervals_ordering_and_clipping():
    point = pd.Series(np.linspace(97.0, 99.5, 24), index=pd.RangeIndex(100, 124))
    out = compute_empirical_intervals(point, _residual_book(), monotone=True)
    assert out.index.equals(point.index)
    assert len(out) == 24
    bound_cols = [c for c in out.columns if c.startswith(("p", "pi")) and c != "point_forecast"]
    assert (out[bound_cols].to_numpy() <= 100.0).all() and (out[bound_cols].to_numpy() >= 0.0).all()
    assert (out["p80_upper"] <= out["p90_upper"] + 1e-12).all()
    assert (out["p90_upper"] <= out["p95_upper"] + 1e-12).all()
    assert (out["pi80_lower"] <= out["pi80_upper"] + 1e-12).all()
    assert (out["pi90_lower"] <= out["pi80_lower"] + 1e-12).all()
    assert (out["pi90_upper"] >= out["pi80_upper"] - 1e-12).all()
    # monotone half-widths: upper bounds minus point never shrink (before clipping saturation)
    unclipped = compute_empirical_intervals(np.full(24, 50.0), _residual_book(), monotone=True, clip=None)
    assert (np.diff(unclipped["p90_upper"].to_numpy()) >= -1e-12).all()


def test_compute_empirical_intervals_options_and_errors():
    book = _residual_book(n_steps=6)
    point = np.full(6, 40.0)
    plain = compute_empirical_intervals(point, book)
    pooled = compute_empirical_intervals(point, book, pool_neighbors=1)
    conformal = compute_empirical_intervals(point, book, finite_sample=True)
    assert (pooled["n_residuals"].iloc[1:-1] == 27).all()
    assert (conformal["p95_upper"] >= plain["p95_upper"] - 1e-9).all()
    # horizons beyond the backtest borrow the nearest covered horizon
    long = compute_empirical_intervals(np.full(8, 40.0), book)
    assert long["p90_upper"].iloc[7] == pytest.approx(long["p90_upper"].iloc[5])
    with pytest.raises(ValueError):
        compute_empirical_intervals(point, book, quantiles=(1.2,))
    with pytest.raises(ValueError):
        compute_empirical_intervals(point, {})
    with pytest.raises(ValueError):
        compute_empirical_intervals(point, book, pool_neighbors=-1)


def test_interval_coverage_table():
    e = np.array([-1.0, 0.0, 1.0])
    out = compute_empirical_intervals(np.full(4, 50.0), {h: e for h in range(1, 5)})
    cov = interval_coverage([50.0, 50.5, 49.5, 52.0], out)
    assert set(cov["bound"]) == {"p80_upper", "p90_upper", "p95_upper", "pi80", "pi90"}
    row = cov.set_index("bound").loc["p90_upper"]
    assert row["nominal"] == pytest.approx(0.9)
    assert row["empirical_coverage"] == pytest.approx(0.75)  # 52.0 exceeds the bound
    with pytest.raises(ValueError):
        interval_coverage([1.0], out)


# --------------------------------------------------------------------------- policy
def test_vm_core_weights():
    meta = pd.DataFrame({"vm_id": ["a", "b", "c", "d"], "virtual_core_bucket": ["2", 8, ">24", "weird"]})
    w = vm_core_weights(["a", "b", "c", "d", "e"], meta)
    assert w.to_dict() == {"a": 2.0, "b": 8.0, "c": 32.0, "d": 2.0, "e": 2.0}
    assert (vm_core_weights(["a", "b"], None) == 2.0).all()


def test_compute_fleet_core_capacity_hand_values():
    cpu = pd.DataFrame(
        {
            "timestamp": [0, 0, 300, 300, 300],
            "vm_id": ["a", "b", "a", "b", "b"],  # last row duplicates (300, b)
            "min_cpu": [0.0] * 5,
            "max_cpu": [80.0, 20.0, 50.0, 40.0, 40.0],
            "avg_cpu": [50.0, 10.0, 25.0, 20.0, 20.0],
        }
    )
    meta = pd.DataFrame({"vm_id": ["a", "b"], "virtual_core_bucket": ["2", "8"]})
    out = compute_fleet_core_capacity(cpu, meta).set_index("timestamp")
    assert out["active_vm_count"].tolist() == [2, 2]
    assert out["total_provisioned_cores"].tolist() == [10.0, 10.0]
    assert out.loc[0, "demanded_cores_avg"] == pytest.approx(2 * 0.5 + 8 * 0.1)
    assert out.loc[0, "core_weighted_cpu_utilization"] == pytest.approx(18.0)
    assert out.loc[0, "demanded_cores_peak"] == pytest.approx(2 * 0.8 + 8 * 0.2)
    assert out.loc[0, "peak_to_avg_ratio"] == pytest.approx(3.2 / 1.8)
    assert out.loc[300, "demanded_cores_avg"] == pytest.approx(0.5 + 1.6)


def test_compute_fleet_core_capacity_synthetic_and_no_metadata(synthetic_readings_df, synthetic_metadata_df):
    with_meta = compute_fleet_core_capacity(synthetic_readings_df, synthetic_metadata_df)
    without = compute_fleet_core_capacity(synthetic_readings_df)
    assert len(with_meta) == len(without) == 80
    assert (without["total_provisioned_cores"] == 100.0).all()  # 50 VMs x default 2 cores
    # 10 VMs per bucket: 10 * (2 + 4 + 8 + 24 + 32)
    assert (with_meta["total_provisioned_cores"] == 700.0).all()
    assert with_meta["core_weighted_cpu_utilization"].between(0, 100).all()
    assert (with_meta["peak_to_avg_ratio"] >= 1.0).all()
    with pytest.raises(KeyError):
        compute_fleet_core_capacity(synthetic_readings_df.drop(columns="avg_cpu"))
    no_max = compute_fleet_core_capacity(synthetic_readings_df.drop(columns="max_cpu"))
    assert no_max["demanded_cores_peak"].isna().all()


@pytest.fixture
def intervals_df():
    return pd.DataFrame(
        {
            "horizon": np.arange(1, 5),
            "point_forecast": np.full(4, 40.0),
            "p80_upper": np.full(4, 50.0),
            "p90_upper": np.full(4, 60.0),
            "p95_upper": np.full(4, 70.0),
        }
    )


def test_capacity_policy_trajectories(intervals_df):
    y = np.full(4, 55.0)
    traj = capacity_policy_trajectories(y, np.full(4, 40.0), intervals_df, total_cores=1000.0)
    assert traj["policy"].nunique() == 6 and len(traj) == 24
    assert set(traj.columns) >= {
        "policy", "horizon", "demand_pct", "provisioned_pct",
        "shortfall_pct", "overprovision_pct", "provisioned_cores",
    }
    p90 = traj[traj["policy"] == RECOMMENDED_POLICY]
    assert p90["horizon"].tolist() == [1, 2, 3, 4]
    assert (p90["provisioned_cores"] == 600.0).all()
    assert (p90["shortfall_pct"] == 0.0).all() and (p90["overprovision_pct"] == 5.0).all()
    assert (traj["provisioned_pct"].between(0, 100)).all()
    with pytest.raises(ValueError):
        capacity_policy_trajectories(y[:3], np.full(4, 40.0), intervals_df)


def test_evaluate_capacity_policies_hand_values(intervals_df):
    out = evaluate_capacity_policies(np.full(4, 55.0), np.full(4, 40.0), intervals_df, total_cores=1000.0)
    by = out.set_index("policy")
    assert len(out) == 6 and out["is_recommended"].sum() == 1
    assert out.loc[out["is_recommended"], "policy"].iloc[0] == RECOMMENDED_POLICY

    point = by.loc["Point Forecast (No Headroom)"]
    assert point["sla_violation_rate_pct"] == 100.0 and point["sla_violations"] == 4
    assert point["mean_shortfall_pct"] == pytest.approx(15.0)
    assert point["asymmetric_cost"] == pytest.approx(5.0 * 15.0)
    assert point["mean_shortfall_cores"] == pytest.approx(150.0)

    p90 = by.loc[RECOMMENDED_POLICY]
    assert p90["sla_violation_rate_pct"] == 0.0 and p90["mean_shortfall_pct"] == 0.0
    assert p90["mean_overprovision_pct"] == pytest.approx(5.0)
    assert p90["asymmetric_cost"] == pytest.approx(5.0)
    assert p90["mean_provisioned_cores"] == pytest.approx(600.0)

    # +10% relative margin: 44 vs demand 55 -> still short by 11
    assert by.loc["Static Safety Margin (+10% relative)", "mean_shortfall_pct"] == pytest.approx(11.0)
    # a burst add-on inflates demand: 55 * 1.1 = 60.5 > p90 (60)
    burst = evaluate_capacity_policies(np.full(4, 55.0), np.full(4, 40.0), intervals_df, peak_burst_addon_pct=10.0)
    assert burst.set_index("policy").loc[RECOMMENDED_POLICY, "sla_violation_rate_pct"] == 100.0


def test_evaluate_capacity_policies_missing_columns_and_validation(intervals_df):
    partial = intervals_df.drop(columns=["p80_upper", "p95_upper"])
    out = evaluate_capacity_policies(np.full(4, 30.0), np.full(4, 40.0), partial)
    assert len(out) == 4  # point + 2 static + p90
    with pytest.raises(ValueError):
        evaluate_capacity_policies(np.full(4, 30.0), np.full(4, 40.0), intervals_df, underprovision_cost_ratio=-1.0)


# ---------------------------------------------------------------------------- quality
def test_data_quality_report_with_metadata(synthetic_readings_df, synthetic_metadata_df, tmp_path):
    r = compute_data_quality_report(synthetic_readings_df, synthetic_metadata_df)
    assert r["total_rows"] == 4000 and r["unique_vms"] == 50 and r["unique_timestamps"] == 80
    assert (r["timestamp_min"], r["timestamp_max"]) == (0, 23700)
    assert r["cadence_diffs"] == {"300": 79} and r["is_regular_cadence"] is True
    assert r["is_balanced_panel"] is True and r["is_constant_active_vms"] is True
    assert r["duplicate_vm_timestamp_pairs"] == 0
    assert r["missing_values_cpu"] == {"min_cpu": 0, "avg_cpu": 0, "max_cpu": 0}
    assert r["ordering_violations"] == {"min_gt_avg": 0, "avg_gt_max": 0, "out_of_0_100_range": 0}
    assert (r["train_window_timestamps"], r["holdout_window_timestamps"]) == (56, 24)
    assert r["vm_id_exact_match"] is True and r["missing_metadata_vms"] == 0
    assert sum(r["category_counts"].values()) == 50
    assert r["virtual_core_bucket_counts"] == {"2": 10, "4": 10, "8": 10, "24": 10, ">24": 10}
    assert r["vms_created_after_start"] == 0 and r["vms_deleted_before_end"] == 0
    assert r["created_timestamp_min"] == 0 and r["deleted_timestamp_max"] == 40500

    path = save_data_quality_report(r, tmp_path / "sub" / "dq.json")
    assert json.loads(path.read_text(encoding="utf-8"))["unique_vms"] == 50


def test_data_quality_report_without_metadata(synthetic_readings_df):
    r = compute_data_quality_report(synthetic_readings_df)
    assert "metadata_rows" not in r and "vm_id_exact_match" not in r
    json.dumps(r)  # fully JSON-serialisable


def test_data_quality_report_flags_problems(synthetic_readings_df):
    bad = synthetic_readings_df.iloc[:-30].copy()  # ragged panel
    bad = pd.concat([bad, bad.iloc[:3]], ignore_index=True)  # duplicate rows
    bad.loc[0, "avg_cpu"] = np.nan
    bad.loc[1, "min_cpu"] = bad.loc[1, "avg_cpu"] + 1
    bad.loc[2, "max_cpu"] = 120.0
    r = compute_data_quality_report(bad)
    assert r["duplicate_vm_timestamp_pairs"] == 3
    assert r["missing_values_cpu"]["avg_cpu"] == 1
    assert r["ordering_violations"]["min_gt_avg"] >= 1
    assert r["ordering_violations"]["out_of_0_100_range"] >= 1
    assert r["is_balanced_panel"] is False


# ------------------------------------------------------------------ cwfc.predict CLI
def test_predict_cli_without_metadata(tmp_path, run_predict, synthetic_data_dir_no_metadata, synthetic_readings_df):
    out = tmp_path / "preds.csv"
    proc = run_predict(synthetic_data_dir_no_metadata, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.exists()
    preds = pd.read_csv(out)
    expected_ts = np.sort(synthetic_readings_df["timestamp"].unique())[-24:]
    assert verify_predictions(preds, expected_timestamps=expected_ts) == []
    assert preds["timestamp"].dtype.kind == "i"  # integer seconds, never datetimes


def test_predict_cli_with_metadata(tmp_path, run_predict, synthetic_data_dir, synthetic_readings_df):
    out = tmp_path / "preds_meta.csv"
    proc = run_predict(synthetic_data_dir, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    expected_ts = np.sort(synthetic_readings_df["timestamp"].unique())[-24:]
    assert verify_predictions(pd.read_csv(out), expected_timestamps=expected_ts) == []


def test_predict_cli_is_deterministic(tmp_path, run_predict, synthetic_data_dir_no_metadata):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert run_predict(synthetic_data_dir_no_metadata, a).returncode == 0
    assert run_predict(synthetic_data_dir_no_metadata, b).returncode == 0
    assert a.read_bytes() == b.read_bytes()
