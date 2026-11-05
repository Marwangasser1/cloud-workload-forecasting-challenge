"""Tests for ``cwfc.targets``.

NOTE: the *actual* target contract implemented in ``cwfc.targets`` (and required by
``SUBMISSION.md`` / ``cwfc.gates``) is ``active_vm_count`` and
``aggregate_cpu_utilization`` (+ diagnostic cross-sectional columns), so that is what is
asserted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.targets import (
    build_aggregate_targets,
    build_segment_targets,
    split_raw_readings_train_holdout,
    split_train_holdout,
)

AGG_COLUMNS = [
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
    "std_cpu",
    "median_cpu",
    "p90_cpu",
    "p95_cpu",
    "mean_max_cpu",
    "p95_max_cpu_t",
    "mean_min_cpu",
    "mean_burst_spread",
]


def _small_readings() -> pd.DataFrame:
    """Deliberately unsorted frame: t=300 listed first, t=0 has 3 VMs, t=300 has 4."""
    rows = [
        # timestamp, vm_id, min, max, avg
        (300, "d", 15.0, 90.0, 40.0),
        (300, "a", 1.0, 20.0, 10.0),
        (300, "b", 2.0, 40.0, 20.0),
        (300, "c", 3.0, 61.0, 30.0),
        (0, "a", 5.0, 20.0, 10.0),
        (0, "b", 10.0, 40.0, 20.0),
        (0, "c", 15.0, 60.0, 30.0),
    ]
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


class TestBuildAggregateTargets:
    def test_columns_and_sorted_integer_timestamps(self):
        out = build_aggregate_targets(_small_readings())
        assert list(out.columns) == AGG_COLUMNS
        assert out["timestamp"].tolist() == [0, 300]
        assert out["timestamp"].dtype == np.int64
        assert out["active_vm_count"].dtype == np.int64

    def test_hand_calculated_values(self):
        out = build_aggregate_targets(_small_readings()).set_index("timestamp")
        t0, t1 = out.loc[0], out.loc[300]

        assert t0["active_vm_count"] == 3
        assert t1["active_vm_count"] == 4
        assert t0["aggregate_cpu_utilization"] == pytest.approx(20.0)
        assert t1["aggregate_cpu_utilization"] == pytest.approx(25.0)
        assert t0["median_cpu"] == pytest.approx(20.0)
        assert t1["median_cpu"] == pytest.approx(25.0)
        assert t0["std_cpu"] == pytest.approx(10.0)  # sample std (ddof=1) of 10,20,30
        assert t0["mean_max_cpu"] == pytest.approx(40.0)
        assert t1["mean_max_cpu"] == pytest.approx(52.75)
        assert t0["mean_min_cpu"] == pytest.approx(10.0)
        # burst spread = max - min averaged over VMs
        assert t0["mean_burst_spread"] == pytest.approx(((15 + 30 + 45) / 3))

        # linear-interpolated percentiles, identical to np.percentile
        assert t0["p95_max_cpu_t"] == pytest.approx(np.percentile([20, 40, 60], 95))
        assert t0["p95_max_cpu_t"] == pytest.approx(58.0)
        assert t1["p95_max_cpu_t"] == pytest.approx(np.percentile([20, 40, 61, 90], 95))
        assert t1["p95_cpu"] == pytest.approx(np.percentile([10, 20, 30, 40], 95))
        assert t1["p90_cpu"] == pytest.approx(np.percentile([10, 20, 30, 40], 90))

    def test_aggregate_is_clipped_to_valid_range(self):
        df = pd.DataFrame(
            {
                "timestamp": [0, 0],
                "vm_id": ["a", "b"],
                "min_cpu": [0.0, 0.0],
                "max_cpu": [150.0, 150.0],
                "avg_cpu": [120.0, 130.0],
            }
        )
        assert build_aggregate_targets(df)["aggregate_cpu_utilization"].iloc[0] == 100.0

    def test_single_vm_has_zero_std(self):
        df = pd.DataFrame(
            {"timestamp": [0], "vm_id": ["a"], "min_cpu": [1.0], "max_cpu": [3.0], "avg_cpu": [2.0]}
        )
        out = build_aggregate_targets(df)
        assert out["std_cpu"].iloc[0] == 0.0
        assert out["active_vm_count"].iloc[0] == 1

    def test_nan_avg_dropped_and_duplicates_counted_once(self):
        df = pd.DataFrame(
            {
                "timestamp": [0, 0, 0, 0],
                "vm_id": ["a", "a", "b", "c"],
                "min_cpu": [0.0, 0.0, 0.0, 0.0],
                "max_cpu": [10.0, 10.0, 10.0, 10.0],
                "avg_cpu": [10.0, 20.0, 30.0, np.nan],  # a duplicated (mean 15), c NaN
            }
        )
        out = build_aggregate_targets(df)
        assert out["active_vm_count"].iloc[0] == 2
        assert out["aggregate_cpu_utilization"].iloc[0] == pytest.approx((15.0 + 30.0) / 2)

    def test_does_not_mutate_input(self):
        df = _small_readings()
        snapshot = df.copy()
        build_aggregate_targets(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_synthetic_panel(self, synthetic_readings_df):
        out = build_aggregate_targets(synthetic_readings_df)
        assert list(out.columns) == AGG_COLUMNS
        assert len(out) == 80
        assert out["timestamp"].tolist() == list(range(0, 80 * 300, 300))
        assert (out["active_vm_count"] == 50).all()
        assert out["aggregate_cpu_utilization"].between(0, 100).all()
        assert np.isfinite(out.drop(columns="timestamp").to_numpy()).all()
        # aggregate equals the plain per-timestamp mean of avg_cpu
        expected = synthetic_readings_df.groupby("timestamp")["avg_cpu"].mean().to_numpy()
        np.testing.assert_allclose(out["aggregate_cpu_utilization"].to_numpy(), expected)
        # synthetic fixture really contains busy VMs
        assert (synthetic_readings_df["max_cpu"] > 60.0).any()


class TestBuildSegmentTargets:
    def test_hand_calculated_segments(self):
        readings = _small_readings()
        meta = pd.DataFrame(
            {"vm_id": ["a", "b", "c"], "vm_category": ["Interactive", "Interactive", "Delay-insensitive"]}
        )  # vm "d" has no metadata -> "Unknown"
        out = build_segment_targets(readings, meta).set_index(["timestamp", "vm_category"])
        assert set(out.columns) == {"active_vm_count", "aggregate_cpu_utilization", "p95_max_cpu_t"}
        assert out.loc[(0, "Interactive"), "aggregate_cpu_utilization"] == pytest.approx(15.0)
        assert out.loc[(0, "Interactive"), "active_vm_count"] == 2
        assert out.loc[(0, "Delay-insensitive"), "aggregate_cpu_utilization"] == pytest.approx(30.0)
        assert out.loc[(300, "Unknown"), "aggregate_cpu_utilization"] == pytest.approx(40.0)
        assert out.loc[(300, "Unknown"), "active_vm_count"] == 1
        assert out.loc[(0, "Interactive"), "p95_max_cpu_t"] == pytest.approx(np.percentile([20, 40], 95))

    def test_segments_partition_the_fleet(self, synthetic_readings_df, synthetic_metadata_df):
        seg = build_segment_targets(synthetic_readings_df, synthetic_metadata_df)
        agg = build_aggregate_targets(synthetic_readings_df)
        assert set(seg["vm_category"]) == {"Interactive", "Delay-insensitive", "Unknown"}
        counts = seg.groupby("timestamp")["active_vm_count"].sum()
        np.testing.assert_array_equal(counts.to_numpy(), agg["active_vm_count"].to_numpy())
        # count-weighted segment means recombine to the fleet mean
        seg = seg.assign(w=seg["aggregate_cpu_utilization"] * seg["active_vm_count"])
        recombined = seg.groupby("timestamp")["w"].sum() / counts
        np.testing.assert_allclose(
            recombined.to_numpy(), agg["aggregate_cpu_utilization"].to_numpy(), rtol=1e-9
        )

    def test_alternate_segment_column(self, synthetic_readings_df, synthetic_metadata_df):
        seg = build_segment_targets(synthetic_readings_df, synthetic_metadata_df, "virtual_core_bucket")
        assert set(seg["virtual_core_bucket"]) == {"2", "4", "8", "24", ">24"}


class TestSplits:
    def test_split_train_holdout_positional(self):
        df = pd.DataFrame({"timestamp": np.arange(30) * 300, "y": np.arange(30.0)})
        train, hold = split_train_holdout(df, n_holdout=6)
        assert len(train) == 24 and len(hold) == 6
        assert train["timestamp"].max() < hold["timestamp"].min()
        assert hold["timestamp"].tolist() == (np.arange(24, 30) * 300).tolist()
        assert train.index.tolist() == list(range(24))
        assert hold.index.tolist() == list(range(6))
        pd.testing.assert_frame_equal(pd.concat([train, hold], ignore_index=True), df)

    def test_split_default_is_24(self):
        df = pd.DataFrame({"timestamp": np.arange(80) * 300, "y": np.arange(80.0)})
        train, hold = split_train_holdout(df)
        assert (len(train), len(hold)) == (56, 24)

    def test_split_returns_copies(self):
        df = pd.DataFrame({"timestamp": np.arange(30), "y": np.arange(30.0)})
        train, hold = split_train_holdout(df, 6)
        hold.loc[:, "y"] = -1.0
        assert (df["y"] >= 0).all()

    @pytest.mark.parametrize("n_holdout", [0, -3])
    def test_split_rejects_non_positive_holdout(self, n_holdout):
        with pytest.raises(ValueError):
            split_train_holdout(pd.DataFrame({"y": range(10)}), n_holdout)

    def test_split_requires_longer_series(self):
        with pytest.raises(ValueError):
            split_train_holdout(pd.DataFrame({"y": range(24)}), 24)

    def test_split_raw_readings(self, synthetic_readings_df):
        train, hold, hold_ts = split_raw_readings_train_holdout(synthetic_readings_df, 24)
        all_ts = np.sort(synthetic_readings_df["timestamp"].unique())
        np.testing.assert_array_equal(hold_ts, all_ts[-24:])
        assert train["timestamp"].nunique() == 56
        assert hold["timestamp"].nunique() == 24
        assert train["timestamp"].max() < hold["timestamp"].min()
        assert len(train) + len(hold) == len(synthetic_readings_df)
        assert len(train) == 56 * 50 and len(hold) == 24 * 50
        assert set(hold["timestamp"]) == set(hold_ts.tolist())

    def test_split_raw_matches_aggregate_split(self, synthetic_readings_df):
        train_raw, _, hold_ts = split_raw_readings_train_holdout(synthetic_readings_df, 24)
        agg_train, agg_hold = split_train_holdout(build_aggregate_targets(synthetic_readings_df), 24)
        pd.testing.assert_frame_equal(
            build_aggregate_targets(train_raw), agg_train, check_exact=False, rtol=1e-12
        )
        np.testing.assert_array_equal(agg_hold["timestamp"].to_numpy(), hold_ts)

    def test_split_raw_validates(self, synthetic_readings_df):
        with pytest.raises(ValueError):
            split_raw_readings_train_holdout(synthetic_readings_df, 0)
        short = synthetic_readings_df[synthetic_readings_df["timestamp"] < 24 * 300]
        with pytest.raises(ValueError):
            split_raw_readings_train_holdout(short, 24)
