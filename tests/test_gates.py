"""Tests for the submission-contract gate (``cwfc.gates``).

The enforced contract (see ``SUBMISSION.md``) is the column set
``timestamp, active_vm_count, aggregate_cpu_utilization`` with exactly ``n_holdout`` rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.gates import REQUIRED_COLUMNS, assert_valid_predictions, verify_predictions

HOLDOUT_TS = np.arange(110, 134, dtype="int64") * 300


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": HOLDOUT_TS.copy(),
            "active_vm_count": np.full(24, 50, dtype="int64"),
            "aggregate_cpu_utilization": np.linspace(30.0, 35.0, 24),
        }
    )


def _has(violations, fragment):
    return any(fragment in v for v in violations)


def test_valid_predictions_pass(valid_df):
    assert verify_predictions(valid_df) == []
    assert verify_predictions(valid_df, expected_timestamps=HOLDOUT_TS) == []
    assert_valid_predictions(valid_df, expected_timestamps=HOLDOUT_TS)  # does not raise


def test_boundary_values_are_valid(valid_df):
    valid_df["aggregate_cpu_utilization"] = [0.0, 100.0] * 12
    valid_df["active_vm_count"] = 0
    assert verify_predictions(valid_df) == []


def test_required_columns_constant():
    assert REQUIRED_COLUMNS == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


@pytest.mark.parametrize("n_rows", [0, 1, 23, 25, 48])
def test_wrong_row_count(valid_df, n_rows):
    df = pd.concat([valid_df] * 2, ignore_index=True).iloc[:n_rows]
    # keep timestamps unique/increasing so only the row-count rule is exercised
    df = df.assign(timestamp=np.arange(n_rows, dtype="int64") * 300)
    assert _has(verify_predictions(df), "expected 24 rows")


def test_custom_n_holdout(valid_df):
    assert verify_predictions(valid_df.iloc[:12], n_holdout=12) == []
    assert _has(verify_predictions(valid_df.iloc[:12]), "expected 24 rows")


def test_missing_column(valid_df):
    v = verify_predictions(valid_df.drop(columns="active_vm_count"))
    assert _has(v, "columns must be exactly")


def test_extra_column(valid_df):
    assert _has(verify_predictions(valid_df.assign(extra=1.0)), "columns must be exactly")


def test_wrong_column_order_or_name(valid_df):
    reordered = valid_df[["timestamp", "aggregate_cpu_utilization", "active_vm_count"]]
    assert _has(verify_predictions(reordered), "columns must be exactly")
    renamed = valid_df.rename(columns={"aggregate_cpu_utilization": "fleet_avg_cpu"})
    assert _has(verify_predictions(renamed), "columns must be exactly")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("col", REQUIRED_COLUMNS)
def test_nan_and_inf_rejected(valid_df, col, bad):
    valid_df[col] = valid_df[col].astype(float)
    valid_df.loc[5, col] = bad
    assert _has(verify_predictions(valid_df), f"column '{col}' contains NaN")


def test_non_numeric_rejected(valid_df):
    valid_df["aggregate_cpu_utilization"] = valid_df["aggregate_cpu_utilization"].astype(object)
    valid_df.loc[3, "aggregate_cpu_utilization"] = "abc"
    assert _has(verify_predictions(valid_df), "non-numeric")


def test_unsorted_timestamps(valid_df):
    valid_df.loc[[3, 4], "timestamp"] = valid_df.loc[[4, 3], "timestamp"].to_numpy()
    assert _has(verify_predictions(valid_df), "strictly increasing")


def test_reversed_timestamps(valid_df):
    assert _has(verify_predictions(valid_df.iloc[::-1].reset_index(drop=True)), "strictly increasing")


def test_duplicate_timestamps(valid_df):
    valid_df.loc[10, "timestamp"] = valid_df.loc[9, "timestamp"]
    v = verify_predictions(valid_df)
    assert _has(v, "duplicates") and _has(v, "strictly increasing")


def test_non_integer_timestamps(valid_df):
    valid_df["timestamp"] = valid_df["timestamp"] + 0.5
    assert _has(verify_predictions(valid_df), "integer-valued")


def test_expected_timestamps_mismatch(valid_df):
    shifted = HOLDOUT_TS + 300
    assert _has(verify_predictions(valid_df, expected_timestamps=shifted), "do not match expected")
    with pytest.raises(ValueError, match="do not match expected"):
        assert_valid_predictions(valid_df, expected_timestamps=shifted)


@pytest.mark.parametrize("bad", [-0.001, 100.001, 250.0, -5.0])
def test_utilization_outside_0_100(valid_df, bad):
    valid_df.loc[7, "aggregate_cpu_utilization"] = bad
    assert _has(verify_predictions(valid_df), "within [0, 100]")


@pytest.mark.parametrize("bad", [-1, -50])
def test_negative_active_vm_count(valid_df, bad):
    valid_df.loc[2, "active_vm_count"] = bad
    assert _has(verify_predictions(valid_df), "active_vm_count must be >= 0")


def test_assert_valid_predictions_raises_with_all_violations(valid_df):
    valid_df.loc[0, "aggregate_cpu_utilization"] = 150.0
    valid_df.loc[1, "active_vm_count"] = -3
    with pytest.raises(ValueError) as exc:
        assert_valid_predictions(valid_df)
    msg = str(exc.value)
    assert "Prediction contract violations" in msg
    assert "within [0, 100]" in msg and "active_vm_count must be >= 0" in msg


def test_verify_does_not_mutate_input(valid_df):
    snapshot = valid_df.copy()
    verify_predictions(valid_df, HOLDOUT_TS)
    pd.testing.assert_frame_equal(valid_df, snapshot)
