from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def verify_predictions(
    pred_df: pd.DataFrame,
    expected_timestamps: Sequence[int] | np.ndarray | None = None,
    n_holdout: int = 24,
) -> list[str]:
    """Return a list of contract violations (empty if the submission is valid)."""
    v: list[str] = []

    if list(pred_df.columns) != REQUIRED_COLUMNS:
        v.append(f"columns must be exactly {REQUIRED_COLUMNS}, got {list(pred_df.columns)}")
        return v  # remaining checks need the right columns

    if len(pred_df) != n_holdout:
        v.append(f"expected {n_holdout} rows, got {len(pred_df)}")

    numeric: dict[str, pd.Series] = {}
    bad_values = False
    for c in REQUIRED_COLUMNS:
        s = pd.to_numeric(pred_df[c], errors="coerce")
        numeric[c] = s
        arr = s.to_numpy(dtype=float)
        if pred_df[c].isna().any() or s.isna().any() or not np.isfinite(arr).all():
            v.append(f"column '{c}' contains NaN, null, non-numeric or infinite values")
            bad_values = True

    ts = numeric["timestamp"]
    if not bad_values:
        if not (ts.to_numpy() == np.floor(ts.to_numpy())).all():
            v.append("timestamp must be integer-valued")
        if ts.duplicated().any():
            v.append("timestamp contains duplicates")
        if not ts.is_monotonic_increasing or (ts.diff().dropna() <= 0).any():
            v.append("timestamp must be strictly increasing")
        if expected_timestamps is not None:
            exp = [int(x) for x in expected_timestamps]
            got = [int(x) for x in ts.to_numpy()]
            if got != exp:
                v.append("timestamp values do not match expected holdout timestamps")
        if not (numeric["active_vm_count"] >= 0).all():
            v.append("active_vm_count must be >= 0")
        u = numeric["aggregate_cpu_utilization"]
        if not ((u >= 0.0) & (u <= 100.0)).all():
            v.append("aggregate_cpu_utilization must be within [0, 100]")
    return v


def assert_valid_predictions(
    pred_df: pd.DataFrame,
    expected_timestamps: Sequence[int] | np.ndarray | None = None,
    n_holdout: int = 24,
) -> None:
    violations = verify_predictions(pred_df, expected_timestamps, n_holdout)
    if violations:
        raise ValueError("Prediction contract violations: " + "; ".join(violations))
