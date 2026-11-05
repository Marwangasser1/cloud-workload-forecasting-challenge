from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: Any
    validation_start: Any
    validation_end: Any
    train_end_idx: int = 0  # exclusive: frame.iloc[:train_end_idx]
    val_start_idx: int = 0
    val_end_idx: int = 0  # exclusive: frame.iloc[val_start_idx:val_end_idx]


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 24,
    min_train_size: int = 48,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    Works for integer-second and datetime timestamps (no conversion applied).
    `horizon=24` is two hours at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = frame[timestamp_col].reset_index(drop=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        out.append(
            TimeSplit(
                train_end=ts.iloc[train_end_idx - 1],
                validation_start=ts.iloc[train_end_idx],
                validation_end=ts.iloc[train_end_idx + horizon - 1],
                train_end_idx=train_end_idx,
                val_start_idx=train_end_idx,
                val_end_idx=train_end_idx + horizon,
            )
        )
        train_end_idx += step
    return out
