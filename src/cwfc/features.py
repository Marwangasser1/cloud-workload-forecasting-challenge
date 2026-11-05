"""Leakage-free feature construction for direct multi-horizon forecasting.

Conventions
-----------
* A *forecast origin* ``t0`` is a positional row index. The information set at the
  origin is ``series_df.iloc[:t0 + 1]`` (rows ``0..t0`` inclusive).
* Targets are ``y[t0 + h]`` for ``h = 1..horizon``.
* Every feature at ``t0`` is a function of rows ``<= t0`` only. The vectorised
  :func:`build_feature_frame` uses exclusively causal pandas operations (``shift``,
  trailing ``rolling``, ``ewm(adjust=False)``) so row ``t`` of its output equals
  ``build_origin_feature_vector(series_df.iloc[:t + 1])`` -- this equivalence is the
  leakage guarantee and is cheap to unit-test.
* ``lag_k`` is the value ``k`` steps *before* the origin (``y[t0 - k]``). When fewer
  than ``k`` earlier rows exist the earliest available value is used instead.
* Rolling statistics use the trailing window *including* ``y[t0]``
  (``min_periods=1``, population std ``ddof=0``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_TARGET = "aggregate_cpu_utilization"

LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
ROLL_MEAN_WINDOWS: tuple[int, ...] = (3, 6, 12, 24)
ROLL_STD_WINDOWS: tuple[int, ...] = (3, 6, 12, 24)
ROLL_MINMAX_WINDOWS: tuple[int, ...] = (6, 12)
DIFF_LAGS: tuple[int, ...] = (1, 3, 6, 12)
EWMA_SPANS: tuple[int, ...] = (3, 6, 12)
CROSS_SECTIONAL_COLUMNS: tuple[str, ...] = (
    "p90_cpu",
    "p95_cpu",
    "mean_max_cpu",
    "mean_burst_spread",
    "std_cpu",
)

# Columns emitted by build_direct_supervised_dataset that are bookkeeping, not features.
META_COLUMNS: tuple[str, ...] = ("origin_index",)
HORIZON_COLUMNS: tuple[str, ...] = ("horizon", "h_sin", "h_cos")


def _level_columns() -> list[str]:
    """Feature columns measured on the same scale as the target level."""
    cols = [f"lag_{k}" for k in LAGS]
    cols += [f"roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
    cols += [f"roll_min_{w}" for w in ROLL_MINMAX_WINDOWS]
    cols += [f"roll_max_{w}" for w in ROLL_MINMAX_WINDOWS]
    cols += [f"ewma_{s}" for s in EWMA_SPANS]
    return cols


LEVEL_COLUMNS: tuple[str, ...] = tuple(_level_columns())


def build_feature_frame(
    series_df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    cross_sectional_cols: Iterable[str] = CROSS_SECTIONAL_COLUMNS,
) -> pd.DataFrame:
    """Causal origin features for *every* row of ``series_df`` (positional index).

    Row ``t`` depends on rows ``0..t`` only.
    """
    if target_col not in series_df.columns:
        raise KeyError(f"target column {target_col!r} not found")
    if series_df.empty:
        raise ValueError("series_df must not be empty")
    y = series_df[target_col].astype(float).reset_index(drop=True)
    if not np.all(np.isfinite(y.to_numpy())):
        raise ValueError(f"{target_col!r} contains non-finite values")
    first = float(y.iloc[0])

    feats: dict[str, pd.Series] = {"y_t0": y}
    for k in LAGS:
        feats[f"lag_{k}"] = y.shift(k).fillna(first)
    for w in ROLL_MEAN_WINDOWS:
        feats[f"roll_mean_{w}"] = y.rolling(w, min_periods=1).mean()
    for w in ROLL_STD_WINDOWS:
        feats[f"roll_std_{w}"] = y.rolling(w, min_periods=1).std(ddof=0).fillna(0.0)
    for w in ROLL_MINMAX_WINDOWS:
        feats[f"roll_min_{w}"] = y.rolling(w, min_periods=1).min()
        feats[f"roll_max_{w}"] = y.rolling(w, min_periods=1).max()
    for k in DIFF_LAGS:
        feats[f"diff_{k}"] = y - feats[f"lag_{k}"]
    for s in EWMA_SPANS:
        feats[f"ewma_{s}"] = y.ewm(span=s, adjust=False).mean()

    for col in cross_sectional_cols:
        if col in series_df.columns and col != target_col:
            xs = series_df[col].astype(float).reset_index(drop=True)
            feats[col] = xs
            if col == "mean_max_cpu":
                feats["mean_max_cpu_roll_mean_6"] = xs.rolling(6, min_periods=1).mean()

    frame = pd.DataFrame(feats)
    frame.index = series_df.index
    return frame


def build_origin_feature_vector(
    history_df: pd.DataFrame, target_col: str = DEFAULT_TARGET
) -> dict[str, float]:
    """Features at the origin ``t0 = len(history_df) - 1``.

    ``history_df`` must contain rows ``0..t0`` only; no future rows are ever read.
    """
    if history_df.empty:
        raise ValueError("history_df must not be empty")
    last = build_feature_frame(history_df, target_col).iloc[-1]
    return {k: float(v) for k, v in last.items()}


def to_relative_features(X: pd.DataFrame) -> pd.DataFrame:
    """Express level-type features as deviations from the current value ``y_t0``.

    Makes residual-over-persistence models (approximately) level invariant, which
    matters when the holdout level differs from typical training levels.
    """
    out = X.copy()
    if "y_t0" not in out.columns:
        return out
    for col in LEVEL_COLUMNS:
        if col in out.columns:
            out[col] = out[col] - out["y_t0"]
    if "mean_max_cpu" in out.columns and "mean_max_cpu_roll_mean_6" in out.columns:
        out["mean_max_cpu_roll_mean_6"] = out["mean_max_cpu_roll_mean_6"] - out["mean_max_cpu"]
    return out


def horizon_encoding(h: np.ndarray | int, period: int) -> dict[str, np.ndarray]:
    h_arr = np.asarray(h, dtype=float)
    angle = 2.0 * np.pi * h_arr / float(max(period, 1))
    return {"horizon": h_arr, "h_sin": np.sin(angle), "h_cos": np.cos(angle)}


def build_direct_supervised_dataset(
    train_df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    horizon: int = 24,
    min_history: int = 12,
    residual: bool = False,
    complete_horizons_only: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Pooled direct multi-horizon training set built **only** from ``train_df``.

    One sample per valid ``(t0, h)`` pair with ``t0 >= min_history - 1``,
    ``1 <= h <= horizon`` and ``t0 + h < len(train_df)``. Origins near the end of the
    window therefore contribute their short horizons. With
    ``complete_horizons_only=True`` only origins with all ``horizon`` targets inside
    ``train_df`` are used.

    Returns
    -------
    X : DataFrame
        Origin features + ``horizon``/``h_sin``/``h_cos`` + ``origin_index``
        (bookkeeping; drop via :data:`META_COLUMNS` before fitting).
    y : ndarray
        ``y[t0 + h]`` or, when ``residual=True``, ``y[t0 + h] - y[t0]``.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if min_history < 1:
        raise ValueError("min_history must be >= 1")
    n = len(train_df)
    feature_frame = build_feature_frame(train_df, target_col).reset_index(drop=True)
    y_all = feature_frame["y_t0"].to_numpy()

    origins: list[int] = []
    horizons: list[int] = []
    for t0 in range(min_history - 1, n - 1):
        max_h = min(horizon, n - 1 - t0)
        if complete_horizons_only and max_h < horizon:
            continue
        origins.extend([t0] * max_h)
        horizons.extend(range(1, max_h + 1))

    if not origins:
        empty = pd.DataFrame(
            columns=[*feature_frame.columns, *HORIZON_COLUMNS, *META_COLUMNS], dtype=float
        )
        return empty, np.empty(0, dtype=float)

    o = np.asarray(origins, dtype=int)
    h = np.asarray(horizons, dtype=int)
    X = feature_frame.iloc[o].reset_index(drop=True)
    for name, values in horizon_encoding(h, horizon).items():
        X[name] = values
    X["origin_index"] = o
    y = y_all[o + h]
    if residual:
        y = y - y_all[o]
    return X, y.astype(float)


def feature_columns(X: pd.DataFrame, include_horizon: bool = True) -> list[str]:
    drop = set(META_COLUMNS)
    if not include_horizon:
        drop |= set(HORIZON_COLUMNS)
    return [c for c in X.columns if c not in drop]
