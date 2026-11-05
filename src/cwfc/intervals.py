"""Horizon-specific empirical prediction intervals from rolling-origin residuals.

Residuals are defined as ``e[f, h] = y_true[f, h] - y_pred[f, h]`` for fold ``f`` and
step ``h`` (1-based). Bounds are ``point_forecast[h] + quantile(e[:, h], q)``, so a
biased model's systematic error is corrected for automatically.

With only a handful of rolling-origin folds per horizon, raw empirical quantiles are
noisy; three optional regularisers are provided:

* ``pool_neighbors`` -- pool residuals from steps ``h - k .. h + k``;
* ``finite_sample``  -- conformal-style level ``ceil((n + 1) q) / n`` (conservative);
* ``monotone``       -- interval half-widths never shrink as the horizon grows.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def residuals_by_horizon(
    y_true_folds: Sequence[Sequence[float]] | np.ndarray,
    y_pred_folds: Sequence[Sequence[float]] | np.ndarray,
) -> dict[int, np.ndarray]:
    """Convert per-fold actual/predicted arrays (``n_folds x H``) to ``{h: residuals}``.

    Ragged folds (shorter than ``H``) are allowed; ``h`` is 1-based.
    """
    if len(y_true_folds) != len(y_pred_folds):
        raise ValueError("y_true_folds and y_pred_folds must have the same number of folds")
    out: dict[int, list[float]] = {}
    for actual, pred in zip(y_true_folds, y_pred_folds):
        a = np.asarray(actual, dtype=float)
        p = np.asarray(pred, dtype=float)
        if a.shape != p.shape:
            raise ValueError("each fold's actual and predicted arrays must have equal shape")
        for h, err in enumerate(a - p, start=1):
            if np.isfinite(err):
                out.setdefault(h, []).append(float(err))
    return {h: np.asarray(v, dtype=float) for h, v in sorted(out.items())}


def _adjusted_level(q: float, n: int, finite_sample: bool) -> float:
    if not finite_sample or n == 0:
        return q
    return min(1.0, math.ceil((n + 1) * q) / n)


def _upper_quantile(e: np.ndarray, q: float, finite_sample: bool) -> float:
    level = _adjusted_level(q, len(e), finite_sample)
    method = "higher" if finite_sample else "linear"
    return float(np.quantile(e, level, method=method))


def _lower_quantile(e: np.ndarray, q: float, finite_sample: bool) -> float:
    """Lower-tail quantile at level ``q`` (< 0.5), mirrored so it is equally conservative."""
    return -_upper_quantile(-e, 1.0 - q, finite_sample)


def _pooled_residuals(
    residuals: Mapping[int, np.ndarray], h: int, pool_neighbors: int
) -> np.ndarray:
    available = sorted(k for k, v in residuals.items() if len(v) > 0)
    if not available:
        raise ValueError("cv_residuals_by_horizon contains no residuals")
    window = [k for k in available if abs(k - h) <= pool_neighbors]
    if not window:
        # Horizon not covered by the backtest: borrow the nearest covered horizon
        # (prefers the longer one on ties, i.e. the more conservative choice).
        nearest = min(available, key=lambda k: (abs(k - h), -k))
        window = [nearest]
    return np.concatenate([np.asarray(residuals[k], dtype=float) for k in window])


def _label(level: float) -> str:
    return f"{int(round(level * 100))}"


def compute_empirical_intervals(
    point_forecast: pd.Series | np.ndarray | Sequence[float],
    cv_residuals_by_horizon: Mapping[int, np.ndarray],
    quantiles: Sequence[float] = (0.80, 0.90, 0.95),
    two_sided_levels: Sequence[float] = (0.80, 0.90),
    clip: tuple[float, float] | None = (0.0, 100.0),
    pool_neighbors: int = 0,
    finite_sample: bool = False,
    monotone: bool = False,
) -> pd.DataFrame:
    """Empirical one-sided capacity quantiles and two-sided intervals per horizon.

    Returns columns ``horizon``, ``point_forecast``, ``p{q}_upper`` for each entry of
    ``quantiles`` (e.g. ``p80_upper``, ``p90_upper``, ``p95_upper``) and
    ``pi{L}_lower`` / ``pi{L}_upper`` for each two-sided level ``L`` (default 80, 90).
    ``p{q}_upper`` is the level-``q`` one-sided upper bound (actual exceeds it with
    probability ``~1 - q``); ``pi{L}`` uses the ``(1-L)/2`` and ``(1+L)/2`` quantiles.
    All bounds are clipped to ``clip`` (``[0, 100]`` by default; ``None`` disables).
    The index mirrors ``point_forecast`` when it is a Series.
    """
    index = point_forecast.index if isinstance(point_forecast, pd.Series) else None
    point = np.asarray(point_forecast, dtype=float).ravel()
    n_steps = len(point)
    for q in (*quantiles, *two_sided_levels):
        if not 0.0 < q < 1.0:
            raise ValueError(f"levels must lie in (0, 1); got {q}")
    if pool_neighbors < 0:
        raise ValueError("pool_neighbors must be >= 0")

    offsets: dict[str, np.ndarray] = {}
    for q in quantiles:
        offsets[f"p{_label(q)}_upper"] = np.empty(n_steps)
    for level in two_sided_levels:
        offsets[f"pi{_label(level)}_lower"] = np.empty(n_steps)
        offsets[f"pi{_label(level)}_upper"] = np.empty(n_steps)
    n_resid = np.empty(n_steps, dtype=int)

    for i in range(n_steps):
        h = i + 1
        e = _pooled_residuals(cv_residuals_by_horizon, h, pool_neighbors)
        n_resid[i] = len(e)
        for q in quantiles:
            offsets[f"p{_label(q)}_upper"][i] = _upper_quantile(e, q, finite_sample)
        for level in two_sided_levels:
            alpha = (1.0 - level) / 2.0
            offsets[f"pi{_label(level)}_lower"][i] = _lower_quantile(e, alpha, finite_sample)
            offsets[f"pi{_label(level)}_upper"][i] = _upper_quantile(e, 1.0 - alpha, finite_sample)

    if monotone:
        for name, off in offsets.items():
            if name.endswith("_upper"):
                offsets[name] = np.maximum.accumulate(off)
            else:
                offsets[name] = np.minimum.accumulate(off)

    out = pd.DataFrame({"horizon": np.arange(1, n_steps + 1), "point_forecast": point})
    for name, off in offsets.items():
        bound = point + off
        if clip is not None:
            bound = np.clip(bound, clip[0], clip[1])
        out[name] = bound
    if clip is not None:
        out["point_forecast"] = np.clip(point, clip[0], clip[1])
    out["n_residuals"] = n_resid
    if index is not None:
        out.index = index
    return out


def interval_coverage(y_true: Sequence[float] | np.ndarray, intervals_df: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage of every bound in ``intervals_df`` against realised values.

    One-sided ``p{q}_upper``: share of steps with ``actual <= bound`` (nominal ``q``).
    Two-sided ``pi{L}``: share inside ``[lower, upper]`` plus mean width (nominal ``L``).
    """
    actual = np.asarray(y_true, dtype=float).ravel()
    if len(actual) != len(intervals_df):
        raise ValueError("y_true and intervals_df must have the same length")
    rows = []
    for col in intervals_df.columns:
        if col.startswith("p") and col.endswith("_upper") and not col.startswith("pi"):
            nominal = int(col[1:-6]) / 100.0
            bound = intervals_df[col].to_numpy(dtype=float)
            rows.append(
                {
                    "bound": col,
                    "nominal": nominal,
                    "empirical_coverage": float(np.mean(actual <= bound)),
                    "mean_width": float(np.mean(bound - intervals_df["point_forecast"].to_numpy(dtype=float))),
                }
            )
    for col in intervals_df.columns:
        if col.startswith("pi") and col.endswith("_lower"):
            label = col[2:-6]
            lower = intervals_df[col].to_numpy(dtype=float)
            upper = intervals_df[f"pi{label}_upper"].to_numpy(dtype=float)
            rows.append(
                {
                    "bound": f"pi{label}",
                    "nominal": int(label) / 100.0,
                    "empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                    "mean_width": float(np.mean(upper - lower)),
                }
            )
    return pd.DataFrame(rows, columns=["bound", "nominal", "empirical_coverage", "mean_width"])
