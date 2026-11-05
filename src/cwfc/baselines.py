"""Simple, transparent baseline forecasters.

All baselines share the interface ``f(train, template, target, ...) -> pd.Series``:

* ``train``    -- history only (rows strictly before the forecast window).
* ``template`` -- frame whose index defines the forecast steps (its target values,
                  if any, are never read).
* ``target``   -- name of the column in ``train`` to forecast.

The returned Series is indexed like ``template`` and has ``dtype=float``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _history(train: pd.DataFrame, target: str) -> np.ndarray:
    if train.empty:
        raise ValueError("train must not be empty")
    values = train[target].astype(float).to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"train[{target!r}] contains non-finite values")
    return values


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    season_length: int = 24,
    fallback: bool = False,
) -> pd.Series:
    """Repeat the last observed season.

    If there is less than one season of history a ``ValueError`` is raised, unless
    ``fallback=True`` in which case persistence is returned instead.
    """
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        if fallback and len(values) > 0:
            return persistence_forecast(train, template, target)
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)


def moving_average_forecast(
    train: pd.DataFrame, template: pd.DataFrame, target: str, window: int = 12
) -> pd.Series:
    """Flat forecast equal to the mean of the last ``min(window, len(train))`` values."""
    if window <= 0:
        raise ValueError("window must be positive")
    values = _history(train, target)
    level = float(np.mean(values[-min(window, len(values)):]))
    return pd.Series(level, index=template.index, dtype=float)


def drift_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    lookback: int | None = None,
    damping: float = 1.0,
    clip: tuple[float, float] | None = None,
) -> pd.Series:
    """Random-walk-with-drift forecast.

    The slope is ``(y_last - y_first) / (n - 1)`` over the last ``lookback`` points
    (the full history when ``lookback is None``). With ``damping < 1`` the cumulative
    drift at step ``h`` is ``slope * sum_{k=1..h} damping**k`` (damped-trend style),
    which keeps long-horizon extrapolations bounded. ``clip`` optionally bounds the
    output (e.g. ``(0, 100)`` for utilization percentages).
    """
    if lookback is not None and lookback < 2:
        raise ValueError("lookback must be >= 2 or None")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    values = _history(train, target)
    window = values if lookback is None else values[-min(lookback, len(values)):]
    last = float(values[-1])
    if len(window) < 2:
        slope = 0.0
    else:
        slope = float(window[-1] - window[0]) / (len(window) - 1)
    steps = np.arange(1, len(template) + 1, dtype=float)
    if damping == 1.0:
        cumulative = steps
    else:
        cumulative = damping * (1.0 - damping**steps) / (1.0 - damping)
    forecast = last + slope * cumulative
    if clip is not None:
        forecast = np.clip(forecast, clip[0], clip[1])
    return pd.Series(forecast, index=template.index, dtype=float)


def ewma_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, span: int = 6) -> pd.Series:
    """Flat forecast equal to the last exponentially weighted moving-average level."""
    if span < 1:
        raise ValueError("span must be >= 1")
    values = _history(train, target)
    level = float(pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1])
    return pd.Series(level, index=template.index, dtype=float)
