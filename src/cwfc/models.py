"""Forecasting models for the aggregate cloud-workload targets.

Every aggregate-level model follows the unified interface::

    fit_predict(train_df, template_df, target) -> pd.Series

* ``train_df``    -- aggregate series rows strictly before the forecast window.
* ``template_df`` -- defines the forecast steps (only ``index``/``len`` are used; any
                     target values it may carry are never read).
* returns a float Series indexed like ``template_df``.

Models are fit from scratch on every call (no hidden state), deterministic, and
fall back to persistence whenever the history is too short or the series is constant.
CPU-utilisation forecasts are clipped to ``[0, 100]``; other targets to ``>= 0``.

The VM-level panel models (:func:`segment_panel_forecast`,
:func:`vm_shrinkage_forecast`, :func:`global_vm_panel_forecast`) take the raw
VM-level CPU readings of the *training window only* instead of the aggregate frame.
"""

from __future__ import annotations

import warnings
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import (
    drift_forecast,
    ewma_forecast,
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
)
from .features import (
    DEFAULT_TARGET,
    build_direct_supervised_dataset,
    build_feature_frame,
    feature_columns,
    horizon_encoding,
    to_relative_features,
)

Forecaster = Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]

MIN_TRAIN_ROWS = 10


# --------------------------------------------------------------------------- helpers
def _bounds_for(target: str) -> tuple[float, float]:
    return (0.0, 100.0) if "cpu" in target.lower() else (0.0, np.inf)


def _finalize(values: np.ndarray, template_df: pd.DataFrame, target: str) -> pd.Series:
    lo, hi = _bounds_for(target)
    arr = np.clip(np.asarray(values, dtype=float), lo, hi)
    return pd.Series(arr, index=template_df.index, dtype=float)


def _values(train_df: pd.DataFrame, target: str) -> np.ndarray:
    if train_df.empty:
        raise ValueError("train_df must not be empty")
    values = train_df[target].astype(float).to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"train_df[{target!r}] contains non-finite values")
    return values


def _is_constant(values: np.ndarray) -> bool:
    return bool(np.ptp(values) <= 1e-12)


def _persistence(train_df: pd.DataFrame, template_df: pd.DataFrame, target: str) -> pd.Series:
    return _finalize(persistence_forecast(train_df, template_df, target).to_numpy(), template_df, target)


def _effective_min_history(n: int, min_history: int) -> int:
    return int(max(1, min(min_history, n // 3)))


# ------------------------------------------------------------------- damped ETS
def _damped_holt_numpy(values: np.ndarray, steps: int) -> np.ndarray:
    """Grid-searched additive damped-trend Holt (fallback when statsmodels is absent)."""
    best_sse, best = np.inf, (1.0, 0.0, 0.9)
    for alpha in np.linspace(0.05, 1.0, 20):
        for beta in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3):
            for phi in (0.8, 0.9, 0.95, 0.98):
                level, trend, sse = values[0], values[1] - values[0], 0.0
                for y in values[1:]:
                    pred = level + phi * trend
                    sse += (y - pred) ** 2
                    new_level = alpha * y + (1 - alpha) * pred
                    trend = beta * (new_level - level) + (1 - beta) * phi * trend
                    level = new_level
                if sse < best_sse:
                    best_sse, best = sse, (alpha, beta, phi)
    alpha, beta, phi = best
    level, trend = values[0], values[1] - values[0]
    for y in values[1:]:
        pred = level + phi * trend
        new_level = alpha * y + (1 - alpha) * pred
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    damp = np.cumsum(phi ** np.arange(1, steps + 1))
    return level + damp * trend


def damped_ets_forecast(
    train_df: pd.DataFrame,
    template_df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
) -> pd.Series:
    """Additive damped-trend exponential smoothing (no seasonality).

    Falls back to simple exponential smoothing if the damped fit fails, and to
    persistence for constant or very short (< 10 rows) series.
    """
    values = _values(train_df, target)
    steps = len(template_df)
    if steps == 0:
        return pd.Series(dtype=float, index=template_df.index)
    if len(values) < MIN_TRAIN_ROWS or _is_constant(values):
        return _persistence(train_df, template_df, target)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    except ImportError:  # pragma: no cover - depends on environment
        return _finalize(_damped_holt_numpy(values, steps), template_df, target)

    forecast: np.ndarray | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast = np.asarray(fit.forecast(steps), dtype=float)
        except Exception:
            forecast = None
        if forecast is None or not np.all(np.isfinite(forecast)):
            try:
                fit = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
                forecast = np.asarray(fit.forecast(steps), dtype=float)
            except Exception:
                forecast = None
    if forecast is None or not np.all(np.isfinite(forecast)):
        return _persistence(train_df, template_df, target)
    return _finalize(forecast, template_df, target)


# ------------------------------------------------------------ direct ML models
def _origin_row(train_df: pd.DataFrame, target: str, cols: list[str]) -> pd.DataFrame:
    """Relative features at the last training row (the forecast origin)."""
    frame = build_feature_frame(train_df, target).iloc[[-1]].reset_index(drop=True)
    return to_relative_features(frame)[cols]


def direct_ridge_forecast(
    train_df: pd.DataFrame,
    template_df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    alpha: float = 10.0,
    min_history: int = 12,
    min_samples_per_horizon: int = 8,
) -> pd.Series:
    """Direct multi-horizon Ridge: one ``StandardScaler + Ridge`` model per step ``h``.

    Each model predicts the residual over persistence ``y[t0+h] - y[t0]`` from
    level-relative origin features, so strong regularisation shrinks the forecast
    toward persistence instead of toward the training mean. Horizons with too few
    in-window training pairs reuse the prediction of the longest fitted horizon.
    """
    steps = len(template_df)
    values = _values(train_df, target)
    if steps == 0:
        return pd.Series(dtype=float, index=template_df.index)
    if len(values) < MIN_TRAIN_ROWS or _is_constant(values):
        return _persistence(train_df, template_df, target)

    X, y = build_direct_supervised_dataset(
        train_df,
        target,
        horizon=steps,
        min_history=_effective_min_history(len(values), min_history),
        residual=True,
    )
    if X.empty:
        return _persistence(train_df, template_df, target)
    cols = feature_columns(X, include_horizon=False)
    X_rel = to_relative_features(X[cols])
    origin = _origin_row(train_df, target, cols)
    horizon_arr = X["horizon"].to_numpy()

    residual_pred = np.zeros(steps, dtype=float)
    last = 0.0
    for h in range(1, steps + 1):
        mask = horizon_arr == h
        if mask.sum() >= min_samples_per_horizon:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(X_rel.loc[mask], y[mask])
            last = float(model.predict(origin)[0])
        residual_pred[h - 1] = last
    return _finalize(values[-1] + residual_pred, template_df, target)


def direct_hgb_forecast(
    train_df: pd.DataFrame,
    template_df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    random_state: int = 42,
    min_history: int = 12,
    min_samples: int = 50,
) -> pd.Series:
    """Pooled direct multi-horizon gradient boosting on ``(origin features, h, sin/cos h)``.

    Predicts ``y[t0+h] - y[t0]`` with a small, heavily regularised
    ``HistGradientBoostingRegressor``; falls back to persistence on small data.
    """
    steps = len(template_df)
    values = _values(train_df, target)
    if steps == 0:
        return pd.Series(dtype=float, index=template_df.index)
    if len(values) < MIN_TRAIN_ROWS or _is_constant(values):
        return _persistence(train_df, template_df, target)

    X, y = build_direct_supervised_dataset(
        train_df,
        target,
        horizon=steps,
        min_history=_effective_min_history(len(values), min_history),
        residual=True,
    )
    if len(X) < min_samples:
        return _persistence(train_df, template_df, target)
    cols = feature_columns(X, include_horizon=True)
    X_rel = to_relative_features(X[cols])
    model = HistGradientBoostingRegressor(
        max_iter=80,
        max_depth=3,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )
    model.fit(X_rel, y)

    base_cols = [c for c in cols if c not in ("horizon", "h_sin", "h_cos")]
    origin = _origin_row(train_df, target, base_cols)
    grid = pd.concat([origin] * steps, ignore_index=True)
    for name, vals in horizon_encoding(np.arange(1, steps + 1), steps).items():
        grid[name] = vals
    residual_pred = model.predict(grid[cols])
    return _finalize(values[-1] + residual_pred, template_df, target)


# -------------------------------------------------------------- hybrid ensemble
def _moving_average_6(train_df: pd.DataFrame, template_df: pd.DataFrame, target: str) -> pd.Series:
    return moving_average_forecast(train_df, template_df, target, window=6)


HYBRID_COMPONENTS: dict[str, Forecaster] = {
    "persistence": persistence_forecast,
    "moving_average_6": _moving_average_6,
    "damped_ets": damped_ets_forecast,
    "direct_ridge": direct_ridge_forecast,
}


def hybrid_ensemble_weights(
    train_df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    horizon: int = 24,
    n_folds: int = 4,
    fold_step: int = 6,
    min_train: int = 48,
    power: float = 1.0,
    components: Mapping[str, Forecaster] | None = None,
) -> dict[str, float]:
    """Inverse-MAE ensemble weights from an *inner* rolling-origin backtest.

    Inner folds are carved exclusively out of ``train_df`` (origins at
    ``len(train_df) - horizon - k * fold_step``), so no holdout value is touched.
    Equal weights are returned when no inner fold fits.
    """
    comps = dict(components or HYBRID_COMPONENTS)
    n = len(train_df)
    ends = [n - horizon - k * fold_step for k in range(n_folds)]
    ends = [e for e in ends if e >= max(min_train, MIN_TRAIN_ROWS)]
    if not ends or horizon < 1:
        return {name: 1.0 / len(comps) for name in comps}
    errors: dict[str, list[float]] = {name: [] for name in comps}
    for end in ends:
        inner_train = train_df.iloc[:end]
        inner_val = train_df.iloc[end : end + horizon]
        actual = inner_val[target].astype(float).to_numpy()
        for name, fn in comps.items():
            pred = fn(inner_train, inner_val, target).to_numpy()
            errors[name].append(float(np.mean(np.abs(actual - pred))))
    mean_err = {name: float(np.mean(v)) for name, v in errors.items()}
    if max(mean_err.values()) <= 1e-12:
        return {name: 1.0 / len(comps) for name in comps}
    inv = {name: 1.0 / (err + 1e-9) ** power for name, err in mean_err.items()}
    total = sum(inv.values())
    return {name: v / total for name, v in inv.items()}


def hybrid_ensemble_forecast(
    train_df: pd.DataFrame,
    template_df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    weights: Mapping[str, float] | None = None,
    adaptive: bool = True,
    n_folds: int = 4,
    fold_step: int = 6,
) -> pd.Series:
    """Weighted blend of persistence, MA(6), damped ETS and direct Ridge.

    ``weights`` (any subset of :data:`HYBRID_COMPONENTS`, renormalised) overrides the
    default; otherwise weights are inverse inner-backtest MAE (``adaptive=True``) or
    equal.
    """
    values = _values(train_df, target)
    if len(template_df) == 0:
        return pd.Series(dtype=float, index=template_df.index)
    if _is_constant(values):
        return _persistence(train_df, template_df, target)
    if weights is None:
        if adaptive:
            weights = hybrid_ensemble_weights(
                train_df, target, horizon=len(template_df), n_folds=n_folds, fold_step=fold_step
            )
        else:
            weights = {name: 1.0 for name in HYBRID_COMPONENTS}
    unknown = set(weights) - set(HYBRID_COMPONENTS)
    if unknown:
        raise KeyError(f"unknown ensemble components: {sorted(unknown)}")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("ensemble weights must sum to a positive value")
    blend = np.zeros(len(template_df), dtype=float)
    for name, w in weights.items():
        if w == 0:
            continue
        blend += (w / total) * HYBRID_COMPONENTS[name](train_df, template_df, target).to_numpy()
    return _finalize(blend, template_df, target)


# ------------------------------------------------------------ active VM count
def active_vm_count_forecast(
    train_df: pd.DataFrame, template_df: pd.DataFrame, target: str = "active_vm_count"
) -> pd.Series:
    """Persistence for a constant population (exact on ``data-v1``), else rounded Ridge."""
    if train_df[target].nunique() == 1:
        return persistence_forecast(train_df, template_df, target)
    fc = direct_ridge_forecast(train_df, template_df, target)
    return fc.clip(lower=0.0).round()


# ------------------------------------------------------------ VM-level panel
_CPU_COLUMNS = ("timestamp", "vm_id", "avg_cpu")


def _check_cpu_frame(cpu_train_df: pd.DataFrame) -> None:
    missing = [c for c in _CPU_COLUMNS if c not in cpu_train_df.columns]
    if missing:
        raise KeyError(f"cpu_train_df missing columns: {missing}")
    if cpu_train_df.empty:
        raise ValueError("cpu_train_df must not be empty")


def assign_vm_segments(
    cpu_train_df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    n_quantile_segments: int = 20,
    min_segment_size: int = 30,
) -> pd.Series:
    """Map each VM to a homogeneous stratum (Series indexed by ``vm_id``).

    * With metadata: ``vm_category | virtual_core_bucket``; strata smaller than
      ``min_segment_size`` are merged into ``vm_category | other``.
    * Without metadata: ``n_quantile_segments`` quantile bins of each VM's mean
      ``avg_cpu`` computed on ``cpu_train_df`` only (training window).
    """
    _check_cpu_frame(cpu_train_df)
    vm_ids = pd.Index(pd.unique(cpu_train_df["vm_id"]), name="vm_id")
    if meta_df is not None and {"vm_id", "vm_category", "virtual_core_bucket"} <= set(meta_df.columns):
        meta = meta_df.drop_duplicates("vm_id").set_index("vm_id").reindex(vm_ids)
        category = meta["vm_category"].astype(object).where(meta["vm_category"].notna(), "Unknown").astype(str)
        cores = meta["virtual_core_bucket"].astype(object).where(meta["virtual_core_bucket"].notna(), "na").astype(str)
        label = category + "|" + cores
        counts = label.map(label.value_counts())
        small = counts < min_segment_size
        label = label.where(~small, category + "|other")
        return label.rename("segment")
    vm_mean = cpu_train_df.groupby("vm_id")["avg_cpu"].mean().reindex(vm_ids)
    n_bins = int(max(1, min(n_quantile_segments, len(vm_mean))))
    bins = pd.qcut(vm_mean.rank(method="first"), n_bins, labels=False)
    return ("q" + bins.astype(int).astype(str).str.zfill(2)).rename("segment")


def segment_panel_forecast(
    cpu_train_df: pd.DataFrame,
    template_df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    horizon: int = 24,
    model: str = "ridge",
    alpha: float = 10.0,
    random_state: int = 42,
    n_quantile_segments: int = 20,
    min_history: int = 12,
    segments: pd.Series | None = None,
) -> pd.Series:
    """Bottom-up stratum panel forecaster for ``aggregate_cpu_utilization``.

    1. VMs are grouped into strata (:func:`assign_vm_segments`).
    2. Each stratum's mean-CPU series gets causal, level-relative origin features plus
       ``seg_expanding_dev`` (expanding mean - current level) and ``seg_log_size``.
    3. A global model pooled across strata predicts ``y_s[t0+h] - y_s[t0]``:
       per-horizon weighted Ridge (``model="ridge"``) or pooled HGB (``model="hgb"``),
       with samples weighted by stratum VM count.
    4. Stratum forecasts (clipped to ``[0, 100]``) are aggregated with exact VM-count
       weights at the last training timestamp.

    Steps beyond ``horizon`` repeat the last modelled step.
    """
    _check_cpu_frame(cpu_train_df)
    steps = len(template_df)
    if steps == 0:
        return pd.Series(dtype=float, index=template_df.index)
    if model not in ("ridge", "hgb"):
        raise ValueError("model must be 'ridge' or 'hgb'")
    seg = segments if segments is not None else assign_vm_segments(cpu_train_df, meta_df, n_quantile_segments)

    df = cpu_train_df.loc[:, list(_CPU_COLUMNS)].copy()
    df["segment"] = df["vm_id"].map(seg).fillna("unassigned")
    seg_series = (
        df.groupby(["timestamp", "segment"])["avg_cpu"].mean().unstack("segment").sort_index().ffill().bfill()
    )
    last_ts = df["timestamp"].max()
    weights = (
        df.loc[df["timestamp"] == last_ts].groupby("segment")["vm_id"].nunique().reindex(seg_series.columns).fillna(0.0)
    )
    n_t = len(seg_series)
    level_now = seg_series.iloc[-1].to_numpy()
    agg_now = float(np.average(level_now, weights=weights.to_numpy())) if weights.sum() > 0 else float(level_now.mean())
    if n_t < MIN_TRAIN_ROWS:
        return _finalize(np.full(steps, agg_now), template_df, "cpu")

    model_h = int(max(1, min(horizon, steps)))
    mh = _effective_min_history(n_t, min_history)
    X_parts, y_parts, w_parts, origin_parts = [], [], [], []
    for s in seg_series.columns:
        sdf = pd.DataFrame({"y": seg_series[s].to_numpy()})
        X, y = build_direct_supervised_dataset(sdf, "y", horizon=model_h, min_history=mh, residual=True)
        expanding = sdf["y"].expanding().mean().to_numpy()
        extra_size = float(np.log1p(weights[s]))
        if not X.empty:
            o = X["origin_index"].to_numpy()
            X = to_relative_features(X)
            X["seg_expanding_dev"] = expanding[o] - X["y_t0"].to_numpy()
            X["seg_log_size"] = extra_size
            X_parts.append(X)
            y_parts.append(y)
            w_parts.append(np.full(len(y), max(float(weights[s]), 1.0)))
        origin = to_relative_features(build_feature_frame(sdf, "y").iloc[[-1]].reset_index(drop=True))
        origin["seg_expanding_dev"] = expanding[-1] - float(sdf["y"].iloc[-1])
        origin["seg_log_size"] = extra_size
        origin_parts.append(origin)
    if not X_parts:
        return _finalize(np.full(steps, agg_now), template_df, "cpu")

    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = np.concatenate(y_parts)
    w_all = np.concatenate(w_parts)
    origins = pd.concat(origin_parts, ignore_index=True)
    n_seg = len(origins)
    residual = np.zeros((n_seg, model_h), dtype=float)

    if model == "ridge":
        cols = feature_columns(X_all, include_horizon=False)
        h_arr = X_all["horizon"].to_numpy()
        last = np.zeros(n_seg)
        for h in range(1, model_h + 1):
            mask = h_arr == h
            if mask.sum() >= 10:
                est = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                est.fit(X_all.loc[mask, cols], y_all[mask], ridge__sample_weight=w_all[mask])
                last = est.predict(origins[cols])
            residual[:, h - 1] = last
    else:
        cols = feature_columns(X_all, include_horizon=True)
        est = HistGradientBoostingRegressor(
            max_iter=80,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
        est.fit(X_all[cols], y_all, sample_weight=w_all)
        for h in range(1, model_h + 1):
            grid = origins.copy()
            for name, vals in horizon_encoding(np.full(n_seg, h), model_h).items():
                grid[name] = vals
            residual[:, h - 1] = est.predict(grid[cols])

    seg_fc = np.clip(level_now[:, None] + residual, 0.0, 100.0)
    w = weights.to_numpy()
    agg = (w[:, None] * seg_fc).sum(axis=0) / w.sum() if w.sum() > 0 else seg_fc.mean(axis=0)
    if steps > model_h:
        agg = np.concatenate([agg, np.full(steps - model_h, agg[-1])])
    return _finalize(agg[:steps], template_df, "cpu")


def _vm_matrix(cpu_train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(V x T)`` avg-CPU matrix (gaps forward/back-filled) + mask of VMs active at the last step."""
    wide = cpu_train_df.pivot_table(index="vm_id", columns="timestamp", values="avg_cpu", aggfunc="mean")
    wide = wide.sort_index(axis=1)
    active_last = wide.iloc[:, -1].notna().to_numpy()
    filled = wide.ffill(axis=1).bfill(axis=1).to_numpy(dtype=float)
    return filled, active_last


def vm_shrinkage_forecast(
    cpu_train_df: pd.DataFrame,
    template_df: pd.DataFrame,
    horizon: int = 24,
    recent_window: int = 12,
    drift_window: int = 12,
    min_history: int = 24,
    origin_step: int = 2,
    ridge_lambda: float = 1e-3,
    return_coefficients: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Vectorised VM-level empirical-shrinkage forecaster.

    For VM ``i`` at origin ``t0`` and step ``h``::

        y_hat[i, t0+h] = y[i, t0]
                         + b_h * (recent_mean[i, t0] - y[i, t0])      # 1-hour mean
                         + c_h * (expanding_mean[i, t0] - y[i, t0])   # long-run level
                         + d_h * h * fleet_drift[t0]                  # global drift

    i.e. a convex-style shrinkage of the last value toward its recent and long-run
    means (implied weight on ``y[i, t0]`` is ``1 - b_h - c_h``) plus a fleet drift term.
    ``(b_h, c_h, d_h)`` are fit by (lightly ridged) least squares over all VMs and
    in-window origins. All features are trailing statistics, so only training rows are
    used. VM forecasts are clipped to ``[0, 100]`` and averaged over VMs active at the
    last training timestamp.
    """
    _check_cpu_frame(cpu_train_df)
    steps = len(template_df)
    if steps == 0:
        empty = pd.Series(dtype=float, index=template_df.index)
        return (empty, pd.DataFrame()) if return_coefficients else empty
    M, active_last = _vm_matrix(cpu_train_df)
    n_vm, n_t = M.shape
    last = M[:, -1]
    if n_t < max(min_history, recent_window) + 2:
        fc = _finalize(np.full(steps, float(last[active_last].mean())), template_df, "cpu")
        return (fc, pd.DataFrame()) if return_coefficients else fc

    csum = np.cumsum(M, axis=1)
    t_idx = np.arange(n_t)
    expanding = csum / (t_idx + 1)
    recent = np.empty_like(M)
    w = recent_window
    recent[:, :w] = csum[:, :w] / (t_idx[:w] + 1)
    recent[:, w:] = (csum[:, w:] - csum[:, :-w]) / w
    fleet = M.mean(axis=0)
    fleet_recent = pd.Series(fleet).rolling(drift_window, min_periods=1).mean().to_numpy()
    drift = np.zeros(n_t)
    drift[drift_window:] = (fleet_recent[drift_window:] - fleet_recent[:-drift_window]) / drift_window

    def design(t0: int, h: int) -> np.ndarray:
        y0 = M[:, t0]
        return np.column_stack(
            [recent[:, t0] - y0, expanding[:, t0] - y0, np.full(n_vm, h * drift[t0])]
        )

    model_h = int(max(1, min(horizon, steps)))
    coefs = np.zeros((model_h, 3))
    n_pairs = np.zeros(model_h, dtype=int)
    prev = np.zeros(3)
    for h in range(1, model_h + 1):
        ztz = np.zeros((3, 3))
        zty = np.zeros(3)
        count = 0
        for t0 in range(min_history - 1, n_t - h, origin_step):
            Z = design(t0, h)
            target = M[:, t0 + h] - M[:, t0]
            ztz += Z.T @ Z
            zty += Z.T @ target
            count += 1
        if count > 0:
            scale = max(np.trace(ztz) / 3.0, 1e-12)
            prev = np.linalg.solve(ztz + ridge_lambda * scale * np.eye(3), zty)
        coefs[h - 1] = prev
        n_pairs[h - 1] = count * n_vm

    agg = np.empty(model_h)
    for h in range(1, model_h + 1):
        pred_vm = np.clip(last + design(n_t - 1, h) @ coefs[h - 1], 0.0, 100.0)
        agg[h - 1] = pred_vm[active_last].mean()
    if steps > model_h:
        agg = np.concatenate([agg, np.full(steps - model_h, agg[-1])])
    fc = _finalize(agg[:steps], template_df, "cpu")
    if not return_coefficients:
        return fc
    table = pd.DataFrame(
        {
            "horizon": np.arange(1, model_h + 1),
            "w_last_value": 1.0 - coefs[:, 0] - coefs[:, 1],
            "w_recent_mean": coefs[:, 0],
            "w_expanding_mean": coefs[:, 1],
            "w_fleet_drift": coefs[:, 2],
            "n_training_pairs": n_pairs,
        }
    )
    return fc, table


def global_vm_panel_forecast(
    cpu_train_df: pd.DataFrame,
    template_df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    horizon: int = 24,
    random_state: int = 42,
    target: str = DEFAULT_TARGET,
    method: str = "blend",
    blend_weight: float = 0.5,
) -> pd.Series:
    """Global panel experiment on VM-level training readings.

    ``method``:
      * ``"segment_ridge"`` / ``"segment_hgb"`` -- bottom-up stratum panel model;
      * ``"vm_shrinkage"`` -- VM-level vectorised empirical shrinkage;
      * ``"blend"`` -- ``blend_weight * segment_ridge + (1 - blend_weight) * vm_shrinkage``.

    For ``target="active_vm_count"`` the per-timestamp distinct-VM count is built from
    ``cpu_train_df``; a constant population yields persistence, otherwise a rounded,
    non-negative direct Ridge forecast.
    """
    _check_cpu_frame(cpu_train_df)
    if target == "active_vm_count":
        counts = cpu_train_df.groupby("timestamp")["vm_id"].nunique().sort_index()
        agg = pd.DataFrame({"active_vm_count": counts.to_numpy(dtype=float)})
        return active_vm_count_forecast(agg, template_df, "active_vm_count")
    if target != DEFAULT_TARGET:
        raise ValueError(f"unsupported target for panel model: {target!r}")

    if method == "segment_ridge":
        return segment_panel_forecast(cpu_train_df, template_df, meta_df, horizon, "ridge", random_state=random_state)
    if method == "segment_hgb":
        return segment_panel_forecast(cpu_train_df, template_df, meta_df, horizon, "hgb", random_state=random_state)
    if method == "vm_shrinkage":
        return vm_shrinkage_forecast(cpu_train_df, template_df, horizon=horizon)
    if method == "blend":
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        seg = segment_panel_forecast(cpu_train_df, template_df, meta_df, horizon, "ridge", random_state=random_state)
        vm = vm_shrinkage_forecast(cpu_train_df, template_df, horizon=horizon)
        return _finalize(blend_weight * seg.to_numpy() + (1 - blend_weight) * vm.to_numpy(), template_df, target)
    raise ValueError(f"unknown method: {method!r}")


# ------------------------------------------------------------------- registry
def _seasonal_naive_24(train_df: pd.DataFrame, template_df: pd.DataFrame, target: str) -> pd.Series:
    return seasonal_naive_forecast(train_df, template_df, target, season_length=24, fallback=True)


def _drift_damped(train_df: pd.DataFrame, template_df: pd.DataFrame, target: str) -> pd.Series:
    return drift_forecast(train_df, template_df, target, lookback=24, damping=0.9, clip=_bounds_for(target))


def _ewma_6(train_df: pd.DataFrame, template_df: pd.DataFrame, target: str) -> pd.Series:
    return ewma_forecast(train_df, template_df, target, span=6)


AGGREGATE_MODELS: dict[str, Forecaster] = {
    "persistence": persistence_forecast,
    "seasonal_naive_24": _seasonal_naive_24,
    "moving_average_6": _moving_average_6,
    "moving_average_12": moving_average_forecast,
    "drift_damped_24": _drift_damped,
    "ewma_6": _ewma_6,
    "damped_ets": damped_ets_forecast,
    "direct_ridge": direct_ridge_forecast,
    "direct_hgb": direct_hgb_forecast,
    "hybrid_ensemble": hybrid_ensemble_forecast,
}
"""Name -> ``fit_predict(train_df, template_df, target)`` for aggregate-frame models."""
