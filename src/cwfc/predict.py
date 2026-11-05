"""Generate the holdout ``predictions.csv`` required by ``SUBMISSION.md``.

Usage::

    python -m cwfc.predict --data challenge_data --output predictions.csv

Protocol (leakage-free)
-----------------------
1. Load VM-level CPU readings (``vm_metadata`` is *not* required).
2. Build the full per-timestamp target frame (``build_aggregate_targets``) and add the
   fleet-level aliases ``fleet_avg_cpu`` / ``fleet_p95_max_cpu`` plus
   ``fleet_busy_vm_fraction``. Every target is a per-timestamp cross-sectional
   statistic, so training rows never depend on holdout readings.
3. Split positionally: the final ``horizon`` timestamps (24 on the challenge data) are
   the holdout. Only their ``timestamp`` values are read; their targets are never used.
4. Fit each model on the training rows only and forecast the ``horizon`` steps.
5. Enforce physical invariants, check the submission contract (``cwfc.gates``) and
   write ``timestamp,active_vm_count,aggregate_cpu_utilization``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .baselines import (
    drift_forecast,
    ewma_forecast,
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
)
from .gates import REQUIRED_COLUMNS, assert_valid_predictions
from .io import load_cpu_readings
from .models import (
    AGGREGATE_MODELS,
    active_vm_count_forecast,
    damped_ets_forecast,
    direct_hgb_forecast,
    direct_ridge_forecast,
    hybrid_ensemble_forecast,
)
from .targets import build_aggregate_targets

Forecaster = Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]

# ----------------------------------------------------------------- constants
MAX_HORIZON = 24  # 2 hours at 5-minute cadence
MIN_MODEL_TRAIN_ROWS = 12  # below this the CPU targets fall back to MA(12)
BUSY_CPU_THRESHOLD = 50.0  # a VM is "busy" when its 5-min average CPU >= 50%

CPU_TARGETS: tuple[str, ...] = ("fleet_avg_cpu", "fleet_p95_max_cpu", "fleet_busy_vm_fraction")
COUNT_TARGET = "active_vm_count"
FLEET_TARGETS: tuple[str, ...] = (*CPU_TARGETS, COUNT_TARGET)

TARGET_LABELS: dict[str, str] = {
    "fleet_avg_cpu": "Fleet mean CPU (%)",
    "fleet_p95_max_cpu": "Fleet P95 of 5-min max CPU (%)",
    "fleet_busy_vm_fraction": f"Busy-VM fraction (avg CPU >= {BUSY_CPU_THRESHOLD:.0f}%)",
    "active_vm_count": "Active VM count",
}


# ------------------------------------------------------------- model registry
def _seasonal_naive(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    return seasonal_naive_forecast(train, template, target, season_length=24, fallback=True)


def _moving_average(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    return moving_average_forecast(train, template, target, window=12)


def _drift(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    clip = (0.0, 100.0) if "cpu" in target.lower() else (0.0, np.inf)
    return drift_forecast(train, template, target, lookback=24, damping=0.9, clip=clip)


def _ewma(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    return ewma_forecast(train, template, target, span=6)


BASELINE_MODELS: dict[str, Forecaster] = {
    "persistence": persistence_forecast,
    "seasonal_naive": _seasonal_naive,
    "moving_average": _moving_average,
    "drift": _drift,
    "ewma": _ewma,
}
CANDIDATE_MODELS: dict[str, Forecaster] = {
    "damped_ets": damped_ets_forecast,
    "direct_ridge": direct_ridge_forecast,
    "direct_hgb": direct_hgb_forecast,
    "hybrid_ensemble": hybrid_ensemble_forecast,
}
MODEL_REGISTRY: dict[str, Forecaster] = {**AGGREGATE_MODELS, **BASELINE_MODELS, **CANDIDATE_MODELS}
"""Every aggregate forecaster addressable by ``--model`` (``fit_predict(train, template, target)``)."""

MODEL_LABELS: dict[str, str] = {
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal Naive",
    "moving_average": "MA(12)",
    "drift": "Drift",
    "ewma": "EWMA",
    "damped_ets": "Damped ETS",
    "direct_ridge": "Direct Ridge",
    "direct_hgb": "Direct HGB",
    "hybrid_ensemble": "Hybrid Ensemble",
}


# ----------------------------------------------------------------- targets
def compute_busy_vm_fraction(readings: pd.DataFrame, threshold: float = BUSY_CPU_THRESHOLD) -> pd.Series:
    """Share of VMs (counted once per timestamp) whose 5-min ``avg_cpu >= threshold``."""
    df = readings.dropna(subset=["avg_cpu"])
    per_vm = df.groupby(["timestamp", "vm_id"], sort=True)["avg_cpu"].mean()
    busy = (per_vm >= threshold).astype(float)
    return busy.groupby(level="timestamp").mean().rename("fleet_busy_vm_fraction")


def build_fleet_targets(readings: pd.DataFrame, busy_threshold: float = BUSY_CPU_THRESHOLD) -> pd.DataFrame:
    """Per-timestamp target frame on the full raw timestamp grid.

    Contains every column of :func:`cwfc.targets.build_aggregate_targets` plus
    ``fleet_avg_cpu`` (= ``aggregate_cpu_utilization``), ``fleet_p95_max_cpu``
    (= ``p95_max_cpu_t``) and ``fleet_busy_vm_fraction``. Timestamps whose readings are
    all missing are kept as NaN rows so the holdout grid matches the raw data exactly.
    """
    agg = build_aggregate_targets(readings).set_index("timestamp")
    agg["fleet_busy_vm_fraction"] = compute_busy_vm_fraction(readings, busy_threshold)
    agg["fleet_avg_cpu"] = agg["aggregate_cpu_utilization"]
    agg["fleet_p95_max_cpu"] = agg["p95_max_cpu_t"]
    all_ts = np.sort(readings["timestamp"].astype("int64").unique())
    agg = agg.reindex(pd.Index(all_ts, name="timestamp"))
    out = agg.reset_index()
    out["timestamp"] = out["timestamp"].astype("int64")
    return out


def holdout_horizon(n_timestamps: int, max_horizon: int = MAX_HORIZON) -> int:
    """24 on the challenge data; ``max(1, n // 3)`` for tiny smoke datasets."""
    if n_timestamps < 2:
        raise ValueError("need at least 2 timestamps to forecast a holdout")
    if n_timestamps > max_horizon:
        return max_horizon
    return int(min(max_horizon, max(1, n_timestamps // 3)))


def prepare_training_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps causally inside the training window (ffill, then bfill for leading rows)."""
    out = train_df.copy().reset_index(drop=True)
    num_cols = [c for c in out.columns if c != "timestamp"]
    out[num_cols] = out[num_cols].astype(float).ffill().bfill().fillna(0.0)
    return out


# ---------------------------------------------------------------- forecasting
def enforce_invariants(fc: pd.DataFrame) -> pd.DataFrame:
    """Clip forecasts to physically valid ranges (in place on a copy)."""
    out = fc.copy()
    if "fleet_avg_cpu" in out:
        out["fleet_avg_cpu"] = out["fleet_avg_cpu"].clip(0.0, 100.0)
    if "fleet_p95_max_cpu" in out:
        p95 = out["fleet_p95_max_cpu"].clip(0.0, 100.0)
        if "fleet_avg_cpu" in out:
            p95 = np.maximum(p95, out["fleet_avg_cpu"])
        out["fleet_p95_max_cpu"] = p95
    if "fleet_busy_vm_fraction" in out:
        out["fleet_busy_vm_fraction"] = out["fleet_busy_vm_fraction"].clip(0.0, 1.0)
    if COUNT_TARGET in out:
        out[COUNT_TARGET] = out[COUNT_TARGET].clip(lower=0.0).round()
    return out


def forecast_cpu_target(
    train_df: pd.DataFrame, template: pd.DataFrame, target: str, model_name: str = "hybrid_ensemble"
) -> pd.Series:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_name!r}; choose from {sorted(MODEL_REGISTRY)}")
    if len(train_df) < MIN_MODEL_TRAIN_ROWS:
        return moving_average_forecast(train_df, template, target, window=12)
    return MODEL_REGISTRY[model_name](train_df, template, target)


def forecast_fleet_targets(
    train_df: pd.DataFrame, horizon: int, model_name: str = "hybrid_ensemble"
) -> pd.DataFrame:
    """Forecast all four fleet targets ``horizon`` steps past the end of ``train_df``."""
    train = prepare_training_frame(train_df)
    template = pd.DataFrame(index=pd.RangeIndex(horizon))
    fc = pd.DataFrame(index=template.index)
    for target in CPU_TARGETS:
        fc[target] = forecast_cpu_target(train, template, target, model_name).to_numpy(dtype=float)
    fc[COUNT_TARGET] = active_vm_count_forecast(train, template, COUNT_TARGET).to_numpy(dtype=float)
    return enforce_invariants(fc)


def generate_predictions(
    data_dir: str | Path,
    output_path: str | Path | None = None,
    model_name: str = "hybrid_ensemble",
) -> pd.DataFrame:
    """Build, validate and (optionally) write the 24-step holdout submission.

    Returns the contract frame ``timestamp, active_vm_count, aggregate_cpu_utilization``.
    """
    readings = load_cpu_readings(data_dir)
    agg_df = build_fleet_targets(readings)
    horizon = holdout_horizon(len(agg_df))

    train_df = agg_df.iloc[:-horizon]
    holdout_timestamps = agg_df["timestamp"].iloc[-horizon:].astype("int64").tolist()  # timestamps only

    fc = forecast_fleet_targets(train_df, horizon, model_name)
    pred_df = pd.DataFrame(
        {
            "timestamp": np.asarray(holdout_timestamps, dtype="int64"),
            "active_vm_count": fc[COUNT_TARGET].to_numpy(dtype=float).round().astype("int64"),
            "aggregate_cpu_utilization": fc["fleet_avg_cpu"].to_numpy(dtype=float),
        }
    )[REQUIRED_COLUMNS]

    assert_valid_predictions(pred_df, expected_timestamps=holdout_timestamps, n_holdout=horizon)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out, index=False)
    return pred_df


# ------------------------------------------------------------------------ CLI
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m cwfc.predict", description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="directory containing vm_cpu_readings.csv(.gz)")
    p.add_argument("--output", default="predictions.csv", help="output CSV path")
    p.add_argument(
        "--model",
        default="hybrid_ensemble",
        choices=sorted(MODEL_REGISTRY),
        help="forecaster for the CPU targets (default: hybrid_ensemble)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        pred = generate_predictions(args.data, args.output, model_name=args.model)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {len(pred)} predictions ({pred['timestamp'].iloc[0]}..{pred['timestamp'].iloc[-1]}) "
        f"to {args.output} using model={args.model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
