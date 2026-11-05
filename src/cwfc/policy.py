"""Operational capacity-planning and headroom-policy simulation.

Converts utilisation forecasts into provisioning decisions and scores them with an
asymmetric cost: running short of capacity (SLA risk) is penalised
``underprovision_cost_ratio`` times more than idle headroom.

All utilisation quantities are fleet percentages in ``[0, 100]``; ``*_cores`` columns
rescale percentage points to virtual cores via ``total_cores / 100``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_CORE_WEIGHTS: dict[str, float] = {"2": 2.0, "4": 4.0, "8": 8.0, "24": 24.0, ">24": 32.0}
DEFAULT_CORES_PER_VM = 2.0
RECOMMENDED_POLICY = "Horizon-Adaptive P90 Upper Bound (Recommended)"


def _normalise_bucket(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def vm_core_weights(
    vm_ids: pd.Index | np.ndarray | pd.Series,
    meta_df: pd.DataFrame | None = None,
    core_weights: Mapping[str, float] | None = None,
    default: float = DEFAULT_CORES_PER_VM,
) -> pd.Series:
    """Virtual-core weight per VM (Series indexed by ``vm_id``).

    ``virtual_core_bucket`` values map via ``core_weights`` (default
    ``2->2, 4->4, 8->8, 24->24, >24->32``); VMs without metadata or with an unknown
    bucket get ``default`` cores.
    """
    ids = pd.Index(pd.unique(np.asarray(vm_ids)), name="vm_id")
    mapping = {_normalise_bucket(k): float(v) for k, v in (core_weights or DEFAULT_CORE_WEIGHTS).items()}
    if meta_df is None or "virtual_core_bucket" not in meta_df.columns:
        return pd.Series(default, index=ids, dtype=float, name="cores")
    buckets = meta_df.drop_duplicates("vm_id").set_index("vm_id")["virtual_core_bucket"].reindex(ids)
    cores = buckets.map(lambda b: mapping.get(_normalise_bucket(b), default)).astype(float)
    return cores.rename("cores")


def compute_fleet_core_capacity(
    cpu_df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    core_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Per-timestamp fleet capacity in virtual cores.

    Columns: ``timestamp``, ``active_vm_count``, ``total_provisioned_cores``,
    ``core_weighted_cpu_utilization`` (``sum(c*avg)/sum(c)``), ``demanded_cores_avg``
    (``sum(c*avg/100)``), ``demanded_cores_peak`` (``sum(c*max/100)``, the 5-minute
    burst demand; NaN if ``max_cpu`` is absent) and ``peak_to_avg_ratio``.
    Duplicate ``(timestamp, vm_id)`` rows are counted once.
    """
    required = {"timestamp", "vm_id", "avg_cpu"}
    missing = required - set(cpu_df.columns)
    if missing:
        raise KeyError(f"cpu_df missing columns: {sorted(missing)}")
    cols = ["timestamp", "vm_id", "avg_cpu"] + (["max_cpu"] if "max_cpu" in cpu_df.columns else [])
    df = cpu_df.loc[:, cols].drop_duplicates(["timestamp", "vm_id"]).copy()
    cores = vm_core_weights(df["vm_id"], meta_df, core_weights)
    df["cores"] = df["vm_id"].map(cores).astype(float)
    df["demand_avg"] = df["cores"] * df["avg_cpu"].astype(float) / 100.0
    agg = {"vm_id": "nunique", "cores": "sum", "demand_avg": "sum"}
    if "max_cpu" in df.columns:
        df["demand_peak"] = df["cores"] * df["max_cpu"].astype(float) / 100.0
        agg["demand_peak"] = "sum"
    out = df.groupby("timestamp", sort=True).agg(agg).reset_index()
    out = out.rename(
        columns={
            "vm_id": "active_vm_count",
            "cores": "total_provisioned_cores",
            "demand_avg": "demanded_cores_avg",
            "demand_peak": "demanded_cores_peak",
        }
    )
    if "demanded_cores_peak" not in out.columns:
        out["demanded_cores_peak"] = np.nan
    out["core_weighted_cpu_utilization"] = 100.0 * out["demanded_cores_avg"] / out["total_provisioned_cores"]
    out["peak_to_avg_ratio"] = out["demanded_cores_peak"] / out["demanded_cores_avg"]
    return out[
        [
            "timestamp",
            "active_vm_count",
            "total_provisioned_cores",
            "core_weighted_cpu_utilization",
            "demanded_cores_avg",
            "demanded_cores_peak",
            "peak_to_avg_ratio",
        ]
    ]


def _policy_levels(point: np.ndarray, intervals_df: pd.DataFrame) -> dict[str, np.ndarray]:
    policies: dict[str, np.ndarray] = {
        "Point Forecast (No Headroom)": point,
        "Static Safety Margin (+5% relative)": point * 1.05,
        "Static Safety Margin (+10% relative)": point * 1.10,
    }
    adaptive = [
        ("p80_upper", "Horizon-Adaptive P80 Upper Bound"),
        ("p90_upper", RECOMMENDED_POLICY),
        ("p95_upper", "Horizon-Adaptive P95 Upper Bound (Conservative)"),
    ]
    for col, name in adaptive:
        if col in intervals_df.columns:
            values = intervals_df[col].to_numpy(dtype=float)
            if len(values) != len(point):
                raise ValueError(f"intervals_df[{col!r}] length does not match the forecast")
            policies[name] = values
    return policies


def capacity_policy_trajectories(
    y_true_cpu: np.ndarray,
    point_pred_cpu: np.ndarray,
    intervals_df: pd.DataFrame,
    total_cores: float = 28388.0,
    peak_burst_addon_pct: float = 0.0,
) -> pd.DataFrame:
    """Long-format per-step view (``policy, horizon, demand_pct, provisioned_pct, ...``) for plotting."""
    demand = np.asarray(y_true_cpu, dtype=float).ravel() * (1.0 + peak_burst_addon_pct / 100.0)
    point = np.asarray(point_pred_cpu, dtype=float).ravel()
    if demand.shape != point.shape:
        raise ValueError("y_true_cpu and point_pred_cpu must have the same length")
    frames = []
    for name, level in _policy_levels(point, intervals_df).items():
        provisioned = np.clip(level, 0.0, 100.0)
        frames.append(
            pd.DataFrame(
                {
                    "policy": name,
                    "horizon": np.arange(1, len(point) + 1),
                    "demand_pct": demand,
                    "provisioned_pct": provisioned,
                    "shortfall_pct": np.maximum(0.0, demand - provisioned),
                    "overprovision_pct": np.maximum(0.0, provisioned - demand),
                    "provisioned_cores": provisioned * total_cores / 100.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def evaluate_capacity_policies(
    y_true_cpu: np.ndarray,
    point_pred_cpu: np.ndarray,
    intervals_df: pd.DataFrame,
    total_cores: float = 28388.0,
    underprovision_cost_ratio: float = 5.0,
    peak_burst_addon_pct: float = 0.0,
) -> pd.DataFrame:
    """Score provisioning policies over the forecast horizon.

    Policies: point forecast; +5% / +10% relative static margins; horizon-adaptive
    P80 / P90 (recommended) / P95 upper bounds from ``intervals_df`` (any missing
    column is skipped). Provisioned levels are clipped to ``[0, 100]``.

    ``peak_burst_addon_pct`` inflates realised demand by a relative percentage to
    stress-test against intra-interval bursts (``demand = y_true * (1 + addon/100)``).

    Metrics: ``sla_violation_rate_pct`` (% of steps with demand > provisioned),
    ``mean/max_shortfall_pct``, ``mean_overprovision_pct``, core equivalents and
    ``asymmetric_cost = ratio * mean_shortfall_pct + mean_overprovision_pct``.
    """
    if underprovision_cost_ratio < 0:
        raise ValueError("underprovision_cost_ratio must be non-negative")
    traj = capacity_policy_trajectories(
        y_true_cpu, point_pred_cpu, intervals_df, total_cores=total_cores, peak_burst_addon_pct=peak_burst_addon_pct
    )
    cores_per_pct = total_cores / 100.0
    rows = []
    for name, grp in traj.groupby("policy", sort=False):
        short = grp["shortfall_pct"].to_numpy()
        over = grp["overprovision_pct"].to_numpy()
        violations = grp["demand_pct"].to_numpy() > grp["provisioned_pct"].to_numpy()
        mean_short, mean_over = float(short.mean()), float(over.mean())
        rows.append(
            {
                "policy": name,
                "mean_provisioned_pct": float(grp["provisioned_pct"].mean()),
                "mean_provisioned_cores": float(grp["provisioned_pct"].mean() * cores_per_pct),
                "sla_violations": int(violations.sum()),
                "sla_violation_rate_pct": float(100.0 * violations.mean()),
                "mean_shortfall_pct": mean_short,
                "max_shortfall_pct": float(short.max()),
                "mean_overprovision_pct": mean_over,
                "mean_shortfall_cores": mean_short * cores_per_pct,
                "mean_overprovision_cores": mean_over * cores_per_pct,
                "asymmetric_cost": underprovision_cost_ratio * mean_short + mean_over,
                "is_recommended": name == RECOMMENDED_POLICY,
            }
        )
    return pd.DataFrame(rows)
