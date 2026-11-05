from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _py(v):
    """Convert numpy scalars to plain Python for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _stats(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {k: None for k in ("min", "p25", "median", "mean", "p75", "p95", "max")}
    return {
        "min": _py(s.min()),
        "p25": _py(s.quantile(0.25)),
        "median": _py(s.median()),
        "mean": _py(s.mean()),
        "p75": _py(s.quantile(0.75)),
        "p95": _py(s.quantile(0.95)),
        "max": _py(s.max()),
    }


def compute_data_quality_report(
    cpu_df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    n_holdout: int = 24,
) -> dict:
    report: dict = {}
    ts_unique = np.sort(cpu_df["timestamp"].unique())
    n_ts = int(len(ts_unique))

    report["total_rows"] = int(len(cpu_df))
    report["unique_vms"] = int(cpu_df["vm_id"].nunique())
    report["unique_timestamps"] = n_ts
    report["timestamp_min"] = _py(ts_unique[0]) if n_ts else None
    report["timestamp_max"] = _py(ts_unique[-1]) if n_ts else None

    diffs = pd.Series(np.diff(ts_unique)).value_counts() if n_ts > 1 else pd.Series(dtype=int)
    report["cadence_diffs"] = {str(int(k)): int(v) for k, v in diffs.items()}
    report["is_regular_cadence"] = bool(len(diffs) <= 1)

    report["missing_values_cpu"] = {
        c: int(cpu_df[c].isna().sum()) for c in ("min_cpu", "avg_cpu", "max_cpu")
    }
    report["duplicate_vm_timestamp_pairs"] = int(
        cpu_df.duplicated(subset=["timestamp", "vm_id"]).sum()
    )

    obs_per_vm = cpu_df.groupby("vm_id")["timestamp"].nunique()
    report["obs_per_vm_min"] = _py(obs_per_vm.min()) if len(obs_per_vm) else None
    report["obs_per_vm_max"] = _py(obs_per_vm.max()) if len(obs_per_vm) else None
    report["is_balanced_panel"] = bool(len(obs_per_vm) > 0 and (obs_per_vm == n_ts).all())

    active = cpu_df.groupby("timestamp")["vm_id"].nunique()
    report["active_vm_count_min"] = _py(active.min()) if len(active) else None
    report["active_vm_count_max"] = _py(active.max()) if len(active) else None
    report["is_constant_active_vms"] = bool(len(active) > 0 and active.nunique() == 1)

    report["cpu_bounds"] = {c: _stats(cpu_df[c]) for c in ("min_cpu", "avg_cpu", "max_cpu")}
    report["ordering_violations"] = {
        "min_gt_avg": int((cpu_df["min_cpu"] > cpu_df["avg_cpu"]).sum()),
        "avg_gt_max": int((cpu_df["avg_cpu"] > cpu_df["max_cpu"]).sum()),
        "out_of_0_100_range": int(
            (
                (cpu_df[["min_cpu", "avg_cpu", "max_cpu"]] < 0)
                | (cpu_df[["min_cpu", "avg_cpu", "max_cpu"]] > 100)
            )
            .any(axis=1)
            .sum()
        ),
    }

    report["train_window_timestamps"] = max(n_ts - n_holdout, 0)
    report["holdout_window_timestamps"] = int(n_holdout)

    if meta_df is not None:
        report["metadata_rows"] = int(len(meta_df))
        cpu_vms = set(cpu_df["vm_id"].unique())
        meta_vms = set(meta_df["vm_id"].unique())
        report["vm_id_exact_match"] = bool(cpu_vms == meta_vms)
        report["missing_metadata_vms"] = int(len(cpu_vms - meta_vms))
        for col, key in (
            ("vm_category", "category_counts"),
            ("virtual_core_bucket", "virtual_core_bucket_counts"),
            ("memory_gb_bucket", "memory_gb_bucket_counts"),
        ):
            if col in meta_df.columns:
                report[key] = {str(k): int(v) for k, v in meta_df[col].value_counts().items()}
        for col, prefix in (
            ("timestamp_vm_created", "created"),
            ("timestamp_vm_deleted", "deleted"),
        ):
            if col in meta_df.columns:
                s = pd.to_numeric(meta_df[col], errors="coerce")
                report[f"{prefix}_timestamp_min"] = _py(s.min())
                report[f"{prefix}_timestamp_max"] = _py(s.max())
        ts_min, ts_max = report["timestamp_min"], report["timestamp_max"]
        if "timestamp_vm_created" in meta_df.columns and ts_min is not None:
            s = pd.to_numeric(meta_df["timestamp_vm_created"], errors="coerce")
            report["vms_created_after_start"] = int((s > ts_min).sum())
        if "timestamp_vm_deleted" in meta_df.columns and ts_max is not None:
            s = pd.to_numeric(meta_df["timestamp_vm_deleted"], errors="coerce")
            report["vms_deleted_before_end"] = int((s <= ts_max).sum())

    return report


def save_data_quality_report(report: dict, output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_py)
    return p
