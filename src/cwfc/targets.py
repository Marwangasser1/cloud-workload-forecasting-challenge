from __future__ import annotations

import numpy as np
import pandas as pd


def _p(q: float):
    def f(s: pd.Series) -> float:
        return float(np.quantile(s.to_numpy(dtype=float), q))

    f.__name__ = f"p{int(q * 100)}"
    return f


def _clean_readings(cpu_df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN avg_cpu and dedupe (timestamp, vm_id) so each VM counts once per t."""
    df = cpu_df.dropna(subset=["avg_cpu"])
    if df.duplicated(subset=["timestamp", "vm_id"]).any():
        df = df.groupby(["timestamp", "vm_id"], as_index=False, sort=False).agg(
            min_cpu=("min_cpu", "mean"),
            max_cpu=("max_cpu", "mean"),
            avg_cpu=("avg_cpu", "mean"),
        )
    return df


def build_aggregate_targets(cpu_df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_readings(cpu_df).copy()
    df["_burst"] = df["max_cpu"] - df["min_cpu"]
    g = df.groupby("timestamp", sort=True)
    out = pd.DataFrame(
        {
            "active_vm_count": g["vm_id"].nunique(),
            "aggregate_cpu_utilization": g["avg_cpu"].mean().clip(0.0, 100.0),
            "std_cpu": g["avg_cpu"].std(ddof=1).fillna(0.0),
            "median_cpu": g["avg_cpu"].median(),
            "p90_cpu": g["avg_cpu"].quantile(0.90),
            "p95_cpu": g["avg_cpu"].quantile(0.95),
            "mean_max_cpu": g["max_cpu"].mean(),
            "p95_max_cpu_t": g["max_cpu"].quantile(0.95),
            "mean_min_cpu": g["min_cpu"].mean(),
            "mean_burst_spread": g["_burst"].mean(),
        }
    )
    out["active_vm_count"] = out["active_vm_count"].astype("int64")
    out = out.sort_index().reset_index()
    out["timestamp"] = out["timestamp"].astype("int64")
    return out


def build_segment_targets(
    cpu_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    segment_col: str = "vm_category",
) -> pd.DataFrame:
    df = _clean_readings(cpu_df)
    seg = meta_df[["vm_id", segment_col]].drop_duplicates(subset=["vm_id"])
    df = df.merge(seg, on="vm_id", how="left")
    df[segment_col] = df[segment_col].fillna("Unknown").astype(str)
    g = df.groupby(["timestamp", segment_col], sort=True)
    out = pd.DataFrame(
        {
            "active_vm_count": g["vm_id"].nunique().astype("int64"),
            "aggregate_cpu_utilization": g["avg_cpu"].mean().clip(0.0, 100.0),
            "p95_max_cpu_t": g["max_cpu"].quantile(0.95),
        }
    )
    return out.reset_index()


def split_train_holdout(
    series_df: pd.DataFrame, n_holdout: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_holdout <= 0:
        raise ValueError("n_holdout must be positive")
    if len(series_df) <= n_holdout:
        raise ValueError(
            f"series length {len(series_df)} must exceed n_holdout={n_holdout}"
        )
    return (
        series_df.iloc[:-n_holdout].copy().reset_index(drop=True),
        series_df.iloc[-n_holdout:].copy().reset_index(drop=True),
    )


def split_raw_readings_train_holdout(
    cpu_df: pd.DataFrame, n_holdout: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if n_holdout <= 0:
        raise ValueError("n_holdout must be positive")
    ts = np.sort(cpu_df["timestamp"].unique())
    if len(ts) <= n_holdout:
        raise ValueError(
            f"{len(ts)} unique timestamps must exceed n_holdout={n_holdout}"
        )
    holdout_ts = ts[-n_holdout:]
    train_ts = ts[:-n_holdout]
    return (
        cpu_df[cpu_df["timestamp"].isin(train_ts)].copy(),
        cpu_df[cpu_df["timestamp"].isin(holdout_ts)].copy(),
        holdout_ts,
    )
