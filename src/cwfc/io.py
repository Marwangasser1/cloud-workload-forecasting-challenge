from __future__ import annotations

from pathlib import Path

import pandas as pd

CPU_REQUIRED_COLUMNS = {"timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"}
META_REQUIRED_COLUMNS = {"vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"}
_META_STR_COLUMNS = ("vm_category", "virtual_core_bucket", "memory_gb_bucket")


def _find_file(data_dir: str | Path, stem: str) -> Path | None:
    d = Path(data_dir)
    for suffix in (".csv.gz", ".csv"):
        p = d / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def load_cpu_readings(data_dir: str | Path) -> pd.DataFrame:
    path = _find_file(data_dir, "vm_cpu_readings")
    if path is None:
        raise FileNotFoundError(
            f"vm_cpu_readings.csv.gz / vm_cpu_readings.csv not found in {data_dir}"
        )
    df = pd.read_csv(path)
    missing = CPU_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"vm_cpu_readings missing required columns: {sorted(missing)}")
    # Timestamps are integer seconds; never convert to datetime.
    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ("min_cpu", "max_cpu", "avg_cpu"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def load_vm_metadata(data_dir: str | Path) -> pd.DataFrame | None:
    path = _find_file(data_dir, "vm_metadata")
    if path is None:
        return None
    df = pd.read_csv(path)
    missing = META_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"vm_metadata missing required columns: {sorted(missing)}")
    for c in _META_STR_COLUMNS:
        df[c] = df[c].astype(str)
    return df
