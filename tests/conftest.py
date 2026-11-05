"""Shared, fast, deterministic synthetic fixtures for the CWFC test-suite."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_VMS = 50
N_TIMESTAMPS = 80
CADENCE = 300
N_BUSY_VMS = 10
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="session")
def synthetic_readings_df() -> pd.DataFrame:
    """50 VMs x 80 five-minute timestamps (0, 300, ..., 23700).

    Guarantees ``0 <= min_cpu <= avg_cpu <= max_cpu <= 100`` and that the first
    ``N_BUSY_VMS`` VMs are "busy" (``max_cpu > 60`` for all their readings).
    """
    rng = np.random.default_rng(1234)
    timestamps = np.arange(N_TIMESTAMPS, dtype="int64") * CADENCE
    t = np.arange(N_TIMESTAMPS, dtype=float)

    base = rng.uniform(8.0, 40.0, size=N_VMS)
    base[:N_BUSY_VMS] = rng.uniform(55.0, 75.0, size=N_BUSY_VMS)  # busy VMs
    amplitude = rng.uniform(2.0, 8.0, size=N_VMS)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=N_VMS)
    fleet_wave = 4.0 * np.sin(2.0 * np.pi * t / 24.0)  # shared diurnal-ish pattern

    avg = (
        base[:, None]
        + amplitude[:, None] * np.sin(2.0 * np.pi * t[None, :] / 36.0 + phase[:, None])
        + fleet_wave[None, :]
        + rng.normal(0.0, 1.5, size=(N_VMS, N_TIMESTAMPS))
    )
    avg = np.clip(avg, 1.0, 95.0)
    min_cpu = np.clip(avg - rng.uniform(0.5, 8.0, size=avg.shape), 0.0, None)
    max_cpu = np.clip(avg + rng.uniform(0.5, 12.0, size=avg.shape), None, 100.0)
    max_cpu[:N_BUSY_VMS] = np.maximum(max_cpu[:N_BUSY_VMS], np.maximum(avg[:N_BUSY_VMS], 62.0))
    max_cpu = np.minimum(np.maximum(max_cpu, avg), 100.0)

    df = pd.DataFrame(
        {
            "timestamp": np.tile(timestamps, N_VMS),
            "vm_id": np.repeat([f"vm_{i:03d}" for i in range(N_VMS)], N_TIMESTAMPS),
            "min_cpu": min_cpu.ravel(),
            "max_cpu": max_cpu.ravel(),
            "avg_cpu": avg.ravel(),
        }
    )
    df["timestamp"] = df["timestamp"].astype("int64")
    assert (df["min_cpu"] <= df["avg_cpu"]).all() and (df["avg_cpu"] <= df["max_cpu"]).all()
    return df.sort_values(["timestamp", "vm_id"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_metadata_df() -> pd.DataFrame:
    categories = ["Interactive", "Delay-insensitive", "Unknown"]
    core_buckets = ["2", "4", "8", "24", ">24"]
    memory_buckets = ["4", "8", "16", "32", ">64"]
    return pd.DataFrame(
        {
            "vm_id": [f"vm_{i:03d}" for i in range(N_VMS)],
            "vm_category": [categories[i % 3] for i in range(N_VMS)],
            "virtual_core_bucket": [core_buckets[i % 5] for i in range(N_VMS)],
            "memory_gb_bucket": [memory_buckets[i % 5] for i in range(N_VMS)],
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": 40500,
        }
    )


@pytest.fixture
def synthetic_data_dir(tmp_path, synthetic_readings_df, synthetic_metadata_df) -> Path:
    d = tmp_path / "data_with_metadata"
    d.mkdir()
    synthetic_readings_df.to_csv(d / "vm_cpu_readings.csv.gz", index=False)
    synthetic_metadata_df.to_csv(d / "vm_metadata.csv.gz", index=False)
    return d


@pytest.fixture
def synthetic_data_dir_no_metadata(tmp_path, synthetic_readings_df) -> Path:
    d = tmp_path / "data_no_metadata"
    d.mkdir()
    synthetic_readings_df.to_csv(d / "vm_cpu_readings.csv.gz", index=False)
    return d


def predict_module_available() -> bool:
    """True once ``cwfc.predict`` exists (it is written in parallel by Wave 2A)."""
    try:
        return importlib.util.find_spec("cwfc.predict") is not None
    except ModuleNotFoundError:
        return False


@pytest.fixture
def run_predict():
    """Run ``python -m cwfc.predict`` as a subprocess; skips if the module is missing."""
    if not predict_module_available():
        pytest.skip("cwfc.predict is not available yet")

    def _run(data_dir: Path, output: Path) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_DIR), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(SRC_DIR)]
        )
        return subprocess.run(
            [sys.executable, "-m", "cwfc.predict", "--data", str(data_dir), "--output", str(output)],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )

    return _run
