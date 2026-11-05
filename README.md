# Cloud Workload Forecasting Challenge (`cwfc`)

End-to-end exploratory analysis, leakage-free multi-horizon forecasting, hierarchical reconciliation, conformalized uncertainty calibration, and virtual-core capacity headroom policy for the **Cloud Workload Forecasting Challenge (`data-v1`)** on Microsoft Azure Public Dataset V2 (`10,000` VMs × `134` five-minute intervals = `1,340,000` observations).

---

## Quick Start (Clean Checkout to `predictions.csv`)

### 1. Environment Setup

Requires **Python 3.10+**. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\pip install -e ".[dev]"
# macOS / Linux:
# .venv/bin/pip install -e ".[dev]"
```

*(Alternatively, install via `pip install -r requirements.txt` and `pip install -e .`.)*

### 2. Dataset Location (`challenge_data/`)

Download the `data-v1.tar.gz` release archive from the challenge GitHub Release and extract it into `challenge_data/` so that the directory contains:

```text
challenge_data/
  vm_cpu_readings.csv.gz
  vm_metadata.csv.gz
  SHA256SUMS
```

*(Note: `cwfc.predict` also supports plain `.csv` files and runs cleanly even when the optional `vm_metadata.csv.gz` file is absent, so reviewer smoke tests on minimal synthetic datasets work out of the box.)*

### 3. Generate the Required Holdout Submission (`predictions.csv`)

Run the single prediction command from the repository root:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This command:
1. Loads `vm_cpu_readings.csv.gz` (preserving raw integer-second timestamps `0..39900`).
2. Constructs the per-timestamp cross-sectional target frame (`build_aggregate_targets`).
3. Applies a strict positional `110 / 24` chronological split: **only the first 110 training timestamps (`0..32700`) are passed to model fitting**; the final 24 holdout timestamps (`33000..39900`) are never read except for their `timestamp` index.
4. Forecasts `active_vm_count` and `aggregate_cpu_utilization` using our `hybrid_ensemble` forecaster (`--model` can also select `persistence`, `moving_average`, `seasonal_naive`, `drift`, `ewma`, `damped_ets`, `direct_ridge`, or `direct_hgb`).
5. Runs the post-hoc submission contract gate (`cwfc.gates.assert_valid_predictions`) to verify exact schema (`timestamp,active_vm_count,aggregate_cpu_utilization`), `24` rows, strictly increasing timestamps matching `33000..39900`, finite values, `active_vm_count >= 0`, and `aggregate_cpu_utilization in [0, 100]` before writing `predictions.csv`.

### 4. Reproduce All Evaluation Tables, Data Quality Audit & Figures

To regenerate `reports/data_quality_report.json`, all 7 benchmark CSV tables in `reports/tables/`, and all 6 publication-quality PNG figures in `reports/figures/`:

```bash
python -m cwfc.evaluate --data challenge_data --reports-dir reports
```

### 5. Run the Test Suite (`161` Automated Tests)

```bash
pytest -v
```

The test suite (`tests/`) includes:
- **Holdout Perturbation Anti-Leakage Proofs (`tests/test_no_leakage.py`):** Verifies that multiplying all holdout CPU readings by `10×` (or replacing them with constants) produces **bitwise identical** training features and 24-step holdout predictions across all baselines, aggregate models, VM-level shrinkage, and the `cwfc.predict` CLI.
- **Submission Contract Gates (`tests/test_gates.py`):** 35 tests covering schema, row count, ordering, NaNs/Infs, and physical bounds.
- **Positional Time-Aware Validation (`tests/test_validation.py`):** Verifies expanding rolling-origin splits on integer-second timestamps.
- **Target Construction, Metrics, Intervals & Policy (`tests/test_targets.py`, `tests/test_metrics.py`, `tests/test_predict_smoke.py`).**

---

## Summary of Main Modeling Decisions & Key Findings

1. **Data Integrity & Cohort Selection Bias (`active_vm_count = 10,000`):**
   - `data-v1` is a balanced panel of `10,000` VMs × `134` timestamps (`300s` cadence, `0` missing values, `0` duplicates, `0` `min <= avg <= max` violations).
   - Every VM in `vm_metadata.csv.gz` has `timestamp_vm_created = 0` and `timestamp_vm_deleted >= 40,500` (the very first VM deletion occurs 10 minutes *after* the holdout window ends at `39,900s`). Consequently, `active_vm_count` is deterministically `10,000` across the entire trace. `active_vm_count_forecast` detects constant training populations automatically and falls back to non-negative rounded Direct Ridge when churn is present.
2. **Causal Level-Relative Feature Engineering:**
   - Because the training window spans `9.08` hours (`110` steps) and includes a mid-trace workload surge (`10.15%` peak CPU around hours 5–7), raw-level tree models fail to extrapolate and 24-hour calendar features overfit. We center all autoregressive lags (`5m–60m`), rolling means (`15m–2h`), and EWMA statistics relative to the current origin level $y_{t_0}$, predicting multi-horizon residual displacements $\Delta y_{t_0, h} = y_{t_0+h} - y_{t_0}$.
3. **Top-Down Hybrid Ensemble & Bottom-Up Stratum Reconciliation:**
   - On the 24-step holdout (`33,000..39,900s`), our **`Hybrid Ensemble`** achieves **`MAE = 0.1061%` CPU** (`RMSE = 0.1423%`, `sMAPE = 1.14%`, `MASE = 0.876`), reducing error by **27.7% vs. Persistence** (`MAE = 0.1468%`) and ranking #1 on 4-fold Rolling CV (`CV MAE = 0.2027%`).
   - Disaggregating the fleet into homogeneous `vm_category × virtual_core_bucket` strata (**`Bottom-Up Stratum Panel Ridge`**) further reduces holdout error to **`MAE = 0.0802%` CPU** (**45.4% error reduction vs. Persistence**).
4. **Conformalized Quantile Intervals & vCore Capacity Headroom Policy:**
   - Provisioning strictly to point forecasts causes SLA shortfalls on **62.5% (15/24)** of holdout steps. Using out-of-fold empirical residuals across the fleet's **`28,388` virtual cores**, our recommended **`Horizon-Adaptive P90 Upper Bound`** policy achieves **`0.0%` SLA violations** on holdout while cutting wasted buffer capacity by **52.8%** compared to a static `+10%` safety margin.

---

## Repository Structure

```text
├── README.md                          # Setup, CLI reproduction, and architecture overview
├── pyproject.toml                     # Package metadata and dependencies
├── requirements.txt                   # Pinned runtime and development dependencies
├── predictions.csv                    # Generated 24-row holdout submission (verified by cwfc.gates)
├── src/cwfc/
│   ├── __init__.py                    # Public package exports
│   ├── config.py                      # Frozen ForecastConfig dataclass
│   ├── io.py                          # Resilient loaders for CPU readings and optional VM metadata
│   ├── quality.py                     # Automated data quality & physical invariant auditor
│   ├── targets.py                     # Fleet aggregate & segment target builders + train/holdout splitters
│   ├── validation.py                  # Positional rolling-origin expanding-window splitter
│   ├── metrics.py                     # MAE, RMSE, sMAPE, NMAE, MASE, Pinball Loss, Coverage, Peak MAE
│   ├── gates.py                       # Post-hoc submission contract verifier (verify_predictions)
│   ├── baselines.py                   # Persistence, Seasonal Naive, MA(12), Damped Drift, EWMA
│   ├── features.py                    # Strictly causal level-relative lag/rolling/EWMA feature builder
│   ├── models.py                      # Damped ETS, Direct Ridge, Direct HGB, Hybrid Ensemble, Stratum & VM Shrinkage
│   ├── intervals.py                   # Conformalized out-of-fold empirical prediction intervals
│   ├── policy.py                      # vCore capacity headroom simulator & asymmetric cost evaluator
│   ├── predict.py                     # CLI entry point: python -m cwfc.predict
│   └── evaluate.py                    # CLI entry point: python -m cwfc.evaluate
├── notebooks/
│   ├── 01_data_understanding.ipynb    # Executed notebook: schema, integrity audit, selection-bias survival curve
│   ├── 02_workload_eda.ipynb          # Executed notebook: target dynamics, ACF, VM skewness, segment EDA
│   └── 03_validation_and_errors.ipynb # Executed notebook: rolling CV, holdout benchmark, reconciliation, policy
├── reports/
│   ├── final_report.md                # Comprehensive 5-page technical report (Sections 1–9)
│   ├── data_quality_report.json       # Programmatic audit summary of the 1.34M-row release
│   ├── tables/                        # 7 CSV benchmark tables (CV, holdout, horizon decay, segments, policy)
│   └── figures/                       # 6 publication-quality PNG figures (180 DPI)
└── tests/                             # 161 pytest unit, contract-gate, and holdout-perturbation leakage tests
```
