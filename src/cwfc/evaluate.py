"""Comprehensive evaluation, backtesting, and visual generation CLI (`cwfc.evaluate`).

Usage::

    python -m cwfc.evaluate --data challenge_data --reports-dir reports

Produces:
- ``reports/data_quality_report.json``
- ``reports/tables/aggregate_cv_metrics.csv``
- ``reports/tables/aggregate_holdout_metrics.csv``
- ``reports/tables/horizon_decay_metrics.csv``
- ``reports/tables/segment_holdout_metrics.csv``
- ``reports/tables/reconciliation_comparison.csv``
- ``reports/tables/interval_coverage.csv``
- ``reports/tables/interval_coverage_and_policy.csv``
- ``reports/figures/fig1_fleet_overview.png``
- ``reports/figures/fig2_vm_heterogeneity_and_segments.png``
- ``reports/figures/fig3_cv_and_holdout_model_comparison.png``
- ``reports/figures/fig4_holdout_forecasts_with_intervals.png``
- ``reports/figures/fig5_error_by_horizon_and_residuals.png``
- ``reports/figures/fig6_capacity_headroom_policy.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import persistence_forecast
from .intervals import compute_empirical_intervals, interval_coverage as eval_interval_coverage, residuals_by_horizon
from .io import load_cpu_readings, load_vm_metadata
from .metrics import compute_all_metrics, mae, pinball_loss, rmse
from .models import (
    active_vm_count_forecast,
    global_vm_panel_forecast,
    hybrid_ensemble_forecast,
    segment_panel_forecast,
    vm_shrinkage_forecast,
)
from .policy import (
    RECOMMENDED_POLICY,
    capacity_policy_trajectories,
    compute_fleet_core_capacity,
    evaluate_capacity_policies,
)
from .predict import (
    BASELINE_MODELS,
    CANDIDATE_MODELS,
    COUNT_TARGET,
    CPU_TARGETS,
    MODEL_LABELS,
    MODEL_REGISTRY,
    TARGET_LABELS,
    build_fleet_targets,
    prepare_training_frame,
)
from .quality import compute_data_quality_report, save_data_quality_report
from .targets import build_segment_targets, split_raw_readings_train_holdout, split_train_holdout
from .validation import rolling_origin_splits

ALL_MODELS_ORDER = [
    "persistence",
    "seasonal_naive",
    "moving_average",
    "drift",
    "ewma",
    "damped_ets",
    "direct_ridge",
    "direct_hgb",
    "hybrid_ensemble",
]


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def evaluate_rolling_cv(train_df: pd.DataFrame, horizon: int = 24, min_train_size: int = 48, step: int = 12) -> pd.DataFrame:
    """Evaluate all baseline and candidate models across 4 rolling-origin folds on the training window."""
    splits = rolling_origin_splits(train_df, horizon=horizon, min_train_size=min_train_size, step=step)
    rows: list[dict[str, object]] = []

    for target in (*CPU_TARGETS, COUNT_TARGET):
        models_to_run = (
            ["persistence", "moving_average", "direct_ridge", "hybrid_ensemble"]
            if target == COUNT_TARGET
            else ALL_MODELS_ORDER
        )
        for model_name in models_to_run:
            fold_metrics: list[dict[str, float]] = []
            for fold_idx, split in enumerate(splits):
                f_train = prepare_training_frame(train_df.iloc[: split.train_end_idx])
                f_test = train_df.iloc[split.val_start_idx : split.val_end_idx].reset_index(drop=True)
                y_true = f_test[target].to_numpy(dtype=float)
                y_train = f_train[target].to_numpy(dtype=float)
                base_pred = persistence_forecast(f_train, f_test, target).to_numpy(dtype=float)

                if target == COUNT_TARGET and model_name == "hybrid_ensemble":
                    y_pred = active_vm_count_forecast(f_train, f_test, target).to_numpy(dtype=float)
                else:
                    y_pred = MODEL_REGISTRY[model_name](f_train, f_test, target).to_numpy(dtype=float)

                peak_thr = float(np.quantile(y_train, 0.75))
                m = compute_all_metrics(y_true, y_pred, base_pred, y_train=y_train, peak_threshold=peak_thr)
                m["fold"] = fold_idx + 1
                fold_metrics.append(m)

            df_f = pd.DataFrame(fold_metrics)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "family": "Baseline" if model_name in BASELINE_MODELS else "Candidate",
                    "n_folds": len(splits),
                    "cv_mae_mean": float(df_f["mae"].mean()),
                    "cv_mae_std": float(df_f["mae"].std(ddof=1)),
                    "cv_rmse_mean": float(df_f["rmse"].mean()),
                    "cv_rmse_std": float(df_f["rmse"].std(ddof=1)),
                    "cv_smape_mean": float(df_f["smape"].mean()),
                    "cv_nmae_mean": float(df_f["normalized_mae"].mean()),
                    "cv_mase_mean": float(df_f["mase"].mean()),
                    "cv_peak_mae_mean": float(df_f["peak_mae"].mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_holdout(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Evaluate all baseline and candidate models strictly on the 24-step holdout window."""
    train = prepare_training_frame(train_df)
    test = holdout_df.reset_index(drop=True)
    preds_store: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for target in (*CPU_TARGETS, COUNT_TARGET):
        y_true = test[target].to_numpy(dtype=float)
        y_train = train[target].to_numpy(dtype=float)
        base_pred = persistence_forecast(train, test, target).to_numpy(dtype=float)
        peak_thr = float(np.quantile(y_train, 0.75))

        models_to_run = (
            ["persistence", "moving_average", "direct_ridge", "hybrid_ensemble"]
            if target == COUNT_TARGET
            else ALL_MODELS_ORDER
        )
        for model_name in models_to_run:
            if target == COUNT_TARGET and model_name == "hybrid_ensemble":
                y_pred = active_vm_count_forecast(train, test, target).to_numpy(dtype=float)
            else:
                y_pred = MODEL_REGISTRY[model_name](train, test, target).to_numpy(dtype=float)
            preds_store[(target, model_name)] = y_pred
            m = compute_all_metrics(y_true, y_pred, base_pred, y_train=y_train, peak_threshold=peak_thr)
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "model_label": MODEL_LABELS.get(model_name, model_name),
                    "family": "Baseline" if model_name in BASELINE_MODELS else "Candidate",
                    "holdout_mae": float(m["mae"]),
                    "holdout_rmse": float(m["rmse"]),
                    "holdout_smape": float(m["smape"]),
                    "holdout_nmae": float(m["normalized_mae"]),
                    "holdout_mase": float(m["mase"]),
                    "holdout_peak_mae": float(m["peak_mae"]),
                }
            )
    return pd.DataFrame(rows), preds_store


def evaluate_horizon_decay(
    holdout_df: pd.DataFrame, preds_store: dict[tuple[str, str], np.ndarray]
) -> pd.DataFrame:
    """Compute MAE and RMSE across short (1-6), medium (7-12), and long (13-24) forecast horizons."""
    buckets = [
        ("Short (h=1..6, 5-30m)", slice(0, 6)),
        ("Medium (h=7..12, 35-60m)", slice(6, 12)),
        ("Long (h=13..24, 65-120m)", slice(12, 24)),
        ("Full Holdout (h=1..24, 5-120m)", slice(0, 24)),
    ]
    focus_models = ["persistence", "moving_average", "damped_ets", "direct_ridge", "direct_hgb", "hybrid_ensemble"]
    rows: list[dict[str, object]] = []
    for target in CPU_TARGETS:
        y_true = holdout_df[target].to_numpy(dtype=float)
        for model_name in focus_models:
            y_pred = preds_store[(target, model_name)]
            for b_label, sl in buckets:
                rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "horizon_bucket": b_label,
                        "mae": mae(y_true[sl], y_pred[sl]),
                        "rmse": rmse(y_true[sl], y_pred[sl]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_segments(
    readings: pd.DataFrame, metadata: pd.DataFrame | None, n_holdout: int = 24
) -> pd.DataFrame:
    """Evaluate holdout forecasting performance across VM workload categories and virtual core buckets."""
    if metadata is None:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    focus_models = ["persistence", "moving_average", "damped_ets", "direct_ridge", "hybrid_ensemble"]

    for seg_col in ("vm_category", "virtual_core_bucket"):
        seg_df = build_segment_targets(readings, metadata, segment_col=seg_col)
        for seg_val, grp in seg_df.groupby(seg_col, sort=True):
            grp_sorted = grp.sort_values("timestamp").reset_index(drop=True)
            train_s, hold_s = split_train_holdout(grp_sorted, n_holdout=n_holdout)
            y_true = hold_s["aggregate_cpu_utilization"].to_numpy(dtype=float)
            y_train = train_s["aggregate_cpu_utilization"].to_numpy(dtype=float)
            base_pred = persistence_forecast(train_s, hold_s, "aggregate_cpu_utilization").to_numpy(dtype=float)
            vm_count = int(hold_s["active_vm_count"].iloc[0])

            for model_name in focus_models:
                y_pred = MODEL_REGISTRY[model_name](train_s, hold_s, "aggregate_cpu_utilization").to_numpy(dtype=float)
                m = compute_all_metrics(y_true, y_pred, base_pred, y_train=y_train)
                rows.append(
                    {
                        "segment_type": seg_col,
                        "segment_value": str(seg_val),
                        "vm_count": vm_count,
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "holdout_mae": float(m["mae"]),
                        "holdout_rmse": float(m["rmse"]),
                        "holdout_nmae": float(m["normalized_mae"]),
                        "holdout_mase": float(m["mase"]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_reconciliation(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    cpu_train_df: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compare Top-Down Aggregate vs Segment Bottom-Up vs VM-Level Empirical Shrinkage on fleet_avg_cpu."""
    train = prepare_training_frame(train_df)
    test = holdout_df.reset_index(drop=True)
    y_true = test["fleet_avg_cpu"].to_numpy(dtype=float)
    y_train = train["fleet_avg_cpu"].to_numpy(dtype=float)
    base_pred = persistence_forecast(train, test, "fleet_avg_cpu").to_numpy(dtype=float)

    paradigms: list[tuple[str, str, np.ndarray]] = [
        (
            "Top-Down Aggregate (Persistence Baseline)",
            "Aggregate Top-Down",
            base_pred,
        ),
        (
            "Top-Down Aggregate (Direct Ridge)",
            "Aggregate Top-Down",
            MODEL_REGISTRY["direct_ridge"](train, test, "fleet_avg_cpu").to_numpy(dtype=float),
        ),
        (
            "Top-Down Aggregate (Hybrid Ensemble)",
            "Aggregate Top-Down",
            MODEL_REGISTRY["hybrid_ensemble"](train, test, "fleet_avg_cpu").to_numpy(dtype=float),
        ),
        (
            "Bottom-Up Stratum Panel (Ridge, Category x Core Bucket)",
            "Segment Bottom-Up",
            segment_panel_forecast(cpu_train_df, test, metadata, horizon=len(test), model="ridge").to_numpy(dtype=float),
        ),
        (
            "Bottom-Up Stratum Panel (HGB, Category x Core Bucket)",
            "Segment Bottom-Up",
            segment_panel_forecast(cpu_train_df, test, metadata, horizon=len(test), model="hgb").to_numpy(dtype=float),
        ),
        (
            "VM-Level Empirical-Bayes Shrinkage (10,000 VMs)",
            "VM-Level Bottom-Up",
            vm_shrinkage_forecast(cpu_train_df, test, horizon=len(test)).to_numpy(dtype=float),
        ),
        (
            "Global VM Panel Blend (50% Stratum Ridge + 50% VM Shrinkage)",
            "Hierarchical Reconciliation",
            global_vm_panel_forecast(cpu_train_df, test, metadata, horizon=len(test), method="blend").to_numpy(dtype=float),
        ),
    ]

    rows: list[dict[str, object]] = []
    pred_map: dict[str, np.ndarray] = {}
    for label, paradigm, y_pred in paradigms:
        pred_map[label] = y_pred
        m = compute_all_metrics(y_true, y_pred, base_pred, y_train=y_train)
        rows.append(
            {
                "approach": label,
                "paradigm": paradigm,
                "holdout_mae": float(m["mae"]),
                "holdout_rmse": float(m["rmse"]),
                "holdout_smape": float(m["smape"]),
                "holdout_nmae": float(m["normalized_mae"]),
                "holdout_mase": float(m["mase"]),
            }
        )
    return pd.DataFrame(rows), pred_map


def evaluate_intervals_and_policy(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    readings: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute calibrated out-of-fold empirical prediction intervals and evaluate vCore capacity policies."""
    dense_splits = rolling_origin_splits(train_df, horizon=24, min_train_size=48, step=4)
    intervals_by_target: dict[str, pd.DataFrame] = {}
    cov_rows: list[pd.DataFrame] = []

    train_clean = prepare_training_frame(train_df)
    test_clean = holdout_df.reset_index(drop=True)

    for target in CPU_TARGETS:
        y_true_folds, y_pred_folds = [], []
        for sp in dense_splits:
            ft = prepare_training_frame(train_df.iloc[: sp.train_end_idx])
            fv = train_df.iloc[sp.val_start_idx : sp.val_end_idx].reset_index(drop=True)
            y_true_folds.append(fv[target].to_numpy(dtype=float))
            y_pred_folds.append(hybrid_ensemble_forecast(ft, fv, target).to_numpy(dtype=float))

        res_book = residuals_by_horizon(y_true_folds, y_pred_folds)
        point_fc = hybrid_ensemble_forecast(train_clean, test_clean, target).to_numpy(dtype=float)
        clip_bounds = (0.0, 1.0) if target == "fleet_busy_vm_fraction" else (0.0, 100.0)
        iv_df = compute_empirical_intervals(
            point_fc,
            res_book,
            quantiles=(0.80, 0.90, 0.95),
            two_sided_levels=(0.80, 0.90),
            clip=clip_bounds,
            pool_neighbors=2,
            finite_sample=True,
            monotone=True,
        )
        intervals_by_target[target] = iv_df
        y_hold = test_clean[target].to_numpy(dtype=float)
        cov = eval_interval_coverage(y_hold, iv_df)
        cov.insert(0, "target", target)
        cov["pinball_loss"] = [
            pinball_loss(y_hold, iv_df[b].to_numpy(dtype=float), nom) if b.endswith("_upper") else np.nan
            for b, nom in zip(cov["bound"], cov["nominal"])
        ]
        cov_rows.append(cov)

    cov_df = pd.concat(cov_rows, ignore_index=True)

    cap_df = compute_fleet_core_capacity(readings, metadata)
    total_cores = float(cap_df["total_provisioned_cores"].iloc[0])
    y_hold_cpu = test_clean["fleet_avg_cpu"].to_numpy(dtype=float)
    point_cpu = intervals_by_target["fleet_avg_cpu"]["point_forecast"].to_numpy(dtype=float)

    policy_standard = evaluate_capacity_policies(
        y_hold_cpu,
        point_cpu,
        intervals_by_target["fleet_avg_cpu"],
        total_cores=total_cores,
        underprovision_cost_ratio=5.0,
        peak_burst_addon_pct=0.0,
    )
    policy_standard.insert(0, "scenario", "Standard 5-Min Mean Demand")

    policy_burst = evaluate_capacity_policies(
        y_hold_cpu,
        point_cpu,
        intervals_by_target["fleet_avg_cpu"],
        total_cores=total_cores,
        underprovision_cost_ratio=5.0,
        peak_burst_addon_pct=5.0,
    )
    policy_burst.insert(0, "scenario", "Burst Stress-Test (+5% Intra-Window Surge)")

    policy_df = pd.concat([policy_standard, policy_burst], ignore_index=True)
    return cov_df, policy_df, intervals_by_target, cap_df


def generate_all_figures(
    agg_df: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    readings: pd.DataFrame,
    metadata: pd.DataFrame | None,
    cv_df: pd.DataFrame,
    holdout_metrics_df: pd.DataFrame,
    preds_store: dict[tuple[str, str], np.ndarray],
    reconcil_df: pd.DataFrame,
    intervals_by_target: dict[str, pd.DataFrame],
    policy_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate all 6 publication-quality PNG figures in `reports/figures/`."""
    _set_plot_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    hours_all = agg_df["timestamp"].to_numpy(dtype=float) / 3600.0
    split_hour = float(train_df["timestamp"].iloc[-1]) / 3600.0

    # --- Figure 1: 4-Panel Fleet Overview ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    panels = [
        ("fleet_avg_cpu", "Fleet Mean CPU Utilization (%)", "#1f77b4"),
        ("fleet_p95_max_cpu", "Fleet P95 of 5-Min Max CPU (%)", "#d62728"),
        ("fleet_busy_vm_fraction", "Busy-VM Fraction (avg CPU >= 50%)", "#ff7f0e"),
        ("active_vm_count", "Active VM Count (Selection-Biased Cohort)", "#2ca02c"),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        y = agg_df[col].to_numpy(dtype=float)
        ax.plot(hours_all[: len(train_df)], y[: len(train_df)], color=color, lw=1.8, label="Train (110 steps, 0..32,700s)")
        ax.plot(
            hours_all[len(train_df) - 1 :],
            y[len(train_df) - 1 :],
            color="#111111",
            lw=2.0,
            ls="-",
            label="Holdout Actual (24 steps, 33,000..39,900s)",
        )
        ax.axvline(split_hour, color="#444444", ls="--", lw=1.2)
        ax.axvspan(split_hour, hours_all[-1], color="#f0f0f0", alpha=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(col)
        ax.legend(loc="best", frameon=True)
    axes[1, 0].set_xlabel("Elapsed Trace Time (Hours from t=0)")
    axes[1, 1].set_xlabel("Elapsed Trace Time (Hours from t=0)")
    fig.suptitle(
        "Figure 1: Cloud Fleet Workload Targets Across the 134 Five-Minute Observation Window (data-v1)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1 = figures_dir / "fig1_fleet_overview.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(p1)

    # --- Figure 2: VM Heterogeneity, Segments & Post-Window Survival ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0))
    vm_means = readings.groupby("vm_id")[["avg_cpu", "max_cpu"]].mean()
    ax = axes[0, 0]
    ax.hist(vm_means["avg_cpu"], bins=50, alpha=0.65, color="#1f77b4", label="Per-VM Mean avg_cpu (%)", density=True)
    ax.hist(vm_means["max_cpu"], bins=50, alpha=0.50, color="#d62728", label="Per-VM Mean max_cpu (%)", density=True)
    ax.axvline(vm_means["avg_cpu"].mean(), color="#1f77b4", ls="--", lw=1.4, label=f"Mean avg_cpu = {vm_means['avg_cpu'].mean():.2f}%")
    ax.axvline(vm_means["max_cpu"].mean(), color="#d62728", ls="--", lw=1.4, label=f"Mean max_cpu = {vm_means['max_cpu'].mean():.2f}%")
    ax.set_title("(a) Cross-Sectional VM Heterogeneity (10,000 VMs)", fontweight="bold")
    ax.set_xlabel("CPU Utilization (%)")
    ax.set_ylabel("Density")
    ax.legend()

    if metadata is not None:
        cat_df = build_segment_targets(readings, metadata, "vm_category")
        ax = axes[0, 1]
        palette_cat = {"Interactive": "#d62728", "Delay-insensitive": "#1f77b4", "Unknown": "#7f7f7f"}
        for cat, grp in cat_df.groupby("vm_category"):
            grp = grp.sort_values("timestamp")
            n_vms = int(grp["active_vm_count"].iloc[0])
            ax.plot(
                grp["timestamp"] / 3600.0,
                grp["aggregate_cpu_utilization"],
                lw=1.7,
                color=palette_cat.get(str(cat), "#2ca02c"),
                label=f"{cat} (n={n_vms:,})",
            )
        ax.axvline(split_hour, color="#444444", ls="--", lw=1.1)
        ax.set_title("(b) Mean CPU Trajectory by Workload Category", fontweight="bold")
        ax.set_xlabel("Elapsed Trace Time (Hours)")
        ax.set_ylabel("Mean CPU (%)")
        ax.legend()

        core_df = build_segment_targets(readings, metadata, "virtual_core_bucket")
        ax = axes[1, 0]
        for bkt in ("2", "4", "8", "24", ">24"):
            grp = core_df[core_df["virtual_core_bucket"] == bkt].sort_values("timestamp")
            if not grp.empty:
                n_vms = int(grp["active_vm_count"].iloc[0])
                ax.plot(grp["timestamp"] / 3600.0, grp["aggregate_cpu_utilization"], lw=1.6, label=f"{bkt} vCores (n={n_vms:,})")
        ax.axvline(split_hour, color="#444444", ls="--", lw=1.1)
        ax.set_title("(c) Mean CPU Trajectory by Virtual Core Bucket", fontweight="bold")
        ax.set_xlabel("Elapsed Trace Time (Hours)")
        ax.set_ylabel("Mean CPU (%)")
        ax.legend(ncol=2)

        ax = axes[1, 1]
        del_times = np.sort(metadata["timestamp_vm_deleted"].to_numpy(dtype=float)) / 3600.0
        eval_grid_h = np.linspace(0.0, 48.0, 400)
        alive_counts = [(del_times > h).sum() for h in eval_grid_h]
        ax.plot(eval_grid_h, alive_counts, color="#2ca02c", lw=2.2, label="Surviving VMs in Cohort")
        ax.axvspan(0.0, hours_all[-1], color="#1f77b4", alpha=0.18, label="Observed Trace (0..11.08h, 10,000 alive)")
        ax.axvline(40500.0 / 3600.0, color="#d62728", ls="--", lw=1.4, label="First Deletion at t=40,500s (11.25h)")
        ax.set_title("(d) Cohort Selection Bias: Post-Window VM Survival Curve", fontweight="bold")
        ax.set_xlabel("Time Since Trace Start (Hours, First 48h Shown)")
        ax.set_ylabel("Active VMs Remaining")
        ax.set_ylim(8500, 10150)
        ax.legend(loc="lower left")

    fig.suptitle(
        "Figure 2: VM-Level Skewness, Workload Stratum Dynamics, and Cohort Survival Mechanism",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = figures_dir / "fig2_vm_heterogeneity_and_segments.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(p2)

    # --- Figure 3: CV vs Holdout Model Comparison + Reconciliation ---
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    for idx, target in enumerate(CPU_TARGETS):
        ax = axes.ravel()[idx]
        cv_sub = cv_df[cv_df["target"] == target].set_index("model").loc[list(ALL_MODELS_ORDER)]
        ho_sub = holdout_metrics_df[holdout_metrics_df["target"] == target].set_index("model").loc[list(ALL_MODELS_ORDER)]
        y_pos = np.arange(len(ALL_MODELS_ORDER))
        labels = [MODEL_LABELS[m] for m in ALL_MODELS_ORDER]
        ax.barh(y_pos - 0.18, cv_sub["cv_mae_mean"], height=0.35, color="#6baed6", label="Rolling CV Mean MAE (4 folds)")
        ax.barh(y_pos + 0.18, ho_sub["holdout_mae"], height=0.35, color="#08519c", label="Holdout MAE (24 steps)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(f"({chr(97 + idx)}) {TARGET_LABELS[target]}", fontweight="bold")
        ax.set_xlabel("Mean Absolute Error (MAE)")
        ax.legend(loc="lower right")

    ax = axes[1, 1]
    rec_sorted = reconcil_df.sort_values("holdout_mae", ascending=True)
    y_pos = np.arange(len(rec_sorted))
    colors = ["#2ca02c" if "Hybrid" in a or "Shrinkage" in a else "#3182bd" for a in rec_sorted["approach"]]
    ax.barh(y_pos, rec_sorted["holdout_mae"], color=colors, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rec_sorted["approach"], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("(d) Top-Down vs Bottom-Up Reconciliation (Fleet Mean CPU)", fontweight="bold")
    ax.set_xlabel("Holdout MAE (% CPU)")

    fig.suptitle(
        "Figure 3: Rolling Cross-Validation vs Holdout Accuracy Across Models and Aggregation Paradigms",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p3 = figures_dir / "fig3_cv_and_holdout_model_comparison.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(p3)

    # --- Figure 4: Holdout Forecasts with Calibrated Prediction Intervals ---
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.2), sharex=True)
    tail_len = 36
    train_tail = train_df.iloc[-tail_len:]
    t_tail_h = train_tail["timestamp"].to_numpy(dtype=float) / 3600.0
    t_hold_h = holdout_df["timestamp"].to_numpy(dtype=float) / 3600.0

    for ax, target in zip(axes, CPU_TARGETS):
        iv = intervals_by_target[target]
        ax.plot(t_tail_h, train_tail[target], color="#444444", lw=1.6, label="Training History (last 3 hours)")
        ax.plot(t_hold_h, holdout_df[target], color="#000000", lw=2.2, marker="o", ms=3.5, label="Holdout Actual")
        ax.plot(t_hold_h, preds_store[(target, "persistence")], color="#7f7f7f", ls="--", lw=1.4, label="Persistence Baseline")
        ax.plot(t_hold_h, preds_store[(target, "moving_average")], color="#ff7f0e", ls="-.", lw=1.5, label="MA(12) Baseline")
        ax.plot(t_hold_h, preds_store[(target, "direct_ridge")], color="#2ca02c", ls=":", lw=1.7, label="Direct Ridge")
        ax.plot(t_hold_h, iv["point_forecast"], color="#1f77b4", lw=2.2, label="Hybrid Ensemble (Selected)")
        ax.fill_between(t_hold_h, iv["pi80_lower"], iv["pi80_upper"], color="#1f77b4", alpha=0.22, label="Calibrated 80% PI")
        ax.fill_between(t_hold_h, iv["pi90_lower"], iv["pi90_upper"], color="#1f77b4", alpha=0.11, label="Calibrated 90% PI")
        ax.axvline(split_hour, color="#333333", ls="--", lw=1.2)
        ax.set_title(TARGET_LABELS[target], fontweight="bold")
        ax.set_ylabel(target)
        ax.legend(loc="upper left", ncol=4, frameon=True)
    axes[-1].set_xlabel("Elapsed Trace Time (Hours from t=0)")
    fig.suptitle(
        "Figure 4: 24-Step Holdout Forecasts (33,000..39,900s) and Conformalized Empirical Prediction Intervals",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p4 = figures_dir / "fig4_holdout_forecasts_with_intervals.png"
    fig.savefig(p4)
    plt.close(fig)
    saved.append(p4)

    # --- Figure 5: Error by Horizon & Residual Diagnostics ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    h_steps = np.arange(1, 25)
    y_true_cpu = holdout_df["fleet_avg_cpu"].to_numpy(dtype=float)
    ax = axes[0]
    for m_name, color, ls in [
        ("persistence", "#7f7f7f", "--"),
        ("moving_average", "#ff7f0e", "-."),
        ("direct_ridge", "#2ca02c", ":"),
        ("hybrid_ensemble", "#1f77b4", "-"),
    ]:
        abs_err = np.abs(y_true_cpu - preds_store[("fleet_avg_cpu", m_name)])
        smooth_err = pd.Series(abs_err).rolling(3, min_periods=1, center=True).mean()
        ax.plot(h_steps, smooth_err, lw=2.0, color=color, ls=ls, marker="o", ms=3, label=MODEL_LABELS[m_name])
    ax.axvline(6.5, color="#888888", ls=":", lw=1.0)
    ax.axvline(12.5, color="#888888", ls=":", lw=1.0)
    ax.set_title("(a) Holdout Absolute Error by Forecast Horizon Step (3-Step Smoothed)", fontweight="bold")
    ax.set_xlabel("Forecast Horizon Step h (5-min intervals, 1..24 = 5..120 min)")
    ax.set_ylabel("Absolute Error (% CPU)")
    ax.legend()

    ax = axes[1]
    res_ens = y_true_cpu - preds_store[("fleet_avg_cpu", "hybrid_ensemble")]
    res_pers = y_true_cpu - preds_store[("fleet_avg_cpu", "persistence")]
    ax.bar(h_steps - 0.18, res_pers, width=0.35, color="#bdbdbd", label="Persistence Residual (Actual - Pred)")
    ax.bar(h_steps + 0.18, res_ens, width=0.35, color="#1f77b4", label="Hybrid Ensemble Residual (Actual - Pred)")
    ax.axhline(0.0, color="#111111", lw=1.2)
    ax.set_title("(b) Holdout Residual Trajectory by Horizon Step (Fleet Mean CPU)", fontweight="bold")
    ax.set_xlabel("Forecast Horizon Step h (1..24)")
    ax.set_ylabel("Residual Error y_true - y_pred (% CPU)")
    ax.legend()

    fig.suptitle(
        "Figure 5: Horizon Degradation Profile and Holdout Residual Diagnostics",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p5 = figures_dir / "fig5_error_by_horizon_and_residuals.png"
    fig.savefig(p5)
    plt.close(fig)
    saved.append(p5)

    # --- Figure 6: Capacity Headroom Policy & Cost-Risk Frontier ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    total_cores = float(cap_df["total_provisioned_cores"].iloc[0])
    iv_cpu = intervals_by_target["fleet_avg_cpu"]
    point_cpu = iv_cpu["point_forecast"].to_numpy(dtype=float)
    traj = capacity_policy_trajectories(y_true_cpu, point_cpu, iv_cpu, total_cores=total_cores)

    ax = axes[0]
    demand_cores = y_true_cpu * total_cores / 100.0
    ax.plot(t_hold_h, demand_cores, color="#000000", lw=2.3, marker="o", ms=4, label="Realised Fleet Core Demand")
    for pol_name, color, ls in [
        ("Point Forecast (No Headroom)", "#d62728", "--"),
        ("Static Safety Margin (+10% relative)", "#ff7f0e", "-."),
        (RECOMMENDED_POLICY, "#2ca02c", "-"),
        ("Horizon-Adaptive P95 Upper Bound (Conservative)", "#1f77b4", ":"),
    ]:
        sub = traj[traj["policy"] == pol_name]
        if not sub.empty:
            ax.plot(t_hold_h, sub["provisioned_cores"], color=color, ls=ls, lw=2.0, label=pol_name)
    ax.fill_between(
        t_hold_h,
        demand_cores,
        traj[traj["policy"] == RECOMMENDED_POLICY]["provisioned_cores"],
        color="#2ca02c",
        alpha=0.15,
        label="P90 Adaptive Safety Buffer",
    )
    ax.set_title(f"(a) Holdout vCore Provisioning Trajectory (Fleet Capacity = {int(total_cores):,} vCores)", fontweight="bold")
    ax.set_xlabel("Holdout Time (Hours from t=0)")
    ax.set_ylabel("Virtual Cores (vCores)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    pol_std = policy_df[policy_df["scenario"] == "Standard 5-Min Mean Demand"].copy()
    y_pos = np.arange(len(pol_std))
    bar_colors = ["#2ca02c" if rec else "#6baed6" for rec in pol_std["is_recommended"]]
    ax.barh(y_pos, pol_std["asymmetric_cost"], color=bar_colors, height=0.55)
    for i, row in enumerate(pol_std.itertuples()):
        ax.text(
            row.asymmetric_cost + 0.08,
            i,
            f"Cost={row.asymmetric_cost:.2f} | SLA viol={row.sla_violation_rate_pct:.1f}%",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pol_std["policy"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, float(pol_std["asymmetric_cost"].max()) * 1.42)
    ax.set_title("(b) Asymmetric Operational Loss (5x Shortfall Penalty + 1x Idle Buffer)", fontweight="bold")
    ax.set_xlabel("Asymmetric Cost (% of Fleet Capacity)")

    fig.suptitle(
        "Figure 6: Operational Capacity Planning & Headroom Policy Simulation Across 28,388 Fleet vCores",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p6 = figures_dir / "fig6_capacity_headroom_policy.png"
    fig.savefig(p6)
    plt.close(fig)
    saved.append(p6)

    return saved


def run_evaluation(data_dir: str | Path, reports_dir: str | Path = "reports") -> dict[str, Path]:
    """Run full data quality audit, CV/holdout evaluation, and save all tables and figures."""
    data_path = Path(data_dir)
    rep_path = Path(reports_dir)
    tables_dir = rep_path / "tables"
    figures_dir = rep_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    readings = load_cpu_readings(data_path)
    metadata = load_vm_metadata(data_path)

    dq_report = compute_data_quality_report(readings, metadata)
    dq_path = save_data_quality_report(dq_report, rep_path / "data_quality_report.json")

    agg_df = build_fleet_targets(readings)
    train_df, holdout_df = split_train_holdout(agg_df, n_holdout=24)
    cpu_train_df, _, _ = split_raw_readings_train_holdout(readings, n_holdout=24)

    cv_df = evaluate_rolling_cv(train_df, horizon=24, min_train_size=48, step=12)
    holdout_metrics_df, preds_store = evaluate_holdout(train_df, holdout_df)
    horizon_df = evaluate_horizon_decay(holdout_df, preds_store)
    segment_df = evaluate_segments(readings, metadata, n_holdout=24)
    reconcil_df, _ = evaluate_reconciliation(train_df, holdout_df, cpu_train_df, metadata)
    cov_df, policy_df, intervals_by_target, cap_df = evaluate_intervals_and_policy(
        train_df, holdout_df, readings, metadata
    )

    t1 = tables_dir / "aggregate_cv_metrics.csv"
    t2 = tables_dir / "aggregate_holdout_metrics.csv"
    t3 = tables_dir / "horizon_decay_metrics.csv"
    t4 = tables_dir / "segment_holdout_metrics.csv"
    t5 = tables_dir / "reconciliation_comparison.csv"
    t6_cov = tables_dir / "interval_coverage.csv"
    t6_pol = tables_dir / "interval_coverage_and_policy.csv"

    cv_df.to_csv(t1, index=False)
    holdout_metrics_df.to_csv(t2, index=False)
    horizon_df.to_csv(t3, index=False)
    segment_df.to_csv(t4, index=False)
    reconcil_df.to_csv(t5, index=False)
    cov_df.to_csv(t6_cov, index=False)
    policy_df.to_csv(t6_pol, index=False)

    generate_all_figures(
        agg_df=agg_df,
        train_df=train_df,
        holdout_df=holdout_df,
        readings=readings,
        metadata=metadata,
        cv_df=cv_df,
        holdout_metrics_df=holdout_metrics_df,
        preds_store=preds_store,
        reconcil_df=reconcil_df,
        intervals_by_target=intervals_by_target,
        policy_df=policy_df,
        cap_df=cap_df,
        figures_dir=figures_dir,
    )

    return {
        "data_quality": dq_path,
        "aggregate_cv_metrics": t1,
        "aggregate_holdout_metrics": t2,
        "horizon_decay_metrics": t3,
        "segment_holdout_metrics": t4,
        "reconciliation_comparison": t5,
        "interval_coverage": t6_cov,
        "interval_coverage_and_policy": t6_pol,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cwfc.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data", help="Directory containing vm_cpu_readings.csv(.gz)")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for JSON, tables, and figures")
    args = parser.parse_args(argv)
    paths = run_evaluation(args.data, args.reports_dir)
    print(f"Completed evaluation. Generated {len(paths)} artifacts in {args.reports_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
