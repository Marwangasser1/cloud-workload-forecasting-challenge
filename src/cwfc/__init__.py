"""Cloud Workload Forecasting Challenge (``cwfc``) utilities.

Leakage-free short-horizon forecasting of fleet-level CPU demand on the Azure Public
Dataset V2 ``data-v1`` subset, plus empirical prediction intervals and a capacity
headroom policy simulator.

Entry points::

    python -m cwfc.predict  --data challenge_data --output predictions.csv
    python -m cwfc.evaluate --data challenge_data --reports-dir reports
"""

from __future__ import annotations

__version__ = "0.2.0"

from .baselines import (
    drift_forecast,
    ewma_forecast,
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
)
from .config import DEFAULT_CONFIG, ForecastConfig
from .features import (
    DEFAULT_TARGET,
    build_direct_supervised_dataset,
    build_feature_frame,
    build_origin_feature_vector,
    feature_columns,
    to_relative_features,
)
from .gates import REQUIRED_COLUMNS, assert_valid_predictions, verify_predictions
from .intervals import compute_empirical_intervals, interval_coverage, residuals_by_horizon
from .io import load_cpu_readings, load_vm_metadata
from .metrics import (
    compute_all_metrics,
    mae,
    mase,
    normalized_mae,
    peak_mae,
    pinball_loss,
    rmse,
    smape,
)
from .models import (
    AGGREGATE_MODELS,
    active_vm_count_forecast,
    assign_vm_segments,
    damped_ets_forecast,
    direct_hgb_forecast,
    direct_ridge_forecast,
    global_vm_panel_forecast,
    hybrid_ensemble_forecast,
    hybrid_ensemble_weights,
    segment_panel_forecast,
    vm_shrinkage_forecast,
)
from .policy import (
    RECOMMENDED_POLICY,
    capacity_policy_trajectories,
    compute_fleet_core_capacity,
    evaluate_capacity_policies,
    vm_core_weights,
)
from .quality import compute_data_quality_report, save_data_quality_report
from .targets import (
    build_aggregate_targets,
    build_segment_targets,
    split_raw_readings_train_holdout,
    split_train_holdout,
)
from .validation import TimeSplit, rolling_origin_splits

__all__ = [
    "__version__",
    # config
    "DEFAULT_CONFIG",
    "ForecastConfig",
    # io / quality
    "load_cpu_readings",
    "load_vm_metadata",
    "compute_data_quality_report",
    "save_data_quality_report",
    # targets / validation
    "build_aggregate_targets",
    "build_segment_targets",
    "split_train_holdout",
    "split_raw_readings_train_holdout",
    "TimeSplit",
    "rolling_origin_splits",
    # metrics
    "mae",
    "rmse",
    "smape",
    "normalized_mae",
    "mase",
    "pinball_loss",
    "peak_mae",
    "compute_all_metrics",
    # gates
    "REQUIRED_COLUMNS",
    "verify_predictions",
    "assert_valid_predictions",
    # baselines
    "persistence_forecast",
    "seasonal_naive_forecast",
    "moving_average_forecast",
    "drift_forecast",
    "ewma_forecast",
    # features
    "DEFAULT_TARGET",
    "build_feature_frame",
    "build_origin_feature_vector",
    "build_direct_supervised_dataset",
    "feature_columns",
    "to_relative_features",
    # models
    "AGGREGATE_MODELS",
    "damped_ets_forecast",
    "direct_ridge_forecast",
    "direct_hgb_forecast",
    "hybrid_ensemble_forecast",
    "hybrid_ensemble_weights",
    "active_vm_count_forecast",
    "assign_vm_segments",
    "segment_panel_forecast",
    "vm_shrinkage_forecast",
    "global_vm_panel_forecast",
    # intervals / policy
    "residuals_by_horizon",
    "compute_empirical_intervals",
    "interval_coverage",
    "RECOMMENDED_POLICY",
    "vm_core_weights",
    "compute_fleet_core_capacity",
    "capacity_policy_trajectories",
    "evaluate_capacity_policies",
]
