from __future__ import annotations

from dataclasses import dataclass, field

_DEFAULT_CORE_WEIGHTS: dict[str, float] = {
    "2": 2.0,
    "4": 4.0,
    "8": 8.0,
    "24": 24.0,
    ">24": 32.0,
}


@dataclass(frozen=True)
class ForecastConfig:
    """Central, dependency-free configuration for the forecasting pipeline."""

    cadence_seconds: int = 300
    n_holdout: int = 24  # 2 hours at 5-minute cadence
    min_train_size: int = 48  # 4 hours minimum history for rolling-origin CV
    cv_step: int = 1
    season_lengths: tuple[int, ...] = (12, 24)
    ma_windows: tuple[int, ...] = (3, 6, 12, 24)
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    rolling_windows: tuple[int, ...] = (3, 6, 12, 24)
    quantiles: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    peak_quantile: float = 0.90
    underprovision_cost_ratio: float = 5.0
    random_seed: int = 42
    core_bucket_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_CORE_WEIGHTS)
    )

    def core_weight(self, bucket: object) -> float:
        """Weight for a virtual-core bucket string; 2.0 for unknown buckets."""
        return float(self.core_bucket_weights.get(str(bucket), 2.0))


DEFAULT_CONFIG = ForecastConfig()
