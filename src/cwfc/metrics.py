from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)


def mase(y_true, y_pred, y_train) -> float:
    """MAE scaled by in-sample one-step naive MAE on y_train."""
    train = np.asarray(y_train, dtype=float)
    err = mae(y_true, y_pred)
    if train.size < 2:
        return float("nan")
    scale = float(np.mean(np.abs(np.diff(train))))
    if scale == 0:
        return 0.0 if err == 0 else float("inf")
    return float(err / scale)


def pinball_loss(y_true, y_quantile, quantile: float) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_quantile, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_quantile must have the same shape")
    d = a - b
    return float(np.mean(np.maximum(quantile * d, (quantile - 1.0) * d)))


def interval_coverage(y_true, lower, upper) -> float:
    a = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((a >= lo) & (a <= hi)))


def peak_mae(y_true, y_pred, peak_threshold: float) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = a >= peak_threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def compute_all_metrics(
    y_true,
    y_pred,
    baseline_pred,
    y_train=None,
    peak_threshold: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "normalized_mae": normalized_mae(y_true, y_pred, baseline_pred),
        "baseline_mae": mae(y_true, baseline_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train)
    if peak_threshold is not None:
        out["peak_mae"] = peak_mae(y_true, y_pred, peak_threshold)
    return out
