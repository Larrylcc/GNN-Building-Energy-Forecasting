from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


def _to_non_negative(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, None)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_true_safe = _to_non_negative(y_true)
    y_pred_safe = _to_non_negative(y_pred)

    mse = float(mean_squared_error(y_true_safe, y_pred_safe))
    mae = float(mean_absolute_error(y_true_safe, y_pred_safe))
    r2 = float(r2_score(y_true_safe, y_pred_safe))
    rmse = float(np.sqrt(mse))
    rmsle = float(np.sqrt(mean_squared_log_error(y_true_safe, y_pred_safe)))

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
        "rmsle": rmsle,
    }


def build_metrics_by_split(split_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for split_name, metrics in split_metrics.items():
        row = {"split": split_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
