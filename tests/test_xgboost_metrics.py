from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from XGBoost.metrics import compute_regression_metrics


def test_metrics_values_on_simple_arrays() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y_pred = np.array([0.0, 1.0, 1.5, 2.5], dtype=np.float64)

    metrics = compute_regression_metrics(y_true, y_pred)

    assert math.isclose(metrics["mse"], 0.125, rel_tol=1e-12)
    assert math.isclose(metrics["mae"], 0.25, rel_tol=1e-12)
    assert math.isclose(metrics["rmse"], math.sqrt(0.125), rel_tol=1e-12)
    assert -1.0 <= metrics["r2"] <= 1.0
    assert metrics["rmsle"] >= 0.0


def test_rmsle_handles_negative_predictions_by_clipping() -> None:
    y_true = np.array([0.0, 1.0, 10.0], dtype=np.float64)
    y_pred = np.array([-3.0, -1.0, 8.0], dtype=np.float64)

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["rmsle"] >= 0.0
