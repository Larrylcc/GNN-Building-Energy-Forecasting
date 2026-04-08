from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from common_preprocessing.config import LOG_TARGET_COLUMN
from XGBoost.config import XGBoostConfig
from XGBoost.metrics import compute_regression_metrics


@dataclass
class TrainResult:
    booster: xgb.Booster
    evals_result: dict
    train_predictions_raw: np.ndarray
    valid_predictions_raw: np.ndarray
    train_metrics: dict[str, float]
    valid_metrics: dict[str, float]
    feature_importance: pd.DataFrame


def make_training_matrix(
    features: pd.DataFrame,
    label: pd.Series | np.ndarray | None = None,
    ref: object | None = None,
):
    matrix_cls = getattr(xgb, "QuantileDMatrix", None)
    feature_names = features.columns.tolist()

    if matrix_cls is not None:
        return matrix_cls(features, label=label, feature_names=feature_names, ref=ref)
    return xgb.DMatrix(features, label=label, feature_names=feature_names)


def _major_version(version_text: str) -> int:
    head = version_text.split(".")[0] if version_text else "0"
    try:
        return int(head)
    except ValueError:
        return 0


def build_xgb_params(config: XGBoostConfig) -> dict:
    params = config.to_training_params()

    if config.use_gpu:
        version_major = _major_version(getattr(xgb, "__version__", "0.0.0"))
        if version_major >= 2:
            params.update(
                {
                    "tree_method": "hist",
                    "device": f"cuda:{config.gpu_device_ordinal}",
                    "sampling_method": "gradient_based",
                }
            )
        else:
            params.update(
                {
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                    "gpu_id": config.gpu_device_ordinal,
                    "sampling_method": "gradient_based",
                }
            )
    else:
        params.update({"tree_method": "hist"})

    return params


def get_prediction_iteration_range(booster: xgb.Booster) -> tuple[int, int]:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return (0, booster.num_boosted_rounds())
    return (0, best_iteration + 1)


def _inverse_transform_predictions(predictions: np.ndarray, training_target_column: str) -> np.ndarray:
    if training_target_column == LOG_TARGET_COLUMN:
        return np.clip(np.expm1(predictions), 0.0, None)
    return np.clip(predictions, 0.0, None)


def extract_feature_importance(booster: xgb.Booster, feature_columns: list[str]) -> pd.DataFrame:
    gain_map = booster.get_score(importance_type="gain")
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": [float(gain_map.get(feature, 0.0)) for feature in feature_columns],
        }
    ).sort_values("importance", ascending=False)


def train_single_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    config: XGBoostConfig,
) -> TrainResult:
    y_train = train_df[config.training_target_column].astype(np.float32)
    y_valid = valid_df[config.training_target_column].astype(np.float32)

    dtrain = make_training_matrix(train_df[feature_columns], label=y_train)
    dvalid = make_training_matrix(valid_df[feature_columns], label=y_valid, ref=dtrain)

    evals_result: dict = {}
    booster = xgb.train(
        params=build_xgb_params(config),
        dtrain=dtrain,
        num_boost_round=config.num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=config.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=config.verbose_eval,
    )

    iteration_range = get_prediction_iteration_range(booster)
    train_pred_model = booster.predict(dtrain, iteration_range=iteration_range)
    valid_pred_model = booster.predict(dvalid, iteration_range=iteration_range)

    train_pred_raw = _inverse_transform_predictions(train_pred_model, config.training_target_column)
    valid_pred_raw = _inverse_transform_predictions(valid_pred_model, config.training_target_column)

    train_true_raw = np.clip(train_df[config.raw_target_column].to_numpy(dtype=np.float64), 0.0, None)
    valid_true_raw = np.clip(valid_df[config.raw_target_column].to_numpy(dtype=np.float64), 0.0, None)

    train_metrics = compute_regression_metrics(train_true_raw, train_pred_raw)
    valid_metrics = compute_regression_metrics(valid_true_raw, valid_pred_raw)

    feature_importance = extract_feature_importance(booster, feature_columns)

    return TrainResult(
        booster=booster,
        evals_result=evals_result,
        train_predictions_raw=train_pred_raw,
        valid_predictions_raw=valid_pred_raw,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        feature_importance=feature_importance,
    )


def predict_test(
    booster: xgb.Booster,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    config: XGBoostConfig,
) -> np.ndarray:
    total_rows = test_df.shape[0]
    predictions_model_scale = np.zeros(total_rows, dtype=np.float32)
    iteration_range = get_prediction_iteration_range(booster)

    for start in range(0, total_rows, config.test_chunk_size):
        stop = min(start + config.test_chunk_size, total_rows)
        chunk_features = test_df.iloc[start:stop][feature_columns]
        dchunk = make_training_matrix(chunk_features)
        chunk_pred = booster.predict(dchunk, iteration_range=iteration_range)
        predictions_model_scale[start:stop] = chunk_pred

    return _inverse_transform_predictions(predictions_model_scale, config.training_target_column)
