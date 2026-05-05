from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from data_preprocess.data_preprocess import CATEGORICAL_COLS, DATA_ROOT, NUMERICAL_COLS, PREPROCESSED_DATA_DIR  # noqa: E402


SOURCE_DATA_PATH = PREPROCESSED_DATA_DIR / "screened_preprocessed_train.csv"
PROCESSED_DATA_PATH = PREPROCESSED_DATA_DIR / "xgboost.csv"

OUTPUT_DIR = Path(r"F:\Desktop\Final\workspace\XGBoost\xgboost_baseline_log1p-minmax_outputs")
MODEL_PATH = OUTPUT_DIR / "xgboost_final_model.json"
MODEL_PARAMS_PATH = OUTPUT_DIR / "xgboost_model_params.json"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "xgboost_feature_importance.csv"
SPLIT_SUMMARY_PATH = OUTPUT_DIR / "xgboost_time_series_split.csv"
VALIDATION_METRICS_PATH = OUTPUT_DIR / "xgboost_validation_metrics.json"
TEST_METRICS_PATH = OUTPUT_DIR / "xgboost_test_metrics.json"
RUN_SUMMARY_PATH = OUTPUT_DIR / "xgboost_run_summary.json"

SEED = 42
EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 3000
GPU_DEVICE_ORDINAL = 0
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def fit_target_minmax(target: np.ndarray) -> tuple[float, float]:
    target_min = float(np.min(target))
    target_max = float(np.max(target))
    return target_min, target_max


def transform_target_minmax(
    target: pd.Series | np.ndarray,
    target_min: float,
    target_max: float,
) -> np.ndarray:
    target_array = np.asarray(target, dtype=np.float32)
    scale = target_max - target_min
    if scale <= 0:
        return np.zeros_like(target_array, dtype=np.float32)
    return ((target_array - target_min) / scale).astype(np.float32)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_ratio = np.divide(
        2.0 * np.abs(y_pred - y_true),
        denominator,
        out=np.zeros_like(y_true, dtype=np.float32),
        where=denominator != 0,
    )
    return float(np.mean(safe_ratio) * 100.0)


def load_source_data(
    source_data_path: Path = SOURCE_DATA_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    if not source_data_path.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {source_data_path}.")

    train_df = pd.read_csv(source_data_path, parse_dates=["timestamp"])
    train_df = train_df.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    feature_cols = [column for column in train_df.columns if column not in {"timestamp", "meter_reading"}]
    return train_df, feature_cols


def preprocess_target_and_save(
    source_data_path: Path = SOURCE_DATA_PATH,
    processed_data_path: Path = PROCESSED_DATA_PATH,
) -> dict[str, object]:
    train_df, feature_cols = load_source_data(source_data_path=source_data_path)
    meter_reading = pd.to_numeric(train_df["meter_reading"], errors="coerce").astype(np.float32)

    if meter_reading.isna().any():
        raise ValueError("The meter_reading column contains missing values and cannot be transformed with log1p.")
    if (meter_reading < 0).any():
        raise ValueError("The meter_reading column contains negative values and cannot be transformed with log1p.")

    meter_reading_log1p = np.log1p(meter_reading.to_numpy(dtype=np.float32))
    target_log1p_min, target_log1p_max = fit_target_minmax(meter_reading_log1p)
    meter_reading_log1p_minmax = transform_target_minmax(
        meter_reading_log1p,
        target_min=target_log1p_min,
        target_max=target_log1p_max,
    )

    processed_df = train_df.copy()
    processed_df["meter_reading"] = meter_reading_log1p_minmax

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_data_path, index=False)

    return {
        "train_df": processed_df,
        "feature_cols": feature_cols,
        "target_log1p_min": target_log1p_min,
        "target_log1p_max": target_log1p_max,
        "processed_data_path": str(processed_data_path),
    }


def make_training_matrix(
    features: pd.DataFrame,
    feature_cols: list[str],
    label: pd.Series | np.ndarray | None = None,
    ref: object | None = None,
):
    matrix_cls = getattr(xgb, "QuantileDMatrix", None)
    if matrix_cls is not None:
        return matrix_cls(features, label=label, feature_names=feature_cols, ref=ref)
    return xgb.DMatrix(features, label=label, feature_names=feature_cols)


def build_xgb_params(seed: int = SEED) -> dict:
    version_text = getattr(xgb, "__version__", "0.0.0")
    major_version = int(version_text.split(".")[0]) if version_text else 0

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 10,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "seed": seed,
        "nthread": 0,
        "verbosity": 1,
    }

    if major_version >= 2:
        params.update(
            {
                "tree_method": "hist",
                "device": f"cuda:{GPU_DEVICE_ORDINAL}",
                "sampling_method": "gradient_based",
            }
        )
    else:
        params.update(
            {
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "gpu_id": GPU_DEVICE_ORDINAL,
                "sampling_method": "gradient_based",
            }
        )

    return params


def assert_gpu_training_ready(feature_cols: list[str]) -> None:
    gpu_params = build_xgb_params(seed=SEED)
    probe_x = pd.DataFrame(np.random.rand(512, len(feature_cols)).astype(np.float32), columns=feature_cols)
    probe_y = np.random.rand(512).astype(np.float32)
    dtrain = make_training_matrix(probe_x, feature_cols=feature_cols, label=probe_y)

    booster = xgb.train(
        params=gpu_params,
        dtrain=dtrain,
        num_boost_round=2,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )
    _ = booster.predict(dtrain)
    print("XGBoost GPU sanity check passed.")

    del probe_x, probe_y, dtrain, booster
    gc.collect()


def get_prediction_iteration_range(booster: xgb.Booster) -> tuple[int, int]:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return (0, booster.num_boosted_rounds())
    return (0, best_iteration + 1)


def extract_feature_importance(booster: xgb.Booster, feature_cols: list[str], split_name: str) -> pd.DataFrame:
    gain_map = booster.get_score(importance_type="gain")

    def normalize_importance(feature: str) -> float:
        raw_value = gain_map.get(feature, 0.0)
        if isinstance(raw_value, list):
            return float(raw_value[0]) if raw_value else 0.0
        return float(raw_value)

    return pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": [normalize_importance(feature) for feature in feature_cols],
            "split": split_name,
        }
    )


def build_time_block_boundaries(
    timestamps: pd.Series,
    n_blocks: int = 10,
) -> tuple[pd.Index, np.ndarray]:
    unique_timestamps = pd.Index(timestamps.dropna().drop_duplicates().sort_values())
    if unique_timestamps.empty:
        raise ValueError("No valid timestamps were found in the training data.")
    if len(unique_timestamps) < n_blocks:
        raise ValueError(f"At least {n_blocks} unique timestamps are required, but only {len(unique_timestamps)} were found.")

    block_boundaries = np.rint(np.linspace(0, len(unique_timestamps), num=n_blocks + 1)).astype(int)
    block_boundaries[0] = 0
    block_boundaries[-1] = len(unique_timestamps)

    for idx in range(1, len(block_boundaries)):
        if block_boundaries[idx] <= block_boundaries[idx - 1]:
            raise ValueError("Failed to build strictly increasing timestamp block boundaries.")

    return unique_timestamps, block_boundaries


def split_train_valid_test_by_time(
    train_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("Train, valid, and test ratios must sum to 1.0.")

    unique_timestamps, block_boundaries = build_time_block_boundaries(train_df["timestamp"], n_blocks=10)
    train_end_timestamp = unique_timestamps[block_boundaries[8] - 1]
    valid_start_timestamp = unique_timestamps[block_boundaries[8]]
    valid_end_timestamp = unique_timestamps[block_boundaries[9] - 1]
    test_start_timestamp = unique_timestamps[block_boundaries[9]]

    train_split_df = train_df[train_df["timestamp"] <= train_end_timestamp].copy()
    valid_split_df = train_df[
        (train_df["timestamp"] >= valid_start_timestamp) & (train_df["timestamp"] <= valid_end_timestamp)
    ].copy()
    test_split_df = train_df[train_df["timestamp"] >= test_start_timestamp].copy()

    if train_split_df.empty or valid_split_df.empty or test_split_df.empty:
        raise ValueError("Time-based split produced an empty train, validation, or test set.")

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "start_timestamp": str(train_split_df["timestamp"].min()),
                "end_timestamp": str(train_split_df["timestamp"].max()),
                "row_count": int(train_split_df.shape[0]),
            },
            {
                "split": "valid",
                "start_timestamp": str(valid_split_df["timestamp"].min()),
                "end_timestamp": str(valid_split_df["timestamp"].max()),
                "row_count": int(valid_split_df.shape[0]),
            },
            {
                "split": "test",
                "start_timestamp": str(test_split_df["timestamp"].min()),
                "end_timestamp": str(test_split_df["timestamp"].max()),
                "row_count": int(test_split_df.shape[0]),
            },
        ]
    )

    return (
        train_split_df.reset_index(drop=True),
        valid_split_df.reset_index(drop=True),
        test_split_df.reset_index(drop=True),
        split_summary,
    )


def train_xgboost_single_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = SEED,
) -> dict[str, object]:
    params = build_xgb_params(seed=seed)
    x_train = train_df[feature_cols]
    x_valid = valid_df[feature_cols]
    y_train = cast(pd.Series, train_df["meter_reading"].astype(np.float32))
    y_valid = cast(pd.Series, valid_df["meter_reading"].astype(np.float32))

    dtrain = make_training_matrix(x_train, feature_cols=feature_cols, label=y_train)
    dvalid = make_training_matrix(x_valid, feature_cols=feature_cols, label=y_valid, ref=dtrain)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100,
    )

    valid_pred = booster.predict(dvalid, iteration_range=get_prediction_iteration_range(booster))
    valid_rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
    best_round = get_prediction_iteration_range(booster)[1]
    feature_importance = extract_feature_importance(booster, feature_cols, "train_valid")

    print(f"Validation RMSE on log1p + min-max target = {valid_rmse:.6f}")

    del x_train, x_valid, y_train, y_valid, dtrain, dvalid, valid_pred
    gc.collect()

    return {
        "model": booster,
        "validation_rmse": valid_rmse,
        "feature_importances": feature_importance,
        "best_round": best_round,
    }


def evaluate_split(
    booster: xgb.Booster,
    split_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    target = cast(pd.Series, split_df["meter_reading"].astype(np.float32))
    dsplit = make_training_matrix(split_df[feature_cols], feature_cols=feature_cols, label=target)
    split_pred = booster.predict(dsplit, iteration_range=get_prediction_iteration_range(booster))

    split_mse = float(mean_squared_error(target, split_pred))
    split_mae = float(mean_absolute_error(target, split_pred))
    split_r2 = float(r2_score(target, split_pred))
    split_rmse = float(np.sqrt(split_mse))
    split_smape = compute_smape(target.to_numpy(dtype=np.float32), split_pred.astype(np.float32))

    del dsplit, target, split_pred
    gc.collect()

    return {
        "mse": split_mse,
        "mae": split_mae,
        "r2": split_r2,
        "rmse": split_rmse,
        "smape": split_smape,
    }


def save_feature_importance(feature_importances: pd.DataFrame, output_path: Path = FEATURE_IMPORTANCE_PATH) -> Path:
    importance_summary = (
        feature_importances.groupby("feature", as_index=False)
        .agg(importance=("importance", "mean"))
        .sort_values(by="importance", ascending=False)
    )
    importance_summary.to_csv(output_path, index=False)
    return output_path


def save_json(data: dict[str, object], output_path: Path) -> Path:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def save_split_summary(split_summary: pd.DataFrame, output_path: Path = SPLIT_SUMMARY_PATH) -> Path:
    split_summary.to_csv(output_path, index=False)
    return output_path


def main(
    data_root: Path = DATA_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_result = preprocess_target_and_save()
    processed_df = cast(pd.DataFrame, preprocessing_result["train_df"])
    feature_cols = cast(list[str], preprocessing_result["feature_cols"])

    assert_gpu_training_ready(feature_cols=feature_cols)
    train_split_df, valid_split_df, test_df, split_summary = split_train_valid_test_by_time(train_df=processed_df)

    training_results = train_xgboost_single_split(
        train_df=train_split_df,
        valid_df=valid_split_df,
        feature_cols=feature_cols,
    )
    final_model = cast(xgb.Booster, training_results["model"])
    validation_metrics = evaluate_split(
        booster=final_model,
        split_df=valid_split_df,
        feature_cols=feature_cols,
    )
    test_metrics = evaluate_split(
        booster=final_model,
        split_df=test_df,
        feature_cols=feature_cols,
    )

    final_model.save_model(MODEL_PATH)
    model_params = build_xgb_params(seed=SEED)
    model_params_artifact = {
        "xgb_params": model_params,
        "best_iteration_round_count": training_results["best_round"],
        "feature_count": len(feature_cols),
        "xgboost_feature_cols": feature_cols,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "num_boost_round_upper_bound": NUM_BOOST_ROUND,
        "source_data_path": str(SOURCE_DATA_PATH),
        "processed_data_path": str(preprocessing_result["processed_data_path"]),
        "target_preprocess": "log1p_then_global_minmax",
        "target_log1p_min": preprocessing_result["target_log1p_min"],
        "target_log1p_max": preprocessing_result["target_log1p_max"],
    }

    model_params_path = save_json(model_params_artifact, MODEL_PARAMS_PATH)
    validation_metrics_path = save_json(validation_metrics, VALIDATION_METRICS_PATH)
    test_metrics_path = save_json(test_metrics, TEST_METRICS_PATH)
    importance_path = save_feature_importance(
        feature_importances=cast(pd.DataFrame, training_results["feature_importances"]),
        output_path=FEATURE_IMPORTANCE_PATH,
    )
    split_summary_path = save_split_summary(split_summary=split_summary, output_path=SPLIT_SUMMARY_PATH)

    artifact_summary = {
        "xgboost_version": getattr(xgb, "__version__", "unknown"),
        "gpu_device_ordinal": GPU_DEVICE_ORDINAL,
        "data_root": str(data_root),
        "source_data_path": str(SOURCE_DATA_PATH),
        "processed_data_path": str(preprocessing_result["processed_data_path"]),
        "output_dir": str(output_dir),
        "preprocessed_data_dir": str(PREPROCESSED_DATA_DIR),
        "target_preprocess": "log1p_then_global_minmax",
        "target_log1p_min": preprocessing_result["target_log1p_min"],
        "target_log1p_max": preprocessing_result["target_log1p_max"],
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "feature_count": len(feature_cols),
        "xgboost_feature_cols": feature_cols,
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS,
        "train_row_count": int(train_split_df.shape[0]),
        "valid_row_count": int(valid_split_df.shape[0]),
        "test_row_count": int(test_df.shape[0]),
        "validation_mse": validation_metrics["mse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_smape": validation_metrics["smape"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_smape": test_metrics["smape"],
        "best_iteration_round_count": training_results["best_round"],
        "model_path": str(MODEL_PATH),
        "model_params_path": str(model_params_path),
        "validation_metrics_path": str(validation_metrics_path),
        "test_metrics_path": str(test_metrics_path),
        "importance_path": str(importance_path),
        "split_summary_path": str(split_summary_path),
    }
    save_json(artifact_summary, RUN_SUMMARY_PATH)
    artifact_summary["summary_path"] = str(RUN_SUMMARY_PATH)

    print(json.dumps(artifact_summary, indent=2))
    return artifact_summary


if __name__ == "__main__":
    main()
