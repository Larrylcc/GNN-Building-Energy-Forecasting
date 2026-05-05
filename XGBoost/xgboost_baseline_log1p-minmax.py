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


TRAIN_DATA_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_train.csv"
VALID_DATA_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_valid.csv"
TEST_DATA_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_test.csv"
PREPROCESSING_SUMMARY_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_summary.json"

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


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_ratio = np.divide(
        2.0 * np.abs(y_pred - y_true),
        denominator,
        out=np.zeros_like(y_true, dtype=np.float32),
        where=denominator != 0,
    )
    return float(np.mean(safe_ratio) * 100.0)


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log1p + min-max split file not found: {path}.")

    split_df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    split_df = split_df.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    split_df["meter_reading"] = pd.to_numeric(split_df["meter_reading"], errors="coerce").astype(np.float32)
    return split_df


def load_preprocessed_splits(
    train_path: Path = TRAIN_DATA_PATH,
    valid_path: Path = VALID_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = load_split_data(train_path)
    valid_df = load_split_data(valid_path)
    test_df = load_split_data(test_path)

    feature_cols = [column for column in train_df.columns if column not in {"timestamp", "meter_reading"}]
    expected_columns = train_df.columns.tolist()

    for split_name, split_df in [("valid", valid_df), ("test", test_df)]:
        if split_df.columns.tolist() != expected_columns:
            raise ValueError(f"The {split_name} split columns do not match the training split columns.")

    return train_df, valid_df, test_df, feature_cols


def load_preprocessing_summary(summary_path: Path = PREPROCESSING_SUMMARY_PATH) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def make_split_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessing_summary: dict[str, object],
) -> pd.DataFrame:
    split_records = preprocessing_summary.get("splits")
    if isinstance(split_records, list) and split_records:
        return pd.DataFrame(split_records)

    return pd.DataFrame(
        [
            build_split_summary_row("train", train_df),
            build_split_summary_row("valid", valid_df),
            build_split_summary_row("test", test_df),
        ]
    )


def build_split_summary_row(split_name: str, split_df: pd.DataFrame) -> dict[str, str | int]:
    return {
        "split": split_name,
        "start_timestamp": str(split_df["timestamp"].min()),
        "end_timestamp": str(split_df["timestamp"].max()),
        "row_count": int(split_df.shape[0]),
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


def build_xgb_params(seed: int = SEED) -> dict[str, object]:
    version_text = getattr(xgb, "__version__", "0.0.0")
    major_version = int(version_text.split(".")[0]) if version_text else 0

    params: dict[str, object] = {
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

    print(f"Validation RMSE on preprocessed log1p + min-max target = {valid_rmse:.6f}")

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
        "smape": split_smape,
        "rmse": split_rmse,
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

    train_df, valid_df, test_df, feature_cols = load_preprocessed_splits()
    preprocessing_summary = load_preprocessing_summary()
    split_summary = make_split_summary(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        preprocessing_summary=preprocessing_summary,
    )

    assert_gpu_training_ready(feature_cols=feature_cols)

    training_results = train_xgboost_single_split(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
    )
    final_model = cast(xgb.Booster, training_results["model"])
    validation_metrics = evaluate_split(
        booster=final_model,
        split_df=valid_df,
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
        "train_data_path": str(TRAIN_DATA_PATH),
        "valid_data_path": str(VALID_DATA_PATH),
        "test_data_path": str(TEST_DATA_PATH),
        "preprocessing_summary_path": str(PREPROCESSING_SUMMARY_PATH),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": preprocessing_summary.get("target_log1p_max"),
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
        "train_data_path": str(TRAIN_DATA_PATH),
        "valid_data_path": str(VALID_DATA_PATH),
        "test_data_path": str(TEST_DATA_PATH),
        "preprocessing_summary_path": str(PREPROCESSING_SUMMARY_PATH),
        "output_dir": str(output_dir),
        "preprocessed_data_dir": str(PREPROCESSED_DATA_DIR),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": preprocessing_summary.get("target_log1p_max"),
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "feature_count": len(feature_cols),
        "xgboost_feature_cols": feature_cols,
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS,
        "train_row_count": int(train_df.shape[0]),
        "valid_row_count": int(valid_df.shape[0]),
        "test_row_count": int(test_df.shape[0]),
        "validation_mse": validation_metrics["mse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_smape": validation_metrics["smape"],
        "validation_rmse": validation_metrics["rmse"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_smape": test_metrics["smape"],
        "test_rmse": test_metrics["rmse"],
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
