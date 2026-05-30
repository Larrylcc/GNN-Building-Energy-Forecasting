from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from data_preprocess.data_preprocess import CATEGORICAL_COLS, DATA_ROOT, NUMERICAL_COLS, PREPROCESSED_DATA_DIR  # noqa: E402
from raw_metric_utils import compute_raw_regression_metrics, inverse_log1p_minmax  # noqa: E402


class MeterConfig(NamedTuple):
    meter_id: int
    meter_name: str


class MeterPaths(NamedTuple):
    data_dir: Path
    output_dir: Path
    train_data_path: Path
    valid_data_path: Path
    test_data_path: Path
    preprocessing_summary_path: Path
    model_path: Path
    model_params_path: Path
    validation_metrics_path: Path
    test_metrics_path: Path
    run_summary_path: Path


OUTPUT_ROOT_DIR = WORKSPACE_ROOT / "XGBoost" / "xgboost_baseline_outputs"
OVERALL_RUN_SUMMARY_PATH = OUTPUT_ROOT_DIR / "xgboost_overall_run_summary.json"

METER_CONFIGS = {
    0: MeterConfig(meter_id=0, meter_name="electricity"),
    1: MeterConfig(meter_id=1, meter_name="chilled_water"),
    2: MeterConfig(meter_id=2, meter_name="steam"),
    3: MeterConfig(meter_id=3, meter_name="hot_water"),
}

SEED = 42
EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 3000
GPU_DEVICE_ORDINAL = 0
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def build_meter_paths(
    config: MeterConfig,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> MeterPaths:
    if config.meter_id == 0:
        data_dir = preprocessed_data_dir
        train_data_path = preprocessed_data_dir / "log1p_minmax_train.csv"
        valid_data_path = preprocessed_data_dir / "log1p_minmax_valid.csv"
        test_data_path = preprocessed_data_dir / "log1p_minmax_test.csv"
        preprocessing_summary_path = preprocessed_data_dir / "log1p_minmax_summary.json"
    else:
        data_dir = preprocessed_data_dir / f"meter_{config.meter_id}"
        train_data_path = data_dir / "log1p_minmax_train.csv"
        valid_data_path = data_dir / "log1p_minmax_valid.csv"
        test_data_path = data_dir / "log1p_minmax_test.csv"
        preprocessing_summary_path = data_dir / "log1p_minmax_summary.json"

    output_dir = output_root_dir / f"meter_{config.meter_id}"

    return MeterPaths(
        data_dir=data_dir,
        output_dir=output_dir,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        test_data_path=test_data_path,
        preprocessing_summary_path=preprocessing_summary_path,
        model_path=output_dir / "xgboost_final_model.json",
        model_params_path=output_dir / "xgboost_model_params.json",
        validation_metrics_path=output_dir / "xgboost_validation_metrics.json",
        test_metrics_path=output_dir / "xgboost_test_metrics.json",
        run_summary_path=output_dir / "xgboost_run_summary.json",
    )


def compute_normalized_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float32, copy=False)
    y_pred = y_pred.astype(np.float32, copy=False)
    row_count = int(y_true.shape[0])
    if row_count == 0:
        raise ValueError("Metric evaluation did not receive any rows.")

    errors = y_pred - y_true
    squared_error_sum = float(np.sum(errors * errors, dtype=np.float64))
    mse = squared_error_sum / row_count
    mae = float(np.mean(np.abs(errors), dtype=np.float64))
    rmse = float(np.sqrt(mse))

    y_sum = float(np.sum(y_true, dtype=np.float64))
    y_squared_sum = float(np.sum(y_true * y_true, dtype=np.float64))
    total_sum_of_squares = y_squared_sum - (y_sum * y_sum / row_count)
    r2 = 0.0 if total_sum_of_squares <= 0.0 else 1.0 - squared_error_sum / total_sum_of_squares

    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_ratio = np.divide(
        2.0 * np.abs(errors),
        denominator,
        out=np.zeros_like(y_true, dtype=np.float32),
        where=denominator != 0,
    )
    smape = float(np.mean(smape_ratio, dtype=np.float64) * 100.0)

    clipped_true = np.clip(y_true, a_min=0.0, a_max=None)
    clipped_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    log_errors = np.log1p(clipped_pred) - np.log1p(clipped_true)
    rmsle = float(np.sqrt(np.mean(log_errors * log_errors, dtype=np.float64)))

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "smape": smape,
        "rmse": rmse,
        "rmsle": rmsle,
    }


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log1p + min-max split file not found: {path}.")

    split_df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    split_df = split_df.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    split_df["meter_reading"] = pd.to_numeric(split_df["meter_reading"], errors="coerce").astype(np.float32)
    return split_df


def load_preprocessed_splits(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
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


def load_preprocessing_summary(summary_path: Path) -> dict[str, object]:
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


def save_json(data: dict[str, object] | list[dict[str, object]], output_path: Path) -> Path:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def train_one_meter(
    config: MeterConfig,
    data_root: Path = DATA_ROOT,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> dict[str, object]:
    paths = build_meter_paths(
        config=config,
        preprocessed_data_dir=preprocessed_data_dir,
        output_root_dir=output_root_dir,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for meter {config.meter_id} ({config.meter_name})...")
    train_df, valid_df, test_df, feature_cols = load_preprocessed_splits(
        train_path=paths.train_data_path,
        valid_path=paths.valid_data_path,
        test_path=paths.test_data_path,
    )
    preprocessing_summary = load_preprocessing_summary(paths.preprocessing_summary_path)
    split_summary = make_split_summary(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        preprocessing_summary=preprocessing_summary,
    )

    x_train = train_df[feature_cols]
    y_train = cast(pd.Series, train_df["meter_reading"].astype(np.float32))
    x_valid = valid_df[feature_cols]
    y_valid = cast(pd.Series, valid_df["meter_reading"].astype(np.float32))

    dtrain = make_training_matrix(x_train, feature_cols=feature_cols, label=y_train)
    dvalid = make_training_matrix(x_valid, feature_cols=feature_cols, label=y_valid, ref=dtrain)

    print(f"Training XGBoost model for Meter {config.meter_id}...")
    params = build_xgb_params(seed=SEED)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=200,
    )

    # Save model
    booster.save_model(paths.model_path)
    best_round = get_prediction_iteration_range(booster)[1]

    # Validation predictions and evaluation
    print("Evaluating model on validation set...")
    valid_pred = booster.predict(dvalid, iteration_range=get_prediction_iteration_range(booster))
    valid_pred_clipped = np.clip(valid_pred, a_min=0.0, a_max=1.0)
    validation_metrics = compute_normalized_regression_metrics(
        y_true=y_valid.to_numpy(),
        y_pred=valid_pred_clipped,
    )

    # Test predictions and evaluation
    print("Evaluating model on test set...")
    x_test = test_df[feature_cols]
    y_test = cast(pd.Series, test_df["meter_reading"].astype(np.float32))
    dtest = make_training_matrix(x_test, feature_cols=feature_cols)
    test_pred = booster.predict(dtest, iteration_range=get_prediction_iteration_range(booster))
    test_pred_clipped = np.clip(test_pred, a_min=0.0, a_max=1.0)
    test_metrics = compute_normalized_regression_metrics(
        y_true=y_test.to_numpy(),
        y_pred=test_pred_clipped,
    )

    # Also compute RAW metrics for reference (using inverse log1p min-max scale)
    target_log1p_min = float(preprocessing_summary.get("target_log1p_min", 0.0))
    target_log1p_max = float(preprocessing_summary.get("target_log1p_max", 1.0))
    
    y_valid_raw = inverse_log1p_minmax(y_valid.to_numpy(), target_log1p_min, target_log1p_max)
    valid_pred_raw = inverse_log1p_minmax(valid_pred_clipped, target_log1p_min, target_log1p_max)
    raw_validation_metrics = compute_raw_regression_metrics(y_true_raw=y_valid_raw, y_pred_raw=valid_pred_raw)

    y_test_raw = inverse_log1p_minmax(y_test.to_numpy(), target_log1p_min, target_log1p_max)
    test_pred_raw = inverse_log1p_minmax(test_pred_clipped, target_log1p_min, target_log1p_max)
    raw_test_metrics = compute_raw_regression_metrics(y_true_raw=y_test_raw, y_pred_raw=test_pred_raw)

    # Save metrics JSON files
    save_json(validation_metrics, paths.validation_metrics_path)
    save_json(test_metrics, paths.test_metrics_path)

    # Build model params metadata
    model_params_artifact = {
        "meter_id": config.meter_id,
        "meter_name": config.meter_name,
        "xgb_params": params,
        "best_iteration_round_count": best_round,
        "feature_count": len(feature_cols),
        "xgboost_feature_cols": feature_cols,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "num_boost_round_upper_bound": NUM_BOOST_ROUND,
        "train_data_path": str(paths.train_data_path),
        "valid_data_path": str(paths.valid_data_path),
        "test_data_path": str(paths.test_data_path),
        "preprocessing_summary_path": str(paths.preprocessing_summary_path),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": target_log1p_min,
        "target_log1p_max": target_log1p_max,
    }
    save_json(model_params_artifact, paths.model_params_path)

    # Build run summary
    artifact_summary = {
        "model_name": "XGBoost",
        "meter_id": config.meter_id,
        "meter_name": config.meter_name,
        "data_root": str(data_root),
        "train_data_path": str(paths.train_data_path),
        "valid_data_path": str(paths.valid_data_path),
        "test_data_path": str(paths.test_data_path),
        "preprocessing_summary_path": str(paths.preprocessing_summary_path),
        "output_dir": str(paths.output_dir),
        "preprocessed_data_dir": str(paths.data_dir),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": target_log1p_min,
        "target_log1p_max": target_log1p_max,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "feature_count": len(feature_cols),
        "xgboost_feature_cols": feature_cols,
        "train_row_count": int(train_df.shape[0]),
        "valid_row_count": int(valid_df.shape[0]),
        "test_row_count": int(test_df.shape[0]),
        "splits": split_summary.to_dict(orient="records"),
        # Normalized metrics (direct model output scale, requested by user)
        "validation_normalized_metrics": validation_metrics,
        "test_normalized_metrics": test_metrics,
        # Raw metrics (inverse-transformed original target scale, for reference)
        "validation_raw_metrics": raw_validation_metrics,
        "test_raw_metrics": raw_test_metrics,
        # Flattened validation/test scores for high level logging
        "validation_mse": validation_metrics["mse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_smape": validation_metrics["smape"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_rmsle": validation_metrics["rmsle"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_smape": test_metrics["smape"],
        "test_rmse": test_metrics["rmse"],
        "test_rmsle": test_metrics["rmsle"],
        "best_iteration_round_count": best_round,
        "model_path": str(paths.model_path),
        "model_params_path": str(paths.model_params_path),
        "validation_metrics_path": str(paths.validation_metrics_path),
        "test_metrics_path": str(paths.test_metrics_path),
    }
    save_json(artifact_summary, paths.run_summary_path)
    print(f"Meter {config.meter_id} ({config.meter_name}) baseline completed successfully.")
    print(f"Validation Normalized RMSE: {validation_metrics['rmse']:.6f} | Test Normalized RMSE: {test_metrics['rmse']:.6f}")

    del train_df, valid_df, test_df, x_train, y_train, x_valid, y_valid, dtrain, dvalid, x_test, y_test, dtest
    gc.collect()

    return artifact_summary


def main() -> list[dict[str, object]]:
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple probe to get columns and assert GPU
    temp_train = load_split_data(build_meter_paths(METER_CONFIGS[0]).train_data_path)
    cols = [col for col in temp_train.columns if col not in {"timestamp", "meter_reading"}]
    del temp_train
    gc.collect()

    assert_gpu_training_ready(feature_cols=cols)

    meter_summaries = []
    for config in METER_CONFIGS.values():
        print("=" * 60)
        print(f"Starting XGBoost for Meter {config.meter_id} ({config.meter_name})")
        print("=" * 60)
        meter_summary = train_one_meter(config=config)
        meter_summaries.append(meter_summary)

    save_json(meter_summaries, OVERALL_RUN_SUMMARY_PATH)
    print("=" * 60)
    print("All meters completed. Overall summary saved.")
    print("=" * 60)
    return meter_summaries


if __name__ == "__main__":
    main()
