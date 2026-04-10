from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import xgboost as xgb
from xgboost.core import XGBoostError


DATA_ROOT = Path(
    r"F:\Desktop\Final\USTB-graduation-project\ASHRAE-Great Energy Predictor III\ashrae-energy-prediction"
)
OUTPUT_DIR = Path(r"F:\Desktop\Final\workspace\XGBoost\origin_true_xgboost_baseline_outputs")

SEED = 42
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 3000
CHUNK_SIZE = 500_000
GPU_DEVICE_ORDINAL = 0

CATEGORICAL_COLS = [
    "site_id",
    "building_id",
    "primary_use",
    "meter",
    "wind_direction",
    "beaufort_scale",
    "month_datetime",
    "weekofyear_datetime",
    "hour_datetime",
    "day_week",
    "day_month_datetime",
    "week_month_datetime",
]

NUMERICAL_COLS = [
    "square_feet",
    "year_built",
    "age",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "floor_count",
    "dayofyear_datetime",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type) not in numerics:
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type).startswith("int"):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(
            f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB "
            f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df


def deg_to_compass(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return int(value / 22.5) % 16


def average_imputation_by_timestamp(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    timestamp_means = df.groupby("timestamp")[column_name].transform("mean")
    df[column_name] = df[column_name].fillna(timestamp_means)
    return df


def compute_beaufort_scale(speed_series: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 33.0, np.inf]
    labels = list(range(13))
    return pd.cut(speed_series, bins=bins, labels=labels, right=False).astype("float32")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    iso_calendar = df["timestamp"].dt.isocalendar()
    df["month_datetime"] = df["timestamp"].dt.month.astype(np.int8)
    df["weekofyear_datetime"] = iso_calendar.week.astype(np.int16)
    df["dayofyear_datetime"] = df["timestamp"].dt.dayofyear.astype(np.int16)
    df["hour_datetime"] = df["timestamp"].dt.hour.astype(np.int8)
    df["day_week"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["day_month_datetime"] = df["timestamp"].dt.day.astype(np.int8)
    df["week_month_datetime"] = np.ceil(df["timestamp"].dt.day / 7.0).astype(np.int8)
    return df


def encode_primary_use(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categories = pd.Index(
        sorted(set(train_df["primary_use"].dropna().tolist()) | set(test_df["primary_use"].dropna().tolist()))
    )
    mapping = {value: idx for idx, value in enumerate(categories)}
    train_df["primary_use"] = train_df["primary_use"].map(mapping).fillna(-1).astype(np.int16)
    test_df["primary_use"] = test_df["primary_use"].map(mapping).fillna(-1).astype(np.int16)
    return train_df, test_df


def finalize_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    df["wind_direction"] = df["wind_direction"].fillna(-1).astype(np.int16)
    df["beaufort_scale"] = df["beaufort_scale"].fillna(-1).astype(np.int8)
    df["floor_count"] = df["floor_count"].fillna(-999).astype(np.int16)
    df["year_built"] = df["year_built"].fillna(-999).astype(np.int16)
    df["age"] = df["age"].fillna(-999).astype(np.int16)
    df["cloud_coverage"] = df["cloud_coverage"].fillna(-999).astype(np.int16)
    df["precip_depth_1_hr"] = df["precip_depth_1_hr"].fillna(-999).astype(np.float32)
    df["air_temperature"] = df["air_temperature"].astype(np.float32)
    df["dew_temperature"] = df["dew_temperature"].astype(np.float32)
    df["square_feet"] = df["square_feet"].astype(np.int32)
    df["building_id"] = df["building_id"].astype(np.int16)
    df["site_id"] = df["site_id"].astype(np.int8)
    df["meter"] = df["meter"].astype(np.int8)
    return df


def load_raw_data(
    data_root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        data_root / "train.csv",
        usecols=["building_id", "meter", "timestamp", "meter_reading"],
        parse_dates=["timestamp"],
    )
    test_df = pd.read_csv(
        data_root / "test.csv",
        usecols=["row_id", "building_id", "meter", "timestamp"],
        parse_dates=["timestamp"],
    )
    weather_train_df = pd.read_csv(data_root / "weather_train.csv", parse_dates=["timestamp"])
    weather_test_df = pd.read_csv(data_root / "weather_test.csv", parse_dates=["timestamp"])
    building_meta_df = pd.read_csv(data_root / "building_metadata.csv")
    sample_submission = pd.read_csv(data_root / "sample_submission.csv")
    return train_df, test_df, weather_train_df, weather_test_df, building_meta_df, sample_submission


def merge_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weather_train_df: pd.DataFrame,
    weather_test_df: pd.DataFrame,
    building_meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.merge(building_meta_df, on="building_id", how="left")
    test_df = test_df.merge(building_meta_df, on="building_id", how="left")
    train_df = train_df.merge(weather_train_df, on=["site_id", "timestamp"], how="left")
    test_df = test_df.merge(weather_test_df, on=["site_id", "timestamp"], how="left")
    return train_df, test_df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = encode_primary_use(train_df, test_df)

    for df in (train_df, test_df):
        df["age"] = df["year_built"].max(skipna=True) - df["year_built"] + 1
        df = average_imputation_by_timestamp(df, "wind_speed")
        df = average_imputation_by_timestamp(df, "wind_direction")
        df["beaufort_scale"] = compute_beaufort_scale(df["wind_speed"])
        df["wind_direction"] = df["wind_direction"].apply(deg_to_compass)
        df = add_time_features(df)
        df = finalize_feature_types(df)
        reduce_mem_usage(df, verbose=True)

    return train_df, test_df


def make_training_matrix(
    features: pd.DataFrame,
    label: pd.Series | np.ndarray | None = None,
    ref: object | None = None,
):
    matrix_cls = getattr(xgb, "QuantileDMatrix", None)
    if matrix_cls is not None:
        return matrix_cls(features, label=label, feature_names=FEATURE_COLS, ref=ref)
    return xgb.DMatrix(features, label=label, feature_names=FEATURE_COLS)


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


def assert_gpu_training_ready() -> None:
    gpu_params = build_xgb_params(seed=SEED)
    probe_x = pd.DataFrame(np.random.rand(512, len(FEATURE_COLS)).astype(np.float32), columns=FEATURE_COLS)
    probe_y = np.random.rand(512).astype(np.float32)
    dtrain = make_training_matrix(probe_x, label=probe_y)

    try:
        booster = xgb.train(
            params=gpu_params,
            dtrain=dtrain,
            num_boost_round=2,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
        _ = booster.predict(dtrain)
        print("XGBoost GPU sanity check passed.")
    except XGBoostError as exc:
        raise RuntimeError(
            "XGBoost GPU training failed. On this Windows machine, please make sure the notebook environment "
            "has a CUDA-enabled xgboost build and can see the RTX 3070."
        ) from exc
    finally:
        del probe_x, probe_y, dtrain
        gc.collect()


def get_prediction_iteration_range(booster: xgb.Booster) -> tuple[int, int]:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return (0, booster.num_boosted_rounds())
    return (0, best_iteration + 1)


def extract_feature_importance(booster: xgb.Booster, feature_cols: list[str], fold: int) -> pd.DataFrame:
    gain_map = booster.get_score(importance_type="gain")
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": [float(gain_map.get(feature, 0.0)) for feature in feature_cols],
            "fold": fold,
        }
    )


def train_xgboost_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> dict:
    target = np.log1p(train_df["meter_reading"].astype(np.float32))
    oof_predictions = np.zeros(train_df.shape[0], dtype=np.float32)
    fold_scores: list[float] = []
    models: list[xgb.Booster] = []
    feature_importance_frames: list[pd.DataFrame] = []

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    params = build_xgb_params(seed=seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_df), start=1):
        x_train = train_df.iloc[train_idx][feature_cols]
        x_valid = train_df.iloc[valid_idx][feature_cols]
        y_train = target.iloc[train_idx]
        y_valid = target.iloc[valid_idx]

        dtrain = make_training_matrix(x_train, label=y_train)
        dvalid = make_training_matrix(x_valid, label=y_valid, ref=dtrain)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=100,
        )

        valid_pred = booster.predict(dvalid, iteration_range=get_prediction_iteration_range(booster))
        oof_predictions[valid_idx] = valid_pred

        fold_rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
        fold_scores.append(fold_rmse)
        models.append(booster)
        feature_importance_frames.append(extract_feature_importance(booster, feature_cols, fold))

        print(f"Fold {fold}: RMSE on log1p target = {fold_rmse:.6f}")

        del x_train, x_valid, y_train, y_valid, dtrain, dvalid, valid_pred
        gc.collect()

    overall_rmse = float(np.sqrt(mean_squared_error(target, oof_predictions)))
    print(f"OOF RMSE on log1p target = {overall_rmse:.6f}")

    return {
        "models": models,
        "oof_predictions": oof_predictions,
        "fold_scores": fold_scores,
        "overall_rmse": overall_rmse,
        "feature_importances": pd.concat(feature_importance_frames, ignore_index=True),
    }


def predict_test_in_chunks(
    models: list[xgb.Booster],
    test_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    predictions_log = np.zeros(test_df.shape[0], dtype=np.float32)

    for start in range(0, test_df.shape[0], chunk_size):
        stop = min(start + chunk_size, test_df.shape[0])
        chunk = test_df.iloc[start:stop][feature_cols]
        dchunk = make_training_matrix(chunk)

        chunk_pred = np.zeros(stop - start, dtype=np.float32)
        for booster in models:
            chunk_pred += booster.predict(dchunk, iteration_range=get_prediction_iteration_range(booster)) / len(models)

        predictions_log[start:stop] = chunk_pred
        print(f"Predicted rows {start:,} to {stop:,}")

        del chunk, dchunk, chunk_pred
        gc.collect()

    return np.clip(np.expm1(predictions_log), 0, None)


def save_feature_importance(feature_importances: pd.DataFrame, output_dir: Path) -> Path:
    importance_summary = (
        feature_importances.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    importance_path = output_dir / "xgboost_feature_importance.csv"
    importance_summary.to_csv(importance_path, index=False)
    return importance_path


def save_metrics(fold_scores: list[float], overall_rmse: float, output_dir: Path) -> Path:
    metrics_df = pd.DataFrame(
        {
            "fold": list(range(1, len(fold_scores) + 1)),
            "rmse_log1p": fold_scores,
        }
    )
    metrics_df.loc[len(metrics_df)] = ["mean", float(np.mean(fold_scores))]
    metrics_df.loc[len(metrics_df)] = ["std", float(np.std(fold_scores))]
    metrics_df.loc[len(metrics_df)] = ["oof", overall_rmse]

    metrics_path = output_dir / "xgboost_cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    return metrics_path


def save_submission(sample_submission: pd.DataFrame, predictions: np.ndarray, output_dir: Path) -> Path:
    submission = sample_submission.copy()
    submission["meter_reading"] = predictions
    submission.loc[submission["meter_reading"] < 0, "meter_reading"] = 0
    submission_path = output_dir / "xgboost_submission.csv"
    submission.to_csv(submission_path, index=False)
    return submission_path


def main(data_root: Path = DATA_ROOT, output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        train_df,
        test_df,
        weather_train_df,
        weather_test_df,
        building_meta_df,
        sample_submission,
    ) = load_raw_data(data_root=data_root)

    train_df, test_df = merge_features(
        train_df=train_df,
        test_df=test_df,
        weather_train_df=weather_train_df,
        weather_test_df=weather_test_df,
        building_meta_df=building_meta_df,
    )

    del weather_train_df, weather_test_df, building_meta_df
    gc.collect()

    train_df, test_df = engineer_features(train_df=train_df, test_df=test_df)
    assert_gpu_training_ready()

    training_results = train_xgboost_cv(train_df=train_df)
    test_predictions = predict_test_in_chunks(models=training_results["models"], test_df=test_df)

    metrics_path = save_metrics(
        fold_scores=training_results["fold_scores"],
        overall_rmse=training_results["overall_rmse"],
        output_dir=output_dir,
    )
    importance_path = save_feature_importance(
        feature_importances=training_results["feature_importances"],
        output_dir=output_dir,
    )
    submission_path = save_submission(
        sample_submission=sample_submission,
        predictions=test_predictions,
        output_dir=output_dir,
    )

    artifact_summary = {
        "xgboost_version": getattr(xgb, "__version__", "unknown"),
        "gpu_device_ordinal": GPU_DEVICE_ORDINAL,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "feature_count": len(FEATURE_COLS),
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS,
        "fold_scores": training_results["fold_scores"],
        "overall_rmse_log1p": training_results["overall_rmse"],
        "metrics_path": str(metrics_path),
        "importance_path": str(importance_path),
        "submission_path": str(submission_path),
    }

    summary_path = output_dir / "xgboost_run_summary.json"
    summary_path.write_text(json.dumps(artifact_summary, indent=2), encoding="utf-8")
    artifact_summary["summary_path"] = str(summary_path)

    print(json.dumps(artifact_summary, indent=2))
    return artifact_summary


if __name__ == "__main__":
    main()
