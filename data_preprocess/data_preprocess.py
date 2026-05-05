#此文件将train.csv每条数据右侧拼接上了对应的建筑信息数据和天气数据，并计算时间序列特征
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    r"F:\Desktop\Final\USTB-graduation-project\ASHRAE-Great Energy Predictor III\ashrae-energy-prediction"
)
PREPROCESSED_DATA_DIR = WORKSPACE_ROOT / "preprocessed_data"
TRAIN_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "preprocessed_train.csv"

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
MISSING_INDICATOR_EXCLUDE_COLS = {"timestamp", "meter_reading"}


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float32", "float64"]
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
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
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


def encode_primary_use(train_df: pd.DataFrame) -> pd.DataFrame:
    categories = pd.Index(sorted(set(train_df["primary_use"].dropna().tolist())))
    mapping = {value: idx for idx, value in enumerate(categories)}
    train_df["primary_use"] = pd.to_numeric(train_df["primary_use"].map(mapping), errors="coerce").astype(np.float32)
    return train_df


def cast_numeric_column(
    df: pd.DataFrame,
    column: str,
    complete_dtype: str | np.dtype,
    missing_dtype: str | np.dtype | None = None,
) -> pd.DataFrame:
    if column not in df.columns:
        return df

    numeric_series = pd.to_numeric(df[column], errors="coerce")
    if numeric_series.isna().any():
        df[column] = numeric_series.astype(missing_dtype or complete_dtype)
    else:
        df[column] = numeric_series.astype(complete_dtype)
    return df


def finalize_base_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["primary_use", "wind_direction", "beaufort_scale"]:
        df = cast_numeric_column(df, column, np.int16, np.float32)

    for column in [
        "year_built",
        "age",
        "cloud_coverage",
        "floor_count",
        "air_temperature",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_speed",
    ]:
        df = cast_numeric_column(df, column, np.float32, np.float32)

    df = cast_numeric_column(df, "square_feet", np.int32, np.float32)
    df = cast_numeric_column(df, "building_id", np.int16, np.float32)
    df = cast_numeric_column(df, "site_id", np.int8, np.float32)
    df = cast_numeric_column(df, "meter", np.int8, np.float32)
    df = cast_numeric_column(df, "month_datetime", np.int8, np.float32)
    df = cast_numeric_column(df, "weekofyear_datetime", np.int16, np.float32)
    df = cast_numeric_column(df, "dayofyear_datetime", np.int16, np.float32)
    df = cast_numeric_column(df, "hour_datetime", np.int8, np.float32)
    df = cast_numeric_column(df, "day_week", np.int8, np.float32)
    df = cast_numeric_column(df, "day_month_datetime", np.int8, np.float32)
    df = cast_numeric_column(df, "week_month_datetime", np.int8, np.float32)
    return df


def add_missing_indicators(train_df: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = sorted(set(train_df.columns) - MISSING_INDICATOR_EXCLUDE_COLS)

    for column in tqdm(candidate_columns, desc="Creating missing indicators", unit="col"):
        missing_mask = train_df[column].isna()
        if missing_mask.any():
            train_df[f"{column}_missing"] = missing_mask.astype(np.int8)

    return train_df


def load_raw_data(
    data_root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        data_root / "train.csv",
        usecols=["building_id", "meter", "timestamp", "meter_reading"],
        parse_dates=["timestamp"],
    )
    weather_train_df = pd.read_csv(data_root / "weather_train.csv", parse_dates=["timestamp"])
    building_meta_df = pd.read_csv(data_root / "building_metadata.csv")
    return train_df, weather_train_df, building_meta_df


def merge_features(
    train_df: pd.DataFrame,
    weather_train_df: pd.DataFrame,
    building_meta_df: pd.DataFrame,
) -> pd.DataFrame:
    train_df = train_df.merge(building_meta_df, on="building_id", how="left")
    train_df = train_df.merge(weather_train_df, on=["site_id", "timestamp"], how="left")
    return train_df


def engineer_features(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = encode_primary_use(train_df)

    for _ in tqdm([0], desc="Engineering base features", unit="split"):
        train_df["age"] = 2016 - train_df["year_built"]
        train_df["beaufort_scale"] = compute_beaufort_scale(train_df["wind_speed"])
        train_df["wind_direction"] = train_df["wind_direction"].apply(deg_to_compass)
        train_df = add_time_features(train_df)
        train_df = finalize_base_feature_types(train_df)

    train_df = add_missing_indicators(train_df)

    for _ in tqdm([0], desc="Optimizing base data memory", unit="split"):
        reduce_mem_usage(train_df, verbose=True)

    return train_df


def save_preprocessed_data(
    train_df: pd.DataFrame,
    output_dir: Path = PREPROCESSED_DATA_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / TRAIN_OUTPUT_PATH.name
    train_df.to_csv(train_path, index=False)
    return train_path


def preprocess_and_save(
    data_root: Path = DATA_ROOT,
    output_dir: Path = PREPROCESSED_DATA_DIR,
) -> dict[str, str]:
    with tqdm(total=5, desc="Base preprocessing pipeline", unit="step") as progress:
        train_df, weather_train_df, building_meta_df = load_raw_data(data_root=data_root)
        progress.update(1)

        train_df = merge_features(
            train_df=train_df,
            weather_train_df=weather_train_df,
            building_meta_df=building_meta_df,
        )
        progress.update(1)

        del weather_train_df, building_meta_df
        gc.collect()
        progress.update(1)

        train_df = engineer_features(train_df=train_df)
        progress.update(1)

        train_path = save_preprocessed_data(train_df=train_df, output_dir=output_dir)
        progress.update(1)

    return {
        "train_path": str(train_path),
    }


def main() -> dict[str, str]:
    result = preprocess_and_save()
    print(result)
    return result


if __name__ == "__main__":
    main()
