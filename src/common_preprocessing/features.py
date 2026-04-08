from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common_preprocessing.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TIMESTAMP_COLUMN
from common_preprocessing.loaders import RawDataBundle


WEATHER_COLUMNS = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]

BUILDING_COLUMNS = ["site_id", "building_id", "primary_use", "square_feet", "year_built", "floor_count"]


def merge_raw_tables(raw_data: RawDataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = raw_data.train.merge(raw_data.building_metadata[BUILDING_COLUMNS], on="building_id", how="left")
    test_df = raw_data.test.merge(raw_data.building_metadata[BUILDING_COLUMNS], on="building_id", how="left")

    train_df = train_df.merge(raw_data.weather_train, on=["site_id", TIMESTAMP_COLUMN], how="left")
    test_df = test_df.merge(raw_data.weather_test, on=["site_id", TIMESTAMP_COLUMN], how="left")

    return train_df, test_df


def encode_primary_use(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    categories = sorted(
        set(train_df["primary_use"].dropna().astype(str).tolist())
        | set(test_df["primary_use"].dropna().astype(str).tolist())
    )
    mapping = {value: idx for idx, value in enumerate(categories)}

    train_df["primary_use"] = train_df["primary_use"].map(mapping).fillna(-1).astype(np.int16)
    test_df["primary_use"] = test_df["primary_use"].map(mapping).fillna(-1).astype(np.int16)
    return mapping


def _add_time_features(df: pd.DataFrame) -> None:
    iso_calendar = df[TIMESTAMP_COLUMN].dt.isocalendar()
    df["month"] = df[TIMESTAMP_COLUMN].dt.month
    df["weekofyear"] = iso_calendar.week
    df["dayofyear"] = df[TIMESTAMP_COLUMN].dt.dayofyear
    df["hour"] = df[TIMESTAMP_COLUMN].dt.hour
    df["dayofweek"] = df[TIMESTAMP_COLUMN].dt.dayofweek
    df["day"] = df[TIMESTAMP_COLUMN].dt.day
    df["week_of_month"] = np.ceil(df[TIMESTAMP_COLUMN].dt.day / 7.0)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)


def _wind_direction_to_bucket(series: pd.Series) -> pd.Series:
    direction = pd.to_numeric(series, errors="coerce")
    bucket = np.floor((np.mod(direction, 360.0)) / 22.5)
    return pd.Series(bucket, index=series.index, dtype="float64")


def _compute_beaufort_scale(speed_series: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 33.0, np.inf]
    labels = list(range(13))
    return pd.cut(speed_series, bins=bins, labels=labels, right=False).astype("float32")


def _fallback_numeric(series: pd.Series, fallback: float) -> float:
    value = pd.to_numeric(series, errors="coerce").median()
    if pd.isna(value):
        return fallback
    return float(value)


def _fill_with_site_median(df: pd.DataFrame, column: str) -> None:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    by_site_median = df.groupby("site_id")[column].transform("median")
    df[column] = df[column].fillna(by_site_median)
    fallback = _fallback_numeric(df[column], fallback=0.0)
    df[column] = df[column].fillna(fallback)


def _clean_weather_columns(df: pd.DataFrame) -> None:
    if "precip_depth_1_hr" in df.columns:
        df.loc[df["precip_depth_1_hr"] < 0, "precip_depth_1_hr"] = np.nan

    for column in WEATHER_COLUMNS:
        _fill_with_site_median(df, column)


def _clean_building_columns(df: pd.DataFrame, reference_year: int) -> None:
    for column in ["square_feet", "year_built", "floor_count"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    year_built_fallback = float(reference_year)
    df["year_built"] = df["year_built"].fillna(_fallback_numeric(df["year_built"], fallback=year_built_fallback))
    df["square_feet"] = df["square_feet"].fillna(_fallback_numeric(df["square_feet"], fallback=0.0))
    df["floor_count"] = df["floor_count"].fillna(_fallback_numeric(df["floor_count"], fallback=0.0))

    df["age"] = (reference_year - df["year_built"] + 1).clip(lower=1)


def _cast_feature_types(df: pd.DataFrame) -> None:
    for column in CATEGORICAL_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(-1).round().astype(np.int16)

    for column in NUMERICAL_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)


def _apply_feature_engineering(df: pd.DataFrame, reference_year: int) -> None:
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    _clean_weather_columns(df)
    _clean_building_columns(df, reference_year=reference_year)
    _add_time_features(df)

    df["wind_direction_bucket"] = _wind_direction_to_bucket(df["wind_direction"])
    df["beaufort_scale"] = _compute_beaufort_scale(df["wind_speed"])

    _cast_feature_types(df)


def engineer_common_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_features = train_df.copy()
    test_features = test_df.copy()

    primary_use_mapping = encode_primary_use(train_features, test_features)

    year_built_all = pd.concat([train_features["year_built"], test_features["year_built"]], axis=0)
    year_built_numeric = pd.to_numeric(year_built_all, errors="coerce").dropna()
    reference_year = int(year_built_numeric.max()) if not year_built_numeric.empty else 2016

    _apply_feature_engineering(train_features, reference_year=reference_year)
    _apply_feature_engineering(test_features, reference_year=reference_year)

    return train_features, test_features, primary_use_mapping


def get_feature_type_summary() -> dict[str, list[str]]:
    return {
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
    }


def to_jsonable_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in mapping.items()}
