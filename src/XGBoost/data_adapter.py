from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common_preprocessing.config import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LOG_TARGET_COLUMN,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
    TEST_ID_COLUMNS,
)


@dataclass
class CommonDatasetBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]
    id_columns: list[str]
    test_id_columns: list[str]
    target_columns: list[str]
    schema: dict[str, Any]


def _read_split(common_data_dir: Path, split_name: str, preferred_format: str) -> pd.DataFrame:
    candidate_formats = [preferred_format]
    for fmt in ["parquet", "csv"]:
        if fmt not in candidate_formats:
            candidate_formats.append(fmt)

    for fmt in candidate_formats:
        path = common_data_dir / f"common_{split_name}.{fmt}"
        if not path.exists():
            continue
        if fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, parse_dates=["timestamp"])

    raise FileNotFoundError(
        f"Could not find common_{split_name} in formats {candidate_formats} under {common_data_dir}."
    )


def _load_schema(common_data_dir: Path) -> dict[str, Any]:
    schema_path = common_data_dir / "common_schema.json"
    if not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _infer_feature_columns(train_df: pd.DataFrame) -> list[str]:
    excluded = {
        ROW_ID_COLUMN,
        *ID_COLUMNS,
        TARGET_COLUMN,
        LOG_TARGET_COLUMN,
    }
    inferred = [column for column in train_df.columns if column not in excluded]
    return inferred


def load_common_datasets(common_data_dir: Path, preferred_format: str = "parquet") -> CommonDatasetBundle:
    schema = _load_schema(common_data_dir)
    train_df = _read_split(common_data_dir, split_name="train", preferred_format=preferred_format)
    valid_df = _read_split(common_data_dir, split_name="valid", preferred_format=preferred_format)
    test_df = _read_split(common_data_dir, split_name="test", preferred_format=preferred_format)

    feature_columns = schema.get("feature_columns", _infer_feature_columns(train_df))
    id_columns = schema.get("id_columns", ID_COLUMNS)
    test_id_columns = schema.get("test_id_columns", TEST_ID_COLUMNS)
    target_columns = schema.get("target_columns", [TARGET_COLUMN, LOG_TARGET_COLUMN])

    missing_features = [column for column in feature_columns if column not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns in common_train: {missing_features}")

    for split_name, frame in [("common_valid", valid_df), ("common_test", test_df)]:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns in {split_name}: {missing}")

    return CommonDatasetBundle(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        feature_columns=feature_columns,
        id_columns=id_columns,
        test_id_columns=test_id_columns,
        target_columns=target_columns,
        schema=schema,
    )


def ensure_timestamp_dtype(bundle: CommonDatasetBundle) -> None:
    for frame in [bundle.train_df, bundle.valid_df, bundle.test_df]:
        if "timestamp" in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")


def default_feature_columns() -> list[str]:
    return FEATURE_COLUMNS
