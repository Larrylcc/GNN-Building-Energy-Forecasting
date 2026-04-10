from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_preprocessing.config import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LOG_TARGET_COLUMN,
    TARGET_COLUMN,
    TEST_ID_COLUMNS,
    TIMESTAMP_COLUMN,
    CommonPreprocessingConfig,
)
from common_preprocessing.features import (
    engineer_common_features,
    get_feature_type_summary,
    merge_raw_tables,
    to_jsonable_mapping,
)
from common_preprocessing.loaders import load_raw_data
from common_preprocessing.split import time_based_train_valid_split


def _unique_in_order(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_columns: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        unique_columns.append(column)
    return unique_columns


def _assert_no_duplicate_columns(df: pd.DataFrame, split_name: str) -> None:
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate columns found in {split_name}: {duplicated}")


def _ensure_columns(df: pd.DataFrame, columns: list[str], split_name: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {split_name}: {missing}")
    output_df = df[columns].copy()
    _assert_no_duplicate_columns(output_df, split_name=split_name)
    return output_df


def _save_dataframe(df: pd.DataFrame, base_name: str, output_dir: Path, output_formats: tuple[str, ...]) -> dict[str, str]:
    output_paths: dict[str, str] = {}

    for fmt in output_formats:
        path = output_dir / f"{base_name}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            try:
                df.to_parquet(path, index=False)
            except ImportError as error:
                raise RuntimeError(
                    "Saving parquet requires an engine such as 'pyarrow' or 'fastparquet'."
                ) from error
        else:
            raise ValueError(f"Unsupported output format '{fmt}'.")
        output_paths[fmt] = str(path)

    return output_paths


def _build_dtype_summary(df: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in df.dtypes.items()}


def run_common_preprocessing_pipeline(config: CommonPreprocessingConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_data(config.data_root)
    merged_train_df, merged_test_df = merge_raw_tables(raw_data)

    train_features_df, test_features_df, primary_use_mapping = engineer_common_features(
        train_df=merged_train_df,
        test_df=merged_test_df,
    )

    train_features_df[TARGET_COLUMN] = (
        pd.to_numeric(train_features_df[TARGET_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0).astype(np.float32)
    )
    train_features_df[LOG_TARGET_COLUMN] = np.log1p(train_features_df[TARGET_COLUMN]).astype(np.float32)

    split_train_df, split_valid_df, split_meta = time_based_train_valid_split(
        train_df=train_features_df,
        valid_ratio=config.valid_ratio,
        timestamp_column=TIMESTAMP_COLUMN,
    )

    train_columns = _unique_in_order([*ID_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN, LOG_TARGET_COLUMN])
    test_columns = _unique_in_order([*TEST_ID_COLUMNS, *FEATURE_COLUMNS])

    common_train_df = _ensure_columns(split_train_df, train_columns, split_name="common_train")
    common_valid_df = _ensure_columns(split_valid_df, train_columns, split_name="common_valid")
    common_test_df = _ensure_columns(test_features_df, test_columns, split_name="common_test")

    common_train_df = common_train_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    common_valid_df = common_valid_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    common_test_df = common_test_df.sort_values([TIMESTAMP_COLUMN, "row_id"]).reset_index(drop=True)

    train_paths = _save_dataframe(common_train_df, "common_train", config.output_dir, config.output_formats)
    valid_paths = _save_dataframe(common_valid_df, "common_valid", config.output_dir, config.output_formats)
    test_paths = _save_dataframe(common_test_df, "common_test", config.output_dir, config.output_formats)

    feature_summary = get_feature_type_summary()
    schema = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data_root": str(config.data_root),
            "output_dir": str(config.output_dir),
            "valid_ratio": config.valid_ratio,
            "random_seed": config.random_seed,
            "output_formats": list(config.output_formats),
        },
        "split": {
            "type": "time",
            "cutoff_timestamp": split_meta.cutoff_timestamp,
            "train_rows": split_meta.train_rows,
            "valid_rows": split_meta.valid_rows,
            "train_unique_timestamps": split_meta.train_unique_timestamps,
            "valid_unique_timestamps": split_meta.valid_unique_timestamps,
        },
        "id_columns": ID_COLUMNS,
        "test_id_columns": TEST_ID_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": [TARGET_COLUMN, LOG_TARGET_COLUMN],
        "categorical_features": feature_summary["categorical_features"],
        "numerical_features": feature_summary["numerical_features"],
        "primary_use_mapping": to_jsonable_mapping(primary_use_mapping),
        "columns_by_split": {
            "train": common_train_df.columns.tolist(),
            "valid": common_valid_df.columns.tolist(),
            "test": common_test_df.columns.tolist(),
        },
        "dtypes": {
            "train": _build_dtype_summary(common_train_df),
            "valid": _build_dtype_summary(common_valid_df),
            "test": _build_dtype_summary(common_test_df),
        },
    }

    schema_path = config.output_dir / "common_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    artifacts: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "train_paths": train_paths,
        "valid_paths": valid_paths,
        "test_paths": test_paths,
        "schema_path": str(schema_path),
        "split": schema["split"],
        "feature_count": len(FEATURE_COLUMNS),
    }

    return artifacts
