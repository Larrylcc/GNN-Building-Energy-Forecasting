from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DATA_DIR = WORKSPACE_ROOT / "preprocessed_data"

SOURCE_DATA_PATH = PREPROCESSED_DATA_DIR / "screened_preprocessed_train.csv"
TRAIN_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_train.csv"
VALID_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_valid.csv"
TEST_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_test.csv"
SUMMARY_OUTPUT_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_summary.json"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
TARGET_COL = "meter_reading"
LOG1P_TARGET_COL = "meter_reading_log1p"


def load_source_data(source_data_path: Path = SOURCE_DATA_PATH) -> pd.DataFrame:
    train_df = pd.read_csv(source_data_path, parse_dates=["timestamp"], low_memory=False)
    train_df = train_df.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    return train_df


def add_log1p_target(train_df: pd.DataFrame) -> pd.DataFrame:
    meter_reading = pd.to_numeric(train_df[TARGET_COL], errors="coerce")

    if meter_reading.isna().any():
        raise ValueError("The meter_reading column contains missing values.")
    if (meter_reading < 0).any():
        raise ValueError("The meter_reading column contains negative values and cannot be transformed with log1p.")

    train_df[LOG1P_TARGET_COL] = np.log1p(meter_reading.to_numpy(dtype=np.float32))
    return train_df


def build_time_split_boundaries(
    timestamps: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.Index, np.ndarray]:
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    unique_timestamps = pd.Index(timestamps.dropna().drop_duplicates().sort_values())
    if unique_timestamps.empty:
        raise ValueError("No valid timestamps were found.")

    train_end_index = int(np.rint(len(unique_timestamps) * train_ratio))
    valid_end_index = int(np.rint(len(unique_timestamps) * (train_ratio + valid_ratio)))
    split_boundaries = np.array([0, train_end_index, valid_end_index, len(unique_timestamps)], dtype=np.int64)

    for idx in range(1, len(split_boundaries)):
        if split_boundaries[idx] <= split_boundaries[idx - 1]:
            raise ValueError("Time-based split produced an empty train, validation, or test timestamp range.")

    return unique_timestamps, split_boundaries


def split_by_timestamp(
    train_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_timestamps, split_boundaries = build_time_split_boundaries(
        timestamps=train_df["timestamp"],
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    train_end_timestamp = unique_timestamps[split_boundaries[1] - 1]
    valid_start_timestamp = unique_timestamps[split_boundaries[1]]
    valid_end_timestamp = unique_timestamps[split_boundaries[2] - 1]
    test_start_timestamp = unique_timestamps[split_boundaries[2]]

    train_split_df = train_df[train_df["timestamp"] <= train_end_timestamp].copy()
    valid_split_df = train_df[
        (train_df["timestamp"] >= valid_start_timestamp) & (train_df["timestamp"] <= valid_end_timestamp)
    ].copy()
    test_split_df = train_df[train_df["timestamp"] >= test_start_timestamp].copy()

    if train_split_df.empty or valid_split_df.empty or test_split_df.empty:
        raise ValueError("Time-based split produced an empty train, validation, or test set.")

    split_summary = pd.DataFrame(
        [
            build_split_summary_row("train", train_split_df),
            build_split_summary_row("valid", valid_split_df),
            build_split_summary_row("test", test_split_df),
        ]
    )

    return (
        train_split_df.reset_index(drop=True),
        valid_split_df.reset_index(drop=True),
        test_split_df.reset_index(drop=True),
        split_summary,
    )


def build_split_summary_row(split_name: str, split_df: pd.DataFrame) -> dict[str, str | int]:
    return {
        "split": split_name,
        "start_timestamp": str(split_df["timestamp"].min()),
        "end_timestamp": str(split_df["timestamp"].max()),
        "row_count": int(split_df.shape[0]),
    }


def fit_train_minmax(train_split_df: pd.DataFrame) -> tuple[float, float]:
    target = train_split_df[LOG1P_TARGET_COL].to_numpy(dtype=np.float32)
    target_min = float(np.min(target))
    target_max = float(np.max(target))
    return target_min, target_max


def apply_target_minmax(
    split_df: pd.DataFrame,
    target_min: float,
    target_max: float,
) -> pd.DataFrame:
    scale = target_max - target_min

    if scale <= 0:
        split_df[TARGET_COL] = np.zeros(split_df.shape[0], dtype=np.float32)
    else:
        target = split_df[LOG1P_TARGET_COL].to_numpy(dtype=np.float32)
        split_df[TARGET_COL] = ((target - target_min) / scale).astype(np.float32)

    split_df = split_df.drop(columns=[LOG1P_TARGET_COL])
    return split_df


def save_json(data: dict[str, object], output_path: Path) -> Path:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def save_outputs(
    train_split_df: pd.DataFrame,
    valid_split_df: pd.DataFrame,
    test_split_df: pd.DataFrame,
    split_summary: pd.DataFrame,
    target_log1p_min: float,
    target_log1p_max: float,
    source_data_path: Path = SOURCE_DATA_PATH,
    train_output_path: Path = TRAIN_OUTPUT_PATH,
    valid_output_path: Path = VALID_OUTPUT_PATH,
    test_output_path: Path = TEST_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
) -> dict[str, object]:
    train_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_split_df.to_csv(train_output_path, index=False)
    valid_split_df.to_csv(valid_output_path, index=False)
    test_split_df.to_csv(test_output_path, index=False)

    split_summary_records = split_summary.to_dict(orient="records")
    summary = {
        "source_data_path": str(source_data_path),
        "train_output_path": str(train_output_path),
        "valid_output_path": str(valid_output_path),
        "test_output_path": str(test_output_path),
        "summary_output_path": str(summary_output_path),
        "target_preprocess": "log1p_then_train_minmax",
        "target_column": TARGET_COL,
        "target_log1p_min": target_log1p_min,
        "target_log1p_max": target_log1p_max,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "total_row_count": int(train_split_df.shape[0] + valid_split_df.shape[0] + test_split_df.shape[0]),
        "train_row_count": int(train_split_df.shape[0]),
        "valid_row_count": int(valid_split_df.shape[0]),
        "test_row_count": int(test_split_df.shape[0]),
        "splits": split_summary_records,
    }
    save_json(summary, summary_output_path)
    return summary


def preprocess_log1p_minmax(
    source_data_path: Path = SOURCE_DATA_PATH,
) -> dict[str, object]:
    with tqdm(total=6, desc="Log1p + train min-max preprocessing", unit="step") as progress:
        train_df = load_source_data(source_data_path=source_data_path)
        progress.update(1)

        train_df = add_log1p_target(train_df=train_df)
        progress.update(1)

        train_split_df, valid_split_df, test_split_df, split_summary = split_by_timestamp(train_df=train_df)
        progress.update(1)

        target_log1p_min, target_log1p_max = fit_train_minmax(train_split_df=train_split_df)
        progress.update(1)

        train_split_df = apply_target_minmax(train_split_df, target_log1p_min, target_log1p_max)
        valid_split_df = apply_target_minmax(valid_split_df, target_log1p_min, target_log1p_max)
        test_split_df = apply_target_minmax(test_split_df, target_log1p_min, target_log1p_max)
        progress.update(1)

        summary = save_outputs(
            train_split_df=train_split_df,
            valid_split_df=valid_split_df,
            test_split_df=test_split_df,
            split_summary=split_summary,
            target_log1p_min=target_log1p_min,
            target_log1p_max=target_log1p_max,
            source_data_path=source_data_path,
        )
        progress.update(1)

    return summary


def main() -> dict[str, object]:
    summary = preprocess_log1p_minmax()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
