from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common_preprocessing.config import TIMESTAMP_COLUMN


@dataclass
class TimeSplitMeta:
    cutoff_timestamp: str
    train_rows: int
    valid_rows: int
    train_unique_timestamps: int
    valid_unique_timestamps: int


def time_based_train_valid_split(
    train_df: pd.DataFrame,
    valid_ratio: float,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, TimeSplitMeta]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in the open interval (0, 1).")

    unique_timestamps = np.sort(train_df[timestamp_column].dropna().unique())
    if unique_timestamps.size < 2:
        raise ValueError("At least two unique timestamps are required for time-based split.")

    cutoff_index = int(np.floor((1.0 - valid_ratio) * unique_timestamps.size))
    cutoff_index = min(max(cutoff_index, 1), unique_timestamps.size - 1)
    cutoff_timestamp = unique_timestamps[cutoff_index]

    train_mask = train_df[timestamp_column] < cutoff_timestamp
    valid_mask = ~train_mask

    split_train = train_df.loc[train_mask].copy()
    split_valid = train_df.loc[valid_mask].copy()

    split_meta = TimeSplitMeta(
        cutoff_timestamp=pd.Timestamp(cutoff_timestamp).isoformat(),
        train_rows=int(split_train.shape[0]),
        valid_rows=int(split_valid.shape[0]),
        train_unique_timestamps=int(split_train[timestamp_column].nunique()),
        valid_unique_timestamps=int(split_valid[timestamp_column].nunique()),
    )

    return split_train, split_valid, split_meta
