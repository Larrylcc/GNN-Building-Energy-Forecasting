from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RawDataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    weather_train: pd.DataFrame
    weather_test: pd.DataFrame
    building_metadata: pd.DataFrame


def load_raw_data(data_root: Path) -> RawDataBundle:
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

    return RawDataBundle(
        train=train_df,
        test=test_df,
        weather_train=weather_train_df,
        weather_test=weather_test_df,
        building_metadata=building_meta_df,
    )
