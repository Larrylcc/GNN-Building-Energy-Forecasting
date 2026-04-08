from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "processed" / "common"

TARGET_COLUMN = "meter_reading"
LOG_TARGET_COLUMN = "target_log1p"
TIMESTAMP_COLUMN = "timestamp"
ROW_ID_COLUMN = "row_id"

ID_COLUMNS = ["building_id", "meter", "site_id", TIMESTAMP_COLUMN]
TEST_ID_COLUMNS = [ROW_ID_COLUMN, *ID_COLUMNS]

CATEGORICAL_FEATURES = [
    "site_id",
    "building_id",
    "primary_use",
    "meter",
    "wind_direction_bucket",
    "beaufort_scale",
    "month",
    "weekofyear",
    "hour",
    "dayofweek",
    "day",
    "week_of_month",
    "is_weekend",
]

NUMERICAL_FEATURES = [
    "square_feet",
    "year_built",
    "age",
    "floor_count",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_speed",
    "dayofyear",
]

FEATURE_COLUMNS = [*CATEGORICAL_FEATURES, *NUMERICAL_FEATURES]

SUPPORTED_OUTPUT_FORMATS = {"csv", "parquet"}


def normalize_output_formats(output_formats: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for fmt in output_formats:
        fmt_lower = fmt.strip().lower()
        if fmt_lower not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output format '{fmt}'. "
                f"Supported formats: {sorted(SUPPORTED_OUTPUT_FORMATS)}"
            )
        if fmt_lower not in normalized:
            normalized.append(fmt_lower)
    if not normalized:
        raise ValueError("At least one output format must be provided.")
    return tuple(normalized)


@dataclass
class CommonPreprocessingConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    valid_ratio: float = 0.2
    random_seed: int = 42
    output_formats: tuple[str, ...] = ("csv", "parquet")

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        self.output_formats = normalize_output_formats(self.output_formats)
        if not 0.0 < self.valid_ratio < 1.0:
            raise ValueError("valid_ratio must be in the open interval (0, 1).")
