from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common_preprocessing.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_ROOT,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CommonPreprocessingConfig,
)
from common_preprocessing.pipeline import run_common_preprocessing_pipeline
from XGBoost.config import DEFAULT_OUTPUT_DIR, XGBoostConfig
from XGBoost.run_xgboost import run_with_config
from XGBoost.trainer import build_xgb_params as _build_xgb_params


SEED = 42
DATA_ROOT = DEFAULT_DATA_ROOT
COMMON_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "common"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

CATEGORICAL_COLS = CATEGORICAL_FEATURES
NUMERICAL_COLS = NUMERICAL_FEATURES
FEATURE_COLS = FEATURE_COLUMNS


def build_xgb_params(seed: int = SEED) -> dict:
    config = XGBoostConfig(seed=seed)
    return _build_xgb_params(config)


def main(data_root: Path = DATA_ROOT, output_dir: Path = OUTPUT_DIR) -> dict:
    preprocessing_config = CommonPreprocessingConfig(
        data_root=Path(data_root),
        output_dir=COMMON_OUTPUT_DIR,
        valid_ratio=0.2,
        output_formats=("csv", "parquet"),
        random_seed=SEED,
    )
    run_common_preprocessing_pipeline(preprocessing_config)

    xgb_config = XGBoostConfig(
        common_data_dir=COMMON_OUTPUT_DIR,
        output_dir=Path(output_dir),
        seed=SEED,
        objective="reg:squarederror",
        eval_metric="rmse",
        training_target_column="target_log1p",
    )
    summary = run_with_config(xgb_config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
