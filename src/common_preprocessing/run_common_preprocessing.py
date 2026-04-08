from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common_preprocessing.config import CommonPreprocessingConfig, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR
from common_preprocessing.pipeline import run_common_preprocessing_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build common train/valid/test datasets for ASHRAE forecasting.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing raw ASHRAE CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save common_train/common_valid/common_test and common_schema.json.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Validation ratio for time-based split.")
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=["csv", "parquet"],
        help="Output formats. Supported: csv parquet",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Reserved for reproducibility metadata.")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    config = CommonPreprocessingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        valid_ratio=args.valid_ratio,
        random_seed=args.random_seed,
        output_formats=tuple(args.output_formats),
    )

    artifacts = run_common_preprocessing_pipeline(config)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))
    return artifacts


if __name__ == "__main__":
    main()
