from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMON_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "common"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "XGBoost"


@dataclass
class XGBoostConfig:
    common_data_dir: Path = DEFAULT_COMMON_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    preferred_input_format: str = "parquet"

    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    training_target_column: str = "target_log1p"
    raw_target_column: str = "meter_reading"

    use_gpu: bool = True
    gpu_device_ordinal: int = 0

    seed: int = 42
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    verbose_eval: int = 100
    test_chunk_size: int = 500_000

    learning_rate: float = 0.05
    max_depth: int = 10
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    max_bin: int = 256

    def __post_init__(self) -> None:
        self.common_data_dir = Path(self.common_data_dir)
        self.output_dir = Path(self.output_dir)
        self.preferred_input_format = self.preferred_input_format.lower()
        if self.preferred_input_format not in {"csv", "parquet"}:
            raise ValueError("preferred_input_format must be one of: csv, parquet")
        if self.training_target_column not in {"target_log1p", "meter_reading"}:
            raise ValueError("training_target_column must be 'target_log1p' or 'meter_reading'.")

    def to_training_params(self) -> dict:
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "max_bin": self.max_bin,
            "seed": self.seed,
            "nthread": 0,
            "verbosity": 1,
        }

    def to_serializable_dict(self) -> dict:
        return {
            "common_data_dir": str(self.common_data_dir),
            "output_dir": str(self.output_dir),
            "preferred_input_format": self.preferred_input_format,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "training_target_column": self.training_target_column,
            "raw_target_column": self.raw_target_column,
            "use_gpu": self.use_gpu,
            "gpu_device_ordinal": self.gpu_device_ordinal,
            "seed": self.seed,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose_eval": self.verbose_eval,
            "test_chunk_size": self.test_chunk_size,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "max_bin": self.max_bin,
        }
