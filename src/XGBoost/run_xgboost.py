from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from XGBoost.config import DEFAULT_COMMON_DATA_DIR, DEFAULT_OUTPUT_DIR, XGBoostConfig
from XGBoost.data_adapter import ensure_timestamp_dtype, load_common_datasets
from XGBoost.metrics import build_metrics_by_split
from XGBoost.trainer import predict_test, train_single_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost on common ASHRAE datasets.")
    parser.add_argument("--common-data-dir", type=Path, default=DEFAULT_COMMON_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preferred-input-format", choices=["csv", "parquet"], default="parquet")

    parser.add_argument("--objective", default="reg:squarederror")
    parser.add_argument("--eval-metric", default="rmse")
    parser.add_argument("--training-target-col", default="target_log1p")

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", dest="use_gpu", action="store_true")
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=None)

    parser.add_argument("--gpu-device-ordinal", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--verbose-eval", type=int, default=100)
    parser.add_argument("--test-chunk-size", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> XGBoostConfig:
    default_config = XGBoostConfig()
    use_gpu = default_config.use_gpu if args.use_gpu is None else args.use_gpu

    return XGBoostConfig(
        common_data_dir=args.common_data_dir,
        output_dir=args.output_dir,
        preferred_input_format=args.preferred_input_format,
        objective=args.objective,
        eval_metric=args.eval_metric,
        training_target_column=args.training_target_col,
        use_gpu=use_gpu,
        gpu_device_ordinal=args.gpu_device_ordinal,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
        test_chunk_size=args.test_chunk_size,
        seed=args.seed,
    )


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_valid_predictions_df(valid_df: pd.DataFrame, valid_pred: pd.Series, id_columns: list[str]) -> pd.DataFrame:
    kept_id_columns = [column for column in id_columns if column in valid_df.columns]
    output = valid_df[kept_id_columns].copy()
    output["meter_reading_true"] = valid_df["meter_reading"].to_numpy()
    output["meter_reading_pred"] = valid_pred.to_numpy()
    return output


def _build_test_predictions_df(test_df: pd.DataFrame, test_pred: pd.Series) -> pd.DataFrame:
    if "row_id" in test_df.columns:
        return pd.DataFrame({"row_id": test_df["row_id"].to_numpy(), "meter_reading": test_pred.to_numpy()})
    return pd.DataFrame({"meter_reading": test_pred.to_numpy()})


def run_with_config(config: XGBoostConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_common_datasets(
        common_data_dir=config.common_data_dir,
        preferred_format=config.preferred_input_format,
    )
    ensure_timestamp_dtype(datasets)

    train_result = train_single_split(
        train_df=datasets.train_df,
        valid_df=datasets.valid_df,
        feature_columns=datasets.feature_columns,
        config=config,
    )

    test_predictions = predict_test(
        booster=train_result.booster,
        test_df=datasets.test_df,
        feature_columns=datasets.feature_columns,
        config=config,
    )

    split_metrics = {
        "train": train_result.train_metrics,
        "valid": train_result.valid_metrics,
    }
    metrics_by_split_df = build_metrics_by_split(split_metrics)

    valid_pred_series = pd.Series(train_result.valid_predictions_raw, name="meter_reading_pred")
    test_pred_series = pd.Series(test_predictions, name="meter_reading")

    valid_predictions_df = _build_valid_predictions_df(
        valid_df=datasets.valid_df,
        valid_pred=valid_pred_series,
        id_columns=datasets.id_columns,
    )
    submission_df = _build_test_predictions_df(datasets.test_df, test_pred_series)

    metrics_json_path = config.output_dir / "metrics.json"
    metrics_csv_path = config.output_dir / "metrics_by_split.csv"
    importance_path = config.output_dir / "xgboost_feature_importance.csv"
    eval_history_path = config.output_dir / "xgboost_eval_history.json"
    valid_predictions_path = config.output_dir / "xgboost_valid_predictions.csv"
    submission_path = config.output_dir / "xgboost_submission.csv"

    _save_json(metrics_json_path, split_metrics)
    metrics_by_split_df.to_csv(metrics_csv_path, index=False)
    train_result.feature_importance.to_csv(importance_path, index=False)
    _save_json(eval_history_path, train_result.evals_result)
    valid_predictions_df.to_csv(valid_predictions_path, index=False)
    submission_df.to_csv(submission_path, index=False)

    summary = {
        "xgboost_version": getattr(xgb, "__version__", "unknown"),
        "config": config.to_serializable_dict(),
        "feature_count": len(datasets.feature_columns),
        "feature_columns": datasets.feature_columns,
        "training_target_column": config.training_target_column,
        "evaluation_metric_space": "original_scale",
        "split_metrics": split_metrics,
        "artifacts": {
            "metrics_json": str(metrics_json_path),
            "metrics_by_split_csv": str(metrics_csv_path),
            "feature_importance_csv": str(importance_path),
            "eval_history_json": str(eval_history_path),
            "valid_predictions_csv": str(valid_predictions_path),
            "submission_csv": str(submission_path),
        },
    }

    summary_path = config.output_dir / "xgboost_run_summary.json"
    _save_json(summary_path, summary)
    summary["artifacts"]["summary_json"] = str(summary_path)

    return summary


def main() -> dict:
    args = parse_args()
    config = build_config(args)
    summary = run_with_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
