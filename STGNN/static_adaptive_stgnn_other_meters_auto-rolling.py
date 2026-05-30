from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from data_preprocess.data_preprocess import CATEGORICAL_COLS, DATA_ROOT, NUMERICAL_COLS, PREPROCESSED_DATA_DIR  # noqa: E402
from STGNN.static_adaptive_graph import (  # noqa: E402
    StaticAdaptiveSTGNNModelConfig,
    StaticAdaptiveZeroInflatedSTGNNGRURegressor,
    StaticGraphConfig,
    build_static_adjacency,
    zero_inflated_masked_loss,
)


def load_base_module():
    module_path = WORKSPACE_ROOT / "STGNN" / "stgnn_gcn_gru_auto-rolling.py"
    spec = importlib.util.spec_from_file_location("static_adaptive_stgnn_base", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load base static adaptive STGNN module from {module_path}.")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


class MeterConfig(NamedTuple):
    meter_id: int
    meter_name: str


class MeterPaths(NamedTuple):
    data_dir: Path
    output_dir: Path
    train_data_path: Path
    valid_data_path: Path
    test_data_path: Path
    preprocessing_summary_path: Path
    model_path: Path
    model_params_path: Path
    training_history_path: Path
    split_summary_path: Path
    validation_metrics_path: Path
    test_metrics_path: Path
    run_summary_path: Path
    static_adjacency_path: Path
    learned_adjacency_path: Path
    prediction_weighted_adjacency_path: Path
    static_topk_edges_path: Path
    learned_topk_edges_path: Path
    prediction_weighted_topk_edges_path: Path


OUTPUT_ROOT_DIR = WORKSPACE_ROOT / "STGNN" / "static_adaptive_stgnn_other_meters_auto-rolling_outputs"
OVERALL_RUN_SUMMARY_PATH = OUTPUT_ROOT_DIR / "static_adaptive_stgnn_other_meters_run_summary.json"
MODEL_NAME = "StaticAdaptiveZeroInflatedSTGNN"
GRAPH_MODE = "static_adaptive"
TARGET_COL = BASE.TARGET_COL

SEED = BASE.SEED
WINDOW_SIZE = BASE.WINDOW_SIZE
STRIDE = BASE.STRIDE
BATCH_SIZE = BASE.BATCH_SIZE
EVAL_BATCH_SIZE = BASE.EVAL_BATCH_SIZE
EPOCHS = BASE.EPOCHS
PATIENCE = BASE.PATIENCE
LEARNING_RATE = BASE.LEARNING_RATE
WEIGHT_DECAY = BASE.WEIGHT_DECAY
GRAD_CLIP_NORM = BASE.GRAD_CLIP_NORM
GCN_DIM = BASE.GCN_DIM
EMBED_DIM = BASE.EMBED_DIM
GRAPH_TOP_K = BASE.GRAPH_TOP_K
GCN_LAYERS = BASE.GCN_LAYERS
HIDDEN_SIZE = BASE.HIDDEN_SIZE
NUM_LAYERS = BASE.NUM_LAYERS
DROPOUT = BASE.DROPOUT
TRAIN_RATIO = BASE.TRAIN_RATIO
VALID_RATIO = BASE.VALID_RATIO
TEST_RATIO = BASE.TEST_RATIO
ONE_HOUR = BASE.ONE_HOUR


METER_CONFIGS = {
    1: MeterConfig(meter_id=1, meter_name="chilled_water"),
    2: MeterConfig(meter_id=2, meter_name="steam"),
    3: MeterConfig(meter_id=3, meter_name="hot_water"),
}


@dataclass(frozen=True)
class ZeroInflatedTrainingConfig:
    window_size: int = WINDOW_SIZE
    stride: int = STRIDE
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    epochs: int = EPOCHS
    patience: int = PATIENCE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip_norm: float = GRAD_CLIP_NORM
    zero_classification_weight: float = 0.2
    positive_regression_weight: float = 2.0
    zero_suppression_weight: float = 0.3
    zero_threshold: float = 0.5
    seed: int = SEED


STGNNModelConfig = StaticAdaptiveSTGNNModelConfig


def build_meter_paths(
    config: MeterConfig,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> MeterPaths:
    data_dir = preprocessed_data_dir / f"meter_{config.meter_id}"
    output_dir = output_root_dir / f"meter_{config.meter_id}"
    prefix = "static_adaptive_zero_inflated_stgnn"

    return MeterPaths(
        data_dir=data_dir,
        output_dir=output_dir,
        train_data_path=data_dir / "log1p_minmax_train.csv",
        valid_data_path=data_dir / "log1p_minmax_valid.csv",
        test_data_path=data_dir / "log1p_minmax_test.csv",
        preprocessing_summary_path=data_dir / "log1p_minmax_summary.json",
        model_path=output_dir / f"{prefix}_final_model.pt",
        model_params_path=output_dir / f"{prefix}_model_params.json",
        training_history_path=output_dir / f"{prefix}_training_history.csv",
        split_summary_path=output_dir / f"{prefix}_time_series_split.csv",
        validation_metrics_path=output_dir / f"{prefix}_validation_metrics.json",
        test_metrics_path=output_dir / f"{prefix}_test_metrics.json",
        run_summary_path=output_dir / f"{prefix}_run_summary.json",
        static_adjacency_path=output_dir / f"{prefix}_static_adjacency.npy",
        learned_adjacency_path=output_dir / f"{prefix}_learned_adjacency.npy",
        prediction_weighted_adjacency_path=output_dir / f"{prefix}_prediction_weighted_adjacency.npy",
        static_topk_edges_path=output_dir / f"{prefix}_static_topk_edges.csv",
        learned_topk_edges_path=output_dir / f"{prefix}_learned_topk_edges.csv",
        prediction_weighted_topk_edges_path=output_dir / f"{prefix}_prediction_weighted_topk_edges.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train zero-inflated static-adaptive STGNN models for other meters.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small synthetic end-to-end verification instead of full training.",
    )
    parser.add_argument(
        "--meter-id",
        type=int,
        choices=tuple(METER_CONFIGS),
        default=None,
        help="Train only one non-electricity meter. Defaults to all three.",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("static", "learned", "static_adaptive"),
        default=GRAPH_MODE,
        help="Graph branch to use for the model.",
    )
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def build_thresholded_predictions(
    predictions: np.ndarray,
    zero_probabilities: np.ndarray,
    zero_threshold: float = 0.5,
) -> np.ndarray:
    thresholded = np.asarray(predictions, dtype=np.float32).copy()
    thresholded[np.asarray(zero_probabilities, dtype=np.float32) >= zero_threshold] = 0.0
    return thresholded


def finalize_zero_state_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
    prediction_on_zero_sum: float,
    zero_target_count: int,
) -> dict[str, float | int]:
    total = true_positive + false_positive + false_negative + true_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    false_positive_denominator = false_positive + true_negative
    false_negative_denominator = false_negative + true_positive

    precision = 0.0 if precision_denominator == 0 else true_positive / precision_denominator
    recall = 0.0 if recall_denominator == 0 else true_positive / recall_denominator
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "zero_true_positive_count": int(true_positive),
        "zero_false_positive_count": int(false_positive),
        "zero_false_negative_count": int(false_negative),
        "zero_true_negative_count": int(true_negative),
        "zero_accuracy": 0.0 if total == 0 else (true_positive + true_negative) / total,
        "zero_precision": float(precision),
        "zero_recall": float(recall),
        "zero_f1": float(f1),
        "zero_false_positive_rate": 0.0
        if false_positive_denominator == 0
        else false_positive / false_positive_denominator,
        "zero_false_negative_rate": 0.0
        if false_negative_denominator == 0
        else false_negative / false_negative_denominator,
        "mean_prediction_on_zero": 0.0 if zero_target_count == 0 else prediction_on_zero_sum / zero_target_count,
    }


def compute_zero_state_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    zero_probabilities: np.ndarray,
    zero_threshold: float = 0.5,
) -> dict[str, float | int]:
    target_zero = np.asarray(targets, dtype=np.float32) <= 0.0
    predicted_zero = np.asarray(zero_probabilities, dtype=np.float32) >= zero_threshold
    prediction_values = np.asarray(predictions, dtype=np.float32)

    true_positive = int(np.sum(target_zero & predicted_zero))
    false_positive = int(np.sum(~target_zero & predicted_zero))
    false_negative = int(np.sum(target_zero & ~predicted_zero))
    true_negative = int(np.sum(~target_zero & ~predicted_zero))
    prediction_on_zero_sum = float(np.sum(prediction_values[target_zero], dtype=np.float64))
    zero_target_count = int(np.sum(target_zero))
    return finalize_zero_state_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        prediction_on_zero_sum=prediction_on_zero_sum,
        zero_target_count=zero_target_count,
    )


def update_regression_sums(
    targets: np.ndarray,
    predictions: np.ndarray,
    sums: dict[str, float],
) -> None:
    errors = predictions - targets
    sums["squared_error_sum"] += float(np.sum(errors * errors, dtype=np.float64))
    sums["absolute_error_sum"] += float(np.sum(np.abs(errors), dtype=np.float64))
    sums["smape_ratio_sum"] += BASE.compute_smape_sum(targets, predictions)
    
    # Calculate RMSLE squared errors
    clipped_true = np.clip(targets, a_min=0.0, a_max=None)
    clipped_pred = np.clip(predictions, a_min=0.0, a_max=None)
    log_errors = np.log1p(clipped_pred) - np.log1p(clipped_true)
    sums["rmsle_squared_error_sum"] += float(np.sum(log_errors * log_errors, dtype=np.float64))


def finalize_metrics(
    row_count: int,
    skipped_row_count: int,
    squared_error_sum: float,
    absolute_error_sum: float,
    smape_ratio_sum: float,
    rmsle_squared_error_sum: float,
    y_sum: float,
    y_squared_sum: float,
) -> dict[str, float | int]:
    if row_count == 0:
        raise ValueError("Rolling evaluation did not produce any predictions.")

    mse = squared_error_sum / row_count
    mae = absolute_error_sum / row_count
    rmse = float(np.sqrt(mse))
    smape = smape_ratio_sum / row_count * 100.0
    rmsle = float(np.sqrt(rmsle_squared_error_sum / row_count))
    total_sum_of_squares = y_squared_sum - (y_sum * y_sum / row_count)
    r2 = 0.0 if total_sum_of_squares <= 0.0 else 1.0 - squared_error_sum / total_sum_of_squares

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "smape": float(smape),
        "rmse": float(rmse),
        "rmsle": float(rmsle),
        "evaluated_row_count": int(row_count),
        "skipped_row_count": int(skipped_row_count),
        "window_count": int(row_count),
    }


def compute_zero_pos_weight(train_data) -> float:
    observed_targets = train_data.targets[train_data.target_mask]
    zero_count = int(np.sum(observed_targets <= 0.0))
    positive_count = int(np.sum(observed_targets > 0.0))
    if zero_count == 0:
        return 1.0
    return float(positive_count / zero_count)


def predict_zero_inflated_global_window(
    model: StaticAdaptiveZeroInflatedSTGNNGRURegressor,
    history_rows,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    sequence = np.expand_dims(np.stack(history_rows, axis=0), axis=0)
    sequence = np.ascontiguousarray(sequence, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(sequence).to(device=device, non_blocking=True).float()
        output = model(batch)
        predictions = output.predictions.detach().cpu().numpy()[0].astype(np.float32)
        zero_probabilities = output.zero_probabilities.detach().cpu().numpy()[0].astype(np.float32)

    predictions = np.nan_to_num(predictions, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    zero_probabilities = np.nan_to_num(zero_probabilities, copy=False, nan=1.0, posinf=1.0, neginf=0.0)
    return predictions, zero_probabilities


def rolling_evaluate_split(
    model: StaticAdaptiveZeroInflatedSTGNNGRURegressor,
    split_data,
    history_parts: list,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    device: torch.device,
    window_size: int,
    eval_batch_size: int,
    zero_threshold: float,
) -> dict[str, float | int]:
    if eval_batch_size != 1:
        raise ValueError("STGNN auto-rolling evaluation must use eval_batch_size=1.")

    meter_feature_index = input_feature_cols.index(TARGET_COL)
    history_rows, last_timestamp = BASE.collect_global_history_suffix(
        first_timestamp=split_data.timestamps[0],
        history_parts=history_parts,
        window_size=window_size,
    )

    evaluated_row_count = 0
    skipped_row_count = int(split_data.unknown_node_row_count)
    soft_sums = {"squared_error_sum": 0.0, "absolute_error_sum": 0.0, "smape_ratio_sum": 0.0, "rmsle_squared_error_sum": 0.0}
    thresholded_sums = {"squared_error_sum": 0.0, "absolute_error_sum": 0.0, "smape_ratio_sum": 0.0, "rmsle_squared_error_sum": 0.0}
    y_sum = 0.0
    y_squared_sum = 0.0
    zero_true_positive = 0
    zero_false_positive = 0
    zero_false_negative = 0
    zero_true_negative = 0
    prediction_on_zero_sum = 0.0
    zero_target_count = 0
    evaluated_timestamp_count = 0
    skipped_timestamp_count = 0

    progress_iter = tqdm(
        range(split_data.timestamps.shape[0]),
        total=int(split_data.timestamps.shape[0]),
        desc=f"Rolling {split_data.name}",
        unit="timestamp",
    )
    for time_index in progress_iter:
        timestamp = split_data.timestamps[time_index]
        current_mask = split_data.target_mask[time_index]
        observed_count = int(np.sum(current_mask))
        expected_previous_timestamp = timestamp - ONE_HOUR

        if last_timestamp is None or last_timestamp != expected_previous_timestamp:
            history_rows.clear()
            current_feature_rows = BASE.build_current_feature_rows(
                split_data=split_data,
                time_index=time_index,
                history_rows=history_rows,
            )
            history_rows.append(current_feature_rows)
            last_timestamp = timestamp
            skipped_row_count += observed_count
            skipped_timestamp_count += 1
            continue

        if len(history_rows) < window_size:
            current_feature_rows = BASE.build_current_feature_rows(
                split_data=split_data,
                time_index=time_index,
                history_rows=history_rows,
            )
            history_rows.append(current_feature_rows)
            last_timestamp = timestamp
            skipped_row_count += observed_count
            skipped_timestamp_count += 1
            continue

        predictions, zero_probabilities = predict_zero_inflated_global_window(
            model=model,
            history_rows=history_rows,
            device=device,
        )

        if observed_count > 0:
            batch_targets = split_data.targets[time_index, current_mask].astype(np.float32, copy=False)
            batch_predictions = predictions[current_mask].astype(np.float32, copy=False)
            batch_zero_probabilities = zero_probabilities[current_mask].astype(np.float32, copy=False)
            batch_thresholded_predictions = build_thresholded_predictions(
                predictions=batch_predictions,
                zero_probabilities=batch_zero_probabilities,
                zero_threshold=zero_threshold,
            )

            evaluated_row_count += int(batch_targets.shape[0])
            update_regression_sums(targets=batch_targets, predictions=batch_predictions, sums=soft_sums)
            update_regression_sums(
                targets=batch_targets,
                predictions=batch_thresholded_predictions,
                sums=thresholded_sums,
            )
            y_sum += float(np.sum(batch_targets, dtype=np.float64))
            y_squared_sum += float(np.sum(batch_targets * batch_targets, dtype=np.float64))

            target_zero = batch_targets <= 0.0
            predicted_zero = batch_zero_probabilities >= zero_threshold
            zero_true_positive += int(np.sum(target_zero & predicted_zero))
            zero_false_positive += int(np.sum(~target_zero & predicted_zero))
            zero_false_negative += int(np.sum(target_zero & ~predicted_zero))
            zero_true_negative += int(np.sum(~target_zero & ~predicted_zero))
            prediction_on_zero_sum += float(np.sum(batch_predictions[target_zero], dtype=np.float64))
            zero_target_count += int(np.sum(target_zero))
            evaluated_timestamp_count += 1

        current_feature_rows = BASE.build_current_feature_rows(
            split_data=split_data,
            time_index=time_index,
            history_rows=history_rows,
        )
        writeback_predictions = np.clip(predictions, 0.0, 1.0)
        predicted_feature_rows = BASE.build_predicted_feature_rows(
            current_feature_rows=current_feature_rows,
            predictions=writeback_predictions,
            meter_feature_index=meter_feature_index,
            feature_means=feature_means,
            feature_stds=feature_stds,
        )
        history_rows.append(predicted_feature_rows)
        last_timestamp = timestamp

    metrics = finalize_metrics(
        row_count=evaluated_row_count,
        skipped_row_count=skipped_row_count,
        squared_error_sum=soft_sums["squared_error_sum"],
        absolute_error_sum=soft_sums["absolute_error_sum"],
        smape_ratio_sum=soft_sums["smape_ratio_sum"],
        rmsle_squared_error_sum=soft_sums["rmsle_squared_error_sum"],
        y_sum=y_sum,
        y_squared_sum=y_squared_sum,
    )
    thresholded_metrics = finalize_metrics(
        row_count=evaluated_row_count,
        skipped_row_count=skipped_row_count,
        squared_error_sum=thresholded_sums["squared_error_sum"],
        absolute_error_sum=thresholded_sums["absolute_error_sum"],
        smape_ratio_sum=thresholded_sums["smape_ratio_sum"],
        rmsle_squared_error_sum=thresholded_sums["rmsle_squared_error_sum"],
        y_sum=y_sum,
        y_squared_sum=y_squared_sum,
    )
    metrics.update({f"thresholded_{key}": value for key, value in thresholded_metrics.items()})
    metrics.update(
        finalize_zero_state_metrics(
            true_positive=zero_true_positive,
            false_positive=zero_false_positive,
            false_negative=zero_false_negative,
            true_negative=zero_true_negative,
            prediction_on_zero_sum=prediction_on_zero_sum,
            zero_target_count=zero_target_count,
        )
    )
    metrics["zero_threshold"] = float(zero_threshold)
    metrics["evaluated_timestamp_count"] = int(evaluated_timestamp_count)
    metrics["skipped_timestamp_count"] = int(skipped_timestamp_count)
    metrics["unknown_node_row_count"] = int(split_data.unknown_node_row_count)
    return metrics


def train_stgnn_model(
    train_data,
    valid_data,
    train_window_starts: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    model_config: STGNNModelConfig,
    static_adjacency: np.ndarray,
    training_config: ZeroInflatedTrainingConfig,
    device: torch.device,
) -> tuple[StaticAdaptiveZeroInflatedSTGNNGRURegressor, pd.DataFrame, int, dict[str, float | int], float]:
    train_dataset = BASE.DenseSequenceWindowDataset(
        input_features=train_data.input_features,
        targets=train_data.targets,
        target_mask=train_data.target_mask,
        window_starts=train_window_starts,
        window_size=training_config.window_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    zero_pos_weight = compute_zero_pos_weight(train_data)
    model = StaticAdaptiveZeroInflatedSTGNNGRURegressor(
        config=model_config,
        static_adjacency=static_adjacency,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_rmse = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] = {}
    best_validation_metrics: dict[str, float | int] = {}
    epochs_without_improvement = 0
    history_records: list[dict[str, float | int]] = []

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        zero_loss_sum = 0.0
        positive_loss_sum = 0.0
        zero_suppression_loss_sum = 0.0

        for batch_x, batch_y, batch_mask in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{training_config.epochs}",
            unit="batch",
        ):
            batch_x = batch_x.to(device=device, non_blocking=True).float()
            batch_y = batch_y.to(device=device, non_blocking=True).float()
            batch_mask = batch_mask.to(device=device, non_blocking=True).bool()
            mask_count = int(batch_mask.sum().item())
            if mask_count == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss_parts = zero_inflated_masked_loss(
                predictions=output.predictions,
                zero_logits=output.zero_logits,
                targets=batch_y,
                target_mask=batch_mask,
                zero_pos_weight=zero_pos_weight,
                zero_classification_weight=training_config.zero_classification_weight,
                positive_regression_weight=training_config.positive_regression_weight,
                zero_suppression_weight=training_config.zero_suppression_weight,
            )
            loss = loss_parts["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            optimizer.step()

            train_loss_sum += float(loss.item()) * mask_count
            zero_loss_sum += float(loss_parts["zero_classification_loss"].item()) * mask_count
            positive_loss_sum += float(loss_parts["positive_regression_loss"].item()) * mask_count
            zero_suppression_loss_sum += float(loss_parts["zero_suppression_loss"].item()) * mask_count
            train_sample_count += mask_count

        if train_sample_count == 0:
            raise ValueError("The training windows did not contain any observed targets.")

        validation_metrics = rolling_evaluate_split(
            model=model,
            split_data=valid_data,
            history_parts=[train_data],
            input_feature_cols=input_feature_cols,
            feature_means=feature_means,
            feature_stds=feature_stds,
            device=device,
            window_size=training_config.window_size,
            eval_batch_size=training_config.eval_batch_size,
            zero_threshold=training_config.zero_threshold,
        )
        history_record = {
            "epoch": epoch,
            "train_loss": float(train_loss_sum / train_sample_count),
            "train_zero_classification_loss": float(zero_loss_sum / train_sample_count),
            "train_positive_regression_loss": float(positive_loss_sum / train_sample_count),
            "train_zero_suppression_loss": float(zero_suppression_loss_sum / train_sample_count),
            "validation_mse": float(validation_metrics["mse"]),
            "validation_mae": float(validation_metrics["mae"]),
            "validation_r2": float(validation_metrics["r2"]),
            "validation_smape": float(validation_metrics["smape"]),
            "validation_rmse": float(validation_metrics["rmse"]),
            "validation_zero_f1": float(validation_metrics["zero_f1"]),
            "validation_mean_prediction_on_zero": float(validation_metrics["mean_prediction_on_zero"]),
            "validation_evaluated_row_count": int(validation_metrics["evaluated_row_count"]),
            "validation_skipped_row_count": int(validation_metrics["skipped_row_count"]),
            "validation_unknown_node_row_count": int(validation_metrics["unknown_node_row_count"]),
        }
        history_records.append(history_record)
        print(json.dumps(history_record, indent=2))

        current_rmse = float(validation_metrics["rmse"])
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config.patience:
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(best_state_dict)
    return model, pd.DataFrame(history_records), best_epoch, best_validation_metrics, zero_pos_weight


def save_model_artifact(
    model: StaticAdaptiveZeroInflatedSTGNNGRURegressor,
    model_config: STGNNModelConfig,
    training_config: ZeroInflatedTrainingConfig,
    node_ids: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    static_adjacency: np.ndarray,
    zero_pos_weight: float,
    output_path: Path,
) -> Path:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "node_ids": node_ids.astype(int).tolist(),
            "input_feature_cols": input_feature_cols,
            "feature_means": feature_means.astype(np.float32),
            "feature_stds": feature_stds.astype(np.float32),
            "static_adjacency": static_adjacency.astype(np.float32),
            "zero_pos_weight": float(zero_pos_weight),
            "model_name": MODEL_NAME,
        },
        output_path,
    )
    return output_path


def save_graph_artifacts(
    model: StaticAdaptiveZeroInflatedSTGNNGRURegressor,
    node_ids: np.ndarray,
    paths: MeterPaths,
) -> dict[str, Path]:
    return BASE.save_graph_artifacts(
        model=model,
        node_ids=node_ids,
        static_adjacency_path=paths.static_adjacency_path,
        learned_adjacency_path=paths.learned_adjacency_path,
        prediction_weighted_adjacency_path=paths.prediction_weighted_adjacency_path,
        static_topk_edges_path=paths.static_topk_edges_path,
        learned_topk_edges_path=paths.learned_topk_edges_path,
        prediction_weighted_topk_edges_path=paths.prediction_weighted_topk_edges_path,
    )


def build_model_params_artifact(
    config: MeterConfig,
    paths: MeterPaths,
    preprocessing_summary: dict[str, object],
    model_config: STGNNModelConfig,
    training_config: ZeroInflatedTrainingConfig,
    node_ids: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    best_epoch: int,
    best_validation_metrics: dict[str, float | int],
    train_window_count: int,
    test_only_node_ids: list[int],
    test_unknown_node_row_count: int,
    graph_artifact_paths: dict[str, Path],
    zero_pos_weight: float,
    prediction_fusion_alpha: list[float],
) -> dict[str, object]:
    return {
        "meter_id": config.meter_id,
        "meter_name": config.meter_name,
        "model_name": MODEL_NAME,
        "graph_mode": model_config.graph_mode,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "zero_pos_weight": float(zero_pos_weight),
        "prediction_fusion_alpha": prediction_fusion_alpha,
        "best_epoch": int(best_epoch),
        "best_validation_rmse": float(best_validation_metrics["rmse"]),
        "best_validation_zero_f1": float(best_validation_metrics["zero_f1"]),
        "node_count": int(node_ids.shape[0]),
        "node_ids": node_ids.astype(int).tolist(),
        "test_only_node_ids": [int(node_id) for node_id in test_only_node_ids],
        "test_unknown_node_row_count": int(test_unknown_node_row_count),
        "feature_count": len(input_feature_cols),
        "stgnn_input_feature_cols": input_feature_cols,
        "input_feature_means": dict(zip(input_feature_cols, [float(value) for value in feature_means])),
        "input_feature_stds": dict(zip(input_feature_cols, [float(value) for value in feature_stds])),
        "target_column": TARGET_COL,
        "metric_scale": "log1p_minmax",
        "train_window_count": int(train_window_count),
        "train_data_path": str(paths.train_data_path),
        "valid_data_path": str(paths.valid_data_path),
        "test_data_path": str(paths.test_data_path),
        "preprocessing_summary_path": str(paths.preprocessing_summary_path),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": preprocessing_summary.get("target_log1p_max"),
        **{key: str(value) for key, value in graph_artifact_paths.items()},
    }


def build_run_summary(
    config: MeterConfig,
    paths: MeterPaths,
    preprocessing_summary: dict[str, object],
    split_summary: pd.DataFrame,
    input_feature_cols: list[str],
    validation_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
    best_epoch: int,
    train_window_count: int,
    train_data,
    valid_data,
    test_data,
    test_only_node_ids: list[int],
    artifact_paths: dict[str, Path],
    graph_artifact_paths: dict[str, Path],
    graph_mode: str,
    zero_pos_weight: float,
    prediction_fusion_alpha: list[float],
    device: torch.device,
) -> dict[str, object]:
    summary = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "meter_id": config.meter_id,
        "meter_name": config.meter_name,
        "model_name": MODEL_NAME,
        "graph_mode": graph_mode,
        "prediction_fusion_alpha": prediction_fusion_alpha,
        "zero_pos_weight": float(zero_pos_weight),
        "metric_scale": "log1p_minmax",
        "data_root": str(DATA_ROOT),
        "train_data_path": str(paths.train_data_path),
        "valid_data_path": str(paths.valid_data_path),
        "test_data_path": str(paths.test_data_path),
        "preprocessing_summary_path": str(paths.preprocessing_summary_path),
        "output_dir": str(paths.output_dir),
        "preprocessed_data_dir": str(paths.data_dir),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": preprocessing_summary.get("target_log1p_max"),
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "node_count": int(train_data.node_ids.shape[0]),
        "test_only_node_ids": [int(node_id) for node_id in test_only_node_ids],
        "feature_count": len(input_feature_cols),
        "stgnn_input_feature_cols": input_feature_cols,
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS,
        "train_raw_row_count": int(train_data.raw_row_count),
        "valid_raw_row_count": int(valid_data.raw_row_count),
        "test_raw_row_count": int(test_data.raw_row_count),
        "train_included_row_count": int(train_data.included_row_count),
        "valid_included_row_count": int(valid_data.included_row_count),
        "test_included_row_count": int(test_data.included_row_count),
        "test_unknown_node_row_count": int(test_data.unknown_node_row_count),
        "train_timestamp_count": int(train_data.timestamps.shape[0]),
        "valid_timestamp_count": int(valid_data.timestamps.shape[0]),
        "test_timestamp_count": int(test_data.timestamps.shape[0]),
        "splits": split_summary.to_dict(orient="records"),
        "train_window_count": int(train_window_count),
        "best_epoch": int(best_epoch),
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{key: str(value) for key, value in artifact_paths.items()},
        **{key: str(value) for key, value in graph_artifact_paths.items()},
    }
    return summary


def train_one_meter(
    config: MeterConfig,
    data_root: Path = DATA_ROOT,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
    graph_mode: str = GRAPH_MODE,
) -> dict[str, object]:
    set_seed(SEED)
    paths = build_meter_paths(
        config=config,
        preprocessed_data_dir=preprocessed_data_dir,
        output_root_dir=output_root_dir,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, input_feature_cols = BASE.load_preprocessed_splits(
        train_path=paths.train_data_path,
        valid_path=paths.valid_data_path,
        test_path=paths.test_data_path,
    )
    preprocessing_summary = BASE.load_preprocessing_summary(paths.preprocessing_summary_path)
    split_summary = BASE.make_split_summary(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        preprocessing_summary=preprocessing_summary,
    )
    feature_means, feature_stds = BASE.fit_input_scaler(train_df=train_df, input_feature_cols=input_feature_cols)
    node_ids = np.sort(train_df["building_id"].unique()).astype(np.int32)
    test_only_node_ids = sorted(set(test_df["building_id"].unique().tolist()) - set(node_ids.tolist()))
    static_graph_config = StaticGraphConfig(top_k=GRAPH_TOP_K)
    static_adjacency = build_static_adjacency(
        train_df=train_df,
        node_ids=node_ids,
        config=static_graph_config,
    )

    train_data = BASE.build_dense_split_data(
        name=f"meter_{config.meter_id}_train",
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    valid_data = BASE.build_dense_split_data(
        name=f"meter_{config.meter_id}_valid",
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    test_data = BASE.build_dense_split_data(
        name=f"meter_{config.meter_id}_test",
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )

    del train_df, valid_df, test_df
    gc.collect()

    training_config = ZeroInflatedTrainingConfig()
    model_config = STGNNModelConfig(
        input_size=len(input_feature_cols),
        node_count=int(node_ids.shape[0]),
        graph_mode=graph_mode,
    )
    train_window_starts = BASE.build_dense_window_starts(
        split_data=train_data,
        window_size=training_config.window_size,
        stride=training_config.stride,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "meter_id": config.meter_id,
                "meter_name": config.meter_name,
                "node_count": int(node_ids.shape[0]),
                "train_window_count": int(train_window_starts.shape[0]),
                "input_feature_count": len(input_feature_cols),
                "test_only_node_ids": test_only_node_ids,
                "test_unknown_node_row_count": int(test_data.unknown_node_row_count),
            },
            indent=2,
        )
    )

    BASE.assert_training_ready(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        train_window_starts=train_window_starts,
        input_feature_cols=input_feature_cols,
        model_config=model_config,
        training_config=training_config,
    )

    model, training_history, best_epoch, best_validation_metrics, zero_pos_weight = train_stgnn_model(
        train_data=train_data,
        valid_data=valid_data,
        train_window_starts=train_window_starts,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        model_config=model_config,
        static_adjacency=static_adjacency,
        training_config=training_config,
        device=device,
    )

    validation_metrics = rolling_evaluate_split(
        model=model,
        split_data=valid_data,
        history_parts=[train_data],
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        device=device,
        window_size=training_config.window_size,
        eval_batch_size=training_config.eval_batch_size,
        zero_threshold=training_config.zero_threshold,
    )
    test_metrics = rolling_evaluate_split(
        model=model,
        split_data=test_data,
        history_parts=[train_data, valid_data],
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        device=device,
        window_size=training_config.window_size,
        eval_batch_size=training_config.eval_batch_size,
        zero_threshold=training_config.zero_threshold,
    )

    model_path = save_model_artifact(
        model=model,
        model_config=model_config,
        training_config=training_config,
        node_ids=node_ids,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        static_adjacency=static_adjacency,
        zero_pos_weight=zero_pos_weight,
        output_path=paths.model_path,
    )
    graph_artifact_paths = save_graph_artifacts(model=model, node_ids=node_ids, paths=paths)
    prediction_fusion_alpha = model.get_prediction_fusion_alpha().astype(float).tolist()
    model_params_path = BASE.save_json(
        build_model_params_artifact(
            config=config,
            paths=paths,
            preprocessing_summary=preprocessing_summary,
            model_config=model_config,
            training_config=training_config,
            node_ids=node_ids,
            input_feature_cols=input_feature_cols,
            feature_means=feature_means,
            feature_stds=feature_stds,
            best_epoch=best_epoch,
            best_validation_metrics=best_validation_metrics,
            train_window_count=int(train_window_starts.shape[0]),
            test_only_node_ids=[int(node_id) for node_id in test_only_node_ids],
            test_unknown_node_row_count=int(test_data.unknown_node_row_count),
            graph_artifact_paths=graph_artifact_paths,
            zero_pos_weight=zero_pos_weight,
            prediction_fusion_alpha=prediction_fusion_alpha,
        ),
        paths.model_params_path,
    )
    training_history_path = BASE.save_training_history(
        training_history=training_history,
        output_path=paths.training_history_path,
    )
    validation_metrics_path = BASE.save_json(validation_metrics, paths.validation_metrics_path)
    test_metrics_path = BASE.save_json(test_metrics, paths.test_metrics_path)
    split_summary_path = BASE.save_split_summary(split_summary=split_summary, output_path=paths.split_summary_path)
    artifact_paths = {
        "model_path": model_path,
        "model_params_path": model_params_path,
        "training_history_path": training_history_path,
        "validation_metrics_path": validation_metrics_path,
        "test_metrics_path": test_metrics_path,
        "split_summary_path": split_summary_path,
    }
    run_summary = build_run_summary(
        config=config,
        paths=paths,
        preprocessing_summary=preprocessing_summary,
        split_summary=split_summary,
        input_feature_cols=input_feature_cols,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        train_window_count=int(train_window_starts.shape[0]),
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        test_only_node_ids=[int(node_id) for node_id in test_only_node_ids],
        artifact_paths=artifact_paths,
        graph_artifact_paths=graph_artifact_paths,
        graph_mode=model_config.graph_mode,
        zero_pos_weight=zero_pos_weight,
        prediction_fusion_alpha=prediction_fusion_alpha,
        device=device,
    )
    run_summary_path = BASE.save_json(run_summary, paths.run_summary_path)
    run_summary["summary_path"] = str(run_summary_path)
    return run_summary


def make_zero_inflated_smoke_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_df, valid_df, test_df, input_feature_cols = BASE.build_smoke_frames()
    for split_df in [train_df, valid_df, test_df]:
        split_df["meter"] = 1
        zero_mask = (pd.to_numeric(split_df["building_id"], errors="coerce") == 1) & (
            pd.to_datetime(split_df["timestamp"]).dt.hour % 2 == 0
        )
        split_df.loc[zero_mask, TARGET_COL] = np.float32(0.0)
    return train_df, valid_df, test_df, input_feature_cols


def run_smoke_test(graph_mode: str = GRAPH_MODE) -> dict[str, object]:
    set_seed(SEED)
    train_df, valid_df, test_df, input_feature_cols = make_zero_inflated_smoke_frames()
    feature_means, feature_stds = BASE.fit_input_scaler(train_df=train_df, input_feature_cols=input_feature_cols)
    node_ids = np.sort(train_df["building_id"].unique()).astype(np.int32)
    static_graph_config = StaticGraphConfig(top_k=2)
    static_adjacency = build_static_adjacency(
        train_df=train_df,
        node_ids=node_ids,
        config=static_graph_config,
    )
    train_data = BASE.build_dense_split_data(
        name="zero_smoke_train",
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    valid_data = BASE.build_dense_split_data(
        name="zero_smoke_valid",
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    test_data = BASE.build_dense_split_data(
        name="zero_smoke_test",
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    training_config = ZeroInflatedTrainingConfig(window_size=4, batch_size=2, eval_batch_size=1, epochs=1, patience=1)
    model_config = STGNNModelConfig(
        input_size=len(input_feature_cols),
        node_count=int(node_ids.shape[0]),
        gcn_dim=4,
        embed_dim=4,
        graph_top_k=2,
        gcn_layers=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        graph_mode=graph_mode,
    )
    train_window_starts = BASE.build_dense_window_starts(
        split_data=train_data,
        window_size=training_config.window_size,
        stride=training_config.stride,
    )
    dataset = BASE.DenseSequenceWindowDataset(
        input_features=train_data.input_features,
        targets=train_data.targets,
        target_mask=train_data.target_mask,
        window_starts=train_window_starts,
        window_size=training_config.window_size,
    )
    sample_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_x, batch_y, batch_mask = next(iter(sample_loader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StaticAdaptiveZeroInflatedSTGNNGRURegressor(
        config=model_config,
        static_adjacency=static_adjacency,
    ).to(device)
    output = model(batch_x.to(device=device).float())
    if tuple(output.predictions.shape) != (2, 3):
        raise AssertionError(f"Unexpected smoke prediction shape: {tuple(output.predictions.shape)}")

    zero_pos_weight = compute_zero_pos_weight(train_data)
    loss_parts = zero_inflated_masked_loss(
        predictions=output.predictions,
        zero_logits=output.zero_logits,
        targets=batch_y.to(device=device).float(),
        target_mask=batch_mask.to(device=device).bool(),
        zero_pos_weight=zero_pos_weight,
    )
    if int(loss_parts["zero_count"]) == 0 or int(loss_parts["positive_count"]) == 0:
        raise AssertionError("Smoke batch did not include both zero and positive targets.")

    model, training_history, best_epoch, best_validation_metrics, zero_pos_weight = train_stgnn_model(
        train_data=train_data,
        valid_data=valid_data,
        train_window_starts=train_window_starts,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        model_config=model_config,
        static_adjacency=static_adjacency,
        training_config=training_config,
        device=device,
    )
    valid_metrics = rolling_evaluate_split(
        model=model,
        split_data=valid_data,
        history_parts=[train_data],
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        device=device,
        window_size=training_config.window_size,
        eval_batch_size=training_config.eval_batch_size,
        zero_threshold=training_config.zero_threshold,
    )
    test_metrics = rolling_evaluate_split(
        model=model,
        split_data=test_data,
        history_parts=[train_data, valid_data],
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        device=device,
        window_size=training_config.window_size,
        eval_batch_size=training_config.eval_batch_size,
        zero_threshold=training_config.zero_threshold,
    )
    required_metric_keys = {
        "thresholded_rmse",
        "zero_accuracy",
        "zero_precision",
        "zero_recall",
        "zero_f1",
        "mean_prediction_on_zero",
    }
    if not required_metric_keys.issubset(valid_metrics):
        raise AssertionError("Validation metrics did not include zero-inflated diagnostics.")

    smoke_summary = {
        "status": "passed",
        "graph_mode": graph_mode,
        "device": str(device),
        "batch_shape": list(batch_x.shape),
        "prediction_shape": list(output.predictions.shape),
        "zero_pos_weight": float(zero_pos_weight),
        "train_window_count": int(train_window_starts.shape[0]),
        "smoke_train_epoch_count": int(training_history.shape[0]),
        "smoke_best_epoch": int(best_epoch),
        "smoke_best_validation_rmse": float(best_validation_metrics["rmse"]),
        "valid_zero_f1": float(valid_metrics["zero_f1"]),
        "test_thresholded_rmse": float(test_metrics["thresholded_rmse"]),
    }
    print(json.dumps(smoke_summary, indent=2))
    return smoke_summary


def train_other_meters(
    meter_id: int | None = None,
    graph_mode: str = GRAPH_MODE,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> list[dict[str, object]]:
    meter_configs = [METER_CONFIGS[meter_id]] if meter_id is not None else list(METER_CONFIGS.values())
    meter_summaries = []
    for config in meter_configs:
        print(f"Training zero-inflated static-adaptive STGNN for meter {config.meter_id} ({config.meter_name})")
        meter_summary = train_one_meter(
            config=config,
            output_root_dir=output_root_dir,
            graph_mode=graph_mode,
        )
        meter_summaries.append(meter_summary)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_root_dir.mkdir(parents=True, exist_ok=True)
    BASE.save_json(meter_summaries, output_root_dir / OVERALL_RUN_SUMMARY_PATH.name)
    return meter_summaries


def main(
    smoke_test: bool = False,
    meter_id: int | None = None,
    graph_mode: str = GRAPH_MODE,
) -> list[dict[str, object]] | dict[str, object]:
    if smoke_test:
        return run_smoke_test(graph_mode=graph_mode)
    summaries = train_other_meters(meter_id=meter_id, graph_mode=graph_mode)
    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    args = parse_args()
    main(
        smoke_test=bool(args.smoke_test),
        meter_id=args.meter_id,
        graph_mode=str(args.graph_mode),
    )
