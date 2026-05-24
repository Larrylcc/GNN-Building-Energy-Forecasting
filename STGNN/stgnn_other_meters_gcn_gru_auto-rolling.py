from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from data_preprocess.data_preprocess import CATEGORICAL_COLS, DATA_ROOT, NUMERICAL_COLS, PREPROCESSED_DATA_DIR  # noqa: E402


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
    adjacency_path: Path
    topk_edges_path: Path


OUTPUT_ROOT_DIR = WORKSPACE_ROOT / "STGNN" / "stgnn_other_meters_gcn_gru_auto-rolling_outputs"
OVERALL_RUN_SUMMARY_PATH = OUTPUT_ROOT_DIR / "stgnn_other_meters_run_summary.json"
SMOKE_TEMPLATE_DATA_PATH = PREPROCESSED_DATA_DIR / "log1p_minmax_train.csv"

METER_CONFIGS = {
    1: MeterConfig(meter_id=1, meter_name="chilled_water"),
    2: MeterConfig(meter_id=2, meter_name="steam"),
    3: MeterConfig(meter_id=3, meter_name="hot_water"),
}


def build_meter_paths(
    config: MeterConfig,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> MeterPaths:
    data_dir = preprocessed_data_dir / f"meter_{config.meter_id}"
    output_dir = output_root_dir / f"meter_{config.meter_id}"

    return MeterPaths(
        data_dir=data_dir,
        output_dir=output_dir,
        train_data_path=data_dir / "log1p_minmax_train.csv",
        valid_data_path=data_dir / "log1p_minmax_valid.csv",
        test_data_path=data_dir / "log1p_minmax_test.csv",
        preprocessing_summary_path=data_dir / "log1p_minmax_summary.json",
        model_path=output_dir / "stgnn_final_model.pt",
        model_params_path=output_dir / "stgnn_model_params.json",
        training_history_path=output_dir / "stgnn_training_history.csv",
        split_summary_path=output_dir / "stgnn_time_series_split.csv",
        validation_metrics_path=output_dir / "stgnn_validation_metrics.json",
        test_metrics_path=output_dir / "stgnn_test_metrics.json",
        run_summary_path=output_dir / "stgnn_run_summary.json",
        adjacency_path=output_dir / "stgnn_learned_adjacency.npy",
        topk_edges_path=output_dir / "stgnn_topk_edges.csv",
    )

SEED = 42
WINDOW_SIZE = 168
STRIDE = 1
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 1
EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
GCN_DIM = 16
EMBED_DIM = 16
GRAPH_TOP_K = 20
GCN_LAYERS = 4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
TARGET_COL = "meter_reading"
ONE_HOUR = np.timedelta64(1, "h")


@dataclass(frozen=True)
class STGNNModelConfig:
    input_size: int
    node_count: int
    gcn_dim: int = GCN_DIM
    embed_dim: int = EMBED_DIM
    graph_top_k: int = GRAPH_TOP_K
    gcn_layers: int = GCN_LAYERS
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT


@dataclass(frozen=True)
class TrainingConfig:
    window_size: int = WINDOW_SIZE
    stride: int = STRIDE
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    epochs: int = EPOCHS
    patience: int = PATIENCE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip_norm: float = GRAD_CLIP_NORM
    seed: int = SEED


@dataclass
class DenseSplitData:
    name: str
    input_features: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    timestamps: np.ndarray
    node_ids: np.ndarray
    raw_row_count: int
    included_row_count: int
    unknown_node_row_count: int


class DenseSequenceWindowDataset(Dataset):
    def __init__(
        self,
        input_features: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
        window_starts: np.ndarray,
        window_size: int,
    ) -> None:
        self.input_features = input_features
        self.targets = targets
        self.target_mask = target_mask
        self.window_starts = window_starts
        self.window_size = window_size

    def __len__(self) -> int:
        return int(self.window_starts.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.window_starts[index])
        end = start + self.window_size
        target_index = end
        return (
            torch.from_numpy(self.input_features[start:end]),
            torch.from_numpy(self.targets[target_index]),
            torch.from_numpy(self.target_mask[target_index]),
        )


class GraphConstructor(nn.Module):
    def __init__(self, node_count: int, embed_dim: int, top_k: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.top_k = top_k
        self.node_embeddings = nn.Parameter(torch.empty(node_count, embed_dim))
        self.register_buffer("eye", torch.eye(node_count, dtype=torch.float32), persistent=False)
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self) -> torch.Tensor:
        normalized_embeddings = nn.functional.normalize(self.node_embeddings, p=2.0, dim=1, eps=1e-12)
        similarity = torch.relu(normalized_embeddings @ normalized_embeddings.transpose(0, 1))
        adjacency = torch.zeros_like(similarity)

        external_top_k = min(self.top_k, max(self.node_count - 1, 0))
        if external_top_k > 0:
            external_scores = similarity.masked_fill(self.eye.bool(), -torch.inf)
            topk_indices = torch.topk(external_scores, k=external_top_k, dim=1).indices
            topk_mask = torch.zeros_like(similarity, dtype=torch.bool)
            topk_mask.scatter_(1, topk_indices, True)
            adjacency = torch.where(topk_mask, similarity, adjacency)

        adjacency = adjacency + self.eye
        row_sums = adjacency.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return adjacency / row_sums


class STGNNGRURegressor(nn.Module):
    def __init__(self, config: STGNNModelConfig) -> None:
        super().__init__()
        self.config = config
        self.graph_constructor = GraphConstructor(
            node_count=config.node_count,
            embed_dim=config.embed_dim,
            top_k=config.graph_top_k,
        )
        self.input_projection = nn.Linear(config.input_size, config.gcn_dim)
        self.gcn_layers = nn.ModuleList(
            nn.Linear(config.gcn_dim, config.gcn_dim) for _ in range(config.gcn_layers)
        )
        self.layer_attention = nn.Parameter(torch.zeros(config.gcn_layers, dtype=torch.float32))
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=config.gcn_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        adjacency = self.graph_constructor()
        hidden = self.input_projection(batch)
        layer_outputs: list[torch.Tensor] = []

        for layer in self.gcn_layers:
            aggregated = torch.einsum("ij,btjf->btif", adjacency, hidden)
            update = torch.relu(layer(aggregated))
            hidden = hidden + self.dropout(update)
            layer_outputs.append(hidden)

        stacked_outputs = torch.stack(layer_outputs, dim=0)
        attention_weights = torch.softmax(self.layer_attention, dim=0).view(-1, 1, 1, 1, 1)
        fused = torch.sum(attention_weights * stacked_outputs, dim=0)

        batch_size, sequence_length, node_count, feature_count = fused.shape
        gru_input = fused.permute(0, 2, 1, 3).contiguous().view(
            batch_size * node_count,
            sequence_length,
            feature_count,
        )
        _, hidden_state = self.gru(gru_input)
        last_hidden = hidden_state[-1]
        predictions = self.head(last_hidden).view(batch_size, node_count)
        return predictions

    def get_adjacency(self) -> torch.Tensor:
        return self.graph_constructor()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STGNN + GCN + GRU models for other meter types.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small synthetic end-to-end verification instead of full training.",
    )
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def compute_smape_sum(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_ratio = np.divide(
        2.0 * np.abs(y_pred - y_true),
        denominator,
        out=np.zeros_like(y_true, dtype=np.float32),
        where=denominator != 0,
    )
    return float(np.sum(safe_ratio, dtype=np.float64))


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    mask = target_mask.to(dtype=predictions.dtype)
    squared_error = (predictions - targets) ** 2
    return torch.sum(squared_error * mask) / torch.sum(mask).clamp_min(1.0)


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log1p + min-max split file not found: {path}.")

    split_df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    split_df = split_df.sort_values(by=["timestamp", "building_id", "meter"], kind="stable").reset_index(drop=True)
    split_df[TARGET_COL] = pd.to_numeric(split_df[TARGET_COL], errors="coerce").astype(np.float32)
    return split_df


def load_preprocessed_splits(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = load_split_data(train_path)
    valid_df = load_split_data(valid_path)
    test_df = load_split_data(test_path)

    expected_columns = train_df.columns.tolist()
    for split_name, split_df in [("valid", valid_df), ("test", test_df)]:
        if split_df.columns.tolist() != expected_columns:
            raise ValueError(f"The {split_name} split columns do not match the training split columns.")

    input_feature_cols = [TARGET_COL] + [column for column in train_df.columns if column not in {"timestamp", TARGET_COL}]
    return train_df, valid_df, test_df, input_feature_cols


def load_preprocessing_summary(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_split_summary_row(split_name: str, split_df: pd.DataFrame) -> dict[str, str | int]:
    return {
        "split": split_name,
        "start_timestamp": str(split_df["timestamp"].min()),
        "end_timestamp": str(split_df["timestamp"].max()),
        "row_count": int(split_df.shape[0]),
    }


def make_split_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessing_summary: dict[str, object],
) -> pd.DataFrame:
    split_records = preprocessing_summary.get("splits")
    if isinstance(split_records, list) and split_records:
        return pd.DataFrame(split_records)

    return pd.DataFrame(
        [
            build_split_summary_row("train", train_df),
            build_split_summary_row("valid", valid_df),
            build_split_summary_row("test", test_df),
        ]
    )


def fit_input_scaler(train_df: pd.DataFrame, input_feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    input_values = train_df[input_feature_cols].to_numpy(dtype=np.float32, copy=True)
    feature_means = np.nanmean(input_values, axis=0).astype(np.float32)
    feature_stds = np.nanstd(input_values, axis=0).astype(np.float32)

    feature_means = np.where(np.isfinite(feature_means), feature_means, 0.0).astype(np.float32)
    feature_stds = np.where(np.isfinite(feature_stds) & (feature_stds > 0.0), feature_stds, 1.0).astype(np.float32)

    del input_values
    gc.collect()
    return feature_means, feature_stds


def transform_input_features(
    split_df: pd.DataFrame,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> np.ndarray:
    input_values = split_df[input_feature_cols].to_numpy(dtype=np.float32, copy=True)
    input_values -= feature_means
    input_values /= feature_stds
    np.nan_to_num(input_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(input_values, dtype=np.float32)


def build_node_lookup(node_ids: np.ndarray, raw_building_ids: np.ndarray) -> np.ndarray:
    max_node_id = int(max(np.max(node_ids), np.max(raw_building_ids)))
    node_lookup = np.full(max_node_id + 1, -1, dtype=np.int32)
    node_lookup[node_ids.astype(np.int64)] = np.arange(node_ids.shape[0], dtype=np.int32)

    node_positions = np.full(raw_building_ids.shape[0], -1, dtype=np.int32)
    non_negative_mask = raw_building_ids >= 0
    in_range_mask = non_negative_mask & (raw_building_ids <= max_node_id)
    node_positions[in_range_mask] = node_lookup[raw_building_ids[in_range_mask]]
    return node_positions


def build_dense_split_data(
    name: str,
    split_df: pd.DataFrame,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    node_ids: np.ndarray,
) -> DenseSplitData:
    timestamps = pd.Index(split_df["timestamp"].drop_duplicates().sort_values()).to_numpy(dtype="datetime64[ns]")
    timestamp_positions = pd.Index(timestamps).get_indexer(split_df["timestamp"])

    raw_building_ids = pd.to_numeric(split_df["building_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    node_positions = build_node_lookup(node_ids=node_ids, raw_building_ids=raw_building_ids)
    included_mask = node_positions >= 0

    time_count = int(timestamps.shape[0])
    node_count = int(node_ids.shape[0])
    feature_count = int(len(input_feature_cols))

    dense_features = np.zeros((time_count, node_count, feature_count), dtype=np.float32)
    dense_targets = np.zeros((time_count, node_count), dtype=np.float32)
    target_mask = np.zeros((time_count, node_count), dtype=bool)

    input_values = transform_input_features(
        split_df=split_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    targets = split_df[TARGET_COL].to_numpy(dtype=np.float32, copy=True)

    included_time_positions = timestamp_positions[included_mask]
    included_node_positions = node_positions[included_mask]
    dense_features[included_time_positions, included_node_positions] = input_values[included_mask]
    dense_targets[included_time_positions, included_node_positions] = targets[included_mask]
    target_mask[included_time_positions, included_node_positions] = True

    del input_values, targets
    gc.collect()

    last_feature_rows = np.zeros((node_count, feature_count), dtype=np.float32)
    for time_index in tqdm(range(time_count), desc=f"Forward filling {name} dense features", unit="timestamp"):
        observed_nodes = target_mask[time_index]
        if np.any(observed_nodes):
            last_feature_rows[observed_nodes] = dense_features[time_index, observed_nodes]
        dense_features[time_index] = last_feature_rows

    return DenseSplitData(
        name=name,
        input_features=np.ascontiguousarray(dense_features, dtype=np.float32),
        targets=np.ascontiguousarray(dense_targets, dtype=np.float32),
        target_mask=np.ascontiguousarray(target_mask),
        timestamps=timestamps,
        node_ids=node_ids.copy(),
        raw_row_count=int(split_df.shape[0]),
        included_row_count=int(np.sum(included_mask)),
        unknown_node_row_count=int(np.sum(~included_mask)),
    )


def build_dense_window_starts(
    split_data: DenseSplitData,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    start_arrays: list[np.ndarray] = []
    timestamps = split_data.timestamps
    gap_offsets = np.flatnonzero(np.diff(timestamps) != ONE_HOUR) + 1
    segment_starts = np.append(np.array([0], dtype=np.int64), gap_offsets)
    segment_ends = np.append(gap_offsets, timestamps.shape[0])

    for segment_start, segment_end in zip(segment_starts, segment_ends):
        if segment_end - segment_start <= window_size:
            continue

        candidate_starts = np.arange(segment_start, segment_end - window_size, stride, dtype=np.int32)
        target_indices = candidate_starts + window_size
        observed_target_mask = np.any(split_data.target_mask[target_indices], axis=1)
        candidate_starts = candidate_starts[observed_target_mask]
        if candidate_starts.shape[0] > 0:
            start_arrays.append(candidate_starts)

    if not start_arrays:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(start_arrays).astype(np.int32, copy=False)


def collect_global_history_suffix(
    first_timestamp: np.datetime64,
    history_parts: list[DenseSplitData],
    window_size: int,
) -> tuple[deque[np.ndarray], np.datetime64 | None]:
    remaining = window_size
    expected_end = first_timestamp - ONE_HOUR
    pieces_from_newest: list[np.ndarray] = []
    last_timestamp: np.datetime64 | None = None

    for history_part in reversed(history_parts):
        timestamps = history_part.timestamps
        local_end_pos = int(np.searchsorted(timestamps, expected_end))

        if local_end_pos >= timestamps.shape[0] or timestamps[local_end_pos] != expected_end:
            break

        local_start_pos = local_end_pos
        collected_count = 1
        while collected_count < remaining and local_start_pos > 0:
            if timestamps[local_start_pos] - timestamps[local_start_pos - 1] != ONE_HOUR:
                break
            local_start_pos -= 1
            collected_count += 1

        pieces_from_newest.append(history_part.input_features[local_start_pos : local_end_pos + 1])

        if last_timestamp is None:
            last_timestamp = history_part.timestamps[local_end_pos]

        remaining -= collected_count
        if remaining == 0:
            break
        if local_start_pos > 0:
            break

        expected_end = timestamps[local_start_pos] - ONE_HOUR

    history_rows: deque[np.ndarray] = deque(maxlen=window_size)
    for piece in reversed(pieces_from_newest):
        for row in piece:
            history_rows.append(row)

    return history_rows, last_timestamp


def predict_global_window(
    model: nn.Module,
    history_rows: deque[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    sequence = np.expand_dims(np.stack(history_rows, axis=0), axis=0)
    sequence = np.ascontiguousarray(sequence, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(sequence).to(device=device, non_blocking=True).float()
        predictions = model(batch).detach().cpu().numpy()[0].astype(np.float32)

    return np.nan_to_num(predictions, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def build_predicted_feature_rows(
    current_feature_rows: np.ndarray,
    predictions: np.ndarray,
    meter_feature_index: int,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
) -> np.ndarray:
    predicted_feature_rows = current_feature_rows.copy()
    normalized_predicted_meter = (predictions - feature_means[meter_feature_index]) / feature_stds[meter_feature_index]
    normalized_predicted_meter = np.nan_to_num(
        normalized_predicted_meter,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    predicted_feature_rows[:, meter_feature_index] = normalized_predicted_meter
    return predicted_feature_rows


def build_current_feature_rows(
    split_data: DenseSplitData,
    time_index: int,
    history_rows: deque[np.ndarray],
) -> np.ndarray:
    if len(history_rows) == 0:
        return split_data.input_features[time_index].copy()

    current_feature_rows = history_rows[-1].copy()
    observed_nodes = split_data.target_mask[time_index]
    if np.any(observed_nodes):
        current_feature_rows[observed_nodes] = split_data.input_features[time_index, observed_nodes]
    return current_feature_rows


def finalize_metrics(
    row_count: int,
    skipped_row_count: int,
    squared_error_sum: float,
    absolute_error_sum: float,
    smape_ratio_sum: float,
    y_sum: float,
    y_squared_sum: float,
) -> dict[str, float | int]:
    if row_count == 0:
        raise ValueError("Rolling evaluation did not produce any predictions.")

    mse = squared_error_sum / row_count
    mae = absolute_error_sum / row_count
    rmse = float(np.sqrt(mse))
    smape = smape_ratio_sum / row_count * 100.0
    total_sum_of_squares = y_squared_sum - (y_sum * y_sum / row_count)
    r2 = 0.0 if total_sum_of_squares <= 0.0 else 1.0 - squared_error_sum / total_sum_of_squares

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "smape": float(smape),
        "rmse": float(rmse),
        "evaluated_row_count": int(row_count),
        "skipped_row_count": int(skipped_row_count),
        "window_count": int(row_count),
    }


def rolling_evaluate_split(
    model: nn.Module,
    split_data: DenseSplitData,
    history_parts: list[DenseSplitData],
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    device: torch.device,
    window_size: int,
    eval_batch_size: int,
) -> dict[str, float | int]:
    if eval_batch_size != 1:
        raise ValueError("STGNN auto-rolling evaluation must use eval_batch_size=1.")

    meter_feature_index = input_feature_cols.index(TARGET_COL)
    history_rows, last_timestamp = collect_global_history_suffix(
        first_timestamp=split_data.timestamps[0],
        history_parts=history_parts,
        window_size=window_size,
    )

    evaluated_row_count = 0
    skipped_row_count = int(split_data.unknown_node_row_count)
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    smape_ratio_sum = 0.0
    y_sum = 0.0
    y_squared_sum = 0.0
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
            current_feature_rows = build_current_feature_rows(
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
            current_feature_rows = build_current_feature_rows(
                split_data=split_data,
                time_index=time_index,
                history_rows=history_rows,
            )
            history_rows.append(current_feature_rows)
            last_timestamp = timestamp
            skipped_row_count += observed_count
            skipped_timestamp_count += 1
            continue

        predictions = predict_global_window(model=model, history_rows=history_rows, device=device)

        if observed_count > 0:
            batch_targets = split_data.targets[time_index, current_mask].astype(np.float32, copy=False)
            batch_predictions = predictions[current_mask].astype(np.float32, copy=False)
            batch_errors = batch_predictions - batch_targets

            evaluated_row_count += int(batch_targets.shape[0])
            squared_error_sum += float(np.sum(batch_errors * batch_errors, dtype=np.float64))
            absolute_error_sum += float(np.sum(np.abs(batch_errors), dtype=np.float64))
            smape_ratio_sum += compute_smape_sum(batch_targets, batch_predictions)
            y_sum += float(np.sum(batch_targets, dtype=np.float64))
            y_squared_sum += float(np.sum(batch_targets * batch_targets, dtype=np.float64))
            evaluated_timestamp_count += 1

        current_feature_rows = build_current_feature_rows(
            split_data=split_data,
            time_index=time_index,
            history_rows=history_rows,
        )
        predicted_feature_rows = build_predicted_feature_rows(
            current_feature_rows=current_feature_rows,
            predictions=predictions,
            meter_feature_index=meter_feature_index,
            feature_means=feature_means,
            feature_stds=feature_stds,
        )
        history_rows.append(predicted_feature_rows)
        last_timestamp = timestamp

    metrics = finalize_metrics(
        row_count=evaluated_row_count,
        skipped_row_count=skipped_row_count,
        squared_error_sum=squared_error_sum,
        absolute_error_sum=absolute_error_sum,
        smape_ratio_sum=smape_ratio_sum,
        y_sum=y_sum,
        y_squared_sum=y_squared_sum,
    )
    metrics["evaluated_timestamp_count"] = int(evaluated_timestamp_count)
    metrics["skipped_timestamp_count"] = int(skipped_timestamp_count)
    metrics["unknown_node_row_count"] = int(split_data.unknown_node_row_count)
    return metrics


def train_stgnn_model(
    train_data: DenseSplitData,
    valid_data: DenseSplitData,
    train_window_starts: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    model_config: STGNNModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> tuple[STGNNGRURegressor, pd.DataFrame, int, dict[str, float | int]]:
    train_dataset = DenseSequenceWindowDataset(
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

    model = STGNNGRURegressor(config=model_config).to(device)
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
            predictions = model(batch_x)
            loss = masked_mse_loss(predictions=predictions, targets=batch_y, target_mask=batch_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            optimizer.step()

            train_loss_sum += float(loss.item()) * mask_count
            train_sample_count += mask_count

        if train_sample_count == 0:
            raise ValueError("The training windows did not contain any observed targets.")

        train_loss = train_loss_sum / train_sample_count
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
        )

        history_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_mse": float(validation_metrics["mse"]),
            "validation_mae": float(validation_metrics["mae"]),
            "validation_r2": float(validation_metrics["r2"]),
            "validation_smape": float(validation_metrics["smape"]),
            "validation_rmse": float(validation_metrics["rmse"]),
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
    return model, pd.DataFrame(history_records), best_epoch, best_validation_metrics


def save_json(data: dict[str, object] | list[dict[str, object]], output_path: Path) -> Path:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def save_split_summary(split_summary: pd.DataFrame, output_path: Path) -> Path:
    split_summary.to_csv(output_path, index=False)
    return output_path


def save_training_history(training_history: pd.DataFrame, output_path: Path) -> Path:
    training_history.to_csv(output_path, index=False)
    return output_path


def save_model_artifact(
    model: STGNNGRURegressor,
    model_config: STGNNModelConfig,
    training_config: TrainingConfig,
    node_ids: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    output_path: Path,
) -> Path:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "node_ids": node_ids.astype(int).tolist(),
            "input_feature_cols": input_feature_cols,
            "input_feature_means": feature_means.tolist(),
            "input_feature_stds": feature_stds.tolist(),
            "target_column": TARGET_COL,
        },
        output_path,
    )
    return output_path


def save_graph_artifacts(
    model: STGNNGRURegressor,
    node_ids: np.ndarray,
    adjacency_path: Path,
    topk_edges_path: Path,
) -> tuple[Path, Path]:
    model.eval()
    with torch.no_grad():
        adjacency = model.get_adjacency().detach().cpu().numpy().astype(np.float32)

    np.save(adjacency_path, adjacency)

    source_indices, target_indices = np.nonzero(adjacency > 0.0)
    edges = pd.DataFrame(
        {
            "source_building_id": node_ids[source_indices].astype(int),
            "target_building_id": node_ids[target_indices].astype(int),
            "source_node_index": source_indices.astype(int),
            "target_node_index": target_indices.astype(int),
            "weight": adjacency[source_indices, target_indices].astype(float),
            "is_self_loop": source_indices == target_indices,
        }
    )
    edges.to_csv(topk_edges_path, index=False)
    return adjacency_path, topk_edges_path


def assert_training_ready(
    train_data: DenseSplitData,
    valid_data: DenseSplitData,
    test_data: DenseSplitData,
    train_window_starts: np.ndarray,
    input_feature_cols: list[str],
    model_config: STGNNModelConfig,
    training_config: TrainingConfig,
) -> None:
    if train_window_starts.shape[0] == 0:
        raise ValueError("The training split produced zero continuous STGNN windows.")
    if train_data.input_features.shape[2] != len(input_feature_cols):
        raise ValueError("Training input feature width does not match input_feature_cols.")
    if model_config.input_size != len(input_feature_cols):
        raise ValueError("STGNN input_size does not match the number of input features.")
    if train_data.input_features.shape[1] != model_config.node_count:
        raise ValueError("Training node count does not match STGNN node_count.")
    if valid_data.input_features.shape[1] != model_config.node_count:
        raise ValueError("Validation node count does not match STGNN node_count.")
    if test_data.input_features.shape[1] != model_config.node_count:
        raise ValueError("Test node count does not match STGNN node_count.")
    if training_config.eval_batch_size != 1:
        raise ValueError("STGNN auto-rolling evaluation must use eval_batch_size=1.")


def build_smoke_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    columns = pd.read_csv(SMOKE_TEMPLATE_DATA_PATH, nrows=0).columns.tolist()

    def make_rows(
        timestamps: pd.DatetimeIndex,
        building_ids: list[int],
        missing_pairs: set[tuple[int, int]] | None = None,
        include_unknown: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        missing_pairs = missing_pairs or set()
        for time_position, timestamp in enumerate(timestamps):
            for building_id in building_ids:
                if (time_position, building_id) in missing_pairs:
                    continue
                row = {column: 0.0 for column in columns}
                row["timestamp"] = timestamp
                row["building_id"] = building_id
                row["meter"] = 0
                row["meter_reading"] = np.float32(0.1 + 0.01 * building_id + 0.001 * time_position)
                row["site_id"] = building_id % 2
                row["primary_use"] = building_id % 3
                row["square_feet"] = 1000 + 100 * building_id
                row["year_built"] = 2000 - building_id
                row["floor_count"] = np.nan if building_id == 2 else 1 + building_id
                row["age"] = 2016 - row["year_built"]
                row["floor_count_missing"] = 1 if building_id == 2 else 0
                row["month_datetime"] = timestamp.month
                row["weekofyear_datetime"] = int(timestamp.isocalendar().week)
                row["dayofyear_datetime"] = timestamp.dayofyear
                row["hour_datetime"] = timestamp.hour
                row["day_week"] = timestamp.dayofweek
                row["day_month_datetime"] = timestamp.day
                row["week_month_datetime"] = int(np.ceil(timestamp.day / 7.0))
                rows.append(row)

            if include_unknown:
                row = {column: 0.0 for column in columns}
                row["timestamp"] = timestamp
                row["building_id"] = 403
                row["meter"] = 0
                row["meter_reading"] = np.float32(0.4 + 0.001 * time_position)
                row["site_id"] = 3
                row["primary_use"] = 0
                row["square_feet"] = 49500
                row["year_built"] = 1962.0
                row["floor_count"] = np.nan
                row["age"] = 54.0
                row["floor_count_missing"] = 1
                row["month_datetime"] = timestamp.month
                row["weekofyear_datetime"] = int(timestamp.isocalendar().week)
                row["dayofyear_datetime"] = timestamp.dayofyear
                row["hour_datetime"] = timestamp.hour
                row["day_week"] = timestamp.dayofweek
                row["day_month_datetime"] = timestamp.day
                row["week_month_datetime"] = int(np.ceil(timestamp.day / 7.0))
                rows.append(row)

        return pd.DataFrame(rows, columns=columns)

    train_timestamps = pd.date_range("2016-01-01 00:00:00", periods=12, freq="h")
    valid_timestamps = pd.date_range(train_timestamps[-1] + pd.Timedelta(hours=1), periods=6, freq="h")
    test_timestamps = pd.date_range(valid_timestamps[-1] + pd.Timedelta(hours=1), periods=6, freq="h")

    train_df = make_rows(train_timestamps, [0, 1, 2], missing_pairs={(3, 1)})
    valid_df = make_rows(valid_timestamps, [0, 1, 2], missing_pairs={(2, 2)})
    test_df = make_rows(test_timestamps, [0, 1, 2], include_unknown=True)

    for split_df in [train_df, valid_df, test_df]:
        split_df["timestamp"] = pd.to_datetime(split_df["timestamp"])
        split_df[TARGET_COL] = pd.to_numeric(split_df[TARGET_COL], errors="coerce").astype(np.float32)

    input_feature_cols = [TARGET_COL] + [column for column in train_df.columns if column not in {"timestamp", TARGET_COL}]
    return train_df, valid_df, test_df, input_feature_cols


def run_smoke_test() -> dict[str, object]:
    set_seed(SEED)
    train_df, valid_df, test_df, input_feature_cols = build_smoke_frames()
    feature_means, feature_stds = fit_input_scaler(train_df=train_df, input_feature_cols=input_feature_cols)
    node_ids = np.sort(train_df["building_id"].unique()).astype(np.int32)

    train_data = build_dense_split_data(
        name="smoke_train",
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    valid_data = build_dense_split_data(
        name="smoke_valid",
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    test_data = build_dense_split_data(
        name="smoke_test",
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )

    training_config = TrainingConfig(window_size=4, batch_size=2, eval_batch_size=1, epochs=1, patience=1)
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
    )
    train_window_starts = build_dense_window_starts(
        split_data=train_data,
        window_size=training_config.window_size,
        stride=training_config.stride,
    )

    dataset = DenseSequenceWindowDataset(
        input_features=train_data.input_features,
        targets=train_data.targets,
        target_mask=train_data.target_mask,
        window_starts=train_window_starts,
        window_size=training_config.window_size,
    )
    sample_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_x, batch_y, batch_mask = next(iter(sample_loader))

    expected_feature_count = len(input_feature_cols)
    if tuple(batch_x.shape) != (2, 4, 3, expected_feature_count):
        raise AssertionError(f"Unexpected smoke batch shape: {tuple(batch_x.shape)}")
    if tuple(batch_y.shape) != (2, 3):
        raise AssertionError(f"Unexpected smoke target shape: {tuple(batch_y.shape)}")
    if tuple(batch_mask.shape) != (2, 3):
        raise AssertionError(f"Unexpected smoke mask shape: {tuple(batch_mask.shape)}")
    if train_data.target_mask[3, 1]:
        raise AssertionError("Smoke missing node was not masked.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = STGNNGRURegressor(config=model_config).to(device)
    predictions = model(batch_x.to(device=device).float())
    if tuple(predictions.shape) != (2, 3):
        raise AssertionError(f"Unexpected smoke prediction shape: {tuple(predictions.shape)}")

    with torch.no_grad():
        adjacency = model.get_adjacency().detach().cpu()
    if not torch.all(torch.diag(adjacency) > 0.0):
        raise AssertionError("Graph self-loops were not preserved.")
    if not torch.allclose(adjacency.sum(dim=1), torch.ones(3), atol=1e-5):
        raise AssertionError("Graph rows are not normalized.")

    test_predictions = torch.tensor([[1.0, 10.0]], dtype=torch.float32)
    test_targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    test_mask = torch.tensor([[True, False]])
    loss_value = masked_mse_loss(test_predictions, test_targets, test_mask)
    if not torch.isclose(loss_value, torch.tensor(1.0)):
        raise AssertionError("Masked MSE counted an unobserved node.")

    meter_feature_index = input_feature_cols.index(TARGET_COL)
    manual_predictions = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    predicted_feature_rows = build_predicted_feature_rows(
        current_feature_rows=valid_data.input_features[0],
        predictions=manual_predictions,
        meter_feature_index=meter_feature_index,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    history_rows: deque[np.ndarray] = deque(maxlen=training_config.window_size)
    history_rows.append(predicted_feature_rows)
    expected_meter_values = (manual_predictions - feature_means[meter_feature_index]) / feature_stds[meter_feature_index]
    if not np.allclose(history_rows[-1][:, meter_feature_index], expected_meter_values.astype(np.float32)):
        raise AssertionError("Rolling prediction writeback did not update the meter feature.")

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
    )
    if test_data.unknown_node_row_count == 0:
        raise AssertionError("Smoke test did not include an unknown node.")
    if int(test_metrics["unknown_node_row_count"]) != test_data.unknown_node_row_count:
        raise AssertionError("Unknown node rows were not tracked in rolling metrics.")

    smoke_summary = {
        "status": "passed",
        "device": str(device),
        "batch_shape": list(batch_x.shape),
        "prediction_shape": list(predictions.shape),
        "train_window_count": int(train_window_starts.shape[0]),
        "valid_evaluated_row_count": int(valid_metrics["evaluated_row_count"]),
        "test_unknown_node_row_count": int(test_metrics["unknown_node_row_count"]),
    }
    print(json.dumps(smoke_summary, indent=2))
    return smoke_summary


def build_model_params_artifact(
    config: MeterConfig,
    paths: MeterPaths,
    preprocessing_summary: dict[str, object],
    model_config: STGNNModelConfig,
    training_config: TrainingConfig,
    node_ids: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    best_epoch: int,
    best_validation_metrics: dict[str, float | int],
    train_window_count: int,
    test_only_node_ids: list[int],
    test_unknown_node_row_count: int,
    adjacency_path: Path,
    topk_edges_path: Path,
) -> dict[str, object]:
    return {
        "meter_id": config.meter_id,
        "meter_name": config.meter_name,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "best_epoch": int(best_epoch),
        "best_validation_rmse": float(best_validation_metrics["rmse"]),
        "node_count": int(node_ids.shape[0]),
        "node_ids": node_ids.astype(int).tolist(),
        "test_only_node_ids": [int(node_id) for node_id in test_only_node_ids],
        "test_unknown_node_row_count": int(test_unknown_node_row_count),
        "feature_count": len(input_feature_cols),
        "stgnn_input_feature_cols": input_feature_cols,
        "input_feature_means": dict(zip(input_feature_cols, [float(value) for value in feature_means])),
        "input_feature_stds": dict(zip(input_feature_cols, [float(value) for value in feature_stds])),
        "target_column": TARGET_COL,
        "train_window_count": int(train_window_count),
        "train_data_path": str(paths.train_data_path),
        "valid_data_path": str(paths.valid_data_path),
        "test_data_path": str(paths.test_data_path),
        "preprocessing_summary_path": str(paths.preprocessing_summary_path),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": preprocessing_summary.get("target_log1p_max"),
        "adjacency_path": str(adjacency_path),
        "topk_edges_path": str(topk_edges_path),
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
    train_data: DenseSplitData,
    valid_data: DenseSplitData,
    test_data: DenseSplitData,
    test_only_node_ids: list[int],
    model_path: Path,
    model_params_path: Path,
    training_history_path: Path,
    validation_metrics_path: Path,
    test_metrics_path: Path,
    split_summary_path: Path,
    adjacency_path: Path,
    topk_edges_path: Path,
    device_text: str,
    cuda_available: bool,
    data_root: Path = DATA_ROOT,
) -> dict[str, object]:
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": device_text,
        "meter_id": config.meter_id,
        "meter_name": config.meter_name,
        "data_root": str(data_root),
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
        "validation_window_count": int(validation_metrics["window_count"]),
        "test_window_count": int(test_metrics["window_count"]),
        "validation_evaluated_row_count": int(validation_metrics["evaluated_row_count"]),
        "validation_skipped_row_count": int(validation_metrics["skipped_row_count"]),
        "test_evaluated_row_count": int(test_metrics["evaluated_row_count"]),
        "test_skipped_row_count": int(test_metrics["skipped_row_count"]),
        "validation_mse": validation_metrics["mse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_smape": validation_metrics["smape"],
        "validation_rmse": validation_metrics["rmse"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_smape": test_metrics["smape"],
        "test_rmse": test_metrics["rmse"],
        "best_epoch": int(best_epoch),
        "model_path": str(model_path),
        "model_params_path": str(model_params_path),
        "training_history_path": str(training_history_path),
        "validation_metrics_path": str(validation_metrics_path),
        "test_metrics_path": str(test_metrics_path),
        "split_summary_path": str(split_summary_path),
        "adjacency_path": str(adjacency_path),
        "topk_edges_path": str(topk_edges_path),
    }


def train_one_meter(
    config: MeterConfig,
    data_root: Path = DATA_ROOT,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> dict[str, object]:
    set_seed(SEED)
    paths = build_meter_paths(
        config=config,
        preprocessed_data_dir=preprocessed_data_dir,
        output_root_dir=output_root_dir,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, input_feature_cols = load_preprocessed_splits(
        train_path=paths.train_data_path,
        valid_path=paths.valid_data_path,
        test_path=paths.test_data_path,
    )
    preprocessing_summary = load_preprocessing_summary(paths.preprocessing_summary_path)
    split_summary = make_split_summary(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        preprocessing_summary=preprocessing_summary,
    )

    feature_means, feature_stds = fit_input_scaler(train_df=train_df, input_feature_cols=input_feature_cols)
    node_ids = np.sort(train_df["building_id"].unique()).astype(np.int32)
    test_only_node_ids = sorted(set(test_df["building_id"].unique().tolist()) - set(node_ids.tolist()))

    train_data = build_dense_split_data(
        name="train",
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    valid_data = build_dense_split_data(
        name="valid",
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    test_data = build_dense_split_data(
        name="test",
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )

    del train_df, valid_df, test_df
    gc.collect()

    training_config = TrainingConfig()
    model_config = STGNNModelConfig(input_size=len(input_feature_cols), node_count=int(node_ids.shape[0]))
    train_window_starts = build_dense_window_starts(
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

    assert_training_ready(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        train_window_starts=train_window_starts,
        input_feature_cols=input_feature_cols,
        model_config=model_config,
        training_config=training_config,
    )

    model, training_history, best_epoch, best_validation_metrics = train_stgnn_model(
        train_data=train_data,
        valid_data=valid_data,
        train_window_starts=train_window_starts,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        model_config=model_config,
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
    )

    model_path = save_model_artifact(
        model=model,
        model_config=model_config,
        training_config=training_config,
        node_ids=node_ids,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        output_path=paths.model_path,
    )
    adjacency_path, topk_edges_path = save_graph_artifacts(
        model=model,
        node_ids=node_ids,
        adjacency_path=paths.adjacency_path,
        topk_edges_path=paths.topk_edges_path,
    )
    model_params_artifact = build_model_params_artifact(
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
        adjacency_path=adjacency_path,
        topk_edges_path=topk_edges_path,
    )

    model_params_path = save_json(model_params_artifact, paths.model_params_path)
    training_history_path = save_training_history(training_history=training_history, output_path=paths.training_history_path)
    validation_metrics_path = save_json(validation_metrics, paths.validation_metrics_path)
    test_metrics_path = save_json(test_metrics, paths.test_metrics_path)
    split_summary_path = save_split_summary(split_summary=split_summary, output_path=paths.split_summary_path)

    artifact_summary = build_run_summary(
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
        model_path=model_path,
        model_params_path=model_params_path,
        training_history_path=training_history_path,
        validation_metrics_path=validation_metrics_path,
        test_metrics_path=test_metrics_path,
        split_summary_path=split_summary_path,
        adjacency_path=adjacency_path,
        topk_edges_path=topk_edges_path,
        device_text=str(device),
        cuda_available=torch.cuda.is_available(),
        data_root=data_root,
    )
    save_json(artifact_summary, paths.run_summary_path)
    artifact_summary["summary_path"] = str(paths.run_summary_path)

    print(json.dumps(artifact_summary, indent=2))

    del train_data, valid_data, test_data, train_window_starts, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return artifact_summary


def train_other_meters(
    data_root: Path = DATA_ROOT,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> list[dict[str, object]]:
    output_root_dir.mkdir(parents=True, exist_ok=True)
    meter_summaries = []

    for config in METER_CONFIGS.values():
        print(f"Training STGNN for meter {config.meter_id} ({config.meter_name})")
        meter_summary = train_one_meter(
            config=config,
            data_root=data_root,
            preprocessed_data_dir=preprocessed_data_dir,
            output_root_dir=output_root_dir,
        )
        meter_summaries.append(meter_summary)

    save_json(meter_summaries, output_root_dir / OVERALL_RUN_SUMMARY_PATH.name)
    return meter_summaries


def main(smoke_test: bool = False) -> list[dict[str, object]] | dict[str, object]:
    if smoke_test:
        return run_smoke_test()

    meter_summaries = train_other_meters()
    print(json.dumps(meter_summaries, indent=2))
    return meter_summaries


if __name__ == "__main__":
    args = parse_args()
    main(smoke_test=bool(args.smoke_test))
