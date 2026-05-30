from __future__ import annotations

import gc
import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from data_preprocess.data_preprocess import CATEGORICAL_COLS, DATA_ROOT, NUMERICAL_COLS, PREPROCESSED_DATA_DIR  # noqa: E402
from raw_metric_utils import RawMetricAccumulator, inverse_log1p_minmax  # noqa: E402


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
    validation_metrics_path: Path
    test_metrics_path: Path
    run_summary_path: Path


OUTPUT_ROOT_DIR = WORKSPACE_ROOT / "TCN" / "tcn_baseline_outputs"
OVERALL_RUN_SUMMARY_PATH = OUTPUT_ROOT_DIR / "tcn_overall_run_summary.json"

METER_CONFIGS = {
    0: MeterConfig(meter_id=0, meter_name="electricity"),
    1: MeterConfig(meter_id=1, meter_name="chilled_water"),
    2: MeterConfig(meter_id=2, meter_name="steam"),
    3: MeterConfig(meter_id=3, meter_name="hot_water"),
}

SEED = 42
WINDOW_SIZE = 168
STRIDE_TRAIN = 4
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 2048
EPOCHS = 2
PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
NUM_CHANNELS = [64, 64, 64, 64, 64]
KERNEL_SIZE = 3
DROPOUT = 0.1
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
TARGET_COL = "meter_reading"
ONE_HOUR = np.timedelta64(1, "h")


@dataclass
class SplitData:
    name: str
    df: pd.DataFrame
    input_features: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray
    building_ids: np.ndarray
    meter_ids: np.ndarray
    group_slices: dict[tuple[int, int], tuple[int, int]]


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        input_features: np.ndarray,
        targets: np.ndarray,
        window_starts: np.ndarray,
        window_size: int,
    ) -> None:
        self.input_features = input_features
        self.targets = targets
        self.window_starts = window_starts
        self.window_size = window_size

    def __len__(self) -> int:
        return int(self.window_starts.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.float32]:
        start = int(self.window_starts[index])
        end = start + self.window_size
        return torch.from_numpy(self.input_features[start:end]), np.float32(self.targets[end])


# ---------------------------------------------------------------------------
# TCN Architecture
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        from torch.nn.utils import weight_norm

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.conv1.bias is not None:
            self.conv1.bias.data.fill_(0)
        if self.conv2.bias is not None:
            self.conv2.bias.data.fill_(0)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            if self.downsample.bias is not None:
                self.downsample.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_size: int, num_channels: list[int], kernel_size: int = KERNEL_SIZE, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(num_channels[-1], 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, L, C) -> transpose to (B, C, L) for Conv1d
        x = batch.transpose(1, 2)
        out = self.tcn(x)
        last_step_out = out[:, :, -1]
        return self.head(last_step_out).squeeze(-1)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def build_meter_paths(
    config: MeterConfig,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> MeterPaths:
    if config.meter_id == 0:
        data_dir = preprocessed_data_dir
        train_data_path = preprocessed_data_dir / "log1p_minmax_train.csv"
        valid_data_path = preprocessed_data_dir / "log1p_minmax_valid.csv"
        test_data_path = preprocessed_data_dir / "log1p_minmax_test.csv"
        preprocessing_summary_path = preprocessed_data_dir / "log1p_minmax_summary.json"
    else:
        data_dir = preprocessed_data_dir / f"meter_{config.meter_id}"
        train_data_path = data_dir / "log1p_minmax_train.csv"
        valid_data_path = data_dir / "log1p_minmax_valid.csv"
        test_data_path = data_dir / "log1p_minmax_test.csv"
        preprocessing_summary_path = data_dir / "log1p_minmax_summary.json"

    output_dir = output_root_dir / f"meter_{config.meter_id}"
    return MeterPaths(
        data_dir=data_dir, output_dir=output_dir,
        train_data_path=train_data_path, valid_data_path=valid_data_path, test_data_path=test_data_path,
        preprocessing_summary_path=preprocessing_summary_path,
        model_path=output_dir / "tcn_model.pt",
        model_params_path=output_dir / "tcn_model_params.json",
        validation_metrics_path=output_dir / "tcn_validation_metrics.json",
        test_metrics_path=output_dir / "tcn_test_metrics.json",
        run_summary_path=output_dir / "tcn_run_summary.json",
    )


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log1p + min-max split file not found: {path}.")
    split_df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    split_df = split_df.sort_values(by=["building_id", "meter", "timestamp"], kind="stable").reset_index(drop=True)
    split_df[TARGET_COL] = pd.to_numeric(split_df[TARGET_COL], errors="coerce").astype(np.float32)
    return split_df


def load_preprocessed_splits(
    train_path: Path, valid_path: Path, test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = load_split_data(train_path)
    valid_df = load_split_data(valid_path)
    test_df = load_split_data(test_path)
    expected_columns = train_df.columns.tolist()
    for split_name, split_df in [("valid", valid_df), ("test", test_df)]:
        if split_df.columns.tolist() != expected_columns:
            raise ValueError(f"The {split_name} split columns do not match the training split columns.")
    input_feature_cols = [TARGET_COL] + [c for c in train_df.columns if c not in {"timestamp", TARGET_COL}]
    return train_df, valid_df, test_df, input_feature_cols


def load_preprocessing_summary(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def make_split_summary(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
    preprocessing_summary: dict[str, object],
) -> pd.DataFrame:
    split_records = preprocessing_summary.get("splits")
    if isinstance(split_records, list) and split_records:
        return pd.DataFrame(split_records)
    return pd.DataFrame([
        build_split_summary_row("train", train_df),
        build_split_summary_row("valid", valid_df),
        build_split_summary_row("test", test_df),
    ])


def build_split_summary_row(split_name: str, split_df: pd.DataFrame) -> dict[str, str | int]:
    return {
        "split": split_name,
        "start_timestamp": str(split_df["timestamp"].min()),
        "end_timestamp": str(split_df["timestamp"].max()),
        "row_count": int(split_df.shape[0]),
    }


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
    split_df: pd.DataFrame, input_feature_cols: list[str],
    feature_means: np.ndarray, feature_stds: np.ndarray,
) -> np.ndarray:
    input_values = split_df[input_feature_cols].to_numpy(dtype=np.float32, copy=True)
    input_values -= feature_means
    input_values /= feature_stds
    np.nan_to_num(input_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(input_values, dtype=np.float32)


def build_group_slices(split_df: pd.DataFrame) -> dict[tuple[int, int], tuple[int, int]]:
    building_ids = split_df["building_id"].to_numpy()
    meter_ids = split_df["meter"].to_numpy()
    row_count = int(split_df.shape[0])
    if row_count == 0:
        return {}
    group_starts_mask = np.empty(row_count, dtype=bool)
    group_starts_mask[0] = True
    group_starts_mask[1:] = (building_ids[1:] != building_ids[:-1]) | (meter_ids[1:] != meter_ids[:-1])
    group_starts = np.flatnonzero(group_starts_mask)
    group_ends = np.append(group_starts[1:], row_count)
    group_slices: dict[tuple[int, int], tuple[int, int]] = {}
    for start, end in zip(group_starts, group_ends):
        key = (int(building_ids[start]), int(meter_ids[start]))
        group_slices[key] = (int(start), int(end))
    return group_slices


def build_split_data(name: str, split_df: pd.DataFrame, input_features: np.ndarray) -> SplitData:
    return SplitData(
        name=name, df=split_df, input_features=input_features,
        targets=split_df[TARGET_COL].to_numpy(dtype=np.float32, copy=True),
        timestamps=split_df["timestamp"].to_numpy(dtype="datetime64[ns]", copy=True),
        building_ids=split_df["building_id"].to_numpy(copy=True),
        meter_ids=split_df["meter"].to_numpy(copy=True),
        group_slices=build_group_slices(split_df),
    )


def build_window_starts(
    split_data: SplitData, window_size: int = WINDOW_SIZE, stride: int = STRIDE_TRAIN,
) -> np.ndarray:
    start_arrays: list[np.ndarray] = []
    for group_start, group_end in split_data.group_slices.values():
        group_length = group_end - group_start
        if group_length <= window_size:
            continue
        group_timestamps = split_data.timestamps[group_start:group_end]
        gap_offsets = np.flatnonzero(np.diff(group_timestamps) != ONE_HOUR) + 1
        segment_starts = np.append(np.array([0], dtype=np.int64), gap_offsets) + group_start
        segment_ends = np.append(gap_offsets, group_length) + group_start
        for segment_start, segment_end in zip(segment_starts, segment_ends):
            if segment_end - segment_start <= window_size:
                continue
            start_arrays.append(np.arange(segment_start, segment_end - window_size, stride, dtype=np.int32))
    if not start_arrays:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(start_arrays).astype(np.int32, copy=False)


def save_json(data: dict[str, object] | list[dict[str, object]], output_path: Path) -> Path:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Auto-rolling evaluation infrastructure
# ---------------------------------------------------------------------------

def compute_smape_sum(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_ratio = np.divide(
        2.0 * np.abs(y_pred - y_true), denominator,
        out=np.zeros_like(y_true, dtype=np.float32), where=denominator != 0,
    )
    return float(np.sum(safe_ratio, dtype=np.float64))


def collect_history_suffix_for_group(
    key: tuple[int, int],
    first_timestamp: np.datetime64,
    history_parts: list[SplitData],
    window_size: int,
) -> tuple[deque[np.ndarray], np.datetime64 | None]:
    remaining = window_size
    expected_end = first_timestamp - ONE_HOUR
    pieces_from_newest: list[np.ndarray] = []
    last_timestamp: np.datetime64 | None = None

    for history_part in reversed(history_parts):
        group_slice = history_part.group_slices.get(key)
        if group_slice is None:
            continue
        group_start, group_end = group_slice
        group_timestamps = history_part.timestamps[group_start:group_end]
        local_end_pos = int(np.searchsorted(group_timestamps, expected_end))
        if local_end_pos >= group_timestamps.shape[0] or group_timestamps[local_end_pos] != expected_end:
            break
        local_start_pos = local_end_pos
        collected_count = 1
        while collected_count < remaining and local_start_pos > 0:
            if group_timestamps[local_start_pos] - group_timestamps[local_start_pos - 1] != ONE_HOUR:
                break
            local_start_pos -= 1
            collected_count += 1
        absolute_start = group_start + local_start_pos
        absolute_end = group_start + local_end_pos + 1
        pieces_from_newest.append(history_part.input_features[absolute_start:absolute_end])
        if last_timestamp is None:
            last_timestamp = history_part.timestamps[absolute_end - 1]
        remaining -= collected_count
        if remaining == 0:
            break
        if local_start_pos > 0:
            break
        expected_end = group_timestamps[local_start_pos] - ONE_HOUR

    history_rows: deque[np.ndarray] = deque(maxlen=window_size)
    for piece in reversed(pieces_from_newest):
        for row in piece:
            history_rows.append(row)
    return history_rows, last_timestamp


def build_initial_histories(
    split_data: SplitData, history_parts: list[SplitData], window_size: int,
) -> tuple[dict[tuple[int, int], deque[np.ndarray]], dict[tuple[int, int], np.datetime64]]:
    histories: dict[tuple[int, int], deque[np.ndarray]] = {}
    last_timestamps: dict[tuple[int, int], np.datetime64] = {}
    for key, (group_start, _) in split_data.group_slices.items():
        history_rows, last_timestamp = collect_history_suffix_for_group(
            key=key, first_timestamp=split_data.timestamps[group_start],
            history_parts=history_parts, window_size=window_size,
        )
        histories[key] = history_rows
        if last_timestamp is not None:
            last_timestamps[key] = last_timestamp
    return histories, last_timestamps


def append_history_row(
    histories: dict[tuple[int, int], deque[np.ndarray]],
    last_timestamps: dict[tuple[int, int], np.datetime64],
    key: tuple[int, int], feature_row: np.ndarray,
    timestamp: np.datetime64, window_size: int,
) -> None:
    history_rows = histories.get(key)
    if history_rows is None:
        history_rows = deque(maxlen=window_size)
        histories[key] = history_rows
    history_rows.append(feature_row)
    last_timestamps[key] = timestamp


def predict_sequence_batch(
    model: nn.Module, sequences: np.ndarray, device: torch.device, batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, sequences.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, sequences.shape[0])
            batch = torch.from_numpy(sequences[batch_start:batch_end]).to(device=device, non_blocking=True).float()
            batch_prediction = model(batch).detach().cpu().numpy().astype(np.float32)
            predictions.append(batch_prediction)
    return np.nan_to_num(np.concatenate(predictions), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def finalize_metrics(
    row_count: int, skipped_row_count: int,
    squared_error_sum: float, absolute_error_sum: float,
    smape_ratio_sum: float, rmsle_squared_error_sum: float,
    y_sum: float, y_squared_sum: float,
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
        "mse": float(mse), "mae": float(mae), "r2": float(r2),
        "smape": float(smape), "rmse": float(rmse), "rmsle": float(rmsle),
        "evaluated_row_count": int(row_count), "skipped_row_count": int(skipped_row_count),
    }


def rolling_evaluate_split(
    model: nn.Module,
    split_data: SplitData,
    history_parts: list[SplitData],
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    device: torch.device,
    window_size: int,
    eval_batch_size: int,
    raw_accumulator: RawMetricAccumulator | None = None,
) -> dict[str, float | int]:
    """Autoregressive rolling evaluation: predicted meter_reading is fed back."""
    meter_feature_index = input_feature_cols.index(TARGET_COL)
    histories, last_timestamps = build_initial_histories(
        split_data=split_data, history_parts=history_parts, window_size=window_size,
    )

    # Sort rows by (timestamp, building_id, meter) so we process chronologically
    timestamp_order = np.lexsort((
        split_data.meter_ids,
        split_data.building_ids,
        split_data.timestamps.astype("datetime64[ns]").astype(np.int64),
    ))
    ordered_timestamps = split_data.timestamps[timestamp_order]
    timestamp_starts = np.append(
        np.array([0], dtype=np.int64),
        np.flatnonzero(ordered_timestamps[1:] != ordered_timestamps[:-1]) + 1,
    )
    timestamp_ends = np.append(timestamp_starts[1:], timestamp_order.shape[0])

    evaluated_row_count = 0
    skipped_row_count = 0
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    smape_ratio_sum = 0.0
    rmsle_squared_error_sum = 0.0
    y_sum = 0.0
    y_squared_sum = 0.0

    total_timestamps = int(timestamp_starts.shape[0])
    log_interval = max(1, total_timestamps // 20)

    for ts_idx, (timestamp_start, timestamp_end) in enumerate(zip(timestamp_starts, timestamp_ends)):
        sequence_rows: list[np.ndarray] = []
        sequence_indices: list[int] = []
        sequence_keys: list[tuple[int, int]] = []

        for ordered_position in range(int(timestamp_start), int(timestamp_end)):
            row_index = int(timestamp_order[ordered_position])
            key = (int(split_data.building_ids[row_index]), int(split_data.meter_ids[row_index]))
            timestamp = split_data.timestamps[row_index]
            expected_previous_timestamp = timestamp - ONE_HOUR
            history_rows = histories.get(key)
            last_timestamp = last_timestamps.get(key)

            # If no continuous history available, reset and skip
            if history_rows is None or last_timestamp is None or last_timestamp != expected_previous_timestamp:
                histories[key] = deque(maxlen=window_size)
                append_history_row(
                    histories=histories, last_timestamps=last_timestamps,
                    key=key, feature_row=split_data.input_features[row_index],
                    timestamp=timestamp, window_size=window_size,
                )
                skipped_row_count += 1
                continue

            # If history buffer not yet full, accumulate and skip
            if len(history_rows) < window_size:
                append_history_row(
                    histories=histories, last_timestamps=last_timestamps,
                    key=key, feature_row=split_data.input_features[row_index],
                    timestamp=timestamp, window_size=window_size,
                )
                skipped_row_count += 1
                continue

            # History is full -> queue for batch prediction
            sequence_rows.append(np.stack(history_rows, axis=0))
            sequence_indices.append(row_index)
            sequence_keys.append(key)

        if not sequence_rows:
            continue

        # Batch predict
        sequence_batch = np.ascontiguousarray(np.stack(sequence_rows, axis=0), dtype=np.float32)
        batch_predictions = predict_sequence_batch(
            model=model, sequences=sequence_batch, device=device, batch_size=eval_batch_size,
        )
        batch_predictions = np.clip(batch_predictions, a_min=0.0, a_max=1.0)
        batch_targets = split_data.targets[np.asarray(sequence_indices, dtype=np.int64)].astype(np.float32, copy=False)
        batch_errors = batch_predictions - batch_targets

        # Accumulate normalized metrics
        evaluated_row_count += int(batch_targets.shape[0])
        squared_error_sum += float(np.sum(batch_errors * batch_errors, dtype=np.float64))
        absolute_error_sum += float(np.sum(np.abs(batch_errors), dtype=np.float64))
        smape_ratio_sum += compute_smape_sum(batch_targets, batch_predictions)

        clipped_true = np.clip(batch_targets, a_min=0.0, a_max=None)
        clipped_pred = np.clip(batch_predictions, a_min=0.0, a_max=None)
        log_errors = np.log1p(clipped_pred) - np.log1p(clipped_true)
        rmsle_squared_error_sum += float(np.sum(log_errors * log_errors, dtype=np.float64))

        y_sum += float(np.sum(batch_targets, dtype=np.float64))
        y_squared_sum += float(np.sum(batch_targets * batch_targets, dtype=np.float64))

        # Accumulate raw metrics if accumulator provided
        if raw_accumulator is not None:
            raw_accumulator.update_normalized(batch_targets, batch_predictions)

        # *** KEY: Feed predictions back into history (autoregressive) ***
        # Standardize the predicted meter_reading to match the z-scored feature space
        normalized_predicted_meter = (batch_predictions - feature_means[meter_feature_index]) / feature_stds[meter_feature_index]
        normalized_predicted_meter = np.nan_to_num(
            normalized_predicted_meter, copy=False, nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32)

        for prediction_position, row_index in enumerate(sequence_indices):
            # Copy the real feature row but replace meter_reading with prediction
            predicted_feature_row = split_data.input_features[row_index].copy()
            predicted_feature_row[meter_feature_index] = normalized_predicted_meter[prediction_position]
            append_history_row(
                histories=histories, last_timestamps=last_timestamps,
                key=sequence_keys[prediction_position],
                feature_row=predicted_feature_row,
                timestamp=split_data.timestamps[row_index],
                window_size=window_size,
            )

        del sequence_batch

        if ts_idx % log_interval == 0:
            print(f"  Rolling {split_data.name}: {ts_idx}/{total_timestamps} timestamps processed, "
                  f"evaluated={evaluated_row_count}, skipped={skipped_row_count}")

    return finalize_metrics(
        row_count=evaluated_row_count, skipped_row_count=skipped_row_count,
        squared_error_sum=squared_error_sum, absolute_error_sum=absolute_error_sum,
        smape_ratio_sum=smape_ratio_sum, rmsle_squared_error_sum=rmsle_squared_error_sum,
        y_sum=y_sum, y_squared_sum=y_squared_sum,
    )


# ---------------------------------------------------------------------------
# Training (still uses teacher-forcing via DataLoader, standard practice)
# ---------------------------------------------------------------------------

def train_one_meter(
    config: MeterConfig,
    device: torch.device,
    data_root: Path = DATA_ROOT,
    preprocessed_data_dir: Path = PREPROCESSED_DATA_DIR,
    output_root_dir: Path = OUTPUT_ROOT_DIR,
) -> dict[str, object]:
    paths = build_meter_paths(config=config, preprocessed_data_dir=preprocessed_data_dir, output_root_dir=output_root_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for meter {config.meter_id} ({config.meter_name})...")
    train_df, valid_df, test_df, input_feature_cols = load_preprocessed_splits(
        train_path=paths.train_data_path, valid_path=paths.valid_data_path, test_path=paths.test_data_path,
    )
    preprocessing_summary = load_preprocessing_summary(paths.preprocessing_summary_path)
    split_summary = make_split_summary(train_df=train_df, valid_df=valid_df, test_df=test_df, preprocessing_summary=preprocessing_summary)

    print("Fitting feature scaler pipeline...")
    feature_means, feature_stds = fit_input_scaler(train_df=train_df, input_feature_cols=input_feature_cols)

    train_input_features = transform_input_features(train_df, input_feature_cols, feature_means, feature_stds)
    valid_input_features = transform_input_features(valid_df, input_feature_cols, feature_means, feature_stds)
    test_input_features = transform_input_features(test_df, input_feature_cols, feature_means, feature_stds)

    train_data = build_split_data("train", train_df, train_input_features)
    valid_data = build_split_data("valid", valid_df, valid_input_features)
    test_data = build_split_data("test", test_df, test_input_features)

    print("Building training sequence sliding windows...")
    train_starts = build_window_starts(train_data, stride=STRIDE_TRAIN)

    train_dataset = SequenceWindowDataset(train_input_features, train_data.targets, train_starts, WINDOW_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)

    print(f"Initializing TCN Model with input size {len(input_feature_cols)}...")
    model = TCNRegressor(
        input_size=len(input_feature_cols), num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE, dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_state_dict = {}
    epochs_without_improvement = 0

    print("Starting PyTorch TCN Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True).float()
            batch_y = batch_y.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            b_size = int(batch_y.numel())
            train_loss_sum += float(loss.item()) * b_size
            train_samples += b_size

        train_loss = train_loss_sum / train_samples

        # Rolling validation
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Running rolling validation...")
        valid_metrics = rolling_evaluate_split(
            model=model, split_data=valid_data, history_parts=[train_data],
            input_feature_cols=input_feature_cols, feature_means=feature_means, feature_stds=feature_stds,
            device=device, window_size=WINDOW_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
        )
        valid_rmse = valid_metrics["rmse"]
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Valid Rolling RMSE: {valid_rmse:.6f}")

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load best weights
    model.load_state_dict(best_state_dict)

    # Save model
    torch.save({
        "model_state_dict": best_state_dict,
        "input_feature_cols": input_feature_cols,
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "num_channels": NUM_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
    }, paths.model_path)

    # Final rolling evaluation on validation and test sets
    target_log1p_min = float(preprocessing_summary.get("target_log1p_min", 0.0))
    target_log1p_max = float(preprocessing_summary.get("target_log1p_max", 1.0))

    print("Final rolling evaluation on validation set...")
    valid_raw_acc = RawMetricAccumulator(target_log1p_min, target_log1p_max)
    validation_metrics = rolling_evaluate_split(
        model=model, split_data=valid_data, history_parts=[train_data],
        input_feature_cols=input_feature_cols, feature_means=feature_means, feature_stds=feature_stds,
        device=device, window_size=WINDOW_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
        raw_accumulator=valid_raw_acc,
    )
    raw_validation_metrics = valid_raw_acc.finalize()

    print("Final rolling evaluation on test set...")
    test_raw_acc = RawMetricAccumulator(target_log1p_min, target_log1p_max)
    test_metrics = rolling_evaluate_split(
        model=model, split_data=test_data, history_parts=[train_data, valid_data],
        input_feature_cols=input_feature_cols, feature_means=feature_means, feature_stds=feature_stds,
        device=device, window_size=WINDOW_SIZE, eval_batch_size=EVAL_BATCH_SIZE,
        raw_accumulator=test_raw_acc,
    )
    raw_test_metrics = test_raw_acc.finalize()

    # Save metrics
    save_json(validation_metrics, paths.validation_metrics_path)
    save_json(test_metrics, paths.test_metrics_path)

    # Build model params
    model_params_artifact = {
        "meter_id": config.meter_id, "meter_name": config.meter_name,
        "num_channels": NUM_CHANNELS, "kernel_size": KERNEL_SIZE,
        "window_size": WINDOW_SIZE, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
        "feature_count": len(input_feature_cols), "tcn_feature_cols": input_feature_cols,
        "train_ratio": TRAIN_RATIO, "valid_ratio": VALID_RATIO, "test_ratio": TEST_RATIO,
        "train_data_path": str(paths.train_data_path),
        "valid_data_path": str(paths.valid_data_path),
        "test_data_path": str(paths.test_data_path),
        "preprocessing_summary_path": str(paths.preprocessing_summary_path),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": target_log1p_min, "target_log1p_max": target_log1p_max,
        "evaluation_mode": "autoregressive_rolling",
    }
    save_json(model_params_artifact, paths.model_params_path)

    # Build run summary
    artifact_summary = {
        "model_name": "TCN",
        "evaluation_mode": "autoregressive_rolling",
        "meter_id": config.meter_id, "meter_name": config.meter_name,
        "data_root": str(data_root),
        "train_data_path": str(paths.train_data_path),
        "valid_data_path": str(paths.valid_data_path),
        "test_data_path": str(paths.test_data_path),
        "preprocessing_summary_path": str(paths.preprocessing_summary_path),
        "output_dir": str(paths.output_dir),
        "preprocessed_data_dir": str(paths.data_dir),
        "target_preprocess": preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": target_log1p_min, "target_log1p_max": target_log1p_max,
        "train_ratio": TRAIN_RATIO, "valid_ratio": VALID_RATIO, "test_ratio": TEST_RATIO,
        "feature_count": len(input_feature_cols), "tcn_feature_cols": input_feature_cols,
        "train_row_count": int(train_df.shape[0]),
        "valid_row_count": int(valid_df.shape[0]),
        "test_row_count": int(test_df.shape[0]),
        "splits": split_summary.to_dict(orient="records"),
        "validation_normalized_metrics": validation_metrics,
        "test_normalized_metrics": test_metrics,
        "validation_raw_metrics": raw_validation_metrics,
        "test_raw_metrics": raw_test_metrics,
        "validation_mse": validation_metrics["mse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "validation_smape": validation_metrics["smape"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_rmsle": validation_metrics["rmsle"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_smape": test_metrics["smape"],
        "test_rmse": test_metrics["rmse"],
        "test_rmsle": test_metrics["rmsle"],
        "model_path": str(paths.model_path),
        "model_params_path": str(paths.model_params_path),
        "validation_metrics_path": str(paths.validation_metrics_path),
        "test_metrics_path": str(paths.test_metrics_path),
    }
    save_json(artifact_summary, paths.run_summary_path)
    print(f"Meter {config.meter_id} ({config.meter_name}) auto-rolling baseline completed.")
    print(f"Validation Rolling RMSE: {validation_metrics['rmse']:.6f} | Test Rolling RMSE: {test_metrics['rmse']:.6f}")

    del train_df, valid_df, test_df, train_input_features, valid_input_features, test_input_features, train_starts
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return artifact_summary


def main() -> list[dict[str, object]]:
    set_seed(SEED)
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for TCN training: {device}")

    meter_summaries = []
    for config in METER_CONFIGS.values():
        print("=" * 60)
        print(f"Starting TCN (Auto-Rolling) for Meter {config.meter_id} ({config.meter_name})")
        print("=" * 60)
        meter_summary = train_one_meter(config=config, device=device)
        meter_summaries.append(meter_summary)

    save_json(meter_summaries, OVERALL_RUN_SUMMARY_PATH)
    print("=" * 60)
    print("All meters completed. Overall summary saved.")
    print("=" * 60)
    return meter_summaries


if __name__ == "__main__":
    main()
