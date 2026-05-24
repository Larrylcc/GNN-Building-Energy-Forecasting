from __future__ import annotations

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


OUTPUT_ROOT_DIR = WORKSPACE_ROOT / "GRU" / "gru_other_meters_auto-rolling_outputs"
OVERALL_RUN_SUMMARY_PATH = OUTPUT_ROOT_DIR / "gru_other_meters_run_summary.json"

METER_CONFIGS = {
    1: MeterConfig(meter_id=1, meter_name="chilled_water"),
    2: MeterConfig(meter_id=2, meter_name="steam"),
    3: MeterConfig(meter_id=3, meter_name="hot_water"),
}

SEED = 42
WINDOW_SIZE = 168
STRIDE = 1
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 2048
EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
TARGET_COL = "meter_reading"
ONE_HOUR = np.timedelta64(1, "h")


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
        model_path=output_dir / "gru_final_model.pt",
        model_params_path=output_dir / "gru_model_params.json",
        training_history_path=output_dir / "gru_training_history.csv",
        split_summary_path=output_dir / "gru_time_series_split.csv",
        validation_metrics_path=output_dir / "gru_validation_metrics.json",
        test_metrics_path=output_dir / "gru_test_metrics.json",
        run_summary_path=output_dir / "gru_run_summary.json",
    )


@dataclass(frozen=True)
class GRUModelConfig:
    input_size: int
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


class GRURegressor(nn.Module):
    def __init__(self, config: GRUModelConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(batch)
        last_hidden = hidden[-1]
        return self.head(last_hidden).squeeze(-1)


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


def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log1p + min-max split file not found: {path}.")

    split_df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    split_df = split_df.sort_values(by=["building_id", "meter", "timestamp"], kind="stable").reset_index(drop=True)
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
        name=name,
        df=split_df,
        input_features=input_features,
        targets=split_df[TARGET_COL].to_numpy(dtype=np.float32, copy=True),
        timestamps=split_df["timestamp"].to_numpy(dtype="datetime64[ns]", copy=True),
        building_ids=split_df["building_id"].to_numpy(copy=True),
        meter_ids=split_df["meter"].to_numpy(copy=True),
        group_slices=build_group_slices(split_df),
    )


def build_window_starts(
    split_data: SplitData,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    start_arrays: list[np.ndarray] = []

    for group_start, group_end in tqdm(
        split_data.group_slices.values(),
        desc=f"Building {split_data.name} GRU windows",
        unit="group",
    ):
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
    split_data: SplitData,
    history_parts: list[SplitData],
    window_size: int,
) -> tuple[dict[tuple[int, int], deque[np.ndarray]], dict[tuple[int, int], np.datetime64]]:
    histories: dict[tuple[int, int], deque[np.ndarray]] = {}
    last_timestamps: dict[tuple[int, int], np.datetime64] = {}

    for key, (group_start, _) in split_data.group_slices.items():
        history_rows, last_timestamp = collect_history_suffix_for_group(
            key=key,
            first_timestamp=split_data.timestamps[group_start],
            history_parts=history_parts,
            window_size=window_size,
        )
        histories[key] = history_rows
        if last_timestamp is not None:
            last_timestamps[key] = last_timestamp

    return histories, last_timestamps


def append_history_row(
    histories: dict[tuple[int, int], deque[np.ndarray]],
    last_timestamps: dict[tuple[int, int], np.datetime64],
    key: tuple[int, int],
    feature_row: np.ndarray,
    timestamp: np.datetime64,
    window_size: int,
) -> None:
    history_rows = histories.get(key)
    if history_rows is None:
        history_rows = deque(maxlen=window_size)
        histories[key] = history_rows
    history_rows.append(feature_row)
    last_timestamps[key] = timestamp


def predict_sequence_batch(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int,
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
    split_data: SplitData,
    history_parts: list[SplitData],
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    device: torch.device,
    window_size: int,
    eval_batch_size: int,
) -> dict[str, float | int]:
    meter_feature_index = input_feature_cols.index(TARGET_COL)
    histories, last_timestamps = build_initial_histories(
        split_data=split_data,
        history_parts=history_parts,
        window_size=window_size,
    )

    timestamp_order = np.lexsort(
        (
            split_data.meter_ids,
            split_data.building_ids,
            split_data.timestamps.astype("datetime64[ns]").astype(np.int64),
        )
    )
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
    y_sum = 0.0
    y_squared_sum = 0.0

    progress_iter = tqdm(
        zip(timestamp_starts, timestamp_ends),
        total=int(timestamp_starts.shape[0]),
        desc=f"Rolling {split_data.name}",
        unit="timestamp",
    )
    for timestamp_start, timestamp_end in progress_iter:
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

            if history_rows is None or last_timestamp is None or last_timestamp != expected_previous_timestamp:
                histories[key] = deque(maxlen=window_size)
                append_history_row(
                    histories=histories,
                    last_timestamps=last_timestamps,
                    key=key,
                    feature_row=split_data.input_features[row_index],
                    timestamp=timestamp,
                    window_size=window_size,
                )
                skipped_row_count += 1
                continue

            if len(history_rows) < window_size:
                append_history_row(
                    histories=histories,
                    last_timestamps=last_timestamps,
                    key=key,
                    feature_row=split_data.input_features[row_index],
                    timestamp=timestamp,
                    window_size=window_size,
                )
                skipped_row_count += 1
                continue

            sequence_rows.append(np.stack(history_rows, axis=0))
            sequence_indices.append(row_index)
            sequence_keys.append(key)

        if not sequence_rows:
            continue

        sequence_batch = np.ascontiguousarray(np.stack(sequence_rows, axis=0), dtype=np.float32)
        batch_predictions = predict_sequence_batch(
            model=model,
            sequences=sequence_batch,
            device=device,
            batch_size=eval_batch_size,
        )
        batch_targets = split_data.targets[np.asarray(sequence_indices, dtype=np.int64)].astype(np.float32, copy=False)
        batch_errors = batch_predictions - batch_targets

        evaluated_row_count += int(batch_targets.shape[0])
        squared_error_sum += float(np.sum(batch_errors * batch_errors, dtype=np.float64))
        absolute_error_sum += float(np.sum(np.abs(batch_errors), dtype=np.float64))
        smape_ratio_sum += compute_smape_sum(batch_targets, batch_predictions)
        y_sum += float(np.sum(batch_targets, dtype=np.float64))
        y_squared_sum += float(np.sum(batch_targets * batch_targets, dtype=np.float64))

        normalized_predicted_meter = (batch_predictions - feature_means[meter_feature_index]) / feature_stds[meter_feature_index]
        normalized_predicted_meter = np.nan_to_num(
            normalized_predicted_meter,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        for prediction_position, row_index in enumerate(sequence_indices):
            predicted_feature_row = split_data.input_features[row_index].copy()
            predicted_feature_row[meter_feature_index] = normalized_predicted_meter[prediction_position]
            append_history_row(
                histories=histories,
                last_timestamps=last_timestamps,
                key=sequence_keys[prediction_position],
                feature_row=predicted_feature_row,
                timestamp=split_data.timestamps[row_index],
                window_size=window_size,
            )

        del sequence_batch

    return finalize_metrics(
        row_count=evaluated_row_count,
        skipped_row_count=skipped_row_count,
        squared_error_sum=squared_error_sum,
        absolute_error_sum=absolute_error_sum,
        smape_ratio_sum=smape_ratio_sum,
        y_sum=y_sum,
        y_squared_sum=y_squared_sum,
    )


def train_gru_model(
    train_data: SplitData,
    valid_data: SplitData,
    train_window_starts: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    model_config: GRUModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> tuple[GRURegressor, pd.DataFrame, int, dict[str, float | int]]:
    train_dataset = SequenceWindowDataset(
        input_features=train_data.input_features,
        targets=train_data.targets,
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

    model = GRURegressor(config=model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_fn = nn.MSELoss()

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

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{training_config.epochs}", unit="batch"):
            batch_x = batch_x.to(device=device, non_blocking=True).float()
            batch_y = batch_y.to(device=device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            optimizer.step()

            batch_size = int(batch_y.numel())
            train_loss_sum += float(loss.item()) * batch_size
            train_sample_count += batch_size

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
    model: GRURegressor,
    model_config: GRUModelConfig,
    training_config: TrainingConfig,
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
            "input_feature_cols": input_feature_cols,
            "input_feature_means": feature_means.tolist(),
            "input_feature_stds": feature_stds.tolist(),
            "target_column": TARGET_COL,
        },
        output_path,
    )
    return output_path


def assert_training_ready(
    train_data: SplitData,
    valid_data: SplitData,
    test_data: SplitData,
    train_window_starts: np.ndarray,
    input_feature_cols: list[str],
    model_config: GRUModelConfig,
) -> None:
    if train_window_starts.shape[0] == 0:
        raise ValueError("The training split produced zero continuous GRU windows.")
    if train_data.input_features.shape[1] != len(input_feature_cols):
        raise ValueError("Training input feature width does not match input_feature_cols.")
    if model_config.input_size != len(input_feature_cols):
        raise ValueError("GRU input_size does not match the number of input features.")
    if valid_data.input_features.shape[1] != model_config.input_size:
        raise ValueError("Validation input feature width does not match GRU input_size.")
    if test_data.input_features.shape[1] != model_config.input_size:
        raise ValueError("Test input feature width does not match GRU input_size.")


def build_model_params_artifact(
    config: MeterConfig,
    paths: MeterPaths,
    preprocessing_summary: dict[str, object],
    model_config: GRUModelConfig,
    training_config: TrainingConfig,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    best_epoch: int,
    best_validation_metrics: dict[str, float | int],
    train_window_count: int,
) -> dict[str, object]:
    return {
        "meter_id": config.meter_id,
        "meter_name": config.meter_name,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "best_epoch": int(best_epoch),
        "best_validation_rmse": float(best_validation_metrics["rmse"]),
        "feature_count": len(input_feature_cols),
        "gru_input_feature_cols": input_feature_cols,
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
    train_row_count: int,
    valid_row_count: int,
    test_row_count: int,
    model_path: Path,
    model_params_path: Path,
    training_history_path: Path,
    validation_metrics_path: Path,
    test_metrics_path: Path,
    split_summary_path: Path,
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
        "feature_count": len(input_feature_cols),
        "gru_input_feature_cols": input_feature_cols,
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS,
        "train_row_count": int(train_row_count),
        "valid_row_count": int(valid_row_count),
        "test_row_count": int(test_row_count),
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

    train_input_features = transform_input_features(
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    valid_input_features = transform_input_features(
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    test_input_features = transform_input_features(
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )

    train_data = build_split_data(name="train", split_df=train_df, input_features=train_input_features)
    valid_data = build_split_data(name="valid", split_df=valid_df, input_features=valid_input_features)
    test_data = build_split_data(name="test", split_df=test_df, input_features=test_input_features)

    training_config = TrainingConfig()
    model_config = GRUModelConfig(input_size=len(input_feature_cols))
    train_window_starts = build_window_starts(
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
                "train_window_count": int(train_window_starts.shape[0]),
                "input_feature_count": len(input_feature_cols),
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
    )

    model, training_history, best_epoch, best_validation_metrics = train_gru_model(
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
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        output_path=paths.model_path,
    )
    model_params_artifact = build_model_params_artifact(
        config=config,
        paths=paths,
        preprocessing_summary=preprocessing_summary,
        model_config=model_config,
        training_config=training_config,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        train_window_count=int(train_window_starts.shape[0]),
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
        train_row_count=int(train_data.df.shape[0]),
        valid_row_count=int(valid_data.df.shape[0]),
        test_row_count=int(test_data.df.shape[0]),
        model_path=model_path,
        model_params_path=model_params_path,
        training_history_path=training_history_path,
        validation_metrics_path=validation_metrics_path,
        test_metrics_path=test_metrics_path,
        split_summary_path=split_summary_path,
        device_text=str(device),
        cuda_available=torch.cuda.is_available(),
        data_root=data_root,
    )
    save_json(artifact_summary, paths.run_summary_path)
    artifact_summary["summary_path"] = str(paths.run_summary_path)

    print(json.dumps(artifact_summary, indent=2))

    del train_df, valid_df, test_df
    del train_input_features, valid_input_features, test_input_features
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
        print(f"Training GRU for meter {config.meter_id} ({config.meter_name})")
        meter_summary = train_one_meter(
            config=config,
            data_root=data_root,
            preprocessed_data_dir=preprocessed_data_dir,
            output_root_dir=output_root_dir,
        )
        meter_summaries.append(meter_summary)

    save_json(meter_summaries, output_root_dir / OVERALL_RUN_SUMMARY_PATH.name)
    return meter_summaries


def main() -> list[dict[str, object]]:
    meter_summaries = train_other_meters()
    print(json.dumps(meter_summaries, indent=2))
    return meter_summaries


if __name__ == "__main__":
    main()
