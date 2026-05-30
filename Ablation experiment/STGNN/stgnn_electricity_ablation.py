from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_STGNN_PATH = WORKSPACE_ROOT / "STGNN" / "stgnn_gcn_gru_auto-rolling.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "electricity_stgnn_ablation_outputs"
SUMMARY_CSV_PATH = OUTPUT_DIR / "electricity_stgnn_ablation_summary.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "electricity_stgnn_ablation_summary.json"
SMOKE_SUMMARY_PATH = OUTPUT_DIR / "smoke_summary.json"
FINAL_SCORE_METRIC = "test_rmsle"
METRIC_SCALE = "log1p_minmax"


def load_source_stgnn_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stgnn_electricity_source", SOURCE_STGNN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source STGNN script: {SOURCE_STGNN_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STGNN = load_source_stgnn_module()


@dataclass(frozen=True)
class AblationSpec:
    removed_module: str
    factory: Callable[[object], nn.Module]
    has_graph_artifact: bool


@dataclass
class PreparedData:
    train_data: object
    valid_data: object
    test_data: object
    input_feature_cols: list[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    node_ids: np.ndarray
    test_only_node_ids: list[int]
    preprocessing_summary: dict[str, object]
    train_window_starts: np.ndarray


class IdentityGraphConstructor(nn.Module):
    def __init__(self, node_count: int) -> None:
        super().__init__()
        self.register_buffer("adjacency", torch.eye(node_count, dtype=torch.float32), persistent=False)

    def forward(self) -> torch.Tensor:
        return self.adjacency


class NoAdaptiveGraphSTGNNGRURegressor(STGNN.STGNNGRURegressor):
    def __init__(self, config: object) -> None:
        super().__init__(config=config)
        self.graph_constructor = IdentityGraphConstructor(node_count=config.node_count)


def run_gcn_layers(model: nn.Module, batch: torch.Tensor) -> list[torch.Tensor]:
    adjacency = model.graph_constructor()
    hidden = model.input_projection(batch)
    layer_outputs: list[torch.Tensor] = []

    for layer in model.gcn_layers:
        aggregated = torch.einsum("ij,btjf->btif", adjacency, hidden)
        update = torch.relu(layer(aggregated))
        hidden = hidden + model.dropout(update)
        layer_outputs.append(hidden)

    return layer_outputs


def apply_gru_head(model: nn.Module, fused: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, node_count, feature_count = fused.shape
    gru_input = fused.permute(0, 2, 1, 3).contiguous().view(
        batch_size * node_count,
        sequence_length,
        feature_count,
    )
    _, hidden_state = model.gru(gru_input)
    last_hidden = hidden_state[-1]
    return model.head(last_hidden).view(batch_size, node_count)


class NoGRUSTGNNRegressor(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config
        self.graph_constructor = STGNN.GraphConstructor(
            node_count=config.node_count,
            embed_dim=config.embed_dim,
            top_k=config.graph_top_k,
        )
        self.input_projection = nn.Linear(config.input_size, config.gcn_dim)
        self.gcn_layers = nn.ModuleList(
            nn.Linear(config.gcn_dim, config.gcn_dim) for _ in range(config.gcn_layers)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.gcn_dim, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        layer_outputs = run_gcn_layers(model=self, batch=batch)
        fused = layer_outputs[-1]
        last_time_step = fused[:, -1, :, :]
        return self.head(last_time_step).squeeze(-1)

    def get_adjacency(self) -> torch.Tensor:
        return self.graph_constructor()


class NoGCNSTGNNGRURegressor(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_size, config.gcn_dim)
        self.gru = nn.GRU(
            input_size=config.gcn_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(batch)
        return apply_gru_head(model=self, fused=projected)


ABLATION_SPECS: dict[str, AblationSpec] = {
    "baseline": AblationSpec(
        removed_module="none",
        factory=lambda config: STGNN.STGNNGRURegressor(config=config),
        has_graph_artifact=True,
    ),
    "no_adaptive_graph": AblationSpec(
        removed_module="adaptive_graph",
        factory=lambda config: NoAdaptiveGraphSTGNNGRURegressor(config=config),
        has_graph_artifact=True,
    ),
    "no_gcn": AblationSpec(
        removed_module="gcn_spatial_message_passing",
        factory=lambda config: NoGCNSTGNNGRURegressor(config=config),
        has_graph_artifact=False,
    ),
    "no_gru": AblationSpec(
        removed_module="gru_temporal_modeling",
        factory=lambda config: NoGRUSTGNNRegressor(config=config),
        has_graph_artifact=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run electricity STGNN ablation experiments.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small tensor-flow test for all ablation variants.",
    )
    return parser.parse_args()


def save_json(data: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def compute_array_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    targets = y_true.astype(np.float32, copy=False).reshape(-1)
    predictions = y_pred.astype(np.float32, copy=False).reshape(-1)
    row_count = int(targets.shape[0])
    errors = predictions - targets

    squared_error_sum = float(np.sum(errors * errors, dtype=np.float64))
    mse = squared_error_sum / row_count
    mae = float(np.mean(np.abs(errors), dtype=np.float64))
    rmse = float(np.sqrt(mse))

    y_sum = float(np.sum(targets, dtype=np.float64))
    y_squared_sum = float(np.sum(targets * targets, dtype=np.float64))
    total_sum_of_squares = y_squared_sum - (y_sum * y_sum / row_count)
    r2 = 0.0 if total_sum_of_squares <= 0.0 else 1.0 - squared_error_sum / total_sum_of_squares

    denominator = np.abs(targets) + np.abs(predictions)
    smape_ratio = np.divide(
        2.0 * np.abs(errors),
        denominator,
        out=np.zeros_like(targets, dtype=np.float32),
        where=denominator != 0,
    )

    clipped_true = np.clip(targets, a_min=0.0, a_max=None)
    clipped_pred = np.clip(predictions, a_min=0.0, a_max=None)
    log_errors = np.log1p(clipped_pred) - np.log1p(clipped_true)

    return {
        "mse": float(mse),
        "mae": mae,
        "r2": float(r2),
        "smape": float(np.mean(smape_ratio, dtype=np.float64) * 100.0),
        "rmse": rmse,
        "rmsle": float(np.sqrt(np.mean(log_errors * log_errors, dtype=np.float64))),
        "evaluated_row_count": row_count,
        "skipped_row_count": 0,
        "window_count": row_count,
    }


def run_smoke_test() -> dict[str, object]:
    STGNN.set_seed(STGNN.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_config = STGNN.STGNNModelConfig(
        input_size=4,
        node_count=3,
        gcn_dim=4,
        embed_dim=4,
        graph_top_k=1,
        gcn_layers=2,
        hidden_size=5,
        num_layers=1,
        dropout=0.0,
    )
    batch = torch.rand(2, 6, 3, 4)
    targets = torch.rand(2, 3)
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    records: list[dict[str, object]] = []

    for variant, spec in ABLATION_SPECS.items():
        STGNN.set_seed(STGNN.SEED)
        model = spec.factory(model_config)
        predictions = model(batch)
        if tuple(predictions.shape) != tuple(targets.shape):
            raise AssertionError(f"{variant} produced shape {tuple(predictions.shape)}, expected {tuple(targets.shape)}.")

        loss = STGNN.masked_mse_loss(predictions=predictions, targets=targets, target_mask=target_mask)
        loss.backward()

        test_metrics = compute_array_metrics(
            y_true=targets.detach().cpu().numpy(),
            y_pred=predictions.detach().cpu().numpy(),
        )
        final_score = float(test_metrics["rmsle"])
        record = {
            "variant": variant,
            "removed_module": spec.removed_module,
            "test_metrics": test_metrics,
            "test_rmsle": final_score,
            "final_score": final_score,
            "final_score_metric": FINAL_SCORE_METRIC,
        }
        if record["final_score"] != record["test_metrics"]["rmsle"]:
            raise AssertionError(f"{variant} final_score does not match test_metrics['rmsle'].")
        records.append(record)

    summary = {
        "metric_scale": METRIC_SCALE,
        "final_score_metric": FINAL_SCORE_METRIC,
        "variant_count": len(records),
        "records": records,
    }
    save_json(summary, SMOKE_SUMMARY_PATH)
    print(json.dumps(summary, indent=2))
    return summary


def prepare_electricity_data(training_config: object) -> PreparedData:
    train_df, valid_df, test_df, input_feature_cols = STGNN.load_preprocessed_splits()
    preprocessing_summary = STGNN.load_preprocessing_summary()
    feature_means, feature_stds = STGNN.fit_input_scaler(
        train_df=train_df,
        input_feature_cols=input_feature_cols,
    )
    node_ids = np.sort(train_df["building_id"].unique()).astype(np.int32)
    test_only_node_ids = sorted(set(test_df["building_id"].unique().tolist()) - set(node_ids.tolist()))

    train_data = STGNN.build_dense_split_data(
        name="train",
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    valid_data = STGNN.build_dense_split_data(
        name="valid",
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    test_data = STGNN.build_dense_split_data(
        name="test",
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )

    del train_df, valid_df, test_df
    gc.collect()

    train_window_starts = STGNN.build_dense_window_starts(
        split_data=train_data,
        window_size=training_config.window_size,
        stride=training_config.stride,
    )

    return PreparedData(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
        test_only_node_ids=[int(node_id) for node_id in test_only_node_ids],
        preprocessing_summary=preprocessing_summary,
        train_window_starts=train_window_starts,
    )


def train_ablation_model(
    variant: str,
    model_factory: Callable[[object], nn.Module],
    train_data: object,
    valid_data: object,
    train_window_starts: np.ndarray,
    input_feature_cols: list[str],
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    model_config: object,
    training_config: object,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, int, dict[str, float | int]]:
    train_dataset = STGNN.DenseSequenceWindowDataset(
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

    model = model_factory(model_config).to(device)
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
            desc=f"{variant} epoch {epoch}/{training_config.epochs}",
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
            loss = STGNN.masked_mse_loss(predictions=predictions, targets=batch_y, target_mask=batch_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            optimizer.step()

            train_loss_sum += float(loss.item()) * mask_count
            train_sample_count += mask_count

        if train_sample_count == 0:
            raise ValueError("The training windows did not contain any observed targets.")

        train_loss = train_loss_sum / train_sample_count
        validation_metrics = STGNN.rolling_evaluate_split(
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
            "validation_rmsle": float(validation_metrics["rmsle"]),
            "validation_evaluated_row_count": int(validation_metrics["evaluated_row_count"]),
            "validation_skipped_row_count": int(validation_metrics["skipped_row_count"]),
            "validation_unknown_node_row_count": int(validation_metrics["unknown_node_row_count"]),
        }
        history_records.append(history_record)
        print(json.dumps({"variant": variant, **history_record}, indent=2))

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


def save_graph_artifacts(model: nn.Module, node_ids: np.ndarray, variant_dir: Path) -> dict[str, str]:
    adjacency_path = variant_dir / "adjacency.npy"
    topk_edges_path = variant_dir / "topk_edges.csv"

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
    return {
        "adjacency_path": str(adjacency_path),
        "topk_edges_path": str(topk_edges_path),
    }


def save_variant_artifacts(
    variant: str,
    spec: AblationSpec,
    model: nn.Module,
    model_config: object,
    training_config: object,
    prepared_data: PreparedData,
    training_history: pd.DataFrame,
    best_epoch: int,
    best_validation_metrics: dict[str, float | int],
    validation_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
) -> dict[str, str]:
    variant_dir = OUTPUT_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    model_path = variant_dir / "model.pt"
    training_history_path = variant_dir / "training_history.csv"
    validation_metrics_path = variant_dir / "validation_metrics.json"
    test_metrics_path = variant_dir / "test_metrics.json"
    model_params_path = variant_dir / "model_params.json"

    graph_paths: dict[str, str] = {}
    if spec.has_graph_artifact:
        graph_paths = save_graph_artifacts(model=model, node_ids=prepared_data.node_ids, variant_dir=variant_dir)

    torch.save(
        {
            "variant": variant,
            "removed_module": spec.removed_module,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "node_ids": prepared_data.node_ids.astype(int).tolist(),
            "input_feature_cols": prepared_data.input_feature_cols,
            "input_feature_means": prepared_data.feature_means.tolist(),
            "input_feature_stds": prepared_data.feature_stds.tolist(),
            "target_column": STGNN.TARGET_COL,
            "metric_scale": METRIC_SCALE,
            "final_score_metric": FINAL_SCORE_METRIC,
        },
        model_path,
    )
    training_history.to_csv(training_history_path, index=False)
    save_json(validation_metrics, validation_metrics_path)
    save_json(test_metrics, test_metrics_path)

    model_params = {
        "variant": variant,
        "removed_module": spec.removed_module,
        "metric_scale": METRIC_SCALE,
        "final_score_metric": FINAL_SCORE_METRIC,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "best_epoch": int(best_epoch),
        "best_validation_rmse": float(best_validation_metrics["rmse"]),
        "best_validation_rmsle": float(best_validation_metrics["rmsle"]),
        "validation_rmse": float(validation_metrics["rmse"]),
        "validation_rmsle": float(validation_metrics["rmsle"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_rmsle": float(test_metrics["rmsle"]),
        "final_score": float(test_metrics["rmsle"]),
        "node_count": int(prepared_data.node_ids.shape[0]),
        "node_ids": prepared_data.node_ids.astype(int).tolist(),
        "test_only_node_ids": prepared_data.test_only_node_ids,
        "test_unknown_node_row_count": int(prepared_data.test_data.unknown_node_row_count),
        "feature_count": len(prepared_data.input_feature_cols),
        "input_feature_cols": prepared_data.input_feature_cols,
        "input_feature_means": dict(
            zip(prepared_data.input_feature_cols, [float(value) for value in prepared_data.feature_means])
        ),
        "input_feature_stds": dict(
            zip(prepared_data.input_feature_cols, [float(value) for value in prepared_data.feature_stds])
        ),
        "target_column": STGNN.TARGET_COL,
        "target_preprocess": prepared_data.preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": prepared_data.preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": prepared_data.preprocessing_summary.get("target_log1p_max"),
        "train_window_count": int(prepared_data.train_window_starts.shape[0]),
        "train_data_path": str(STGNN.TRAIN_DATA_PATH),
        "valid_data_path": str(STGNN.VALID_DATA_PATH),
        "test_data_path": str(STGNN.TEST_DATA_PATH),
        "preprocessing_summary_path": str(STGNN.PREPROCESSING_SUMMARY_PATH),
        "model_path": str(model_path),
        "training_history_path": str(training_history_path),
        "validation_metrics_path": str(validation_metrics_path),
        "test_metrics_path": str(test_metrics_path),
    }
    model_params.update(graph_paths)
    save_json(model_params, model_params_path)

    return {
        "model_path": str(model_path),
        "training_history_path": str(training_history_path),
        "validation_metrics_path": str(validation_metrics_path),
        "test_metrics_path": str(test_metrics_path),
        "model_params_path": str(model_params_path),
        **graph_paths,
    }


def build_summary_record(
    variant: str,
    spec: AblationSpec,
    best_epoch: int,
    validation_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
    baseline_score: float,
) -> dict[str, object]:
    final_score = float(test_metrics["rmsle"])
    return {
        "variant": variant,
        "removed_module": spec.removed_module,
        "best_epoch": int(best_epoch),
        "validation_rmse": float(validation_metrics["rmse"]),
        "validation_rmsle": float(validation_metrics["rmsle"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_rmsle": float(test_metrics["rmsle"]),
        "final_score": final_score,
        "delta_final_score_vs_baseline": float(final_score - baseline_score),
        "evaluated_row_count": int(test_metrics["evaluated_row_count"]),
        "skipped_row_count": int(test_metrics["skipped_row_count"]),
        "output_dir": str(OUTPUT_DIR / variant),
    }


def save_experiment_summary(records: list[dict[str, object]], prepared_data: PreparedData) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(SUMMARY_CSV_PATH, index=False)

    summary = {
        "metric_scale": METRIC_SCALE,
        "final_score_metric": FINAL_SCORE_METRIC,
        "source_stgnn_path": str(SOURCE_STGNN_PATH),
        "output_dir": str(OUTPUT_DIR),
        "train_data_path": str(STGNN.TRAIN_DATA_PATH),
        "valid_data_path": str(STGNN.VALID_DATA_PATH),
        "test_data_path": str(STGNN.TEST_DATA_PATH),
        "preprocessing_summary_path": str(STGNN.PREPROCESSING_SUMMARY_PATH),
        "target_preprocess": prepared_data.preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": prepared_data.preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": prepared_data.preprocessing_summary.get("target_log1p_max"),
        "node_count": int(prepared_data.node_ids.shape[0]),
        "train_window_count": int(prepared_data.train_window_starts.shape[0]),
        "records": records,
        "summary_csv_path": str(SUMMARY_CSV_PATH),
    }
    save_json(summary, SUMMARY_JSON_PATH)
    return summary


def run_ablation_experiment() -> dict[str, object]:
    STGNN.set_seed(STGNN.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_config = STGNN.TrainingConfig()
    prepared_data = prepare_electricity_data(training_config=training_config)
    model_config = STGNN.STGNNModelConfig(
        input_size=len(prepared_data.input_feature_cols),
        node_count=int(prepared_data.node_ids.shape[0]),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    STGNN.assert_training_ready(
        train_data=prepared_data.train_data,
        valid_data=prepared_data.valid_data,
        test_data=prepared_data.test_data,
        train_window_starts=prepared_data.train_window_starts,
        input_feature_cols=prepared_data.input_feature_cols,
        model_config=model_config,
        training_config=training_config,
    )

    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "node_count": int(prepared_data.node_ids.shape[0]),
                "train_window_count": int(prepared_data.train_window_starts.shape[0]),
                "input_feature_count": len(prepared_data.input_feature_cols),
                "variants": list(ABLATION_SPECS),
                "final_score_metric": FINAL_SCORE_METRIC,
            },
            indent=2,
        )
    )

    records: list[dict[str, object]] = []
    baseline_score = 0.0

    for variant, spec in ABLATION_SPECS.items():
        print(json.dumps({"starting_variant": variant, "removed_module": spec.removed_module}, indent=2))
        STGNN.set_seed(training_config.seed)
        model, training_history, best_epoch, best_validation_metrics = train_ablation_model(
            variant=variant,
            model_factory=spec.factory,
            train_data=prepared_data.train_data,
            valid_data=prepared_data.valid_data,
            train_window_starts=prepared_data.train_window_starts,
            input_feature_cols=prepared_data.input_feature_cols,
            feature_means=prepared_data.feature_means,
            feature_stds=prepared_data.feature_stds,
            model_config=model_config,
            training_config=training_config,
            device=device,
        )

        validation_metrics = STGNN.rolling_evaluate_split(
            model=model,
            split_data=prepared_data.valid_data,
            history_parts=[prepared_data.train_data],
            input_feature_cols=prepared_data.input_feature_cols,
            feature_means=prepared_data.feature_means,
            feature_stds=prepared_data.feature_stds,
            device=device,
            window_size=training_config.window_size,
            eval_batch_size=training_config.eval_batch_size,
        )
        test_metrics = STGNN.rolling_evaluate_split(
            model=model,
            split_data=prepared_data.test_data,
            history_parts=[prepared_data.train_data, prepared_data.valid_data],
            input_feature_cols=prepared_data.input_feature_cols,
            feature_means=prepared_data.feature_means,
            feature_stds=prepared_data.feature_stds,
            device=device,
            window_size=training_config.window_size,
            eval_batch_size=training_config.eval_batch_size,
        )

        if variant == "baseline":
            baseline_score = float(test_metrics["rmsle"])

        artifacts = save_variant_artifacts(
            variant=variant,
            spec=spec,
            model=model,
            model_config=model_config,
            training_config=training_config,
            prepared_data=prepared_data,
            training_history=training_history,
            best_epoch=best_epoch,
            best_validation_metrics=best_validation_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )
        record = build_summary_record(
            variant=variant,
            spec=spec,
            best_epoch=best_epoch,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            baseline_score=baseline_score,
        )
        records.append(record)
        print(json.dumps({"completed_variant": variant, **record, "artifacts": artifacts}, indent=2))

        del model, training_history
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = save_experiment_summary(records=records, prepared_data=prepared_data)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> dict[str, object]:
    args = parse_args()
    if args.smoke_test:
        return run_smoke_test()
    return run_ablation_experiment()


if __name__ == "__main__":
    main()
