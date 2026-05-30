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
SOURCE_OTHER_METERS_PATH = WORKSPACE_ROOT / "STGNN" / "static_adaptive_stgnn_other_meters_auto-rolling.py"
SOURCE_GRAPH_PATH = WORKSPACE_ROOT / "STGNN" / "static_adaptive_graph.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "static_adaptive_other_meters_ablation_outputs"
SUMMARY_CSV_PATH = OUTPUT_DIR / "static_adaptive_other_meters_ablation_summary.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "static_adaptive_other_meters_ablation_summary.json"
SMOKE_SUMMARY_PATH = OUTPUT_DIR / "smoke_summary.json"
FINAL_SCORE_METRIC = "test_rmsle"
METRIC_SCALE = "log1p_minmax"


def load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OTHER = load_module("static_adaptive_other_meters_source", SOURCE_OTHER_METERS_PATH)
GRAPH = load_module("static_adaptive_graph_ablation_source", SOURCE_GRAPH_PATH)


@dataclass(frozen=True)
class AblationSpec:
    removed_module: str
    factory: Callable[[object, np.ndarray], nn.Module]
    loss_kind: str
    save_static_graph: bool = True
    save_learned_graph: bool = True
    save_prediction_weighted_graph: bool = True


@dataclass
class PreparedMeterData:
    meter_config: object
    paths: object
    train_data: object
    valid_data: object
    test_data: object
    input_feature_cols: list[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    node_ids: np.ndarray
    test_only_node_ids: list[int]
    preprocessing_summary: dict[str, object]
    split_summary: pd.DataFrame
    static_adjacency: np.ndarray
    train_window_starts: np.ndarray


class AblationZeroInflatedBranch(nn.Module):
    def __init__(
        self,
        config: object,
        use_gcn: bool = True,
        use_layer_attention: bool = True,
        use_gru: bool = True,
        zero_inflated: bool = True,
    ) -> None:
        super().__init__()
        self.use_gcn = use_gcn
        self.use_layer_attention = use_layer_attention
        self.use_gru = use_gru
        self.zero_inflated = zero_inflated

        self.input_projection = nn.Linear(config.input_size, config.gcn_dim)
        if use_gcn:
            self.gcn_layers = nn.ModuleList(
                nn.Linear(config.gcn_dim, config.gcn_dim) for _ in range(config.gcn_layers)
            )
            if use_layer_attention:
                self.layer_attention = nn.Parameter(torch.zeros(config.gcn_layers, dtype=torch.float32))
            self.dropout = nn.Dropout(config.dropout)

        encoded_dim = config.hidden_size if use_gru else config.gcn_dim
        if use_gru:
            self.gru = nn.GRU(
                input_size=config.gcn_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
        if zero_inflated:
            self.zero_head = nn.Linear(encoded_dim, 1)
        self.value_head = nn.Linear(encoded_dim, 1)

    def encode_spatial_temporal(self, batch: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        hidden = self.input_projection(batch)

        if self.use_gcn:
            layer_outputs: list[torch.Tensor] = []
            for layer in self.gcn_layers:
                aggregated = torch.einsum("ij,btjf->btif", adjacency, hidden)
                update = torch.relu(layer(aggregated))
                hidden = hidden + self.dropout(update)
                layer_outputs.append(hidden)

            if self.use_layer_attention:
                stacked_outputs = torch.stack(layer_outputs, dim=0)
                attention_weights = torch.softmax(self.layer_attention, dim=0).view(-1, 1, 1, 1, 1)
                hidden = torch.sum(attention_weights * stacked_outputs, dim=0)
            else:
                hidden = layer_outputs[-1]

        batch_size, sequence_length, node_count, feature_count = hidden.shape
        if not self.use_gru:
            return hidden[:, -1, :, :].reshape(batch_size * node_count, feature_count), batch_size, node_count

        gru_input = hidden.permute(0, 2, 1, 3).contiguous().view(
            batch_size * node_count,
            sequence_length,
            feature_count,
        )
        _, hidden_state = self.gru(gru_input)
        return hidden_state[-1], batch_size, node_count

    def forward(self, batch: torch.Tensor, adjacency: torch.Tensor) -> object:
        encoded, batch_size, node_count = self.encode_spatial_temporal(batch=batch, adjacency=adjacency)
        values = nn.functional.softplus(self.value_head(encoded)).view(batch_size, node_count)
        if self.zero_inflated:
            zero_logits = self.zero_head(encoded).view(batch_size, node_count)
            zero_probabilities = torch.sigmoid(zero_logits)
            predictions = (1.0 - zero_probabilities) * values
        else:
            zero_logits = values.new_full(values.shape, -20.0)
            zero_probabilities = torch.sigmoid(zero_logits)
            predictions = values

        return GRAPH.ZeroInflatedSTGNNOutput(
            predictions=predictions,
            zero_logits=zero_logits,
            zero_probabilities=zero_probabilities,
            values=values,
        )


class StaticAdaptiveAblationModel(nn.Module):
    def __init__(
        self,
        config: object,
        static_adjacency: np.ndarray,
        branch_mode: str = "both",
        fixed_fusion_alpha: float | None = None,
        use_gcn: bool = True,
        use_layer_attention: bool = True,
        use_gru: bool = True,
        zero_inflated: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.branch_mode = branch_mode
        self.fixed_fusion_alpha = fixed_fusion_alpha
        self.graph_constructor = GRAPH.StaticAdaptiveGraphConstructor(
            node_count=config.node_count,
            embed_dim=config.embed_dim,
            top_k=config.graph_top_k,
            static_adjacency=static_adjacency,
            graph_mode="static_adaptive",
        )
        if branch_mode in {"static", "both"}:
            self.static_branch = AblationZeroInflatedBranch(
                config=config,
                use_gcn=use_gcn,
                use_layer_attention=use_layer_attention,
                use_gru=use_gru,
                zero_inflated=zero_inflated,
            )
        if branch_mode in {"learned", "both"}:
            self.learned_branch = AblationZeroInflatedBranch(
                config=config,
                use_gcn=use_gcn,
                use_layer_attention=use_layer_attention,
                use_gru=use_gru,
                zero_inflated=zero_inflated,
            )
        if branch_mode == "both" and fixed_fusion_alpha is None:
            self.prediction_fusion_logits = nn.Parameter(torch.full((1,), 2.0, dtype=torch.float32))

    def forward(self, batch: torch.Tensor) -> object:
        if self.branch_mode == "static":
            static_output = self.static_branch(batch, self.graph_constructor.static_adjacency)
            alpha = torch.zeros((1, 1), dtype=static_output.predictions.dtype, device=static_output.predictions.device)
            return GRAPH.ZeroInflatedSTGNNOutput(
                predictions=static_output.predictions,
                zero_logits=static_output.zero_logits,
                zero_probabilities=static_output.zero_probabilities,
                values=static_output.values,
                static_predictions=static_output.predictions,
                learned_predictions=None,
                fusion_alpha=alpha,
            )

        if self.branch_mode == "learned":
            learned_output = self.learned_branch(batch, self.graph_constructor.learned_adjacency())
            alpha = torch.ones((1, 1), dtype=learned_output.predictions.dtype, device=learned_output.predictions.device)
            return GRAPH.ZeroInflatedSTGNNOutput(
                predictions=learned_output.predictions,
                zero_logits=learned_output.zero_logits,
                zero_probabilities=learned_output.zero_probabilities,
                values=learned_output.values,
                static_predictions=None,
                learned_predictions=learned_output.predictions,
                fusion_alpha=alpha,
            )

        static_output = self.static_branch(batch, self.graph_constructor.static_adjacency)
        learned_output = self.learned_branch(batch, self.graph_constructor.learned_adjacency())
        if self.fixed_fusion_alpha is None:
            alpha = torch.sigmoid(self.prediction_fusion_logits).view(1, 1)
        else:
            alpha = static_output.predictions.new_full((1, 1), float(self.fixed_fusion_alpha))

        predictions = alpha * learned_output.predictions + (1.0 - alpha) * static_output.predictions
        zero_logits = alpha * learned_output.zero_logits + (1.0 - alpha) * static_output.zero_logits
        values = alpha * learned_output.values + (1.0 - alpha) * static_output.values
        zero_probabilities = torch.sigmoid(zero_logits)
        return GRAPH.ZeroInflatedSTGNNOutput(
            predictions=predictions,
            zero_logits=zero_logits,
            zero_probabilities=zero_probabilities,
            values=values,
            static_predictions=static_output.predictions,
            learned_predictions=learned_output.predictions,
            fusion_alpha=alpha,
        )

    def get_static_adjacency(self) -> torch.Tensor:
        return self.graph_constructor.static_adjacency

    def get_learned_adjacency(self) -> torch.Tensor:
        return self.graph_constructor.learned_adjacency()

    def get_adjacency(self) -> torch.Tensor:
        static_adjacency = self.get_static_adjacency()
        learned_adjacency = self.get_learned_adjacency()
        if self.branch_mode == "static":
            return static_adjacency
        if self.branch_mode == "learned":
            return learned_adjacency
        alpha = torch.as_tensor(
            self.get_prediction_fusion_alpha(),
            dtype=static_adjacency.dtype,
            device=static_adjacency.device,
        ).view(1, 1)
        return alpha * learned_adjacency + (1.0 - alpha) * static_adjacency

    def get_prediction_fusion_alpha(self) -> np.ndarray:
        if self.branch_mode == "static":
            return np.array([0.0], dtype=np.float32)
        if self.branch_mode == "learned":
            return np.array([1.0], dtype=np.float32)
        if self.fixed_fusion_alpha is not None:
            return np.array([float(self.fixed_fusion_alpha)], dtype=np.float32)
        return torch.sigmoid(self.prediction_fusion_logits).detach().cpu().numpy().astype(np.float32)


ABLATION_SPECS: dict[str, AblationSpec] = {
    "baseline": AblationSpec(
        removed_module="none",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            branch_mode="both",
        ),
        loss_kind="zero_inflated",
        save_static_graph=True,
        save_learned_graph=True,
        save_prediction_weighted_graph=True,
    ),
    "no_static_graph": AblationSpec(
        removed_module="static_graph_branch",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            branch_mode="learned",
        ),
        loss_kind="zero_inflated",
        save_static_graph=False,
    ),
    "no_learned_graph": AblationSpec(
        removed_module="learned_graph_branch",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            branch_mode="static",
        ),
        loss_kind="zero_inflated",
        save_learned_graph=False,
    ),
    "no_graph_fusion": AblationSpec(
        removed_module="learnable_prediction_fusion",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            branch_mode="both",
            fixed_fusion_alpha=0.5,
        ),
        loss_kind="zero_inflated",
    ),
    "no_gcn": AblationSpec(
        removed_module="gcn_spatial_message_passing",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            use_gcn=False,
        ),
        loss_kind="zero_inflated",
    ),
    "no_layer_attention": AblationSpec(
        removed_module="layer_attention_fusion",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            use_layer_attention=False,
        ),
        loss_kind="zero_inflated",
    ),
    "no_gru": AblationSpec(
        removed_module="gru_temporal_modeling",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            use_gru=False,
        ),
        loss_kind="zero_inflated",
    ),
    "no_zero_inflation": AblationSpec(
        removed_module="zero_classifier_gate_and_zero_specific_loss",
        factory=lambda config, static_adjacency: StaticAdaptiveAblationModel(
            config=config,
            static_adjacency=static_adjacency,
            zero_inflated=False,
        ),
        loss_kind="regression",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run other-meter static-adaptive STGNN ablation experiments.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small tensor-flow test for all ablation variants.",
    )
    parser.add_argument(
        "--meter-id",
        type=int,
        choices=tuple(OTHER.METER_CONFIGS),
        default=None,
        help="Run only one non-electricity meter. Defaults to meter 1, 2, and 3.",
    )
    return parser.parse_args()


def save_json(data: dict[str, object] | list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def regression_masked_loss(predictions: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    mask = target_mask.bool()
    return nn.functional.smooth_l1_loss(predictions[mask], targets[mask], beta=0.5)


def compute_loss_parts(
    spec: AblationSpec,
    output: object,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    training_config: object,
    zero_pos_weight: float,
) -> dict[str, torch.Tensor | int]:
    if spec.loss_kind == "regression":
        loss = regression_masked_loss(
            predictions=output.predictions,
            targets=targets,
            target_mask=target_mask,
        )
        zero = loss * 0.0
        return {
            "loss": loss,
            "zero_classification_loss": zero,
            "positive_regression_loss": loss,
            "zero_suppression_loss": zero,
            "observed_count": int(target_mask.sum().item()),
            "zero_count": int(((targets <= 0.0) & target_mask.bool()).sum().item()),
            "positive_count": int(((targets > 0.0) & target_mask.bool()).sum().item()),
        }

    return OTHER.zero_inflated_masked_loss(
        predictions=output.predictions,
        zero_logits=output.zero_logits,
        targets=targets,
        target_mask=target_mask,
        zero_pos_weight=zero_pos_weight,
        zero_classification_weight=training_config.zero_classification_weight,
        positive_regression_weight=training_config.positive_regression_weight,
        zero_suppression_weight=training_config.zero_suppression_weight,
    )


def prepare_meter_data(config: object, training_config: object) -> PreparedMeterData:
    paths = OTHER.build_meter_paths(
        config=config,
        output_root_dir=OUTPUT_DIR,
    )
    train_df, valid_df, test_df, input_feature_cols = OTHER.BASE.load_preprocessed_splits(
        train_path=paths.train_data_path,
        valid_path=paths.valid_data_path,
        test_path=paths.test_data_path,
    )
    preprocessing_summary = OTHER.BASE.load_preprocessing_summary(paths.preprocessing_summary_path)
    split_summary = OTHER.BASE.make_split_summary(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        preprocessing_summary=preprocessing_summary,
    )
    feature_means, feature_stds = OTHER.BASE.fit_input_scaler(
        train_df=train_df,
        input_feature_cols=input_feature_cols,
    )
    node_ids = np.sort(train_df["building_id"].unique()).astype(np.int32)
    test_only_node_ids = sorted(set(test_df["building_id"].unique().tolist()) - set(node_ids.tolist()))
    static_adjacency = OTHER.build_static_adjacency(
        train_df=train_df,
        node_ids=node_ids,
        config=OTHER.StaticGraphConfig(top_k=OTHER.GRAPH_TOP_K),
    )

    train_data = OTHER.BASE.build_dense_split_data(
        name=f"meter_{config.meter_id}_train",
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    valid_data = OTHER.BASE.build_dense_split_data(
        name=f"meter_{config.meter_id}_valid",
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    test_data = OTHER.BASE.build_dense_split_data(
        name=f"meter_{config.meter_id}_test",
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )

    del train_df, valid_df, test_df
    gc.collect()

    train_window_starts = OTHER.BASE.build_dense_window_starts(
        split_data=train_data,
        window_size=training_config.window_size,
        stride=training_config.stride,
    )

    return PreparedMeterData(
        meter_config=config,
        paths=paths,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
        test_only_node_ids=[int(node_id) for node_id in test_only_node_ids],
        preprocessing_summary=preprocessing_summary,
        split_summary=split_summary,
        static_adjacency=static_adjacency,
        train_window_starts=train_window_starts,
    )


def train_ablation_model(
    variant: str,
    spec: AblationSpec,
    prepared_data: PreparedMeterData,
    model_config: object,
    training_config: object,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, int, dict[str, float | int], float]:
    train_dataset = OTHER.BASE.DenseSequenceWindowDataset(
        input_features=prepared_data.train_data.input_features,
        targets=prepared_data.train_data.targets,
        target_mask=prepared_data.train_data.target_mask,
        window_starts=prepared_data.train_window_starts,
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

    zero_pos_weight = OTHER.compute_zero_pos_weight(prepared_data.train_data)
    model = spec.factory(model_config, prepared_data.static_adjacency).to(device)
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
            desc=f"{variant} meter {prepared_data.meter_config.meter_id} epoch {epoch}/{training_config.epochs}",
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
            loss_parts = compute_loss_parts(
                spec=spec,
                output=output,
                targets=batch_y,
                target_mask=batch_mask,
                training_config=training_config,
                zero_pos_weight=zero_pos_weight,
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

        validation_metrics = OTHER.rolling_evaluate_split(
            model=model,
            split_data=prepared_data.valid_data,
            history_parts=[prepared_data.train_data],
            input_feature_cols=prepared_data.input_feature_cols,
            feature_means=prepared_data.feature_means,
            feature_stds=prepared_data.feature_stds,
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
            "validation_rmsle": float(validation_metrics["rmsle"]),
            "validation_thresholded_rmsle": float(validation_metrics["thresholded_rmsle"]),
            "validation_zero_f1": float(validation_metrics["zero_f1"]),
            "validation_mean_prediction_on_zero": float(validation_metrics["mean_prediction_on_zero"]),
            "validation_evaluated_row_count": int(validation_metrics["evaluated_row_count"]),
            "validation_skipped_row_count": int(validation_metrics["skipped_row_count"]),
            "validation_unknown_node_row_count": int(validation_metrics["unknown_node_row_count"]),
        }
        history_records.append(history_record)
        print(
            json.dumps(
                {
                    "variant": variant,
                    "meter_id": prepared_data.meter_config.meter_id,
                    **history_record,
                },
                indent=2,
            )
        )

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


def save_adjacency_artifact(adjacency: np.ndarray, node_ids: np.ndarray, npy_path: Path, csv_path: Path) -> None:
    np.save(npy_path, adjacency.astype(np.float32))
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
    edges.to_csv(csv_path, index=False)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def save_graph_artifacts(
    model: nn.Module,
    spec: AblationSpec,
    node_ids: np.ndarray,
    variant_dir: Path,
) -> dict[str, str]:
    graph_paths: dict[str, str] = {}
    model.eval()
    with torch.no_grad():
        if spec.save_static_graph:
            static_adjacency = tensor_to_numpy(model.get_static_adjacency())
            static_npy = variant_dir / "static_adjacency.npy"
            static_csv = variant_dir / "static_adjacency.csv"
            save_adjacency_artifact(
                adjacency=static_adjacency,
                node_ids=node_ids,
                npy_path=static_npy,
                csv_path=static_csv,
            )
            graph_paths["static_adjacency_path"] = str(static_npy)
            graph_paths["static_adjacency_csv_path"] = str(static_csv)

        if spec.save_learned_graph:
            learned_adjacency = tensor_to_numpy(model.get_learned_adjacency())
            learned_npy = variant_dir / "learned_adjacency.npy"
            learned_csv = variant_dir / "learned_adjacency.csv"
            save_adjacency_artifact(
                adjacency=learned_adjacency,
                node_ids=node_ids,
                npy_path=learned_npy,
                csv_path=learned_csv,
            )
            graph_paths["learned_adjacency_path"] = str(learned_npy)
            graph_paths["learned_adjacency_csv_path"] = str(learned_csv)

        if spec.save_prediction_weighted_graph:
            weighted_adjacency = tensor_to_numpy(model.get_adjacency())
            weighted_npy = variant_dir / "prediction_weighted_adjacency.npy"
            weighted_csv = variant_dir / "prediction_weighted_adjacency.csv"
            save_adjacency_artifact(
                adjacency=weighted_adjacency,
                node_ids=node_ids,
                npy_path=weighted_npy,
                csv_path=weighted_csv,
            )
            graph_paths["prediction_weighted_adjacency_path"] = str(weighted_npy)
            graph_paths["prediction_weighted_adjacency_csv_path"] = str(weighted_csv)

    return graph_paths


def save_variant_artifacts(
    variant: str,
    spec: AblationSpec,
    model: nn.Module,
    model_config: object,
    training_config: object,
    prepared_data: PreparedMeterData,
    training_history: pd.DataFrame,
    best_epoch: int,
    best_validation_metrics: dict[str, float | int],
    validation_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
    zero_pos_weight: float,
) -> dict[str, str]:
    variant_dir = OUTPUT_DIR / variant / f"meter_{prepared_data.meter_config.meter_id}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    model_path = variant_dir / "model.pt"
    training_history_path = variant_dir / "training_history.csv"
    validation_metrics_path = variant_dir / "validation_metrics.json"
    test_metrics_path = variant_dir / "test_metrics.json"
    model_params_path = variant_dir / "model_params.json"
    run_summary_path = variant_dir / "run_summary.json"

    graph_paths = save_graph_artifacts(
        model=model,
        spec=spec,
        node_ids=prepared_data.node_ids,
        variant_dir=variant_dir,
    )
    prediction_fusion_alpha = model.get_prediction_fusion_alpha().astype(float).tolist()

    torch.save(
        {
            "variant": variant,
            "removed_module": spec.removed_module,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "node_ids": prepared_data.node_ids.astype(int).tolist(),
            "input_feature_cols": prepared_data.input_feature_cols,
            "feature_means": prepared_data.feature_means.astype(np.float32),
            "feature_stds": prepared_data.feature_stds.astype(np.float32),
            "static_adjacency": prepared_data.static_adjacency.astype(np.float32),
            "zero_pos_weight": float(zero_pos_weight),
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
        "meter_id": int(prepared_data.meter_config.meter_id),
        "meter_name": prepared_data.meter_config.meter_name,
        "metric_scale": METRIC_SCALE,
        "final_score_metric": FINAL_SCORE_METRIC,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "zero_pos_weight": float(zero_pos_weight),
        "prediction_fusion_alpha": prediction_fusion_alpha,
        "best_epoch": int(best_epoch),
        "best_validation_rmse": float(best_validation_metrics["rmse"]),
        "best_validation_rmsle": float(best_validation_metrics["rmsle"]),
        "best_validation_zero_f1": float(best_validation_metrics["zero_f1"]),
        "validation_rmse": float(validation_metrics["rmse"]),
        "validation_rmsle": float(validation_metrics["rmsle"]),
        "validation_thresholded_rmsle": float(validation_metrics["thresholded_rmsle"]),
        "validation_zero_f1": float(validation_metrics["zero_f1"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_rmsle": float(test_metrics["rmsle"]),
        "test_thresholded_rmsle": float(test_metrics["thresholded_rmsle"]),
        "test_zero_f1": float(test_metrics["zero_f1"]),
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
        "target_column": OTHER.TARGET_COL,
        "target_preprocess": prepared_data.preprocessing_summary.get("target_preprocess", "log1p_then_train_minmax"),
        "target_log1p_min": prepared_data.preprocessing_summary.get("target_log1p_min"),
        "target_log1p_max": prepared_data.preprocessing_summary.get("target_log1p_max"),
        "train_window_count": int(prepared_data.train_window_starts.shape[0]),
        "train_data_path": str(prepared_data.paths.train_data_path),
        "valid_data_path": str(prepared_data.paths.valid_data_path),
        "test_data_path": str(prepared_data.paths.test_data_path),
        "preprocessing_summary_path": str(prepared_data.paths.preprocessing_summary_path),
        "model_path": str(model_path),
        "training_history_path": str(training_history_path),
        "validation_metrics_path": str(validation_metrics_path),
        "test_metrics_path": str(test_metrics_path),
        "run_summary_path": str(run_summary_path),
    }
    model_params.update(graph_paths)
    save_json(model_params, model_params_path)

    run_summary = {
        **model_params,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "train_raw_row_count": int(prepared_data.train_data.raw_row_count),
        "valid_raw_row_count": int(prepared_data.valid_data.raw_row_count),
        "test_raw_row_count": int(prepared_data.test_data.raw_row_count),
        "train_included_row_count": int(prepared_data.train_data.included_row_count),
        "valid_included_row_count": int(prepared_data.valid_data.included_row_count),
        "test_included_row_count": int(prepared_data.test_data.included_row_count),
        "splits": prepared_data.split_summary.to_dict(orient="records"),
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_json(run_summary, run_summary_path)

    return {
        "model_path": str(model_path),
        "training_history_path": str(training_history_path),
        "validation_metrics_path": str(validation_metrics_path),
        "test_metrics_path": str(test_metrics_path),
        "model_params_path": str(model_params_path),
        "run_summary_path": str(run_summary_path),
        **graph_paths,
    }


def build_summary_record(
    variant: str,
    spec: AblationSpec,
    prepared_data: PreparedMeterData,
    best_epoch: int,
    validation_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
) -> dict[str, object]:
    final_score = float(test_metrics["rmsle"])
    return {
        "variant": variant,
        "removed_module": spec.removed_module,
        "meter_id": int(prepared_data.meter_config.meter_id),
        "meter_name": prepared_data.meter_config.meter_name,
        "best_epoch": int(best_epoch),
        "validation_rmse": float(validation_metrics["rmse"]),
        "validation_rmsle": float(validation_metrics["rmsle"]),
        "validation_zero_f1": float(validation_metrics["zero_f1"]),
        "test_mse": float(test_metrics["mse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_r2": float(test_metrics["r2"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_rmsle": float(test_metrics["rmsle"]),
        "test_thresholded_rmsle": float(test_metrics["thresholded_rmsle"]),
        "test_zero_f1": float(test_metrics["zero_f1"]),
        "final_score": final_score,
        "evaluated_row_count": int(test_metrics["evaluated_row_count"]),
        "skipped_row_count": int(test_metrics["skipped_row_count"]),
        "output_dir": str(OUTPUT_DIR / variant / f"meter_{prepared_data.meter_config.meter_id}"),
    }


def build_macro_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_df = pd.DataFrame(records)
    macro_records = []
    for (variant, removed_module), group in summary_df.groupby(["variant", "removed_module"]):
        n = group["evaluated_row_count"]
        N = n.sum()

        test_mse = (group["test_mse"] * n).sum() / N
        test_mae = (group["test_mae"] * n).sum() / N

        test_rmse = np.sqrt((group["test_rmse"]**2 * n).sum() / N)
        test_rmsle = np.sqrt((group["test_rmsle"]**2 * n).sum() / N)

        sse = group["test_mse"] * n
        sst = np.where(group["test_r2"] < 1.0, sse / (1 - group["test_r2"]), 0.0)
        sst_total = sst.sum()
        test_r2 = 1.0 - (sse.sum() / sst_total) if sst_total != 0 else 1.0

        macro_records.append({
            "variant": variant,
            "removed_module": removed_module,
            "macro_test_mse": float(test_mse),
            "macro_test_mae": float(test_mae),
            "macro_test_r2": float(test_r2),
            "macro_test_rmse": float(test_rmse),
            "macro_test_rmsle": float(test_rmsle),
            "macro_final_score": float(test_rmsle),
            "total_evaluated_row_count": int(N),
        })
    return macro_records


def save_experiment_summary(records: list[dict[str, object]]) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(SUMMARY_CSV_PATH, index=False)
    macro_records = build_macro_records(records=records)
    summary = {
        "metric_scale": METRIC_SCALE,
        "final_score_metric": FINAL_SCORE_METRIC,
        "source_other_meters_path": str(SOURCE_OTHER_METERS_PATH),
        "source_graph_path": str(SOURCE_GRAPH_PATH),
        "output_dir": str(OUTPUT_DIR),
        "records": records,
        "macro_summary": macro_records,
        "summary_csv_path": str(SUMMARY_CSV_PATH),
    }
    save_json(summary, SUMMARY_JSON_PATH)
    return summary


def run_smoke_test() -> dict[str, object]:
    OTHER.set_seed(OTHER.SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, input_feature_cols = OTHER.make_zero_inflated_smoke_frames()
    feature_means, feature_stds = OTHER.BASE.fit_input_scaler(train_df=train_df, input_feature_cols=input_feature_cols)
    node_ids = np.sort(train_df["building_id"].unique()).astype(np.int32)
    static_adjacency = OTHER.build_static_adjacency(
        train_df=train_df,
        node_ids=node_ids,
        config=OTHER.StaticGraphConfig(top_k=2),
    )
    train_data = OTHER.BASE.build_dense_split_data(
        name="ablation_zero_smoke_train",
        split_df=train_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    valid_data = OTHER.BASE.build_dense_split_data(
        name="ablation_zero_smoke_valid",
        split_df=valid_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )
    test_data = OTHER.BASE.build_dense_split_data(
        name="ablation_zero_smoke_test",
        split_df=test_df,
        input_feature_cols=input_feature_cols,
        feature_means=feature_means,
        feature_stds=feature_stds,
        node_ids=node_ids,
    )

    training_config = OTHER.ZeroInflatedTrainingConfig(
        window_size=4,
        batch_size=2,
        eval_batch_size=1,
        epochs=1,
        patience=1,
    )
    model_config = OTHER.STGNNModelConfig(
        input_size=len(input_feature_cols),
        node_count=int(node_ids.shape[0]),
        gcn_dim=4,
        embed_dim=4,
        graph_top_k=2,
        gcn_layers=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        graph_mode=OTHER.GRAPH_MODE,
    )
    train_window_starts = OTHER.BASE.build_dense_window_starts(
        split_data=train_data,
        window_size=training_config.window_size,
        stride=training_config.stride,
    )
    dataset = OTHER.BASE.DenseSequenceWindowDataset(
        input_features=train_data.input_features,
        targets=train_data.targets,
        target_mask=train_data.target_mask,
        window_starts=train_window_starts,
        window_size=training_config.window_size,
    )
    sample_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_x, batch_y, batch_mask = next(iter(sample_loader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, object]] = []
    zero_pos_weight = OTHER.compute_zero_pos_weight(train_data)
    for variant, spec in ABLATION_SPECS.items():
        OTHER.set_seed(OTHER.SEED)
        model = spec.factory(model_config, static_adjacency).to(device)
        output = model(batch_x.to(device=device).float())
        if tuple(output.predictions.shape) != (2, int(node_ids.shape[0])):
            raise AssertionError(f"{variant} produced unexpected prediction shape: {tuple(output.predictions.shape)}")

        loss_parts = compute_loss_parts(
            spec=spec,
            output=output,
            targets=batch_y.to(device=device).float(),
            target_mask=batch_mask.to(device=device).bool(),
            training_config=training_config,
            zero_pos_weight=zero_pos_weight,
        )
        loss_parts["loss"].backward()

        valid_metrics = OTHER.rolling_evaluate_split(
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
        test_metrics = OTHER.rolling_evaluate_split(
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
        required_metric_keys = {"rmsle", "thresholded_rmsle", "zero_f1"}
        if not required_metric_keys.issubset(test_metrics):
            raise AssertionError(f"{variant} rolling metrics are missing required keys.")

        final_score = float(test_metrics["rmsle"])
        record = {
            "variant": variant,
            "removed_module": spec.removed_module,
            "prediction_shape": list(output.predictions.shape),
            "loss": float(loss_parts["loss"].item()),
            "valid_rmsle": float(valid_metrics["rmsle"]),
            "test_rmsle": final_score,
            "test_thresholded_rmsle": float(test_metrics["thresholded_rmsle"]),
            "test_zero_f1": float(test_metrics["zero_f1"]),
            "final_score": final_score,
            "final_score_metric": FINAL_SCORE_METRIC,
        }
        if record["final_score"] != record["test_rmsle"]:
            raise AssertionError(f"{variant} final_score does not match test_rmsle.")
        records.append(record)

        del model
        gc.collect()

    summary = {
        "status": "passed",
        "metric_scale": METRIC_SCALE,
        "final_score_metric": FINAL_SCORE_METRIC,
        "device": str(device),
        "batch_shape": list(batch_x.shape),
        "variant_count": len(records),
        "records": records,
    }
    save_json(summary, SMOKE_SUMMARY_PATH)
    print(json.dumps(summary, indent=2))
    return summary


def run_ablation_experiment(meter_id: int | None = None) -> dict[str, object]:
    training_config = OTHER.ZeroInflatedTrainingConfig()
    meter_configs = [OTHER.METER_CONFIGS[meter_id]] if meter_id is not None else list(OTHER.METER_CONFIGS.values())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_records: list[dict[str, object]] = []

    for meter_config in meter_configs:
        OTHER.set_seed(training_config.seed)
        prepared_data = prepare_meter_data(config=meter_config, training_config=training_config)
        model_config = OTHER.STGNNModelConfig(
            input_size=len(prepared_data.input_feature_cols),
            node_count=int(prepared_data.node_ids.shape[0]),
            graph_mode=OTHER.GRAPH_MODE,
        )
        OTHER.BASE.assert_training_ready(
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
                    "meter_id": meter_config.meter_id,
                    "meter_name": meter_config.meter_name,
                    "node_count": int(prepared_data.node_ids.shape[0]),
                    "train_window_count": int(prepared_data.train_window_starts.shape[0]),
                    "input_feature_count": len(prepared_data.input_feature_cols),
                    "variants": list(ABLATION_SPECS),
                    "final_score_metric": FINAL_SCORE_METRIC,
                },
                indent=2,
            )
        )

        for variant, spec in ABLATION_SPECS.items():
            print(
                json.dumps(
                    {
                        "starting_variant": variant,
                        "removed_module": spec.removed_module,
                        "meter_id": meter_config.meter_id,
                    },
                    indent=2,
                )
            )
            OTHER.set_seed(training_config.seed)
            model, training_history, best_epoch, best_validation_metrics, zero_pos_weight = train_ablation_model(
                variant=variant,
                spec=spec,
                prepared_data=prepared_data,
                model_config=model_config,
                training_config=training_config,
                device=device,
            )
            validation_metrics = OTHER.rolling_evaluate_split(
                model=model,
                split_data=prepared_data.valid_data,
                history_parts=[prepared_data.train_data],
                input_feature_cols=prepared_data.input_feature_cols,
                feature_means=prepared_data.feature_means,
                feature_stds=prepared_data.feature_stds,
                device=device,
                window_size=training_config.window_size,
                eval_batch_size=training_config.eval_batch_size,
                zero_threshold=training_config.zero_threshold,
            )
            test_metrics = OTHER.rolling_evaluate_split(
                model=model,
                split_data=prepared_data.test_data,
                history_parts=[prepared_data.train_data, prepared_data.valid_data],
                input_feature_cols=prepared_data.input_feature_cols,
                feature_means=prepared_data.feature_means,
                feature_stds=prepared_data.feature_stds,
                device=device,
                window_size=training_config.window_size,
                eval_batch_size=training_config.eval_batch_size,
                zero_threshold=training_config.zero_threshold,
            )



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
                zero_pos_weight=zero_pos_weight,
            )
            record = build_summary_record(
                variant=variant,
                spec=spec,
                prepared_data=prepared_data,
                best_epoch=best_epoch,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
            )
            all_records.append(record)
            print(json.dumps({"completed_variant": variant, **record, "artifacts": artifacts}, indent=2))

            del model, training_history
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del prepared_data
        gc.collect()

    summary = save_experiment_summary(records=all_records)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> dict[str, object]:
    args = parse_args()
    if args.smoke_test:
        return run_smoke_test()
    return run_ablation_experiment(meter_id=args.meter_id)


if __name__ == "__main__":
    main()
