from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn


GraphMode = Literal["static", "learned", "static_adaptive"]


@dataclass(frozen=True)
class StaticGraphConfig:
    top_k: int = 20
    site_weight: float = 0.35
    primary_use_weight: float = 0.30
    size_weight: float = 0.15
    age_weight: float = 0.10
    floor_weight: float = 0.10
    age_scale: float = 10.0
    floor_scale: float = 3.0


@dataclass(frozen=True)
class StaticAdaptiveSTGNNModelConfig:
    input_size: int
    node_count: int
    gcn_dim: int = 16
    embed_dim: int = 16
    graph_top_k: int = 20
    gcn_layers: int = 4
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    graph_mode: GraphMode = "static_adaptive"


@dataclass(frozen=True)
class PredictionLevelFusionOutput:
    predictions: torch.Tensor
    static_predictions: torch.Tensor | None
    learned_predictions: torch.Tensor | None
    fusion_alpha: torch.Tensor


@dataclass(frozen=True)
class ZeroInflatedSTGNNOutput:
    predictions: torch.Tensor
    zero_logits: torch.Tensor
    zero_probabilities: torch.Tensor
    values: torch.Tensor
    static_predictions: torch.Tensor | None = None
    learned_predictions: torch.Tensor | None = None
    fusion_alpha: torch.Tensor | None = None


def _first_per_building(train_df: pd.DataFrame, node_ids: np.ndarray) -> pd.DataFrame:
    candidate_columns = ["building_id", "site_id", "primary_use", "square_feet", "year_built", "floor_count", "age"]
    static_columns = [column for column in candidate_columns if column in train_df.columns]
    static_df = train_df[static_columns].drop_duplicates("building_id", keep="first").set_index("building_id")
    return static_df.reindex(node_ids.astype(int)).reset_index()


def _numeric_array(static_df: pd.DataFrame, column: str, default_value: float) -> np.ndarray:
    if column not in static_df.columns:
        return np.full(len(static_df), default_value, dtype=np.float32)
    values = pd.to_numeric(static_df[column], errors="coerce")
    if values.notna().any():
        fill_value = values.median()
    else:
        fill_value = default_value
    return values.fillna(fill_value).to_numpy(dtype=np.float32)


def _categorical_array(static_df: pd.DataFrame, column: str, default_value: object) -> np.ndarray:
    if column not in static_df.columns:
        return np.full(len(static_df), default_value, dtype=object)
    values = static_df[column].astype(object)
    return values.where(values.notna(), default_value).to_numpy(dtype=object)


def build_static_adjacency(
    train_df: pd.DataFrame,
    node_ids: np.ndarray,
    config: StaticGraphConfig,
) -> np.ndarray:
    static_df = _first_per_building(train_df=train_df, node_ids=node_ids)
    node_count = len(static_df)

    site_id = _categorical_array(static_df, "site_id", "__missing_site_id__")
    primary_use = _categorical_array(static_df, "primary_use", "__missing_primary_use__")
    square_feet = _numeric_array(static_df, "square_feet", 0.0)
    size = np.log1p(np.maximum(square_feet, 0.0))
    if "age" in static_df.columns:
        age = _numeric_array(static_df, "age", 0.0)
    else:
        age = 2016.0 - _numeric_array(static_df, "year_built", 2016.0)
    floor_count = _numeric_array(static_df, "floor_count", 0.0)

    site_similarity = (site_id[:, None] == site_id[None, :]).astype(np.float32)
    primary_use_similarity = (primary_use[:, None] == primary_use[None, :]).astype(np.float32)
    size_similarity = np.exp(-np.abs(size[:, None] - size[None, :])).astype(np.float32)
    age_similarity = np.exp(-np.abs(age[:, None] - age[None, :]) / config.age_scale).astype(np.float32)
    floor_similarity = np.exp(
        -np.abs(floor_count[:, None] - floor_count[None, :]) / config.floor_scale
    ).astype(np.float32)

    adjacency = (
        config.site_weight * site_similarity
        + config.primary_use_weight * primary_use_similarity
        + config.size_weight * size_similarity
        + config.age_weight * age_similarity
        + config.floor_weight * floor_similarity
    ).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    top_k = min(max(int(config.top_k), 0), max(node_count - 1, 0))
    if top_k == 0:
        adjacency = np.zeros_like(adjacency, dtype=np.float32)
    elif top_k < node_count - 1:
        topk_adjacency = np.zeros_like(adjacency, dtype=np.float32)
        for row_index in range(node_count):
            top_indices = np.argsort(adjacency[row_index])[-top_k:]
            topk_adjacency[row_index, top_indices] = adjacency[row_index, top_indices]
        adjacency = topk_adjacency

    np.fill_diagonal(adjacency, 1.0)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / row_sums
    return adjacency.astype(np.float32)


class StaticAdaptiveGraphConstructor(nn.Module):
    def __init__(
        self,
        node_count: int,
        embed_dim: int,
        top_k: int,
        static_adjacency: np.ndarray | torch.Tensor,
        graph_mode: GraphMode,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.top_k = top_k
        self.graph_mode = graph_mode
        self.node_embeddings = nn.Parameter(torch.empty(node_count, embed_dim))
        self.register_buffer("eye", torch.eye(node_count, dtype=torch.float32), persistent=False)

        static_tensor = torch.as_tensor(static_adjacency, dtype=torch.float32)
        static_tensor = static_tensor / static_tensor.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("static_adjacency", static_tensor)
        nn.init.xavier_uniform_(self.node_embeddings)

    def learned_adjacency(self) -> torch.Tensor:
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

    def forward(self) -> torch.Tensor:
        if self.graph_mode == "static":
            return self.static_adjacency
        return self.learned_adjacency()


def _encode_stgnn_last_hidden(
    batch: torch.Tensor,
    adjacency: torch.Tensor,
    input_projection: nn.Linear,
    gcn_layers: nn.ModuleList,
    layer_attention: nn.Parameter,
    dropout: nn.Dropout,
    gru: nn.GRU,
) -> tuple[torch.Tensor, int, int]:
    hidden = input_projection(batch)
    layer_outputs: list[torch.Tensor] = []

    for layer in gcn_layers:
        aggregated = torch.einsum("ij,btjf->btif", adjacency, hidden)
        update = torch.relu(layer(aggregated))
        hidden = hidden + dropout(update)
        layer_outputs.append(hidden)

    stacked_outputs = torch.stack(layer_outputs, dim=0)
    attention_weights = torch.softmax(layer_attention, dim=0).view(-1, 1, 1, 1, 1)
    attended = torch.sum(attention_weights * stacked_outputs, dim=0)

    batch_size, sequence_length, node_count, feature_count = attended.shape
    gru_input = attended.permute(0, 2, 1, 3).contiguous().view(
        batch_size * node_count,
        sequence_length,
        feature_count,
    )
    _, hidden_state = gru(gru_input)
    return hidden_state[-1], batch_size, node_count


class _STGNNRegressionBranch(nn.Module):
    def __init__(self, config: StaticAdaptiveSTGNNModelConfig) -> None:
        super().__init__()
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

    def forward(self, batch: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        last_hidden, batch_size, node_count = _encode_stgnn_last_hidden(
            batch=batch,
            adjacency=adjacency,
            input_projection=self.input_projection,
            gcn_layers=self.gcn_layers,
            layer_attention=self.layer_attention,
            dropout=self.dropout,
            gru=self.gru,
        )
        return self.head(last_hidden).view(batch_size, node_count)


class _STGNNZeroInflatedBranch(nn.Module):
    def __init__(self, config: StaticAdaptiveSTGNNModelConfig) -> None:
        super().__init__()
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
        self.zero_head = nn.Linear(config.hidden_size, 1)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, batch: torch.Tensor, adjacency: torch.Tensor) -> ZeroInflatedSTGNNOutput:
        last_hidden, batch_size, node_count = _encode_stgnn_last_hidden(
            batch=batch,
            adjacency=adjacency,
            input_projection=self.input_projection,
            gcn_layers=self.gcn_layers,
            layer_attention=self.layer_attention,
            dropout=self.dropout,
            gru=self.gru,
        )
        zero_logits = self.zero_head(last_hidden).view(batch_size, node_count)
        zero_probabilities = torch.sigmoid(zero_logits)
        values = nn.functional.softplus(self.value_head(last_hidden)).view(batch_size, node_count)
        predictions = (1.0 - zero_probabilities) * values
        return ZeroInflatedSTGNNOutput(
            predictions=predictions,
            zero_logits=zero_logits,
            zero_probabilities=zero_probabilities,
            values=values,
        )


class StaticAdaptiveSTGNNGRURegressor(nn.Module):
    def __init__(
        self,
        config: StaticAdaptiveSTGNNModelConfig,
        static_adjacency: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.graph_constructor = StaticAdaptiveGraphConstructor(
            node_count=config.node_count,
            embed_dim=config.embed_dim,
            top_k=config.graph_top_k,
            static_adjacency=static_adjacency,
            graph_mode=config.graph_mode,
        )
        self.static_branch = _STGNNRegressionBranch(config)
        self.learned_branch = _STGNNRegressionBranch(config)
        self.prediction_fusion_logits = nn.Parameter(torch.full((1,), 2.0, dtype=torch.float32))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.predict_branches(batch).predictions

    def _fusion_alpha_tensor(self) -> torch.Tensor:
        return torch.sigmoid(self.prediction_fusion_logits).view(1, 1)

    def predict_branches(self, batch: torch.Tensor) -> PredictionLevelFusionOutput:
        if self.config.graph_mode == "static":
            static_predictions = self.static_branch(batch, self.graph_constructor.static_adjacency)
            alpha = torch.zeros((1, 1), dtype=static_predictions.dtype, device=static_predictions.device)
            return PredictionLevelFusionOutput(
                predictions=static_predictions,
                static_predictions=static_predictions,
                learned_predictions=None,
                fusion_alpha=alpha,
            )
        if self.config.graph_mode == "learned":
            learned_predictions = self.learned_branch(batch, self.graph_constructor.learned_adjacency())
            alpha = torch.ones((1, 1), dtype=learned_predictions.dtype, device=learned_predictions.device)
            return PredictionLevelFusionOutput(
                predictions=learned_predictions,
                static_predictions=None,
                learned_predictions=learned_predictions,
                fusion_alpha=alpha,
            )

        static_predictions = self.static_branch(batch, self.graph_constructor.static_adjacency)
        learned_predictions = self.learned_branch(batch, self.graph_constructor.learned_adjacency())
        alpha = self._fusion_alpha_tensor()
        predictions = alpha * learned_predictions + (1.0 - alpha) * static_predictions

        return PredictionLevelFusionOutput(
            predictions=predictions,
            static_predictions=static_predictions,
            learned_predictions=learned_predictions,
            fusion_alpha=alpha,
        )

    def _to_numpy_float32(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def get_adjacency(self) -> torch.Tensor:
        return self.get_prediction_weighted_adjacency()

    def get_learned_adjacency(self) -> torch.Tensor:
        return self.graph_constructor.learned_adjacency()

    def get_static_adjacency(self) -> torch.Tensor:
        return self.graph_constructor.static_adjacency

    def get_prediction_weighted_adjacency(self) -> torch.Tensor:
        static_adjacency = self.get_static_adjacency()
        learned_adjacency = self.get_learned_adjacency()
        if self.config.graph_mode == "static":
            return static_adjacency
        if self.config.graph_mode == "learned":
            return learned_adjacency
        alpha = torch.as_tensor(
            self.get_prediction_fusion_alpha(),
            dtype=static_adjacency.dtype,
            device=static_adjacency.device,
        ).view(1, 1)
        return alpha * learned_adjacency + (1.0 - alpha) * static_adjacency

    def get_prediction_fusion_alpha(self) -> np.ndarray:
        if self.config.graph_mode == "static":
            return np.array([0.0], dtype=np.float32)
        if self.config.graph_mode == "learned":
            return np.array([1.0], dtype=np.float32)
        return self._to_numpy_float32(torch.sigmoid(self.prediction_fusion_logits).flatten())


class StaticAdaptiveZeroInflatedSTGNNGRURegressor(nn.Module):
    def __init__(
        self,
        config: StaticAdaptiveSTGNNModelConfig,
        static_adjacency: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.graph_constructor = StaticAdaptiveGraphConstructor(
            node_count=config.node_count,
            embed_dim=config.embed_dim,
            top_k=config.graph_top_k,
            static_adjacency=static_adjacency,
            graph_mode=config.graph_mode,
        )
        self.static_branch = _STGNNZeroInflatedBranch(config)
        self.learned_branch = _STGNNZeroInflatedBranch(config)
        self.prediction_fusion_logits = nn.Parameter(torch.full((1,), 2.0, dtype=torch.float32))

    def forward(self, batch: torch.Tensor) -> ZeroInflatedSTGNNOutput:
        static_output = self.static_branch(batch, self.graph_constructor.static_adjacency)
        learned_output = self.learned_branch(batch, self.graph_constructor.learned_adjacency())

        if self.config.graph_mode == "static":
            alpha = torch.zeros((1, 1), dtype=static_output.predictions.dtype, device=static_output.predictions.device)
            predictions = static_output.predictions
            zero_logits = static_output.zero_logits
            values = static_output.values
        elif self.config.graph_mode == "learned":
            alpha = torch.ones((1, 1), dtype=learned_output.predictions.dtype, device=learned_output.predictions.device)
            predictions = learned_output.predictions
            zero_logits = learned_output.zero_logits
            values = learned_output.values
        else:
            alpha = torch.sigmoid(self.prediction_fusion_logits).view(1, 1)
            predictions = alpha * learned_output.predictions + (1.0 - alpha) * static_output.predictions
            zero_logits = alpha * learned_output.zero_logits + (1.0 - alpha) * static_output.zero_logits
            values = alpha * learned_output.values + (1.0 - alpha) * static_output.values

        zero_probabilities = torch.sigmoid(zero_logits)
        return ZeroInflatedSTGNNOutput(
            predictions=predictions,
            zero_logits=zero_logits,
            zero_probabilities=zero_probabilities,
            values=values,
            static_predictions=static_output.predictions,
            learned_predictions=learned_output.predictions,
            fusion_alpha=alpha,
        )

    def _to_numpy_float32(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def get_adjacency(self) -> torch.Tensor:
        return self.get_prediction_weighted_adjacency()

    def get_learned_adjacency(self) -> torch.Tensor:
        return self.graph_constructor.learned_adjacency()

    def get_static_adjacency(self) -> torch.Tensor:
        return self.graph_constructor.static_adjacency

    def get_prediction_weighted_adjacency(self) -> torch.Tensor:
        static_adjacency = self.get_static_adjacency()
        learned_adjacency = self.get_learned_adjacency()
        if self.config.graph_mode == "static":
            return static_adjacency
        if self.config.graph_mode == "learned":
            return learned_adjacency
        alpha = torch.as_tensor(
            self.get_prediction_fusion_alpha(),
            dtype=static_adjacency.dtype,
            device=static_adjacency.device,
        ).view(1, 1)
        return alpha * learned_adjacency + (1.0 - alpha) * static_adjacency

    def get_prediction_fusion_alpha(self) -> np.ndarray:
        return self._to_numpy_float32(torch.sigmoid(self.prediction_fusion_logits).flatten())


def zero_inflated_masked_loss(
    predictions: torch.Tensor,
    zero_logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    zero_pos_weight: float | torch.Tensor | None = None,
    zero_classification_weight: float = 1.0,
    positive_regression_weight: float = 1.0,
    zero_suppression_weight: float = 1.0,
) -> dict[str, torch.Tensor | int]:
    mask = target_mask.bool()
    observed_count = int(mask.sum().item())
    zero_mask = mask & (targets <= 0.0)
    positive_mask = mask & (targets > 0.0)
    zero_count = int(zero_mask.sum().item())
    positive_count = int(positive_mask.sum().item())

    zero_loss = predictions.sum() * 0.0
    positive_loss = predictions.sum() * 0.0
    zero_classification_loss = predictions.sum() * 0.0

    if observed_count > 0:
        observed_zero_targets = zero_mask[mask].to(dtype=predictions.dtype)
        pos_weight = None
        if zero_pos_weight is not None:
            pos_weight = torch.as_tensor(zero_pos_weight, dtype=predictions.dtype, device=predictions.device)
        zero_classification_loss = nn.functional.binary_cross_entropy_with_logits(
            zero_logits[mask],
            observed_zero_targets,
            pos_weight=pos_weight,
        )

    if positive_count > 0:
        positive_loss = nn.functional.smooth_l1_loss(
            predictions[positive_mask],
            targets[positive_mask],
            beta=0.5,
        )

    if zero_count > 0:
        zero_loss = nn.functional.smooth_l1_loss(
            predictions[zero_mask],
            torch.zeros_like(predictions[zero_mask]),
            beta=0.1,
        )

    total_loss = (
        zero_classification_weight * zero_classification_loss
        + positive_regression_weight * positive_loss
        + zero_suppression_weight * zero_loss
    )
    return {
        "loss": total_loss,
        "zero_classification_loss": zero_classification_loss,
        "positive_regression_loss": positive_loss,
        "zero_suppression_loss": zero_loss,
        "observed_count": observed_count,
        "zero_count": zero_count,
        "positive_count": positive_count,
    }
