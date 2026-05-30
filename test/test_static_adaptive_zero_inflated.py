from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from STGNN.static_adaptive_graph import (
    StaticAdaptiveSTGNNModelConfig,
    StaticAdaptiveSTGNNGRURegressor,
    StaticAdaptiveZeroInflatedSTGNNGRURegressor,
    zero_inflated_masked_loss,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StaticAdaptiveZeroInflatedTest(unittest.TestCase):
    def test_regressor_learned_mode_does_not_call_static_branch(self) -> None:
        class RaisingBranch(torch.nn.Module):
            def forward(self, batch: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
                raise AssertionError("learned graph mode should not evaluate the static branch")

        config = StaticAdaptiveSTGNNModelConfig(
            input_size=3,
            node_count=4,
            gcn_dim=4,
            embed_dim=4,
            graph_top_k=2,
            gcn_layers=2,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            graph_mode="learned",
        )
        model = StaticAdaptiveSTGNNGRURegressor(
            config=config,
            static_adjacency=torch.eye(4, dtype=torch.float32),
        )
        model.static_branch = RaisingBranch()

        output = model.predict_branches(torch.randn(2, 5, 4, 3))

        self.assertIsNone(output.static_predictions)
        self.assertEqual(tuple(output.learned_predictions.shape), (2, 4))
        self.assertTrue(bool(torch.allclose(output.predictions, output.learned_predictions)))
        self.assertEqual(model.get_prediction_fusion_alpha().astype(float).tolist(), [1.0])

    def test_regressor_static_adaptive_fuses_static_and_learned_predictions(self) -> None:
        config = StaticAdaptiveSTGNNModelConfig(
            input_size=3,
            node_count=4,
            gcn_dim=4,
            embed_dim=4,
            graph_top_k=2,
            gcn_layers=2,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            graph_mode="static_adaptive",
        )
        model = StaticAdaptiveSTGNNGRURegressor(
            config=config,
            static_adjacency=torch.eye(4, dtype=torch.float32),
        )
        batch = torch.randn(2, 5, 4, 3)

        output = model.predict_branches(batch)
        forward_predictions = model(batch)
        expected = (
            output.fusion_alpha * output.learned_predictions
            + (1.0 - output.fusion_alpha) * output.static_predictions
        )

        self.assertEqual(tuple(output.predictions.shape), (2, 4))
        self.assertEqual(tuple(output.static_predictions.shape), (2, 4))
        self.assertEqual(tuple(output.learned_predictions.shape), (2, 4))
        self.assertTrue(bool(torch.allclose(output.predictions, expected)))
        self.assertTrue(bool(torch.allclose(forward_predictions, output.predictions)))
        self.assertTrue(0.0 <= float(output.fusion_alpha.item()) <= 1.0)

    def test_zero_inflated_regressor_outputs_prediction_and_zero_state(self) -> None:
        config = StaticAdaptiveSTGNNModelConfig(
            input_size=3,
            node_count=4,
            gcn_dim=4,
            embed_dim=4,
            graph_top_k=2,
            gcn_layers=2,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            graph_mode="static",
        )
        static_adjacency = torch.eye(4, dtype=torch.float32)
        model = StaticAdaptiveZeroInflatedSTGNNGRURegressor(
            config=config,
            static_adjacency=static_adjacency,
        )

        output = model(torch.randn(2, 5, 4, 3))

        self.assertEqual(tuple(output.predictions.shape), (2, 4))
        self.assertEqual(tuple(output.zero_logits.shape), (2, 4))
        self.assertEqual(tuple(output.zero_probabilities.shape), (2, 4))
        self.assertEqual(tuple(output.values.shape), (2, 4))
        self.assertTrue(bool(torch.all(output.predictions >= 0.0)))
        self.assertTrue(bool(torch.all(output.values >= 0.0)))
        self.assertTrue(bool(torch.all((output.zero_probabilities >= 0.0) & (output.zero_probabilities <= 1.0))))

    def test_zero_inflated_static_adaptive_fuses_static_and_learned_predictions(self) -> None:
        config = StaticAdaptiveSTGNNModelConfig(
            input_size=3,
            node_count=4,
            gcn_dim=4,
            embed_dim=4,
            graph_top_k=2,
            gcn_layers=2,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            graph_mode="static_adaptive",
        )
        model = StaticAdaptiveZeroInflatedSTGNNGRURegressor(
            config=config,
            static_adjacency=torch.eye(4, dtype=torch.float32),
        )

        output = model(torch.randn(2, 5, 4, 3))
        expected = (
            output.fusion_alpha * output.learned_predictions
            + (1.0 - output.fusion_alpha) * output.static_predictions
        )

        self.assertEqual(tuple(output.static_predictions.shape), (2, 4))
        self.assertEqual(tuple(output.learned_predictions.shape), (2, 4))
        self.assertTrue(bool(torch.allclose(output.predictions, expected)))
        self.assertTrue(0.0 <= float(output.fusion_alpha.item()) <= 1.0)

    def test_zero_inflated_loss_uses_only_observed_zero_and_positive_targets(self) -> None:
        predictions = torch.tensor([[2.0, 3.0, 999.0], [999.0, 4.0, 5.0]], dtype=torch.float32)
        zero_logits = torch.zeros_like(predictions)
        targets = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0]], dtype=torch.float32)
        target_mask = torch.tensor([[True, True, False], [False, True, True]])

        loss_parts = zero_inflated_masked_loss(
            predictions=predictions,
            zero_logits=zero_logits,
            targets=targets,
            target_mask=target_mask,
        )

        expected_bce = math.log(2.0)
        expected_positive_mse = ((3.0 - 1.0) ** 2 + (5.0 - 2.0) ** 2) / 2.0
        expected_zero_mse = (2.0**2 + 4.0**2) / 2.0
        expected_total = expected_bce + expected_positive_mse + expected_zero_mse

        self.assertEqual(loss_parts["observed_count"], 4)
        self.assertEqual(loss_parts["zero_count"], 2)
        self.assertEqual(loss_parts["positive_count"], 2)
        self.assertTrue(torch.isclose(loss_parts["zero_classification_loss"], torch.tensor(expected_bce)))
        self.assertTrue(torch.isclose(loss_parts["positive_regression_loss"], torch.tensor(expected_positive_mse)))
        self.assertTrue(torch.isclose(loss_parts["zero_suppression_loss"], torch.tensor(expected_zero_mse)))
        self.assertTrue(torch.isclose(loss_parts["loss"], torch.tensor(expected_total)))


class StaticAdaptiveOtherMetersMetricsTest(unittest.TestCase):
    def test_electricity_static_adaptive_outputs_stay_under_gnn_directory(self) -> None:
        module = load_module(
            "static_adaptive_stgnn_gcn_gru_auto_rolling",
            WORKSPACE_ROOT / "STGNN" / "stgnn_gcn_gru_auto-rolling.py",
        )

        self.assertEqual(module.OUTPUT_DIR, WORKSPACE_ROOT / "STGNN" / "stgnn_gcn_gru_auto-rolling_outputs")

    def test_electricity_graph_artifact_names_match_prediction_level_fusion(self) -> None:
        module = load_module(
            "static_adaptive_stgnn_gcn_gru_auto_rolling_artifacts",
            WORKSPACE_ROOT / "STGNN" / "stgnn_gcn_gru_auto-rolling.py",
        )

        self.assertEqual(module.GRAPH_MODE, "learned")
        self.assertFalse(hasattr(module, "FUSED_ADJACENCY_PATH"))
        self.assertEqual(module.ADJACENCY_PATH, module.LEARNED_ADJACENCY_PATH)
        self.assertEqual(module.TOPK_EDGES_PATH, module.LEARNED_TOPK_EDGES_PATH)

    def test_other_meter_graph_artifact_names_match_prediction_level_fusion(self) -> None:
        module = load_module(
            "static_adaptive_stgnn_other_meters_auto_rolling_artifacts",
            WORKSPACE_ROOT / "STGNN" / "static_adaptive_stgnn_other_meters_auto-rolling.py",
        )

        paths = module.build_meter_paths(module.MeterConfig(meter_id=1, meter_name="chilled_water"))

        self.assertFalse(hasattr(paths, "fused_adjacency_path"))
        self.assertTrue(hasattr(paths, "prediction_weighted_adjacency_path"))
        self.assertTrue(hasattr(paths, "prediction_weighted_topk_edges_path"))
        self.assertIn("prediction_weighted", paths.prediction_weighted_adjacency_path.name)

    def test_thresholded_predictions_zero_when_probability_crosses_threshold(self) -> None:
        module = load_module(
            "static_adaptive_stgnn_other_meters_auto_rolling",
            WORKSPACE_ROOT / "STGNN" / "static_adaptive_stgnn_other_meters_auto-rolling.py",
        )

        thresholded = module.build_thresholded_predictions(
            predictions=np.array([0.2, 0.4, 0.6], dtype=np.float32),
            zero_probabilities=np.array([0.49, 0.5, 0.9], dtype=np.float32),
            zero_threshold=0.5,
        )

        np.testing.assert_allclose(thresholded, np.array([0.2, 0.0, 0.0], dtype=np.float32))

    def test_zero_state_metrics_count_false_zero_directions(self) -> None:
        module = load_module(
            "static_adaptive_stgnn_other_meters_auto_rolling",
            WORKSPACE_ROOT / "STGNN" / "static_adaptive_stgnn_other_meters_auto-rolling.py",
        )

        metrics = module.compute_zero_state_metrics(
            targets=np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32),
            predictions=np.array([0.1, 0.0, 0.0, 2.0], dtype=np.float32),
            zero_probabilities=np.array([0.8, 0.4, 0.7, 0.2], dtype=np.float32),
            zero_threshold=0.5,
        )

        self.assertEqual(metrics["zero_true_positive_count"], 1)
        self.assertEqual(metrics["zero_false_positive_count"], 1)
        self.assertEqual(metrics["zero_false_negative_count"], 1)
        self.assertEqual(metrics["zero_true_negative_count"], 1)
        self.assertAlmostEqual(metrics["zero_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["zero_precision"], 0.5)
        self.assertAlmostEqual(metrics["zero_recall"], 0.5)
        self.assertAlmostEqual(metrics["zero_f1"], 0.5)
        self.assertAlmostEqual(metrics["zero_false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["zero_false_negative_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_prediction_on_zero"], 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
