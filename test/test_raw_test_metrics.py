import importlib.util
import json
import sys
import unittest
from collections import deque
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RawMetricUtilsTest(unittest.TestCase):
    def test_inverse_log1p_minmax_restores_raw_values(self) -> None:
        module = load_module("raw_metric_utils", WORKSPACE_ROOT / "raw_metric_utils.py")

        raw_values = module.inverse_log1p_minmax(
            np.array([0.0, 1.0], dtype=np.float32),
            target_log1p_min=0.0,
            target_log1p_max=np.log1p(9.0),
        )

        np.testing.assert_allclose(raw_values, np.array([0.0, 9.0], dtype=np.float32), rtol=1e-6)

    def test_raw_metrics_include_rmsle_with_negative_prediction_clipping(self) -> None:
        module = load_module("raw_metric_utils", WORKSPACE_ROOT / "raw_metric_utils.py")

        metrics = module.compute_raw_regression_metrics(
            y_true_raw=np.array([0.0, 3.0], dtype=np.float32),
            y_pred_raw=np.array([-5.0, 1.0], dtype=np.float32),
        )

        self.assertEqual(set(metrics), {"mse", "mae", "r2", "smape", "rmse", "rmsle"})
        self.assertAlmostEqual(metrics["mse"], 14.5)
        self.assertAlmostEqual(metrics["mae"], 3.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(14.5))
        expected_rmsle = np.sqrt(((np.log1p(0.0) - np.log1p(0.0)) ** 2 + (np.log1p(1.0) - np.log1p(3.0)) ** 2) / 2.0)
        self.assertAlmostEqual(metrics["rmsle"], expected_rmsle)


class RawEvaluatorMetadataTest(unittest.TestCase):
    def test_xgboost_reads_saved_feature_columns_and_best_round(self) -> None:
        module = load_module(
            "xgboost_raw_test_metrics",
            WORKSPACE_ROOT / "XGBoost" / "xgboost_raw_test_metrics.py",
        )

        with TemporaryDirectory() as temp_dir:
            params_path = Path(temp_dir) / "xgboost_model_params.json"
            params_path.write_text(
                json.dumps(
                    {
                        "xgboost_feature_cols": ["building_id", "meter", "feature"],
                        "best_iteration_round_count": 7,
                    }
                ),
                encoding="utf-8",
            )

            feature_cols, best_round = module.load_feature_columns_and_best_round(params_path)

        self.assertEqual(feature_cols, ["building_id", "meter", "feature"])
        self.assertEqual(best_round, 7)

    def test_gru_builds_raw_metrics_artifact_with_row_counts(self) -> None:
        module = load_module(
            "gru_raw_test_metrics",
            WORKSPACE_ROOT / "GRU" / "gru_raw_test_metrics.py",
        )

        artifact = module.build_raw_metrics_artifact(
            metrics={"mse": 1.0, "mae": 2.0, "r2": 3.0, "smape": 4.0, "rmse": 5.0, "rmsle": 6.0},
            metadata={"evaluated_row_count": 10, "skipped_row_count": 2, "window_count": 10},
            paths={
                "model_path": "model.pt",
                "model_params_path": "params.json",
                "test_data_path": "test.csv",
                "preprocessing_summary_path": "summary.json",
            },
            preprocessing_summary={
                "target_preprocess": "log1p_then_train_minmax",
                "target_log1p_min": 0.0,
                "target_log1p_max": 9.0,
            },
        )

        self.assertEqual(artifact["metric_scale"], "raw_meter_reading_after_inverse_transform")
        self.assertEqual(artifact["rmsle"], 6.0)
        self.assertEqual(artifact["evaluated_row_count"], 10)
        self.assertEqual(artifact["model_path"], "model.pt")

    def test_gru_raw_rolling_counts_rows_and_scores_raw_space(self) -> None:
        module = load_module(
            "gru_raw_test_metrics",
            WORKSPACE_ROOT / "GRU" / "gru_raw_test_metrics.py",
        )
        timestamp = np.array(["2016-01-01T01:00:00"], dtype="datetime64[ns]")

        class FakeGRUModule:
            TARGET_COL = "meter_reading"
            deque = deque

            @staticmethod
            def build_initial_histories(split_data, history_parts, window_size):
                key = (int(split_data.building_ids[0]), int(split_data.meter_ids[0]))
                return {key: deque([np.array([0.0], dtype=np.float32)], maxlen=window_size)}, {key: timestamp[0] - np.timedelta64(1, "h")}

            @staticmethod
            def append_history_row(histories, last_timestamps, key, feature_row, timestamp, window_size):
                histories.setdefault(key, deque(maxlen=window_size)).append(feature_row)
                last_timestamps[key] = timestamp

            @staticmethod
            def predict_sequence_batch(model, sequences, device, batch_size):
                return np.array([0.0], dtype=np.float32)

        split_data = SimpleNamespace(
            name="test",
            input_features=np.array([[1.0]], dtype=np.float32),
            targets=np.array([1.0], dtype=np.float32),
            timestamps=timestamp,
            building_ids=np.array([1]),
            meter_ids=np.array([0]),
        )

        metrics = module.rolling_evaluate_split_raw(
            module=FakeGRUModule,
            model=object(),
            split_data=split_data,
            history_parts=[],
            input_feature_cols=["meter_reading"],
            feature_means=np.array([0.0], dtype=np.float32),
            feature_stds=np.array([1.0], dtype=np.float32),
            target_log1p_min=0.0,
            target_log1p_max=np.log1p(9.0),
            device="cpu",
            window_size=1,
            eval_batch_size=1,
        )

        self.assertEqual(metrics["evaluated_row_count"], 1)
        self.assertEqual(metrics["skipped_row_count"], 0)
        self.assertEqual(metrics["window_count"], 1)
        self.assertAlmostEqual(metrics["rmse"], 9.0)

    def test_stgnn_builds_raw_metrics_artifact_with_unknown_node_count(self) -> None:
        module = load_module(
            "stgnn_raw_test_metrics",
            WORKSPACE_ROOT / "STGNN" / "stgnn_raw_test_metrics.py",
        )

        artifact = module.build_raw_metrics_artifact(
            metrics={"mse": 1.0, "mae": 2.0, "r2": 3.0, "smape": 4.0, "rmse": 5.0, "rmsle": 6.0},
            metadata={
                "evaluated_row_count": 10,
                "skipped_row_count": 3,
                "window_count": 10,
                "unknown_node_row_count": 1,
                "evaluated_timestamp_count": 8,
                "skipped_timestamp_count": 2,
            },
            paths={
                "model_path": "model.pt",
                "model_params_path": "params.json",
                "test_data_path": "test.csv",
                "preprocessing_summary_path": "summary.json",
            },
            preprocessing_summary={
                "target_preprocess": "log1p_then_train_minmax",
                "target_log1p_min": 0.0,
                "target_log1p_max": 9.0,
            },
        )

        self.assertEqual(artifact["metric_scale"], "raw_meter_reading_after_inverse_transform")
        self.assertEqual(artifact["rmsle"], 6.0)
        self.assertEqual(artifact["unknown_node_row_count"], 1)
        self.assertEqual(artifact["evaluated_timestamp_count"], 8)

    def test_stgnn_raw_rolling_counts_observed_and_unknown_rows(self) -> None:
        module = load_module(
            "stgnn_raw_test_metrics",
            WORKSPACE_ROOT / "STGNN" / "stgnn_raw_test_metrics.py",
        )
        timestamp = np.array(["2016-01-01T01:00:00"], dtype="datetime64[ns]")

        class FakeSTGNNModule:
            TARGET_COL = "meter_reading"

            @staticmethod
            def collect_global_history_suffix(first_timestamp, history_parts, window_size):
                return deque([np.array([[0.0]], dtype=np.float32)], maxlen=window_size), first_timestamp - np.timedelta64(1, "h")

            @staticmethod
            def predict_global_window(model, history_rows, device):
                return np.array([0.0], dtype=np.float32)

            @staticmethod
            def build_current_feature_rows(split_data, time_index, history_rows):
                return split_data.input_features[time_index].copy()

            @staticmethod
            def build_predicted_feature_rows(current_feature_rows, predictions, meter_feature_index, feature_means, feature_stds):
                predicted_feature_rows = current_feature_rows.copy()
                predicted_feature_rows[:, meter_feature_index] = predictions
                return predicted_feature_rows

        split_data = SimpleNamespace(
            name="test",
            input_features=np.array([[[1.0]]], dtype=np.float32),
            targets=np.array([[1.0]], dtype=np.float32),
            target_mask=np.array([[True]]),
            timestamps=timestamp,
            unknown_node_row_count=2,
        )

        metrics = module.rolling_evaluate_split_raw(
            module=FakeSTGNNModule,
            model=object(),
            split_data=split_data,
            history_parts=[],
            input_feature_cols=["meter_reading"],
            feature_means=np.array([0.0], dtype=np.float32),
            feature_stds=np.array([1.0], dtype=np.float32),
            target_log1p_min=0.0,
            target_log1p_max=np.log1p(9.0),
            device="cpu",
            window_size=1,
        )

        self.assertEqual(metrics["evaluated_row_count"], 1)
        self.assertEqual(metrics["skipped_row_count"], 2)
        self.assertEqual(metrics["unknown_node_row_count"], 2)
        self.assertEqual(metrics["evaluated_timestamp_count"], 1)
        self.assertAlmostEqual(metrics["rmse"], 9.0)


if __name__ == "__main__":
    unittest.main()
