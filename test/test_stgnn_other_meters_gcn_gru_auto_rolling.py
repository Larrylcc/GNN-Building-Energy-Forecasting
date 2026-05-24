import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "STGNN" / "stgnn_other_meters_gcn_gru_auto-rolling.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stgnn_other_meters_gcn_gru_auto_rolling", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class STGNNOtherMetersTest(unittest.TestCase):
    def test_meter_config_builds_expected_paths(self) -> None:
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = module.MeterConfig(meter_id=1, meter_name="chilled_water")
            paths = module.build_meter_paths(
                config=config,
                preprocessed_data_dir=root / "preprocessed_data",
                output_root_dir=root / "outputs",
            )

            self.assertEqual(paths.train_data_path, root / "preprocessed_data" / "meter_1" / "log1p_minmax_train.csv")
            self.assertEqual(paths.valid_data_path, root / "preprocessed_data" / "meter_1" / "log1p_minmax_valid.csv")
            self.assertEqual(paths.test_data_path, root / "preprocessed_data" / "meter_1" / "log1p_minmax_test.csv")
            self.assertEqual(paths.preprocessing_summary_path, root / "preprocessed_data" / "meter_1" / "log1p_minmax_summary.json")
            self.assertEqual(paths.output_dir, root / "outputs" / "meter_1")
            self.assertEqual(paths.model_path, root / "outputs" / "meter_1" / "stgnn_final_model.pt")
            self.assertEqual(paths.adjacency_path, root / "outputs" / "meter_1" / "stgnn_learned_adjacency.npy")
            self.assertEqual(paths.topk_edges_path, root / "outputs" / "meter_1" / "stgnn_topk_edges.csv")

    def test_load_preprocessed_splits_returns_stgnn_input_features(self) -> None:
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.csv"
            valid_path = root / "valid.csv"
            test_path = root / "test.csv"
            split_df = pd.DataFrame(
                {
                    "timestamp": ["2016-01-01 01:00:00", "2016-01-01 00:00:00"],
                    "meter_reading": [0.2, 0.1],
                    "building_id": [10, 10],
                    "meter": [1, 1],
                    "feature": [5.0, 4.0],
                }
            )
            split_df.to_csv(train_path, index=False)
            split_df.to_csv(valid_path, index=False)
            split_df.to_csv(test_path, index=False)

            train_df, valid_df, test_df, input_feature_cols = module.load_preprocessed_splits(
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
            )

            self.assertEqual(input_feature_cols, ["meter_reading", "building_id", "meter", "feature"])
            self.assertEqual(train_df["timestamp"].tolist(), sorted(train_df["timestamp"].tolist()))
            self.assertEqual(str(train_df["meter_reading"].dtype), "float32")
            self.assertEqual(valid_df.columns.tolist(), train_df.columns.tolist())
            self.assertEqual(test_df.columns.tolist(), train_df.columns.tolist())

    def test_load_preprocessed_splits_rejects_column_mismatch(self) -> None:
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.csv"
            valid_path = root / "valid.csv"
            test_path = root / "test.csv"
            pd.DataFrame(
                {
                    "timestamp": ["2016-01-01 00:00:00"],
                    "meter_reading": [0.1],
                    "building_id": [10],
                    "meter": [1],
                    "feature": [4.0],
                }
            ).to_csv(train_path, index=False)
            pd.DataFrame(
                {
                    "timestamp": ["2016-01-01 00:00:00"],
                    "meter_reading": [0.1],
                    "building_id": [10],
                    "meter": [1],
                }
            ).to_csv(valid_path, index=False)
            pd.DataFrame(
                {
                    "timestamp": ["2016-01-01 00:00:00"],
                    "meter_reading": [0.1],
                    "building_id": [10],
                    "meter": [1],
                    "feature": [4.0],
                }
            ).to_csv(test_path, index=False)

            with self.assertRaisesRegex(ValueError, "valid split columns"):
                module.load_preprocessed_splits(train_path=train_path, valid_path=valid_path, test_path=test_path)

    def test_dense_split_uses_train_nodes_and_tracks_unknown_rows(self) -> None:
        module = load_module()

        split_df = pd.DataFrame(
            {
                "timestamp": ["2016-01-01 00:00:00", "2016-01-01 00:00:00", "2016-01-01 01:00:00"],
                "meter_reading": [0.1, 0.2, 0.3],
                "building_id": [10, 11, 12],
                "meter": [1, 1, 1],
                "feature": [4.0, 5.0, 6.0],
            }
        )
        split_df["timestamp"] = pd.to_datetime(split_df["timestamp"])
        input_feature_cols = ["meter_reading", "building_id", "meter", "feature"]
        dense_data = module.build_dense_split_data(
            name="tiny",
            split_df=split_df,
            input_feature_cols=input_feature_cols,
            feature_means=np.zeros(4, dtype=np.float32),
            feature_stds=np.ones(4, dtype=np.float32),
            node_ids=np.array([10, 11], dtype=np.int32),
        )

        self.assertEqual(dense_data.raw_row_count, 3)
        self.assertEqual(dense_data.included_row_count, 2)
        self.assertEqual(dense_data.unknown_node_row_count, 1)
        self.assertEqual(dense_data.input_features.shape, (2, 2, 4))
        self.assertTrue(dense_data.target_mask[0, 0])
        self.assertTrue(dense_data.target_mask[0, 1])
        self.assertFalse(dense_data.target_mask[1, 0])
        self.assertFalse(dense_data.target_mask[1, 1])

    def test_build_artifacts_include_meter_graph_and_target_metadata(self) -> None:
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = module.MeterConfig(meter_id=3, meter_name="hot_water")
            paths = module.build_meter_paths(
                config=config,
                preprocessed_data_dir=root / "preprocessed_data",
                output_root_dir=root / "outputs",
            )
            model_config = module.STGNNModelConfig(input_size=4, node_count=2, gcn_dim=4, embed_dim=4, gcn_layers=1)
            training_config = module.TrainingConfig(window_size=4, batch_size=2, eval_batch_size=1, epochs=1, patience=1)
            preprocessing_summary = {
                "target_preprocess": "log1p_then_train_minmax",
                "target_log1p_min": 0.0,
                "target_log1p_max": 11.5,
            }
            validation_metrics = {
                "mse": 1.0,
                "mae": 2.0,
                "r2": 3.0,
                "smape": 4.0,
                "rmse": 5.0,
                "evaluated_row_count": 6,
                "skipped_row_count": 2,
                "window_count": 6,
                "unknown_node_row_count": 0,
            }
            test_metrics = {
                "mse": 6.0,
                "mae": 7.0,
                "r2": 8.0,
                "smape": 9.0,
                "rmse": 10.0,
                "evaluated_row_count": 7,
                "skipped_row_count": 3,
                "window_count": 7,
                "unknown_node_row_count": 1,
            }
            timestamps = np.array(["2016-01-01T00:00:00", "2016-01-01T01:00:00"], dtype="datetime64[ns]")
            split_data = module.DenseSplitData(
                name="split",
                input_features=np.zeros((2, 2, 4), dtype=np.float32),
                targets=np.zeros((2, 2), dtype=np.float32),
                target_mask=np.zeros((2, 2), dtype=bool),
                timestamps=timestamps,
                node_ids=np.array([10, 11], dtype=np.int32),
                raw_row_count=8,
                included_row_count=7,
                unknown_node_row_count=1,
            )

            model_params = module.build_model_params_artifact(
                config=config,
                paths=paths,
                preprocessing_summary=preprocessing_summary,
                model_config=model_config,
                training_config=training_config,
                node_ids=np.array([10, 11], dtype=np.int32),
                input_feature_cols=["meter_reading", "building_id", "meter", "feature"],
                feature_means=np.zeros(4, dtype=np.float32),
                feature_stds=np.ones(4, dtype=np.float32),
                best_epoch=1,
                best_validation_metrics=validation_metrics,
                train_window_count=12,
                test_only_node_ids=[99],
                test_unknown_node_row_count=1,
                adjacency_path=paths.adjacency_path,
                topk_edges_path=paths.topk_edges_path,
            )
            run_summary = module.build_run_summary(
                config=config,
                paths=paths,
                preprocessing_summary=preprocessing_summary,
                split_summary=pd.DataFrame([{"split": "train", "start_timestamp": "a", "end_timestamp": "b", "row_count": 8}]),
                input_feature_cols=["meter_reading", "building_id", "meter", "feature"],
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                best_epoch=1,
                train_window_count=12,
                train_data=split_data,
                valid_data=split_data,
                test_data=split_data,
                test_only_node_ids=[99],
                model_path=paths.model_path,
                model_params_path=paths.model_params_path,
                training_history_path=paths.training_history_path,
                validation_metrics_path=paths.validation_metrics_path,
                test_metrics_path=paths.test_metrics_path,
                split_summary_path=paths.split_summary_path,
                adjacency_path=paths.adjacency_path,
                topk_edges_path=paths.topk_edges_path,
                device_text="cuda:0",
                cuda_available=True,
            )

            self.assertEqual(model_params["meter_id"], 3)
            self.assertEqual(model_params["target_log1p_max"], 11.5)
            self.assertEqual(model_params["node_ids"], [10, 11])
            self.assertEqual(model_params["test_only_node_ids"], [99])
            self.assertEqual(run_summary["meter_name"], "hot_water")
            self.assertEqual(run_summary["node_count"], 2)
            self.assertEqual(run_summary["test_unknown_node_row_count"], 1)
            self.assertEqual(run_summary["test_rmse"], 10.0)
            self.assertEqual(run_summary["adjacency_path"], str(paths.adjacency_path))


if __name__ == "__main__":
    unittest.main()
