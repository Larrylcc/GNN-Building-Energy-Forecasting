import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "GRU" / "gru_other_meters_auto-rolling.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gru_other_meters_auto_rolling", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GRUOtherMetersTest(unittest.TestCase):
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
            self.assertEqual(paths.model_path, root / "outputs" / "meter_1" / "gru_final_model.pt")

    def test_load_preprocessed_splits_returns_gru_input_features(self) -> None:
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

    def test_build_run_summary_includes_meter_and_artifact_metadata(self) -> None:
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = module.MeterConfig(meter_id=3, meter_name="hot_water")
            paths = module.build_meter_paths(
                config=config,
                preprocessed_data_dir=root / "preprocessed_data",
                output_root_dir=root / "outputs",
            )
            summary = module.build_run_summary(
                config=config,
                paths=paths,
                preprocessing_summary={
                    "target_preprocess": "log1p_then_train_minmax",
                    "target_log1p_min": 0.0,
                    "target_log1p_max": 11.5,
                },
                split_summary=pd.DataFrame(
                    [
                        {"split": "train", "start_timestamp": "a", "end_timestamp": "b", "row_count": 8},
                        {"split": "valid", "start_timestamp": "c", "end_timestamp": "d", "row_count": 1},
                        {"split": "test", "start_timestamp": "e", "end_timestamp": "f", "row_count": 1},
                    ]
                ),
                input_feature_cols=["meter_reading", "building_id", "meter", "feature"],
                validation_metrics={
                    "mse": 1.0,
                    "mae": 2.0,
                    "r2": 3.0,
                    "smape": 4.0,
                    "rmse": 5.0,
                    "evaluated_row_count": 6,
                    "skipped_row_count": 2,
                    "window_count": 6,
                },
                test_metrics={
                    "mse": 6.0,
                    "mae": 7.0,
                    "r2": 8.0,
                    "smape": 9.0,
                    "rmse": 10.0,
                    "evaluated_row_count": 7,
                    "skipped_row_count": 3,
                    "window_count": 7,
                },
                best_epoch=2,
                train_window_count=12,
                train_row_count=8,
                valid_row_count=1,
                test_row_count=1,
                model_path=paths.model_path,
                model_params_path=paths.model_params_path,
                training_history_path=paths.training_history_path,
                validation_metrics_path=paths.validation_metrics_path,
                test_metrics_path=paths.test_metrics_path,
                split_summary_path=paths.split_summary_path,
                device_text="cuda:0",
                cuda_available=True,
            )

            self.assertEqual(summary["meter_id"], 3)
            self.assertEqual(summary["meter_name"], "hot_water")
            self.assertEqual(summary["target_log1p_max"], 11.5)
            self.assertEqual(summary["feature_count"], 4)
            self.assertEqual(summary["train_window_count"], 12)
            self.assertEqual(summary["test_rmse"], 10.0)
            self.assertEqual(summary["model_path"], str(paths.model_path))


if __name__ == "__main__":
    unittest.main()
