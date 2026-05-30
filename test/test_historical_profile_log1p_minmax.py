import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = WORKSPACE_ROOT / "HistoricalProfile" / "historical_profile_common.py"
BASELINE_PATH = WORKSPACE_ROOT / "HistoricalProfile" / "historical_profile_baseline_log1p-minmax.py"
OTHER_METERS_PATH = WORKSPACE_ROOT / "HistoricalProfile" / "historical_profile_other_meters_log1p-minmax.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HistoricalProfileTest(unittest.TestCase):
    def test_baseline_paths_match_expected_artifacts(self) -> None:
        module = load_module("historical_profile_baseline", BASELINE_PATH)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = module.build_paths(
                preprocessed_data_dir=root / "preprocessed_data",
                output_dir=root / "outputs",
            )

            self.assertEqual(paths.train_data_path, root / "preprocessed_data" / "log1p_minmax_train.csv")
            self.assertEqual(paths.valid_data_path, root / "preprocessed_data" / "log1p_minmax_valid.csv")
            self.assertEqual(paths.test_data_path, root / "preprocessed_data" / "log1p_minmax_test.csv")
            self.assertEqual(paths.preprocessing_summary_path, root / "preprocessed_data" / "log1p_minmax_summary.json")
            self.assertEqual(paths.output_dir, root / "outputs")
            self.assertEqual(paths.profile_tables_path, root / "outputs" / "historical_profile_profile_tables.csv")
            self.assertEqual(paths.model_params_path, root / "outputs" / "historical_profile_model_params.json")

    def test_other_meter_paths_match_expected_artifacts(self) -> None:
        module = load_module("historical_profile_other_meters", OTHER_METERS_PATH)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = module.MeterConfig(meter_id=2, meter_name="steam")
            paths = module.build_meter_paths(
                config=config,
                preprocessed_data_dir=root / "preprocessed_data",
                output_root_dir=root / "outputs",
            )

            self.assertEqual(paths.train_data_path, root / "preprocessed_data" / "meter_2" / "log1p_minmax_train.csv")
            self.assertEqual(paths.valid_data_path, root / "preprocessed_data" / "meter_2" / "log1p_minmax_valid.csv")
            self.assertEqual(paths.test_data_path, root / "preprocessed_data" / "meter_2" / "log1p_minmax_test.csv")
            self.assertEqual(paths.preprocessing_summary_path, root / "preprocessed_data" / "meter_2" / "log1p_minmax_summary.json")
            self.assertEqual(paths.output_dir, root / "outputs" / "meter_2")
            self.assertEqual(paths.run_summary_path, root / "outputs" / "meter_2" / "historical_profile_run_summary.json")

    def test_load_preprocessed_splits_parses_sorts_and_validates_columns(self) -> None:
        module = load_module("historical_profile_common", COMMON_PATH)

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
                    "meter": [0, 0],
                    "day_week": [4, 4],
                    "hour_datetime": [1, 0],
                }
            )
            split_df.to_csv(train_path, index=False)
            split_df.to_csv(valid_path, index=False)
            split_df.to_csv(test_path, index=False)

            train_df, valid_df, test_df = module.load_preprocessed_splits(
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
            )

            self.assertTrue(pd.api.types.is_datetime64_any_dtype(train_df["timestamp"]))
            self.assertEqual(train_df["timestamp"].tolist(), sorted(train_df["timestamp"].tolist()))
            self.assertEqual(str(train_df["meter_reading"].dtype), "float32")
            self.assertEqual(valid_df.columns.tolist(), train_df.columns.tolist())
            self.assertEqual(test_df.columns.tolist(), train_df.columns.tolist())

            bad_valid_path = root / "bad_valid.csv"
            split_df.drop(columns=["day_week"]).to_csv(bad_valid_path, index=False)
            with self.assertRaisesRegex(ValueError, "valid split columns"):
                module.load_preprocessed_splits(
                    train_path=train_path,
                    valid_path=bad_valid_path,
                    test_path=test_path,
                )

    def test_hierarchical_profile_predictions_use_ordered_fallbacks(self) -> None:
        module = load_module("historical_profile_common", COMMON_PATH)

        train_df = pd.DataFrame(
            {
                "building_id": [1, 1, 1, 2],
                "meter": [0, 0, 0, 0],
                "day_week": [1, 1, 2, 1],
                "hour_datetime": [8, 8, 9, 8],
                "meter_reading": [0.2, 0.4, 0.6, 0.8],
            }
        )
        predict_df = pd.DataFrame(
            {
                "building_id": [1, 1, 3, 9],
                "meter": [0, 0, 0, 7],
                "day_week": [1, 5, 1, 0],
                "hour_datetime": [8, 8, 8, 0],
                "meter_reading": [0.0, 0.0, 0.0, 0.0],
            }
        )

        model = module.fit_historical_profile(train_df)
        predictions, levels = module.predict_with_historical_profile(model, predict_df)

        np.testing.assert_allclose(predictions, np.array([0.3, 0.3, 0.46666667, 0.5], dtype=np.float32))
        self.assertEqual(
            levels,
            [
                "building_meter_weekday_hour",
                "building_meter_hour",
                "meter_weekday_hour",
                "global_mean",
            ],
        )

    def test_evaluate_predictions_matches_expected_metrics(self) -> None:
        module = load_module("historical_profile_common", COMMON_PATH)

        metrics = module.evaluate_predictions(
            y_true=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            y_pred=np.array([1.0, 2.5, 2.0], dtype=np.float32),
        )

        self.assertAlmostEqual(metrics["mse"], 0.4166666667)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.4166666667))
        self.assertAlmostEqual(metrics["r2"], 0.375)
        self.assertAlmostEqual(metrics["smape"], 20.740740, places=5)
        expected_rmsle = np.sqrt(
            np.mean(
                (
                    np.log1p(np.array([1.0, 2.5, 2.0], dtype=np.float32))
                    - np.log1p(np.array([1.0, 2.0, 3.0], dtype=np.float32))
                )
                ** 2
            )
        )
        self.assertAlmostEqual(metrics["rmsle"], expected_rmsle)
        self.assertEqual(metrics["evaluated_row_count"], 3)
        self.assertEqual(metrics["skipped_row_count"], 0)

    def test_inverse_transform_target_restores_raw_meter_reading(self) -> None:
        module = load_module("historical_profile_common", COMMON_PATH)

        raw_values = module.inverse_transform_target(
            values=np.array([0.0, 1.0], dtype=np.float32),
            target_log1p_min=0.0,
            target_log1p_max=np.log1p(9.0),
        )

        np.testing.assert_allclose(raw_values, np.array([0.0, 9.0], dtype=np.float32), rtol=1e-6)

    def test_evaluate_split_uses_raw_meter_reading_space(self) -> None:
        module = load_module("historical_profile_common", COMMON_PATH)

        train_df = pd.DataFrame(
            {
                "building_id": [1],
                "meter": [0],
                "day_week": [0],
                "hour_datetime": [0],
                "meter_reading": [0.0],
            }
        )
        split_df = pd.DataFrame(
            {
                "building_id": [1],
                "meter": [0],
                "day_week": [0],
                "hour_datetime": [0],
                "meter_reading": [1.0],
            }
        )

        model = module.fit_historical_profile(train_df)
        metrics = module.evaluate_split(
            model=model,
            split_df=split_df,
            target_log1p_min=0.0,
            target_log1p_max=np.log1p(9.0),
        )

        self.assertAlmostEqual(metrics["mse"], 81.0)
        self.assertAlmostEqual(metrics["mae"], 9.0)
        self.assertAlmostEqual(metrics["rmse"], 9.0)
        self.assertAlmostEqual(metrics["smape"], 200.0)
        self.assertAlmostEqual(metrics["rmsle"], np.log1p(9.0), places=6)
        self.assertEqual(metrics["evaluated_row_count"], 1)
        self.assertEqual(metrics["skipped_row_count"], 0)

    def test_build_run_summary_includes_metrics_and_artifact_paths(self) -> None:
        module = load_module("historical_profile_common", COMMON_PATH)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = module.MeterConfig(meter_id=3, meter_name="hot_water")
            paths = module.build_meter_paths(
                config=config,
                preprocessed_data_dir=root / "preprocessed_data",
                output_root_dir=root / "outputs",
            )
            split_summary = pd.DataFrame(
                [
                    {"split": "train", "start_timestamp": "a", "end_timestamp": "b", "row_count": 8},
                    {"split": "valid", "start_timestamp": "c", "end_timestamp": "d", "row_count": 1},
                    {"split": "test", "start_timestamp": "e", "end_timestamp": "f", "row_count": 1},
                ]
            )

            summary = module.build_run_summary(
                config=config,
                paths=paths,
                preprocessing_summary={
                    "target_preprocess": "log1p_then_train_minmax",
                    "target_log1p_min": 0.0,
                    "target_log1p_max": 11.5,
                },
                split_summary=split_summary,
                profile_levels=["building_meter_weekday_hour", "global_mean"],
                validation_metrics={
                    "mse": 1.0,
                    "mae": 2.0,
                    "r2": 3.0,
                    "smape": 4.0,
                    "rmse": 5.0,
                    "rmsle": 5.5,
                    "evaluated_row_count": 6,
                    "skipped_row_count": 0,
                },
                test_metrics={
                    "mse": 6.0,
                    "mae": 7.0,
                    "r2": 8.0,
                    "smape": 9.0,
                    "rmse": 10.0,
                    "rmsle": 10.5,
                    "evaluated_row_count": 7,
                    "skipped_row_count": 0,
                },
                train_row_count=8,
                valid_row_count=1,
                test_row_count=1,
                profile_table_row_count=12,
                profile_tables_path=paths.profile_tables_path,
                model_params_path=paths.model_params_path,
                validation_metrics_path=paths.validation_metrics_path,
                test_metrics_path=paths.test_metrics_path,
                split_summary_path=paths.split_summary_path,
            )

            self.assertEqual(summary["meter_id"], 3)
            self.assertEqual(summary["meter_name"], "hot_water")
            self.assertEqual(summary["target_log1p_max"], 11.5)
            self.assertEqual(summary["profile_strategy"], "hierarchical_mean")
            self.assertEqual(summary["metric_scale"], "raw_meter_reading_after_inverse_transform")
            self.assertEqual(summary["test_rmse"], 10.0)
            self.assertEqual(summary["test_rmsle"], 10.5)
            self.assertEqual(summary["validation_rmsle"], 5.5)
            self.assertEqual(summary["test_skipped_row_count"], 0)
            self.assertEqual(summary["profile_tables_path"], str(paths.profile_tables_path))


if __name__ == "__main__":
    unittest.main()
