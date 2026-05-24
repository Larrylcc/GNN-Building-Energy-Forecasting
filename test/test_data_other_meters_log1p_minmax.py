import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data_preprocess.data_other_meters_log1p_minmax import preprocess_meter_log1p_minmax


class OtherMetersLog1pMinmaxTest(unittest.TestCase):
    def test_preprocess_single_meter_outputs_filtered_splits_and_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "preprocessed_train.csv"
            output_dir = root / "meter_1"

            rows = []
            for hour in range(10):
                rows.append(
                    {
                        "building_id": 100,
                        "meter": 1,
                        "timestamp": f"2016-01-01 {hour:02d}:00:00",
                        "meter_reading": float(hour + 1),
                        "feature": hour,
                    }
                )
            rows.extend(
                [
                    {
                        "building_id": 200,
                        "meter": 0,
                        "timestamp": "2016-01-01 00:00:00",
                        "meter_reading": 99.0,
                        "feature": 99,
                    },
                    {
                        "building_id": 803,
                        "meter": 1,
                        "timestamp": "2016-01-01 00:00:00",
                        "meter_reading": 99.0,
                        "feature": 99,
                    },
                ]
            )
            pd.DataFrame(rows).to_csv(source_path, index=False)

            summary = preprocess_meter_log1p_minmax(
                meter_id=1,
                meter_name="chilled_water",
                source_data_path=source_path,
                output_dir=output_dir,
            )

            train_df = pd.read_csv(output_dir / "log1p_minmax_train.csv", parse_dates=["timestamp"])
            valid_df = pd.read_csv(output_dir / "log1p_minmax_valid.csv", parse_dates=["timestamp"])
            test_df = pd.read_csv(output_dir / "log1p_minmax_test.csv", parse_dates=["timestamp"])
            saved_summary = json.loads((output_dir / "log1p_minmax_summary.json").read_text(encoding="utf-8"))

            combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

            self.assertEqual(summary["meter_id"], 1)
            self.assertEqual(saved_summary["meter_name"], "chilled_water")
            self.assertEqual(set(combined_df["meter"]), {1})
            self.assertNotIn(803, set(combined_df["building_id"]))
            self.assertEqual(summary["total_row_count"], 10)
            self.assertEqual((len(train_df), len(valid_df), len(test_df)), (8, 1, 1))
            self.assertLess(train_df["timestamp"].max(), valid_df["timestamp"].min())
            self.assertLess(valid_df["timestamp"].max(), test_df["timestamp"].min())
            self.assertEqual(
                saved_summary["total_row_count"],
                saved_summary["train_row_count"] + saved_summary["valid_row_count"] + saved_summary["test_row_count"],
            )


if __name__ == "__main__":
    unittest.main()
