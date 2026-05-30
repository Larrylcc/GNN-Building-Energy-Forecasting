import os
import json

metrics = ["mse", "mae", "r2", "rmse", "rmsle"]
models = ["My Model", "GRU", "Ridge", "TCN", "XGBoost"]

paths = {
    0: {
        "My Model": r"F:\Desktop\Final\workspace\STGNN\stgnn_gcn_gru_auto-rolling_outputs\stgnn_test_metrics.json",
        "GRU": r"F:\Desktop\Final\workspace\GRU\Auto-rolling-Backup\gru_baseline_auto-rolling_outputs\gru_test_metrics.json",
        "Ridge": r"F:\Desktop\Final\workspace\Ridge Regression\ridge_regression_outputs\meter_0\ridge_test_metrics.json",
        "TCN": r"F:\Desktop\Final\workspace\TCN\tcn_baseline_outputs\meter_0\tcn_test_metrics.json",
        "XGBoost": r"F:\Desktop\Final\workspace\XGBoost\xgboost_baseline_outputs\meter_0\xgboost_test_metrics.json",
    },
    1: {
        "My Model": r"F:\Desktop\Final\workspace\STGNN\static_adaptive_stgnn_other_meters_auto-rolling_outputs\meter_1\static_adaptive_zero_inflated_stgnn_test_metrics.json",
        "GRU": r"F:\Desktop\Final\workspace\GRU\Auto-rolling-Backup\gru_other_meters_auto-rolling_outputs\meter_1\gru_test_metrics.json",
        "Ridge": r"F:\Desktop\Final\workspace\Ridge Regression\ridge_regression_outputs\meter_1\ridge_test_metrics.json",
        "TCN": r"F:\Desktop\Final\workspace\TCN\tcn_baseline_outputs\meter_1\tcn_test_metrics.json",
        "XGBoost": r"F:\Desktop\Final\workspace\XGBoost\xgboost_baseline_outputs\meter_1\xgboost_test_metrics.json",
    },
    2: {
        "My Model": r"F:\Desktop\Final\workspace\STGNN\static_adaptive_stgnn_other_meters_auto-rolling_outputs\meter_2\static_adaptive_zero_inflated_stgnn_test_metrics.json",
        "GRU": r"F:\Desktop\Final\workspace\GRU\Auto-rolling-Backup\gru_other_meters_auto-rolling_outputs\meter_2\gru_test_metrics.json",
        "Ridge": r"F:\Desktop\Final\workspace\Ridge Regression\ridge_regression_outputs\meter_2\ridge_test_metrics.json",
        "TCN": r"F:\Desktop\Final\workspace\TCN\tcn_baseline_outputs\meter_2\tcn_test_metrics.json",
        "XGBoost": r"F:\Desktop\Final\workspace\XGBoost\xgboost_baseline_outputs\meter_2\xgboost_test_metrics.json",
    },
    3: {
        "My Model": r"F:\Desktop\Final\workspace\STGNN\static_adaptive_stgnn_other_meters_auto-rolling_outputs\meter_3\static_adaptive_zero_inflated_stgnn_test_metrics.json",
        "GRU": r"F:\Desktop\Final\workspace\GRU\Auto-rolling-Backup\gru_other_meters_auto-rolling_outputs\meter_3\gru_test_metrics.json",
        "Ridge": r"F:\Desktop\Final\workspace\Ridge Regression\ridge_regression_outputs\meter_3\ridge_test_metrics.json",
        "TCN": r"F:\Desktop\Final\workspace\TCN\tcn_baseline_outputs\meter_3\tcn_test_metrics.json",
        "XGBoost": r"F:\Desktop\Final\workspace\XGBoost\xgboost_baseline_outputs\meter_3\xgboost_test_metrics.json",
    }
}

meter_names = {
    0: "电表 (Electricity - Meter 0)",
    1: "冷水表 (Chilled Water - Meter 1)",
    2: "蒸汽表 (Steam - Meter 2)",
    3: "热水表 (Hot Water - Meter 3)"
}

for meter_id, name in meter_names.items():
    print(f"### {name}")
    print("| Metric | " + " | ".join(models) + " |")
    print("|---" + "|---" * len(models) + "|")
    
    # load data
    data = {}
    for model in models:
        path = paths[meter_id][model]
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                if model == "My Model" and meter_id in [1, 2, 3]:
                    if "thresholded_mse" in d:
                        data[model] = {m: d[f"thresholded_{m}"] for m in metrics}
                    else:
                        data[model] = {m: d[m] for m in metrics}
                else:
                    data[model] = {m: d.get(m, "N/A") for m in metrics}
        else:
            data[model] = {m: "N/A" for m in metrics}

    for m in metrics:
        row = [m.upper()]
        
        # find best value
        best_val = None
        for model in models:
            val = data[model][m]
            if val != "N/A":
                if best_val is None:
                    best_val = val
                else:
                    if m.upper() == "R2":
                        best_val = max(best_val, val)
                    else:
                        best_val = min(best_val, val)
                        
        for model in models:
            val = data[model][m]
            if val == "N/A":
                row.append("N/A")
            elif val == best_val:
                row.append(f"**{val:.4f}**")
            else:
                row.append(f"{val:.4f}")
        print("| " + " | ".join(row) + " |")
    print("\n")
