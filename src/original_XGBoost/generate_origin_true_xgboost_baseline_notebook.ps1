function Split-NotebookSource {
    param([string]$Text)

    $normalized = $Text -replace "`r`n", "`n"
    $parts = $normalized -split "`n", 0, "SimpleMatch"
    $result = @()

    for ($i = 0; $i -lt $parts.Count; $i++) {
        if ($i -lt $parts.Count - 1) {
            $result += ($parts[$i] + "`n")
        }
        else {
            $result += $parts[$i]
        }
    }

    return $result
}

function New-MarkdownCell {
    param([string]$Text)

    return [ordered]@{
        cell_type = "markdown"
        metadata = @{}
        source = Split-NotebookSource $Text
    }
}

function New-CodeCell {
    param([string]$Text)

    return [ordered]@{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = Split-NotebookSource $Text
    }
}

$cells = @()

$cells += New-MarkdownCell @"
# True XGBoost GPU Baseline for ASHRAE

This notebook is the XGBoost baseline that mirrors the LightGBM baseline structure.

Key points:

- Predict the real target `meter_reading`.
- Train on `log1p(meter_reading)`.
- Use `XGBoost` with GPU acceleration for the Windows + RTX 3070 environment.
- Keep the main merge and feature engineering flow aligned with the LightGBM baseline for fair comparison.
"@

$cells += New-CodeCell @"
from pathlib import Path

from origin_true_xgboost_baseline import (
    CATEGORICAL_COLS,
    DATA_ROOT,
    FEATURE_COLS,
    NUMERICAL_COLS,
    OUTPUT_DIR,
    build_xgb_params,
    main,
)

DATA_ROOT, OUTPUT_DIR
"@

$cells += New-CodeCell @"
print(f"Feature count: {len(FEATURE_COLS)}")
print("Categorical columns:")
print(CATEGORICAL_COLS)
print("Numerical columns:")
print(NUMERICAL_COLS)
print("GPU params:")
print(build_xgb_params())
"@

$cells += New-CodeCell @"
artifacts = main(data_root=DATA_ROOT, output_dir=OUTPUT_DIR)
artifacts
"@

$cells += New-CodeCell @"
import pandas as pd

metrics_df = pd.read_csv(Path(artifacts["metrics_path"]))
importance_df = pd.read_csv(Path(artifacts["importance_path"]))

metrics_df
"@

$cells += New-CodeCell @"
importance_df.head(20)
"@

$notebook = [ordered]@{
    cells = $cells
    metadata = [ordered]@{
        kernelspec = [ordered]@{
            display_name = "Python 3"
            language = "python"
            name = "python3"
        }
        language_info = [ordered]@{
            name = "python"
            version = "3.10"
        }
    }
    nbformat = 4
    nbformat_minor = 5
}

$outputPath = Join-Path $PSScriptRoot "origin_true_xgboost_baseline.ipynb"
$notebook | ConvertTo-Json -Depth 100 | Set-Content -Path $outputPath -Encoding UTF8
Write-Output $outputPath
