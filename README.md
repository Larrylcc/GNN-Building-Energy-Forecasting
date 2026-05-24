# ASHRAE Building Energy Prediction

[中文说明](README.zh-CN.md)

This repository contains an end-to-end graduation project for the ASHRAE Great Energy Predictor III dataset. It builds a local energy-consumption forecasting workflow around data preprocessing, target transformation, time-based validation, three model families, and a FastAPI dashboard for presenting the dataset, preprocessing logic, model results, and comparisons.

The main task focuses on electricity-meter prediction (`meter == 0`). The project also contains extended experiments for chilled water (`meter == 1`), steam (`meter == 2`), and hot water (`meter == 3`).

## Project Scope

- Merge ASHRAE `train.csv`, `building_metadata.csv`, and `weather_train.csv`.
- Engineer building, weather, calendar, missingness, wind-direction, and Beaufort-scale features.
- Screen abnormal buildings and split data by timestamp into train/validation/test sets.
- Transform the target with `log1p`, then apply train-set min-max scaling.
- Train and evaluate implemented XGBoost, GRU, and STGNN/GCN-GRU models, while documenting a broader seven-baseline comparison ladder for the thesis.
- Save model artifacts, training summaries, metrics, feature importance, learned graph outputs, and dashboard JSON files.

## Dataset and Environment

The project is configured for the local ASHRAE dataset path:

```text
F:\Desktop\Final\USTB-graduation-project\ASHRAE-Great Energy Predictor III\ashrae-energy-prediction
```

Expected raw files:

- `train.csv`
- `building_metadata.csv`
- `weather_train.csv`

The prepared Conda environment is:

```text
graduation_project_env
F:\applications\Miniconda\envs\graduation_project_env\bin
```

Main Python dependencies used by the project include `pandas`, `numpy`, `tqdm`, `scikit-learn`, `xgboost`, `torch`, `fastapi`, `uvicorn`, and `jinja2`.

## Repository Layout

```text
.
├── data_preprocess/        # Raw-data merge, screening, target transform, split generation
├── XGBoost/                # XGBoost baseline and other-meter experiments
├── GRU/                    # Sequence GRU baseline and other-meter experiments
├── STGNN/                  # Learnable graph GCN + GRU experiments
├── preprocessed_data/      # Generated train/valid/test CSVs and preprocessing summaries
├── web/                    # FastAPI dashboard, templates, static assets, small JSON data
├── test/                   # Unit tests for preprocessing and model helper logic
├── *.ipynb                 # Exploratory notebooks
└── info.md                 # Local path and project-note reference
```

## Data Pipeline

1. `data_preprocess/data_preprocess.py`
   - Loads raw ASHRAE training, building, and weather files.
   - Joins building metadata on `building_id`.
   - Joins weather data on `site_id` and `timestamp`.
   - Encodes `primary_use`, computes building `age`, converts wind direction to 16 compass bins, maps wind speed to Beaufort scale, and extracts calendar features.
   - Preserves missing values and adds `_missing` indicator columns for fields with missing values.
   - Writes `preprocessed_data/preprocessed_train.csv`.

2. `data_preprocess/data_screening.py`
   - Keeps electricity-meter rows (`meter == 0`).
   - Removes pre-identified abnormal building IDs.
   - Writes `preprocessed_data/screened_preprocessed_train.csv`.

3. `data_preprocess/data_log1p_minmax.py`
   - Applies `log1p` to `meter_reading`.
   - Fits min-max scaling on the training time range only.
   - Splits unique timestamps into 80% train, 10% validation, and 10% test.
   - Writes `log1p_minmax_train.csv`, `log1p_minmax_valid.csv`, `log1p_minmax_test.csv`, and `log1p_minmax_summary.json`.

4. `data_preprocess/data_other_meters_log1p_minmax.py`
   - Repeats the target transform and time split for meters 1, 2, and 3.
   - Writes outputs under `preprocessed_data/meter_1`, `preprocessed_data/meter_2`, and `preprocessed_data/meter_3`.

## Baseline Selection Rationale

The thesis baselines are organized as a progressive comparison from simple statistical regularities to spatiotemporal graph neural networks. This makes the final optimized model compete not only with a weak reference method, but also with increasingly stronger model families: statistical profiles, shallow linear models, strong tabular machine-learning models, pure temporal deep-learning models, and graph-based spatiotemporal models.

In this repository, implemented experiment artifacts are present for XGBoost, GRU, and STGNN. Historical Profile, Ridge Regression, LightGBM, and TCN are documented here as part of the broader baseline design for the thesis comparison.

| Baseline | Technical route | Why it is included and what limitation it exposes |
| --- | --- | --- |
| Historical Profile | Basic statistical and historical-pattern method. It predicts from grouped historical means or medians by building, meter, hour, weekday, or similar periodic keys. | It tests how far repeated historical patterns can go. It cannot actively use weather shifts, building attributes, or nonlinear relationships, and it is weak for missing history, abnormal consumption, and cold-start buildings. |
| Ridge Regression | Linear supervised learning and shallow statistical modeling. It uses L2-regularized linear regression over building, weather, and time features. | It gives an interpretable linear reference, but its expressive power is limited. Categorical features and interactions require manual encoding, and it struggles with nonlinear usage patterns, lag effects, and temporal dependence. |
| XGBoost | Gradient-boosted tree model from traditional machine learning. It learns nonlinear feature relationships through tree ensembles. | It is a strong tabular baseline, but it depends heavily on feature engineering and does not naturally model continuous sequence state, long-range temporal dependence, cross-building relationships, or spatial correlation. |
| LightGBM | Efficient gradient-boosted decision-tree route for large tabular data. It is fast and strong on nonlinear, categorical, and statistical features. | Like XGBoost, it remains a tabular sample-level model. It lacks explicit temporal memory and direct modeling of relationships between buildings or sites. |
| GRU | Recurrent neural network route for pure temporal deep learning. It uses gating to model dependencies within historical windows. | It can represent single-sequence temporal dependence, but training is sequential, long-range memory can still decay, and it does not explicitly use spatial or graph relationships between buildings or sites. |
| TCN | Convolutional sequence-modeling route for pure temporal deep learning. It uses causal and dilated convolutions to capture longer historical contexts. | It is more parallel than GRU and has a stable receptive field, but it still mainly focuses on single-node or single-sequence temporal patterns. It does not directly model building relationships, and its window and convolution design require tuning. |
| STGNN | Spatiotemporal graph neural-network route. It combines graph structure for building or node relationships with GRU or another temporal module for dynamics. | It is the predecessor of the optimized final model. It introduces spatiotemporal relations, but the original version may have insufficient graph construction, coarse adjacency representation, limited temporal modeling capacity, or weak dynamic-relationship modeling, leaving clear room for structural improvement. |

The same baselines can be grouped as a comparison ladder:

| Level | Models |
| --- | --- |
| Basic statistical regularities | Historical Profile |
| Shallow interpretable models | Ridge Regression |
| Strong tabular machine learning | XGBoost, LightGBM |
| Pure temporal deep learning | GRU, TCN |
| Spatiotemporal graph deep learning | STGNN |

## Models

### XGBoost

Scripts:

- `XGBoost/xgboost_baseline_log1p-minmax.py`
- `XGBoost/xgboost_other_meters_log1p-minmax.py`

The XGBoost pipeline represents the strong tabular machine-learning route. It trains boosted decision trees, exports feature importance, and reports validation/test metrics on the transformed target scale.

### GRU

Scripts:

- `GRU/gru_baseline_auto-rolling.py`
- `GRU/gru_other_meters_auto-rolling.py`

The GRU pipeline represents the pure temporal recurrent-learning route. During validation and test evaluation, it performs auto-rolling prediction so recent predicted target values can feed later windows.

### STGNN / GCN-GRU

Scripts:

- `STGNN/stgnn_gcn_gru_auto-rolling.py`
- `STGNN/stgnn_other_meters_gcn_gru_auto-rolling.py`

The STGNN pipeline builds dense timestamp-by-building tensors, learns a graph from node embeddings, applies GCN layers over the learned adjacency matrix, and feeds temporal representations into a GRU. It saves learned adjacency and top-k edge artifacts in addition to model and metric files.

## Current Results

All metrics below are calculated on the transformed target scale: `log1p(meter_reading)` followed by train-set min-max scaling. They are not raw `meter_reading` metrics.

### Electricity Baseline (`meter == 0`)

| Model | Validation RMSE | Test RMSE | Test MAE | Test R2 | Test SMAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.081779 | 0.082954 | 0.053966 | 0.759591 | 16.004551 |
| GRU | 0.105317 | 0.100397 | 0.068428 | 0.641110 | 19.022067 |
| STGNN | 0.098942 | 0.074652 | 0.050243 | 0.805375 | 17.812272 |

For the electricity baseline, STGNN has the best test MSE/MAE/RMSE/R2, while XGBoost has the best test SMAPE and validation metrics.

### Other Meters

| Model | Meter | Test RMSE | Test MAE | Test R2 | Test SMAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| XGBoost | 1 chilled_water | 0.115432 | 0.078847 | 0.583520 | 71.632767 |
| GRU | 1 chilled_water | 0.154703 | 0.119630 | 0.249908 | 94.587402 |
| STGNN | 1 chilled_water | 0.170315 | 0.145925 | 0.094255 | 92.380926 |
| XGBoost | 2 steam | 0.093814 | 0.071584 | 0.468174 | 30.457222 |
| GRU | 2 steam | 0.146466 | 0.105872 | -0.301018 | 33.226240 |
| STGNN | 2 steam | 0.182987 | 0.141270 | -1.022973 | 58.358830 |
| XGBoost | 3 hot_water | 0.196946 | 0.160810 | 0.117737 | 76.101013 |
| GRU | 3 hot_water | 0.175723 | 0.114942 | 0.305309 | 67.413746 |
| STGNN | 3 hot_water | 0.197349 | 0.158705 | 0.114121 | 71.173619 |

## Running the Project

Activate the project environment and move to the workspace:

```powershell
conda activate graduation_project_env
cd F:\Desktop\Final\workspace
```

Run preprocessing:

```powershell
python data_preprocess\data_preprocess.py
python data_preprocess\data_screening.py
python data_preprocess\data_log1p_minmax.py
python data_preprocess\data_other_meters_log1p_minmax.py
```

Run model training:

```powershell
python XGBoost\xgboost_baseline_log1p-minmax.py
python XGBoost\xgboost_other_meters_log1p-minmax.py
python GRU\gru_baseline_auto-rolling.py
python GRU\gru_other_meters_auto-rolling.py
python STGNN\stgnn_gcn_gru_auto-rolling.py
python STGNN\stgnn_other_meters_gcn_gru_auto-rolling.py
```

Run the STGNN smoke tests when you only want to verify tensor construction and model flow:

```powershell
python STGNN\stgnn_gcn_gru_auto-rolling.py --smoke-test
python STGNN\stgnn_other_meters_gcn_gru_auto-rolling.py --smoke-test
```

Run unit tests:

```powershell
python -m unittest discover -s test -p "test_*.py"
```

## Web Dashboard

The FastAPI dashboard reads compact JSON files under `web/data` instead of loading large CSV files at request time.

Start the dashboard:

```powershell
python -m uvicorn web.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

Available pages:

- `/dataset`
- `/preprocessing`
- `/xgboost`
- `/gru`
- `/stgnn`
- `/comparison`

The helper `web/scripts/build_web_data.py` refreshes the dataset, preprocessing, and XGBoost overview JSON files. The existing GRU, STGNN, and comparison JSON files are already present under `web/data`.

## Main Artifacts

- Preprocessed electricity splits: `preprocessed_data/log1p_minmax_*.csv`
- Other-meter splits: `preprocessed_data/meter_*/log1p_minmax_*.csv`
- XGBoost outputs: `XGBoost/*_outputs*/`
- GRU outputs: `GRU/*_outputs/`
- STGNN outputs: `STGNN/*_outputs/`
- Dashboard data: `web/data/*.json`
- Dashboard templates: `web/templates/*.html`

## Notes and Limitations

- Several scripts contain absolute local paths. If the workspace or dataset path changes, update the constants at the top of the corresponding scripts.
- Generated CSVs and model artifacts can be large. The `.gitignore` excludes `preprocessed_data/*`.
- Metrics are reported on the transformed target scale. Use the stored `target_log1p_min` and `target_log1p_max` values if inverse-transforming predictions back toward the original reading scale.
- The web dashboard is a presentation layer; model training and preprocessing are handled by the standalone scripts.
