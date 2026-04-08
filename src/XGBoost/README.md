# XGBoost 方案说明

## 1. 方案概述

本方案面向 ASHRAE 建筑能耗预测任务，采用 `XGBoost` 回归模型。

- 训练数据输入：公共预处理产物 `common_train/common_valid/common_test`
- 默认训练目标：`target_log1p = log1p(meter_reading)`
- 默认训练 objective：`reg:squarederror`
- 训练监控指标：`rmse`（log 空间）
- 主评估指标：在原始能耗尺度上计算 `MSE/MAE/R2/RMSE/RMSLE`

## 2. 目录结构

`src/XGBoost/` 下核心文件：

- `config.py`：XGBoost 配置与超参数
- `data_adapter.py`：读取公共数据并校验字段
- `trainer.py`：训练、早停、特征重要性、测试集分块预测
- `metrics.py`：统一指标计算函数
- `run_xgboost.py`：命令行入口脚本

## 3. 数据与特征

本方案不直接读取原始 `train.csv/test.csv`，而是读取公共数据模块输出的中间数据。

- 训练集：`data/processed/common/common_train.(csv|parquet)`
- 验证集：`data/processed/common/common_valid.(csv|parquet)`
- 测试集：`data/processed/common/common_test.(csv|parquet)`
- Schema：`data/processed/common/common_schema.json`

公共数据中包含：

- 建筑静态特征：`site_id/building_id/primary_use/square_feet/year_built/floor_count/age`
- 天气特征：`air_temperature/cloud_coverage/dew_temperature/precip_depth_1_hr/sea_level_pressure/wind_speed` 等
- 时间特征：`month/weekofyear/dayofyear/hour/dayofweek/day/week_of_month/is_weekend`
- 目标列（train/valid）：`meter_reading` + `target_log1p`

## 4. 训练流程

1. 读取 common train/valid/test。
2. 选择 `feature_columns` 构建 `DMatrix/QuantileDMatrix`。
3. 使用 train 训练，并在 valid 上 early stopping。
4. 对 train/valid/test 预测；若训练目标是 `target_log1p`，则通过 `expm1` 反变换回原始尺度。
5. 在原始尺度上计算并保存 `MSE/MAE/R2/RMSE/RMSLE`。
6. 输出结果到 `results/XGBoost/`。

## 5. 默认超参数

默认配置位于 `src/XGBoost/config.py`：

- `num_boost_round=3000`
- `early_stopping_rounds=100`
- `learning_rate(eta)=0.05`
- `max_depth=10`
- `min_child_weight=5`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_alpha=0.1`
- `reg_lambda=1.0`
- `max_bin=256`
- `seed=42`

GPU 开关：

- 默认 `use_gpu=True`
- 可通过 `--no-gpu` 切换到 CPU

## 6. 为什么默认使用 log1p+MSE 作为训练 loss

默认训练方式：

- 训练目标：`target_log1p = log1p(meter_reading)`
- Objective：`reg:squarederror`

选择原因：

- `meter_reading` 分布长尾明显，直接在原始尺度训练时容易被大值主导。
- `log1p` 变换可压缩长尾，提高训练稳定性和收敛速度。
- `reg:squarederror` 在 XGBoost 中成熟稳定，支持良好的 early stopping 行为。
- 与 `RMSLE` 的思想一致：都强调相对误差而非绝对误差。

## 7. 评估指标定义（原始尺度）

- `MSE = mean((y_true - y_pred)^2)`
- `MAE = mean(|y_true - y_pred|)`
- `R2 = 1 - SSE/SST`
- `RMSE = sqrt(MSE)`
- `RMSLE = sqrt(mean((log1p(y_true) - log1p(y_pred))^2))`

说明：实现中会将预测值裁剪到非负后再计算指标，保证物理含义一致。

## 8. 输出文件

`results/XGBoost/` 下输出：

- `metrics.json`：train/valid 的五项指标
- `metrics_by_split.csv`：按 split 展示指标
- `xgboost_eval_history.json`：训练过程 eval 历史
- `xgboost_feature_importance.csv`：按 gain 统计的特征重要性
- `xgboost_valid_predictions.csv`：验证集真实值与预测值
- `xgboost_submission.csv`：测试集预测（`row_id,meter_reading`）
- `xgboost_run_summary.json`：运行摘要与产物路径

## 9. 运行方式

先生成公共数据：

```bash
python src/common_preprocessing/run_common_preprocessing.py \
  --data-root data \
  --output-dir data/processed/common \
  --valid-ratio 0.2 \
  --output-formats csv parquet
```

再运行 XGBoost：

```bash
python src/XGBoost/run_xgboost.py \
  --common-data-dir data/processed/common \
  --output-dir results/XGBoost \
  --objective reg:squarederror \
  --eval-metric rmse \
  --use-gpu
```

如需 CPU：

```bash
python src/XGBoost/run_xgboost.py --no-gpu
```

## 10. 可扩展点

- 切换训练目标列：`--training-target-col meter_reading`
- 切换 objective/eval_metric：`--objective` / `--eval-metric`
- 调整早停与训练轮数：`--early-stopping-rounds` / `--num-boost-round`
- 调整 test 分块大小：`--test-chunk-size`

以上参数都集中在 `config.py` 与 CLI，可在不改动主流程逻辑的情况下替换实验配置。
