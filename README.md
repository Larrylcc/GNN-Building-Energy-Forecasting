# GNN Building Energy Forecasting

本仓库当前包含：

- 公共预处理流水线：`src/common_preprocessing/`
- XGBoost 方案：`src/XGBoost/`

## 快速开始

1. 生成公共中间数据：

```bash
python src/common_preprocessing/run_common_preprocessing.py \
  --data-root data \
  --output-dir data/processed/common \
  --valid-ratio 0.2 \
  --output-formats csv parquet
```

2. 运行 XGBoost：

```bash
python src/XGBoost/run_xgboost.py \
  --common-data-dir data/processed/common \
  --output-dir results/XGBoost \
  --use-gpu
```

XGBoost 详细方案说明见：`src/XGBoost/README.md`。
