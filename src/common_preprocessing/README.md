# Common Preprocessing

该模块将 ASHRAE 原始数据统一处理为可复用中间数据，供不同模型共享。

## 输出

默认输出目录：`data/processed/common/`

- `common_train.csv` / `common_train.parquet`
- `common_valid.csv` / `common_valid.parquet`
- `common_test.csv` / `common_test.parquet`
- `common_schema.json`

## 主要处理步骤

1. 读取原始 `train/test/weather/building_metadata`。
2. 进行表连接，补齐建筑静态与天气特征。
3. 提取时间特征（month/week/hour/day 等）。
4. 清洗天气缺失值与异常值（如 `precip_depth_1_hr < 0`）。
5. 构造派生特征（`age`、`wind_direction_bucket`、`beaufort_scale`）。
6. 按时间切分 train/valid（默认 valid_ratio=0.2）。
7. 生成 `target_log1p`（train/valid）。
8. 按统一列顺序和类型落盘。

## CLI

```bash
python src/common_preprocessing/run_common_preprocessing.py \
  --data-root data \
  --output-dir data/processed/common \
  --valid-ratio 0.2 \
  --output-formats csv parquet \
  --random-seed 42
```
