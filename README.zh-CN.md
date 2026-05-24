# ASHRAE 建筑能耗预测项目

[English README](README.md)

本仓库是围绕 ASHRAE Great Energy Predictor III 数据集完成的建筑能耗预测毕业设计项目。项目覆盖从原始数据预处理、目标变换、时间序列切分，到 XGBoost、GRU、STGNN/GCN-GRU 建模评估，以及 FastAPI 可视化展示的完整流程。

主任务是电表预测（`meter == 0`）。仓库中也包含冷水（`meter == 1`）、蒸汽（`meter == 2`）、热水（`meter == 3`）三个其他仪表类型的扩展实验。

## 项目范围

- 合并 ASHRAE 的 `train.csv`、`building_metadata.csv`、`weather_train.csv`。
- 构建建筑、天气、时间、缺失值标记、风向编码、蒲福风级等特征。
- 剔除异常建筑，并按时间顺序划分训练集、验证集、测试集。
- 对目标值先做 `log1p`，再使用训练集目标范围做 min-max 归一化。
- 训练并评估已实现的 XGBoost、GRU、STGNN/GCN-GRU 模型，同时为论文整理更完整的七类 baseline 递进对照。
- 保存模型文件、训练摘要、指标、特征重要性、学习得到的图结构和 Web 展示 JSON。

## 数据集与环境

项目当前使用的本地 ASHRAE 数据集路径为：

```text
F:\Desktop\Final\USTB-graduation-project\ASHRAE-Great Energy Predictor III\ashrae-energy-prediction
```

期望包含的原始文件：

- `train.csv`
- `building_metadata.csv`
- `weather_train.csv`

已经准备好的 Conda 环境为：

```text
graduation_project_env
F:\applications\Miniconda\envs\graduation_project_env\bin
```

项目主要依赖包括 `pandas`、`numpy`、`tqdm`、`scikit-learn`、`xgboost`、`torch`、`fastapi`、`uvicorn`、`jinja2`。

## 仓库结构

```text
.
├── data_preprocess/        # 原始数据合并、筛选、目标变换、数据切分
├── XGBoost/                # XGBoost 基线与其他仪表实验
├── GRU/                    # 序列 GRU 基线与其他仪表实验
├── STGNN/                  # 可学习图 GCN + GRU 实验
├── preprocessed_data/      # 生成的 train/valid/test CSV 和预处理摘要
├── web/                    # FastAPI 展示系统、模板、静态资源、小型 JSON 数据
├── test/                   # 预处理和模型辅助逻辑的单元测试
├── *.ipynb                 # 探索性 Notebook
└── info.md                 # 本地路径和项目说明参考
```

## 数据处理流程

1. `data_preprocess/data_preprocess.py`
   - 读取 ASHRAE 原始训练表、建筑元数据和天气表。
   - 按 `building_id` 拼接建筑元数据。
   - 按 `site_id` 和 `timestamp` 拼接天气数据。
   - 编码 `primary_use`，计算建筑 `age`，将风向映射到 16 个方向编码，将风速映射到蒲福风级，并提取时间特征。
   - 保留原始缺失值，同时为存在缺失的字段新增 `_missing` 二值标记列。
   - 输出 `preprocessed_data/preprocessed_train.csv`。

2. `data_preprocess/data_screening.py`
   - 仅保留电表样本（`meter == 0`）。
   - 剔除预先识别的异常建筑 ID。
   - 输出 `preprocessed_data/screened_preprocessed_train.csv`。

3. `data_preprocess/data_log1p_minmax.py`
   - 对 `meter_reading` 执行 `log1p` 变换。
   - 仅基于训练时间段拟合 min-max 归一化参数。
   - 按唯一时间戳划分 80% 训练集、10% 验证集、10% 测试集。
   - 输出 `log1p_minmax_train.csv`、`log1p_minmax_valid.csv`、`log1p_minmax_test.csv` 和 `log1p_minmax_summary.json`。

4. `data_preprocess/data_other_meters_log1p_minmax.py`
   - 对 `meter == 1/2/3` 分别执行目标变换和时间切分。
   - 输出到 `preprocessed_data/meter_1`、`preprocessed_data/meter_2`、`preprocessed_data/meter_3`。

## Baseline 选择理由

论文中的 baseline 设计可以定位为“从简单统计规律到时空图神经网络”的递进对照。这样最终优化模型不是只和一个较弱方法比较，而是依次面对统计方法、浅层线性模型、强表格机器学习模型、纯时序深度学习模型，以及时空图深度学习模型。

本仓库当前已经保留 XGBoost、GRU、STGNN 的实现脚本和实验产物。Historical Profile、Ridge Regression、LightGBM、TCN 在此作为论文 baseline 设计的一部分进行说明。

| Baseline | 代表的技术路线 | 选择理由与暴露出的缺陷 |
| --- | --- | --- |
| Historical Profile | 基础统计/历史模式方法。按建筑、表计、小时、星期等历史分组统计均值或中位数进行预测。 | 用来检验历史周期规律本身能达到的效果上限。它只能复用历史模式，无法主动利用天气变化、建筑属性变化和非线性关系；对缺失历史、异常用能、冷启动建筑较弱。 |
| Ridge Regression | 线性监督学习/浅层统计模型。用 L2 正则化线性回归建模建筑属性、天气、时间特征与能耗之间的关系。 | 提供可解释的线性参考，但表达能力有限，只能学习近似线性关系；类别特征和复杂交互需要人工编码，难以捕捉非线性、滞后效应和时序依赖。 |
| XGBoost | 传统机器学习中的梯度提升树模型。通过树模型集成学习非线性特征关系。 | 作为强表格模型基线，可以衡量特征工程加非线性树模型的能力；但它依赖特征工程，不能天然建模连续时间序列状态，对长时依赖、跨建筑关系和空间相关性表达不足。 |
| LightGBM | 高效梯度提升决策树路线。适合大规模表格数据，训练速度快，对非线性和类别/统计特征较强。 | 与 XGBoost 共同构成强表格模型对照；但本质仍是静态样本级映射，缺少显式时序记忆和建筑间关联建模。 |
| GRU | 纯时序深度学习中的循环神经网络路线。通过门控机制建模历史窗口中的时间依赖。 | 用来检验纯时序记忆是否足够。它能建模单序列时间依赖，但训练存在顺序计算瓶颈，长距离依赖仍可能衰减，也没有显式利用建筑之间、站点之间的空间或图结构关系。 |
| TCN | 纯时序深度学习中的卷积序列建模路线。通过因果卷积和膨胀卷积捕捉较长历史窗口。 | 作为 GRU 的并行卷积时序对照，感受野更稳定；但它仍主要关注单节点/单序列时间模式，不直接建模建筑间关联，窗口长度和卷积结构也需要调参。 |
| STGNN | 时空图神经网络路线。结合图结构建模建筑/节点间关系，并结合 GRU 或时序模块建模时间动态。 | 作为最终优化模型的前身，它能引入时空关系；但原始版本可能存在图构建不充分、邻接关系表达粗糙、时序模块能力有限、动态图关系不足等问题，为优化模型提供结构改进空间。 |

这些 baseline 可以进一步归纳为如下递进层次：

| 层次 | 模型 |
| --- | --- |
| 基础统计规律 | Historical Profile |
| 浅层可解释模型 | Ridge Regression |
| 强表格机器学习 | XGBoost、LightGBM |
| 纯时序深度学习 | GRU、TCN |
| 时空图深度学习 | STGNN |

## 模型方案

### XGBoost

脚本：

- `XGBoost/xgboost_baseline_log1p-minmax.py`
- `XGBoost/xgboost_other_meters_log1p-minmax.py`

XGBoost 方案代表强表格机器学习路线。它通过梯度提升树学习非线性特征关系，导出特征重要性，并在变换后的目标尺度上输出验证集和测试集指标。

### GRU

脚本：

- `GRU/gru_baseline_auto-rolling.py`
- `GRU/gru_other_meters_auto-rolling.py`

GRU 方案代表纯时序循环神经网络路线。验证和测试阶段采用 auto-rolling 预测方式，近期预测出的目标值会进入后续窗口。

### STGNN / GCN-GRU

脚本：

- `STGNN/stgnn_gcn_gru_auto-rolling.py`
- `STGNN/stgnn_other_meters_gcn_gru_auto-rolling.py`

STGNN 方案将数据组织为“时间戳 × 建筑节点”的稠密张量，通过节点嵌入学习图结构，在学习得到的邻接矩阵上执行 GCN，再将时间表示输入 GRU。除模型和指标外，该方案还保存学习到的邻接矩阵和 top-k 边文件。

## 当前实验结果

下表所有指标都在变换后的目标尺度上计算：`log1p(meter_reading)` 后接训练集 min-max 归一化。它们不是原始 `meter_reading` 尺度下的指标。

### 电表基线任务（`meter == 0`）

| 模型 | 验证 RMSE | 测试 RMSE | 测试 MAE | 测试 R2 | 测试 SMAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.081779 | 0.082954 | 0.053966 | 0.759591 | 16.004551 |
| GRU | 0.105317 | 0.100397 | 0.068428 | 0.641110 | 19.022067 |
| STGNN | 0.098942 | 0.074652 | 0.050243 | 0.805375 | 17.812272 |

在电表基线任务中，STGNN 的测试集 MSE/MAE/RMSE/R2 最优；XGBoost 的测试集 SMAPE 和验证集指标更优。

### 其他仪表任务

| 模型 | 仪表 | 测试 RMSE | 测试 MAE | 测试 R2 | 测试 SMAPE |
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

## 运行方式

激活项目环境并进入工作目录：

```powershell
conda activate graduation_project_env
cd F:\Desktop\Final\workspace
```

执行预处理：

```powershell
python data_preprocess\data_preprocess.py
python data_preprocess\data_screening.py
python data_preprocess\data_log1p_minmax.py
python data_preprocess\data_other_meters_log1p_minmax.py
```

执行模型训练：

```powershell
python XGBoost\xgboost_baseline_log1p-minmax.py
python XGBoost\xgboost_other_meters_log1p-minmax.py
python GRU\gru_baseline_auto-rolling.py
python GRU\gru_other_meters_auto-rolling.py
python STGNN\stgnn_gcn_gru_auto-rolling.py
python STGNN\stgnn_other_meters_gcn_gru_auto-rolling.py
```

如果只想快速检查 STGNN 的张量构造和模型流程，可运行 smoke test：

```powershell
python STGNN\stgnn_gcn_gru_auto-rolling.py --smoke-test
python STGNN\stgnn_other_meters_gcn_gru_auto-rolling.py --smoke-test
```

运行单元测试：

```powershell
python -m unittest discover -s test -p "test_*.py"
```

## Web 展示系统

FastAPI 展示系统只读取 `web/data` 下的小型 JSON 文件，运行时不会直接加载大型 CSV。

启动展示系统：

```powershell
python -m uvicorn web.app:app --reload
```

然后访问：

```text
http://127.0.0.1:8000
```

可用页面：

- `/dataset`
- `/preprocessing`
- `/xgboost`
- `/gru`
- `/stgnn`
- `/comparison`

辅助脚本 `web/scripts/build_web_data.py` 可刷新数据集、预处理、XGBoost 三个 overview JSON。现有 GRU、STGNN 和 comparison JSON 已经位于 `web/data` 目录下。

## 主要产物

- 电表预处理切分：`preprocessed_data/log1p_minmax_*.csv`
- 其他仪表切分：`preprocessed_data/meter_*/log1p_minmax_*.csv`
- XGBoost 输出：`XGBoost/*_outputs*/`
- GRU 输出：`GRU/*_outputs/`
- STGNN 输出：`STGNN/*_outputs/`
- 展示系统数据：`web/data/*.json`
- 展示系统模板：`web/templates/*.html`

## 注意事项与限制

- 部分脚本中包含本机绝对路径。如果工作目录或数据集目录发生变化，需要修改对应脚本顶部的常量。
- 生成的 CSV 和模型文件可能较大。当前 `.gitignore` 已排除 `preprocessed_data/*`。
- 指标是在变换后的目标尺度上计算的。如果要还原到原始读数尺度，需要结合保存的 `target_log1p_min` 和 `target_log1p_max`。
- Web 展示系统只是展示层；数据预处理和模型训练由独立脚本完成。
