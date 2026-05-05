from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WEB_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = WEB_DIR.parent
DEFAULT_DATA_ROOT = Path(
    r"F:\Desktop\Final\USTB-graduation-project\ASHRAE-Great Energy Predictor III\ashrae-energy-prediction"
)
PREPROCESSED_DATA_DIR = WORKSPACE_ROOT / "preprocessed_data"
XGBOOST_OUTPUT_DIR = WORKSPACE_ROOT / "XGBoost" / "xgboost_baseline_log1p-minmax_outputs"
DATASET_OUTPUT_PATH = WEB_DIR / "data" / "dataset_overview.json"
PREPROCESSING_OUTPUT_PATH = WEB_DIR / "data" / "preprocessing_overview.json"
XGBOOST_OUTPUT_PATH = WEB_DIR / "data" / "xgboost_overview.json"

METER_LABELS = {
    0: "electricity",
    1: "chilled water",
    2: "steam",
    3: "hot water",
}

FEATURE_ZH_NAMES = {
    "building_id": "建筑编号",
    "meter": "仪表类型",
    "timestamp": "时间戳",
    "meter_reading": "能耗读数目标值",
    "site_id": "站点编号",
    "primary_use": "建筑主要用途",
    "square_feet": "建筑面积",
    "year_built": "建造年份",
    "floor_count": "楼层数",
    "air_temperature": "空气温度",
    "cloud_coverage": "云量覆盖",
    "dew_temperature": "露点温度",
    "precip_depth_1_hr": "1小时降水深度",
    "sea_level_pressure": "海平面气压",
    "wind_direction": "风向编码",
    "wind_speed": "风速",
    "age": "建筑年龄",
    "beaufort_scale": "蒲福风级",
    "month_datetime": "月份",
    "weekofyear_datetime": "年内周序号",
    "dayofyear_datetime": "年内日序号",
    "hour_datetime": "小时",
    "day_week": "星期",
    "day_month_datetime": "月内日期",
    "week_month_datetime": "月内周序号",
    "age_missing": "建筑年龄缺失标记",
    "air_temperature_missing": "空气温度缺失标记",
    "beaufort_scale_missing": "蒲福风级缺失标记",
    "cloud_coverage_missing": "云量覆盖缺失标记",
    "dew_temperature_missing": "露点温度缺失标记",
    "floor_count_missing": "楼层数缺失标记",
    "precip_depth_1_hr_missing": "1小时降水深度缺失标记",
    "sea_level_pressure_missing": "海平面气压缺失标记",
    "wind_direction_missing": "风向编码缺失标记",
    "wind_speed_missing": "风速缺失标记",
    "year_built_missing": "建造年份缺失标记",
}

FIELD_GROUPS = [
    {
        "name": "train.csv",
        "description": "训练集主表，记录建筑、仪表类型、时间戳和目标能耗读数。",
        "fields": [
            {"name": "building_id", "type": "int", "description": "建筑编号，与建筑元数据表关联。"},
            {"name": "meter", "type": "int", "description": "仪表类型，0/1/2/3 分别代表电、冷水、蒸汽、热水。"},
            {"name": "timestamp", "type": "datetime", "description": "小时级时间戳。"},
            {"name": "meter_reading", "type": "float", "description": "目标变量，建筑能耗读数。"},
        ],
    },
    {
        "name": "building_metadata.csv",
        "description": "建筑静态属性表，提供建筑用途、面积、建造年份等信息。",
        "fields": [
            {"name": "site_id", "type": "int", "description": "站点编号，用于关联天气数据。"},
            {"name": "building_id", "type": "int", "description": "建筑编号。"},
            {"name": "primary_use", "type": "category", "description": "建筑主要用途。"},
            {"name": "square_feet", "type": "int", "description": "建筑面积。"},
            {"name": "year_built", "type": "float", "description": "建造年份，存在缺失。"},
            {"name": "floor_count", "type": "float", "description": "楼层数，存在较多缺失。"},
        ],
    },
    {
        "name": "weather_train.csv",
        "description": "训练期天气表，按站点和小时记录气温、云量、降水、风速等天气变量。",
        "fields": [
            {"name": "site_id", "type": "int", "description": "站点编号。"},
            {"name": "timestamp", "type": "datetime", "description": "小时级时间戳。"},
            {"name": "air_temperature", "type": "float", "description": "空气温度。"},
            {"name": "cloud_coverage", "type": "float", "description": "云量覆盖。"},
            {"name": "dew_temperature", "type": "float", "description": "露点温度。"},
            {"name": "precip_depth_1_hr", "type": "float", "description": "1 小时降水深度。"},
            {"name": "sea_level_pressure", "type": "float", "description": "海平面气压。"},
            {"name": "wind_direction", "type": "float", "description": "风向角度。"},
            {"name": "wind_speed", "type": "float", "description": "风速。"},
        ],
    },
]

CATEGORICAL_FEATURES = [
    "site_id",
    "building_id",
    "primary_use",
    "meter",
    "wind_direction",
    "beaufort_scale",
    "month_datetime",
    "weekofyear_datetime",
    "hour_datetime",
    "day_week",
    "day_month_datetime",
    "week_month_datetime",
]

NUMERICAL_FEATURES = [
    "square_feet",
    "year_built",
    "age",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "floor_count",
    "dayofyear_datetime",
]

EXCLUDED_BUILDING_IDS = [
    803,
    801,
    799,
    1088,
    993,
    794,
    881,
    904,
    921,
    927,
    954,
    955,
    983,
    1168,
]

PROCESSING_STAGES = [
    {
        "step": 1,
        "title": "拼接建筑和天气信息",
        "description": "将训练主表依次拼接建筑静态属性和站点天气观测，并在此阶段生成基础特征。",
    },
    {
        "step": 2,
        "title": "缺失值标记与填充",
        "description": "保留 NaN，不进行数值填充；对存在缺失的字段新增二值缺失指示列。",
    },
    {
        "step": 3,
        "title": "任务筛选与异常建筑剔除",
        "description": "保留电表任务样本，并剔除已识别的高读数异常建筑。",
    },
    {
        "step": 4,
        "title": "目标变换与时间切分",
        "description": "将目标值变换为模型训练目标，并按时间顺序切分 train/valid/test。",
    },
]

ENGINEERED_FEATURES = [
    {
        "feature": "primary_use",
        "name_zh": FEATURE_ZH_NAMES["primary_use"],
        "formula": "primary_use_code = index(sorted(unique(primary_use)))",
        "description": "将建筑用途类别排序后编码为整数。",
    },
    {
        "feature": "age",
        "name_zh": FEATURE_ZH_NAMES["age"],
        "formula": "age = 2016 - year_built",
        "description": "以 2016 年为基准计算建筑年龄。",
    },
    {
        "feature": "beaufort_scale",
        "name_zh": FEATURE_ZH_NAMES["beaufort_scale"],
        "formula": "beaufort_scale = cut(wind_speed, bins)",
        "description": "按风速区间映射为 0 到 12 的蒲福风级。",
    },
    {
        "feature": "wind_direction",
        "name_zh": FEATURE_ZH_NAMES["wind_direction"],
        "formula": "wind_direction_code = int(wind_direction / 22.5) % 16",
        "description": "将 0 到 360 度风向角映射到 16 个方向编码。",
    },
    {
        "feature": "month_datetime",
        "name_zh": FEATURE_ZH_NAMES["month_datetime"],
        "formula": "month_datetime = timestamp.month",
        "description": "从时间戳提取月份。",
    },
    {
        "feature": "weekofyear_datetime",
        "name_zh": FEATURE_ZH_NAMES["weekofyear_datetime"],
        "formula": "weekofyear_datetime = timestamp.isocalendar().week",
        "description": "从时间戳提取 ISO 年内周序号。",
    },
    {
        "feature": "dayofyear_datetime",
        "name_zh": FEATURE_ZH_NAMES["dayofyear_datetime"],
        "formula": "dayofyear_datetime = timestamp.dayofyear",
        "description": "从时间戳提取年内第几天。",
    },
    {
        "feature": "hour_datetime",
        "name_zh": FEATURE_ZH_NAMES["hour_datetime"],
        "formula": "hour_datetime = timestamp.hour",
        "description": "从时间戳提取小时。",
    },
    {
        "feature": "day_week",
        "name_zh": FEATURE_ZH_NAMES["day_week"],
        "formula": "day_week = timestamp.dayofweek",
        "description": "从时间戳提取星期序号。",
    },
    {
        "feature": "day_month_datetime",
        "name_zh": FEATURE_ZH_NAMES["day_month_datetime"],
        "formula": "day_month_datetime = timestamp.day",
        "description": "从时间戳提取月内日期。",
    },
    {
        "feature": "week_month_datetime",
        "name_zh": FEATURE_ZH_NAMES["week_month_datetime"],
        "formula": "week_month_datetime = ceil(timestamp.day / 7)",
        "description": "按月内日期计算月内第几周。",
    },
]

BEAUFORT_SCALE_BINS = [
    {"code": 0, "range": "wind_speed < 0.3"},
    {"code": 1, "range": "0.3 <= wind_speed < 1.6"},
    {"code": 2, "range": "1.6 <= wind_speed < 3.4"},
    {"code": 3, "range": "3.4 <= wind_speed < 5.5"},
    {"code": 4, "range": "5.5 <= wind_speed < 8.0"},
    {"code": 5, "range": "8.0 <= wind_speed < 10.8"},
    {"code": 6, "range": "10.8 <= wind_speed < 13.9"},
    {"code": 7, "range": "13.9 <= wind_speed < 17.2"},
    {"code": 8, "range": "17.2 <= wind_speed < 20.8"},
    {"code": 9, "range": "20.8 <= wind_speed < 24.5"},
    {"code": 10, "range": "24.5 <= wind_speed < 28.5"},
    {"code": 11, "range": "28.5 <= wind_speed < 33.0"},
    {"code": 12, "range": "33.0 <= wind_speed"},
]


def dataframe_sample(df: pd.DataFrame, row_count: int) -> list[dict]:
    sample = df.head(row_count).copy()
    sample = sample.where(pd.notna(sample), None)
    return sample.to_dict(orient="records")


def read_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def describe_feature(feature: str) -> dict:
    return {
        "feature": feature,
        "name_zh": FEATURE_ZH_NAMES.get(feature, feature),
        "label": f"{feature}（{FEATURE_ZH_NAMES.get(feature, feature)}）",
    }


def count_csv_rows(path: Path, chunk_size: int, usecols: list[str] | None = None) -> int:
    row_count = 0
    chunks = pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False)
    for chunk in chunks:
        row_count += len(chunk)
    return int(row_count)


def build_wind_direction_bins() -> list[dict]:
    direction_names = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    bins = []
    for code, label in enumerate(direction_names):
        lower = code * 22.5
        upper = (code + 1) * 22.5
        angle = math.radians((lower + 11.25) - 90)
        radius = 40
        bins.append(
            {
                "code": code,
                "label": label,
                "lower": lower,
                "upper": upper,
                "mid_angle": lower + 11.25,
                "x_percent": round(50 + math.cos(angle) * radius, 3),
                "y_percent": round(50 + math.sin(angle) * radius, 3),
                "formula": f"{lower:.1f} <= wind_direction < {upper:.1f}",
            }
        )
    return bins


def transform_target_value(raw_value: float, target_min: float, target_max: float) -> dict:
    log1p_value = math.log1p(raw_value)
    scale = target_max - target_min
    minmax_value = 0.0 if scale <= 0 else (log1p_value - target_min) / scale
    return {
        "raw_meter_reading": round(raw_value, 4),
        "log1p": round(log1p_value, 6),
        "minmax": round(minmax_value, 6),
    }


def summarize_preprocessed_source(
    path: Path,
    chunk_size: int,
    target_min: float,
    target_max: float,
    sample_size: int,
) -> dict:
    row_count = 0
    candidate_rows: list[pd.DataFrame] = []
    usecols = [
        "building_id",
        "meter",
        "timestamp",
        "meter_reading",
        "site_id",
        "primary_use",
        "square_feet",
    ]
    chunks = pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False)

    for chunk in chunks:
        row_count += len(chunk)
        meter_series = pd.to_numeric(chunk["meter"], errors="coerce")
        building_id_series = pd.to_numeric(chunk["building_id"], errors="coerce")
        filtered = chunk.loc[
            (meter_series == 0) & (building_id_series.isin(EXCLUDED_BUILDING_IDS))
        ].copy()

        if filtered.empty:
            continue

        filtered["meter_reading_numeric"] = pd.to_numeric(filtered["meter_reading"], errors="coerce")
        candidate_rows.append(filtered.nlargest(sample_size, "meter_reading_numeric"))

    if candidate_rows:
        removed_sample = pd.concat(candidate_rows, ignore_index=True).nlargest(
            sample_size,
            "meter_reading_numeric",
        )
    else:
        removed_sample = pd.DataFrame(columns=usecols + ["meter_reading_numeric"])

    removed_examples = []
    for row in removed_sample.to_dict(orient="records"):
        raw_value = float(row["meter_reading_numeric"])
        transformed = transform_target_value(raw_value, target_min=target_min, target_max=target_max)
        removed_examples.append(
            {
                "building_id": int(row["building_id"]),
                "timestamp": row["timestamp"],
                "meter_reading_raw": transformed["raw_meter_reading"],
                "log1p_if_kept": transformed["log1p"],
                "minmax_if_kept": transformed["minmax"],
                "site_id": int(row["site_id"]),
                "primary_use": int(row["primary_use"]),
                "square_feet": int(row["square_feet"]),
            }
        )

    return {
        "row_count": int(row_count),
        "removed_high_reading_examples": removed_examples,
    }


def summarize_split_file(path: Path, split_info: dict, sample_size: int) -> dict:
    sample_rows = pd.read_csv(path, nrows=sample_size)
    return {
        "filename": path.name,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "split": split_info["split"],
        "row_count": int(split_info["row_count"]),
        "column_count": int(sample_rows.shape[1]),
        "start_timestamp": split_info["start_timestamp"],
        "end_timestamp": split_info["end_timestamp"],
        "sample_rows": dataframe_sample(sample_rows, sample_size),
    }


def missing_summary_from_counts(
    missing_counts: pd.Series,
    row_count: int,
    top_n: int = 8,
) -> list[dict]:
    summary = (
        missing_counts.sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    summary["missing_count"] = summary["missing_count"].astype(int)
    summary["missing_percent"] = (summary["missing_count"] / row_count * 100).round(4)
    return summary.to_dict(orient="records")


def summarize_train_csv(path: Path, sample_size: int, chunk_size: int) -> dict:
    sample_rows = pd.read_csv(path, nrows=sample_size)
    row_count = 0
    missing_counts: pd.Series | None = None
    meter_counts: dict[int, int] = {}
    building_ids: set[int] = set()
    timestamp_min = ""
    timestamp_max = ""
    reading_min = None
    reading_max = None
    reading_sum = 0.0

    chunks = pd.read_csv(path, chunksize=chunk_size)
    for chunk in chunks:
        row_count += len(chunk)
        chunk_missing = chunk.isna().sum()
        missing_counts = chunk_missing if missing_counts is None else missing_counts.add(chunk_missing, fill_value=0)

        for meter, count in chunk["meter"].value_counts().items():
            meter_counts[int(meter)] = meter_counts.get(int(meter), 0) + int(count)

        building_ids.update(int(value) for value in chunk["building_id"].dropna().unique())
        chunk_timestamp_min = str(chunk["timestamp"].min())
        chunk_timestamp_max = str(chunk["timestamp"].max())
        timestamp_min = chunk_timestamp_min if not timestamp_min else min(timestamp_min, chunk_timestamp_min)
        timestamp_max = chunk_timestamp_max if not timestamp_max else max(timestamp_max, chunk_timestamp_max)

        chunk_reading = pd.to_numeric(chunk["meter_reading"], errors="coerce")
        chunk_min = float(chunk_reading.min())
        chunk_max = float(chunk_reading.max())
        reading_min = chunk_min if reading_min is None else min(reading_min, chunk_min)
        reading_max = chunk_max if reading_max is None else max(reading_max, chunk_max)
        reading_sum += float(chunk_reading.sum())

    meter_distribution = [
        {
            "meter": meter,
            "label": METER_LABELS.get(meter, str(meter)),
            "count": meter_counts[meter],
            "percent": round(meter_counts[meter] / row_count * 100, 2),
        }
        for meter in sorted(meter_counts)
    ]

    return {
        "filename": path.name,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "row_count": row_count,
        "column_count": len(sample_rows.columns),
        "columns": list(sample_rows.columns),
        "time_range": {"start": timestamp_min, "end": timestamp_max},
        "building_count": len(building_ids),
        "meter_distribution": meter_distribution,
        "meter_reading_stats": {
            "min": round(float(reading_min), 4),
            "max": round(float(reading_max), 4),
            "mean": round(reading_sum / row_count, 4),
        },
        "missing_summary": missing_summary_from_counts(missing_counts, row_count),
        "sample_rows": dataframe_sample(sample_rows, sample_size),
    }


def summarize_weather_csv(path: Path, sample_size: int) -> dict:
    weather_df = pd.read_csv(path)
    missing_counts = weather_df.isna().sum()
    return {
        "filename": path.name,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "row_count": int(weather_df.shape[0]),
        "column_count": int(weather_df.shape[1]),
        "columns": list(weather_df.columns),
        "site_count": int(weather_df["site_id"].nunique()),
        "time_range": {"start": str(weather_df["timestamp"].min()), "end": str(weather_df["timestamp"].max())},
        "missing_summary": missing_summary_from_counts(missing_counts, int(weather_df.shape[0])),
        "sample_rows": dataframe_sample(weather_df, sample_size),
    }


def summarize_building_metadata(path: Path, sample_size: int) -> dict:
    building_df = pd.read_csv(path)
    missing_counts = building_df.isna().sum()
    primary_use_counts = (
        building_df["primary_use"]
        .value_counts()
        .head(10)
        .rename_axis("primary_use")
        .reset_index(name="count")
    )
    primary_use_counts["percent"] = (primary_use_counts["count"] / building_df.shape[0] * 100).round(2)

    return {
        "filename": path.name,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "row_count": int(building_df.shape[0]),
        "column_count": int(building_df.shape[1]),
        "columns": list(building_df.columns),
        "building_count": int(building_df["building_id"].nunique()),
        "site_count": int(building_df["site_id"].nunique()),
        "primary_use_count": int(building_df["primary_use"].nunique()),
        "primary_use_distribution": primary_use_counts.to_dict(orient="records"),
        "missing_summary": missing_summary_from_counts(missing_counts, int(building_df.shape[0])),
        "sample_rows": dataframe_sample(building_df, sample_size),
    }


def build_dataset_overview(data_root: Path, sample_size: int, chunk_size: int) -> dict:
    train_summary = summarize_train_csv(data_root / "train.csv", sample_size=sample_size, chunk_size=chunk_size)
    weather_summary = summarize_weather_csv(data_root / "weather_train.csv", sample_size=sample_size)
    building_summary = summarize_building_metadata(data_root / "building_metadata.csv", sample_size=sample_size)

    files = [
        {
            "filename": "train.csv",
            "purpose": "小时级训练标签",
            "row_count": train_summary["row_count"],
            "column_count": train_summary["column_count"],
            "size_mb": train_summary["size_mb"],
        },
        {
            "filename": "weather_train.csv",
            "purpose": "训练期站点天气",
            "row_count": weather_summary["row_count"],
            "column_count": weather_summary["column_count"],
            "size_mb": weather_summary["size_mb"],
        },
        {
            "filename": "building_metadata.csv",
            "purpose": "建筑静态属性",
            "row_count": building_summary["row_count"],
            "column_count": building_summary["column_count"],
            "size_mb": building_summary["size_mb"],
        },
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "competition": "ASHRAE Great Energy Predictor III",
            "project_topic": "建筑能耗预测",
            "runtime_policy": "FastAPI 运行时只读取 web/data 下的小型 JSON 文件，不读取原始大 CSV。",
        },
        "headline_stats": {
            "train_rows": train_summary["row_count"],
            "buildings": building_summary["building_count"],
            "sites": building_summary["site_count"],
            "primary_uses": building_summary["primary_use_count"],
            "time_start": train_summary["time_range"]["start"],
            "time_end": train_summary["time_range"]["end"],
        },
        "files": files,
        "field_groups": FIELD_GROUPS,
        "train": train_summary,
        "weather_train": weather_summary,
        "building_metadata": building_summary,
        "notes": [
            "数据概况参考 ashrae-start-here-a-gentle-introduction.ipynb 中 Data、Glimpse of Data、Missing Values 部分。",
        ],
    }


def build_preprocessing_overview(
    preprocessed_dir: Path,
    dataset_overview: dict,
    sample_size: int,
    chunk_size: int,
) -> dict:
    preprocessed_train_path = preprocessed_dir / "preprocessed_train.csv"
    screened_train_path = preprocessed_dir / "screened_preprocessed_train.csv"
    summary_path = preprocessed_dir / "log1p_minmax_summary.json"
    train_path = preprocessed_dir / "log1p_minmax_train.csv"
    valid_path = preprocessed_dir / "log1p_minmax_valid.csv"
    test_path = preprocessed_dir / "log1p_minmax_test.csv"

    log1p_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_log1p_min = float(log1p_summary["target_log1p_min"])
    target_log1p_max = float(log1p_summary["target_log1p_max"])
    raw_train_rows = int(dataset_overview["headline_stats"]["train_rows"])
    preprocessed_source_summary = summarize_preprocessed_source(
        preprocessed_train_path,
        chunk_size=chunk_size,
        target_min=target_log1p_min,
        target_max=target_log1p_max,
        sample_size=sample_size,
    )
    preprocessed_rows = preprocessed_source_summary["row_count"]
    screened_rows = count_csv_rows(screened_train_path, chunk_size=chunk_size, usecols=["building_id"])
    final_columns = read_columns(train_path)
    preprocessed_columns = read_columns(preprocessed_train_path)
    screened_columns = read_columns(screened_train_path)
    missing_indicator_columns = [column for column in final_columns if column.endswith("_missing")]

    split_by_name = {item["split"]: item for item in log1p_summary["splits"]}
    split_files = [
        summarize_split_file(train_path, split_by_name["train"], sample_size=sample_size),
        summarize_split_file(valid_path, split_by_name["valid"], sample_size=sample_size),
        summarize_split_file(test_path, split_by_name["test"], sample_size=sample_size),
    ]

    row_flow = [
        {
            "label": "原始 train.csv",
            "count": raw_train_rows,
            "description": "小时级原始训练记录。",
        },
        {
            "label": "合并与特征工程后",
            "count": preprocessed_rows,
            "description": "合并建筑和天气数据，并生成基础特征。",
        },
        {
            "label": "筛选后",
            "count": screened_rows,
            "description": "仅保留电表数据并剔除异常建筑。",
        },
        {
            "label": "最终切分总量",
            "count": int(log1p_summary["total_row_count"]),
            "description": "完成 log1p + train min-max 后的 train/valid/test 总行数。",
        },
    ]

    screening_removed = preprocessed_rows - screened_rows
    target_examples = [
        transform_target_value(raw_value, target_log1p_min, target_log1p_max)
        for raw_value in [0.0, 10.0, 100.0, 1000.0, 5000.0, 8000.0]
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "headline_stats": {
            "raw_train_rows": raw_train_rows,
            "screened_rows": screened_rows,
            "final_column_count": len(final_columns),
            "split_ratio": "8:1:1",
            "missing_indicator_count": len(missing_indicator_columns),
        },
        "source_scripts": [
            "data_preprocess.py",
            "data_screening.py",
            "data_log1p_minmax.py",
        ],
        "pipeline_steps": PROCESSING_STAGES,
        "row_flow": row_flow,
        "merge_and_features": {
            "join_formulas": [
                "train_df = train_df.merge(building_metadata, on='building_id', how='left')",
                "train_df = train_df.merge(weather_train, on=['site_id', 'timestamp'], how='left')",
            ],
            "engineered_features": ENGINEERED_FEATURES,
            "wind_direction_formula": "wind_direction_code = int(wind_direction / 22.5) % 16",
            "wind_direction_bins": build_wind_direction_bins(),
            "beaufort_scale_formula": "beaufort_scale = pd.cut(wind_speed, bins, labels=0..12, right=False)",
            "beaufort_scale_bins": BEAUFORT_SCALE_BINS,
        },
        "feature_engineering": {
            "categorical_features": [describe_feature(feature) for feature in CATEGORICAL_FEATURES],
            "numerical_features": [describe_feature(feature) for feature in NUMERICAL_FEATURES],
            "engineered_features": ENGINEERED_FEATURES,
            "feature_count_after_screening": len(screened_columns) - 2,
            "final_columns": final_columns,
        },
        "missing_value_handling": {
            "strategy": "保留 NaN，不直接填补；对存在缺失的字段新增二值缺失指示列。",
            "formula": "missing_flag_c = 1 if isna(c) else 0",
            "fill_strategy": "当前三个预处理脚本没有执行 fillna 或插值填充，因此没有填充值计算公式；缺失值由原始 NaN 和 _missing 指示列共同表达。",
            "excluded_from_missing_indicators": ["timestamp", "meter_reading"],
            "missing_indicator_columns": missing_indicator_columns,
        },
        "screening": {
            "input_file": preprocessed_train_path.name,
            "output_file": screened_train_path.name,
            "input_row_count": preprocessed_rows,
            "kept_row_count": screened_rows,
            "removed_row_count": screening_removed,
            "kept_percent": round(screened_rows / preprocessed_rows * 100, 2),
            "filter_formula": "keep = (meter == 0) and (building_id not in excluded_building_ids)",
            "outlier_reason": "这些建筑在电表任务中出现过异常大的读数，容易主导目标分布，因此在筛选阶段剔除。",
            "rules": [
                "保留 meter == 0 的电表记录。",
                "剔除预先识别的异常建筑 ID。",
            ],
            "excluded_building_ids": EXCLUDED_BUILDING_IDS,
            "removed_high_reading_examples": preprocessed_source_summary["removed_high_reading_examples"],
        },
        "target_transform": {
            "target_preprocess": log1p_summary["target_preprocess"],
            "source_column": "meter_reading",
            "temporary_column": "meter_reading_log1p",
            "output_column": "meter_reading",
            "target_log1p_min": target_log1p_min,
            "target_log1p_max": target_log1p_max,
            "formulas": [
                "y_log = ln(1 + y)",
                "y_scaled = (y_log - min_train_log) / (max_train_log - min_train_log)",
            ],
            "parameter_notes": {
                "target_log1p_min": "训练集原始 meter_reading 经过 log1p 后的最小值，位于 min-max 归一化之前。",
                "target_log1p_max": "训练集原始 meter_reading 经过 log1p 后的最大值，位于 min-max 归一化之前。",
            },
            "example_transformations": target_examples,
            "description": "最终 CSV 中的 meter_reading 已经写回为 log1p + train min-max 后的模型目标值，不再是原始能耗读数。",
        },
        "split_summary": {
            "train_ratio": log1p_summary["train_ratio"],
            "valid_ratio": log1p_summary["valid_ratio"],
            "test_ratio": log1p_summary["test_ratio"],
            "total_row_count": int(log1p_summary["total_row_count"]),
            "files": split_files,
        },
        "artifacts": {
            "preprocessed_columns": preprocessed_columns,
            "screened_columns": screened_columns,
            "final_output_files": [
                train_path.name,
                valid_path.name,
                test_path.name,
            ],
        },
    }


def build_feature_group(features: list[str]) -> dict:
    return {
        "count": len(features),
        "features": [describe_feature(feature) for feature in features],
    }


def metric_block(summary: dict, prefix: str) -> dict:
    return {
        "mse": float(summary[f"{prefix}_mse"]),
        "mae": float(summary[f"{prefix}_mae"]),
        "rmse": float(summary[f"{prefix}_rmse"]),
        "r2": float(summary[f"{prefix}_r2"]),
        "smape": float(summary[f"{prefix}_smape"]),
    }


def build_parameter_cards(params: dict) -> list[dict]:
    parameter_descriptions = {
        "objective": "回归任务目标函数，最小化平方误差。",
        "eval_metric": "训练和早停监控使用的评估指标。",
        "tree_method": "使用直方图近似加速分裂点搜索。",
        "device": "训练脚本指定的计算设备。",
        "eta": "学习率，控制每棵树对最终预测的贡献。",
        "max_depth": "单棵树最大深度。",
        "subsample": "每轮训练抽取的样本比例。",
        "colsample_bytree": "每棵树抽取的特征比例。",
        "max_bin": "直方图分箱数量。",
        "reg_alpha": "L1 正则化系数。",
        "reg_lambda": "L2 正则化系数。",
        "sampling_method": "样本采样方法。",
    }
    ordered_keys = [
        "objective",
        "eval_metric",
        "tree_method",
        "device",
        "eta",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "max_bin",
        "reg_alpha",
        "reg_lambda",
        "sampling_method",
    ]
    return [
        {
            "name": key,
            "value": params[key],
            "description": parameter_descriptions[key],
        }
        for key in ordered_keys
        if key in params
    ]


def build_xgboost_overview(output_dir: Path) -> dict:
    run_summary = json.loads((output_dir / "xgboost_run_summary.json").read_text(encoding="utf-8"))
    model_params = json.loads((output_dir / "xgboost_model_params.json").read_text(encoding="utf-8"))
    split_records = pd.read_csv(output_dir / "xgboost_time_series_split.csv").to_dict(orient="records")
    split_summary = [
        {
            "split": row["split"],
            "start_timestamp": row["start_timestamp"],
            "end_timestamp": row["end_timestamp"],
            "row_count": int(row["row_count"]),
        }
        for row in split_records
    ]
    importance_df = pd.read_csv(output_dir / "xgboost_feature_importance.csv")
    xgboost_major_version = int(str(run_summary["xgboost_version"]).split(".")[0])

    feature_cols = list(run_summary["xgboost_feature_cols"])
    categorical_features = list(run_summary["categorical_features"])
    numerical_features = list(run_summary["numerical_features"])
    missing_indicator_features = [feature for feature in feature_cols if feature.endswith("_missing")]
    grouped_features = set(categorical_features + numerical_features + missing_indicator_features)
    other_features = [feature for feature in feature_cols if feature not in grouped_features]

    importance_df = importance_df.sort_values(by="importance", ascending=False).head(12)
    max_importance = float(importance_df["importance"].max())
    top_importance = []
    for row in importance_df.to_dict(orient="records"):
        importance = float(row["importance"])
        top_importance.append(
            {
                "feature": row["feature"],
                "label": describe_feature(row["feature"])["label"],
                "importance": importance,
                "width_percent": round(importance / max_importance * 100, 2),
            }
        )

    xgb_params = dict(model_params["xgb_params"])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "headline_stats": {
            "feature_count": int(run_summary["feature_count"]),
            "train_rows": int(run_summary["train_row_count"]),
            "best_iteration_round_count": int(run_summary["best_iteration_round_count"]),
            "test_rmse": float(run_summary["test_rmse"]),
        },
        "source_artifacts": [
            "xgboost_run_summary.json",
            "xgboost_model_params.json",
            "xgboost_time_series_split.csv",
            "xgboost_feature_importance.csv",
        ],
        "runtime_policy": "FastAPI 运行时只读取 web/data/xgboost_overview.json，不读取大 CSV，也不加载 xgboost_final_model.json。",
        "data_input": {
            "target_scale": "log1p + train min-max 后的 meter_reading",
            "excluded_columns": ["timestamp", "meter_reading"],
            "feature_expression": "feature_cols = train_df.columns - {timestamp, meter_reading}",
            "matrix_expression": "DMatrix(X, label=y)；X 为 34 个输入特征，y 为变换后的目标值。",
            "split_files": [
                {
                    "split": "train",
                    "filename": Path(run_summary["train_data_path"]).name,
                    "row_count": int(run_summary["train_row_count"]),
                },
                {
                    "split": "valid",
                    "filename": Path(run_summary["valid_data_path"]).name,
                    "row_count": int(run_summary["valid_row_count"]),
                },
                {
                    "split": "test",
                    "filename": Path(run_summary["test_data_path"]).name,
                    "row_count": int(run_summary["test_row_count"]),
                },
            ],
            "time_splits": split_summary,
            "feature_groups": [
                {
                    "name": "分类特征",
                    "description": "站点、建筑、用途、仪表、风向和时间离散特征。",
                    **build_feature_group(categorical_features),
                },
                {
                    "name": "数值特征",
                    "description": "建筑面积、年份、天气观测和年内日序号等连续变量。",
                    **build_feature_group(numerical_features),
                },
                {
                    "name": "缺失指示特征",
                    "description": "对应字段是否缺失的二值标记。",
                    **build_feature_group(missing_indicator_features),
                },
                {
                    "name": "其他输入特征",
                    "description": "未归入前三组但实际进入模型的字段。",
                    **build_feature_group(other_features),
                },
            ],
        },
        "model": {
            "xgboost_version": run_summary["xgboost_version"],
            "algorithm": "hist tree boosting",
            "matrix_type": "QuantileDMatrix" if xgboost_major_version >= 2 else "DMatrix",
            "prediction_scale": "y_scaled",
            "prediction_note": "模型输出为多棵回归树预测值的加和，目标尺度是 log1p + train min-max 后的 meter_reading。",
            "training_rounds": {
                "num_boost_round_upper_bound": int(model_params["num_boost_round_upper_bound"]),
                "early_stopping_rounds": int(model_params["early_stopping_rounds"]),
                "best_iteration_round_count": int(model_params["best_iteration_round_count"]),
            },
            "params": xgb_params,
            "parameter_cards": build_parameter_cards(xgb_params),
            "flow_steps": [
                "读取 log1p_minmax_train / valid / test 小时级切分文件。",
                "移除 timestamp 与 meter_reading，得到 34 维特征矩阵 X。",
                "将 X 和变换后的目标 y 组织为 QuantileDMatrix。",
                "每轮新增一棵回归树，按梯度方向修正上一轮误差。",
                "多棵树输出求和得到 y_scaled 预测值。",
                "在 valid/test 上用同一目标尺度计算 MSE、MAE、RMSE、R2、SMAPE。",
            ],
        },
        "metrics": {
            "scale_note": "所有指标均在 log1p + train min-max 后的目标尺度上计算，不是原始 meter_reading 尺度。",
            "validation": metric_block(run_summary, "validation"),
            "test": metric_block(run_summary, "test"),
            "formulas": [
                {"name": "MSE", "description": "预测值与真实值误差平方的平均值。"},
                {"name": "RMSE", "description": "MSE 开平方，与目标值处于同一尺度。"},
                {"name": "MAE", "description": "预测值与真实值绝对误差的平均值。"},
                {"name": "R2", "description": "模型相对均值基线解释目标方差的比例。"},
                {"name": "SMAPE", "description": "对称平均绝对百分比误差，训练脚本中乘以 100 输出百分数。"},
            ],
        },
        "feature_importance": {
            "importance_type": "gain",
            "top_features": top_importance,
            "note": "此处只展示少量 gain 排名靠前特征，完整解释性分析留给特征解释页面。",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small JSON files for the FastAPI web dashboard.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--preprocessed-dir", type=Path, default=PREPROCESSED_DATA_DIR)
    parser.add_argument("--xgboost-output-dir", type=Path, default=XGBOOST_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--output-path", type=Path, default=DATASET_OUTPUT_PATH)
    parser.add_argument("--preprocessing-output-path", type=Path, default=PREPROCESSING_OUTPUT_PATH)
    parser.add_argument("--xgboost-output-path", type=Path, default=XGBOOST_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_overview = build_dataset_overview(
        data_root=args.data_root,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(dataset_overview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_path}")

    preprocessing_overview = build_preprocessing_overview(
        preprocessed_dir=args.preprocessed_dir,
        dataset_overview=dataset_overview,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
    )
    args.preprocessing_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.preprocessing_output_path.write_text(
        json.dumps(preprocessing_overview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.preprocessing_output_path}")

    xgboost_overview = build_xgboost_overview(output_dir=args.xgboost_output_dir)
    args.xgboost_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.xgboost_output_path.write_text(
        json.dumps(xgboost_overview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.xgboost_output_path}")


if __name__ == "__main__":
    main()
