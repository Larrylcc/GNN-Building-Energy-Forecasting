from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


app = FastAPI(title="ASHRAE 建筑能耗预测展示系统")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def load_json(filename: str) -> dict:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


DATASET_OVERVIEW = load_json("dataset_overview.json")
PREPROCESSING_OVERVIEW = load_json("preprocessing_overview.json")
XGBOOST_OVERVIEW = load_json("xgboost_overview.json")
GRU_OVERVIEW = load_json("gru_overview.json")
STGNN_OVERVIEW = load_json("stgnn_overview.json")
COMPARISON_OVERVIEW = load_json("comparison_overview.json")


PAGES = [
    {"key": "dataset", "label": "数据集概况", "url": "/dataset", "ready": True},
    {"key": "preprocessing", "label": "数据预处理", "url": "/preprocessing", "ready": True},
    {"key": "xgboost", "label": "XGBoost", "url": "/xgboost", "ready": True},
    {"key": "gru", "label": "GRU", "url": "/gru", "ready": True},
    {"key": "stgnn", "label": "STGNN", "url": "/stgnn", "ready": True},
    {"key": "comparison", "label": "结果对比", "url": "/comparison", "ready": True},
]


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/dataset")


@app.get("/dataset")
def dataset_page(request: Request):
    return templates.TemplateResponse(
        request,
        "dataset.html",
        {
            "active_page": "dataset",
            "pages": PAGES,
            "dataset": DATASET_OVERVIEW,
        },
    )


@app.get("/preprocessing")
def preprocessing_page(request: Request):
    return templates.TemplateResponse(
        request,
        "preprocessing.html",
        {
            "active_page": "preprocessing",
            "pages": PAGES,
            "preprocessing": PREPROCESSING_OVERVIEW,
        },
    )


@app.get("/xgboost")
def xgboost_page(request: Request):
    return templates.TemplateResponse(
        request,
        "xgboost.html",
        {
            "active_page": "xgboost",
            "pages": PAGES,
            "xgboost": XGBOOST_OVERVIEW,
        },
    )


@app.get("/gru")
def gru_page(request: Request):
    return templates.TemplateResponse(
        request,
        "gru.html",
        {
            "active_page": "gru",
            "pages": PAGES,
            "gru": GRU_OVERVIEW,
        },
    )


@app.get("/stgnn")
def stgnn_page(request: Request):
    return templates.TemplateResponse(
        request,
        "stgnn.html",
        {
            "active_page": "stgnn",
            "pages": PAGES,
            "stgnn": STGNN_OVERVIEW,
        },
    )


@app.get("/comparison")
def comparison_page(request: Request):
    return templates.TemplateResponse(
        request,
        "comparison.html",
        {
            "active_page": "comparison",
            "pages": PAGES,
            "comparison": COMPARISON_OVERVIEW,
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "completed_pages": ["dataset", "preprocessing", "xgboost", "gru", "stgnn", "comparison"],
        "data_generated_at": DATASET_OVERVIEW["generated_at"],
        "runtime_data_files": [
            "dataset_overview.json",
            "preprocessing_overview.json",
            "xgboost_overview.json",
            "gru_overview.json",
            "stgnn_overview.json",
            "comparison_overview.json",
        ],
    }


@app.get("/api/dataset")
def dataset_api() -> dict:
    return DATASET_OVERVIEW


@app.get("/api/preprocessing")
def preprocessing_api() -> dict:
    return PREPROCESSING_OVERVIEW


@app.get("/api/xgboost")
def xgboost_api() -> dict:
    return XGBOOST_OVERVIEW


@app.get("/api/gru")
def gru_api() -> dict:
    return GRU_OVERVIEW


@app.get("/api/stgnn")
def stgnn_api() -> dict:
    return STGNN_OVERVIEW


@app.get("/api/comparison")
def comparison_api() -> dict:
    return COMPARISON_OVERVIEW


@app.get("/{page_key}")
def planned_page(request: Request, page_key: str):
    page_keys = {page["key"] for page in PAGES}
    if page_key not in page_keys:
        return templates.TemplateResponse(
            request,
            "planned.html",
            {
                "active_page": "",
                "pages": PAGES,
                "title": "页面不存在",
                "message": "当前路由不属于本展示系统。",
            },
            status_code=404,
        )

    page = next(item for item in PAGES if item["key"] == page_key)
    return templates.TemplateResponse(
        request,
        "planned.html",
        {
            "active_page": page_key,
            "pages": PAGES,
            "title": page["label"],
            "message": "该页面将在你审批当前页面后继续实现。",
        },
    )
