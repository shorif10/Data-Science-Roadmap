"""FastAPI service.

    CHURN_MODEL_PATH=artifacts/model/model.joblib uvicorn churn.api:app --port 8000

The model is loaded once at startup (lifespan), not per request. Endpoints are plain `def`,
so FastAPI runs the CPU-bound scoring in its worker threadpool instead of blocking the event loop.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .data import DataValidationError
from .predict import ChurnModel

DEFAULT_MODEL_PATH = "artifacts/model/model.joblib"
DEFAULT_MAX_BATCH = 1000


class CustomerFeatures(BaseModel):
    """One customer. Unknown fields are rejected (typos should fail loudly)."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(None, description="Echoed back; not used by the model.")
    tenure_months: int = Field(..., ge=0, le=120)
    contract: str = Field(..., description="Month-to-month, One year or Two year (case-insensitive).")
    internet_service: str | None = Field(None, description="DSL, Fiber or None; null = unknown.")
    monthly_charges: float = Field(..., ge=0, le=1000)
    total_charges: float | None = Field(None, ge=0, description="null = unknown (imputed).")
    support_calls: int = Field(..., ge=0, le=100)
    paperless_billing: Literal["Yes", "No"]
    autopay: Literal["Yes", "No"]
    senior_citizen: Literal[0, 1]


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float = Field(..., ge=0, le=1)
    churn_flag: bool
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    n: int
    threshold: float
    model_version: str
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    threshold: float


def create_app(model_path: str | None = None, max_batch_size: int | None = None) -> FastAPI:
    path = model_path or os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH)
    max_batch = max_batch_size or int(os.environ.get("CHURN_MAX_BATCH", DEFAULT_MAX_BATCH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = ChurnModel.load(path)  # fail fast: no model, no service
        yield

    app = FastAPI(title="Churn prediction service", version="1.0.0", lifespan=lifespan)

    def _model(request: Request) -> ChurnModel:
        return request.app.state.model

    def _score(model: ChurnModel, customers: list[CustomerFeatures]) -> list[Prediction]:
        rows = [c.model_dump() for c in customers]
        try:
            results = model.predict(rows)
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.problems) from exc
        return [Prediction(**r, threshold=model.threshold, model_version=model.model_version) for r in results]

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        m = _model(request)
        return Health(status="ok", model_version=m.model_version, threshold=m.threshold)

    @app.post("/predict", response_model=Prediction)
    def predict(customer: CustomerFeatures, request: Request) -> Prediction:
        return _score(_model(request), [customer])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
        if len(body.customers) > max_batch:
            raise HTTPException(status_code=413, detail=f"batch too large: {len(body.customers)} > {max_batch}")
        m = _model(request)
        preds = _score(m, body.customers)
        return BatchResponse(n=len(preds), threshold=m.threshold, model_version=m.model_version, predictions=preds)

    return app


# `uvicorn churn.api:app` needs a module-level app. The model path is read from the environment.
app = create_app()
