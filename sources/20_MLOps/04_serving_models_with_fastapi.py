# %% [markdown]
# # Serving Models with FastAPI
#
# **Why this matters.** A model in a notebook helps one person. A model behind an HTTP endpoint helps the checkout page, the CRM, and the nightly job at the same time, and that is where the real engineering begins: rejecting malformed requests with a clear message, loading the model once instead of per request, answering health checks, batching, measuring latency, and changing the API without breaking existing callers.
# You already know how to build web services; this notebook applies that to a model.
#
# **You will learn**
# - wrap a scikit-learn pipeline in a **FastAPI** app with **pydantic** request/response schemas
# - validate input strictly (types, ranges, allowed values, unknown fields) and read a 422 error
# - load the model **once** at startup (lifespan), and expose `/health` and `/ready`
# - serve single and **batch** predictions, and measure **latency** (p50/p95) honestly
# - **version** an API (`/v1`, `/v2`) so a change does not break callers, and return safe 500 errors
# - test everything with `TestClient`, with no network and no server process
#
# **Prerequisites:** [Model packaging and registry](03_model_packaging_and_registry.ipynb) and basic HTTP/JSON knowledge.
#
# > Data used: `datasets/churn.csv` (synthetic). We train a small pipeline in the notebook and serve it from a temp folder.

# %%
import atexit
import json
import shutil
import tempfile
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="serving_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

# %% [markdown]
# ## 1. A model to serve
# A compact pipeline whose first step is the cleaning (so the service cannot forget it). It takes the **raw request fields** and returns a churn probability. We save it with `joblib` into the temp folder; the service will load it from there.

# %%
raw = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "contract", "internet_service",
            "paperless_billing", "autopay", "senior_citizen"]
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CAT = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]

def clean_frame(df):
    out = df[FEATURES].copy()
    out["contract"] = out["contract"].astype("str").str.strip().str.lower()
    out["internet_service"] = out["internet_service"].fillna("unknown")
    for c in ("paperless_billing", "autopay", "senior_citizen"):
        out[c] = out[c].astype("str")
    return out

def build_model():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("clean", FunctionTransformer(clean_frame)), ("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))])

train_raw, test_raw = train_test_split(raw, test_size=0.25, stratify=raw["churned"], random_state=42)
model = build_model().fit(train_raw, train_raw["churned"])
print(f"test ROC-AUC: {roc_auc_score(test_raw['churned'], model.predict_proba(test_raw)[:, 1]):.3f}")
MODEL_PATH = TMP / "model.joblib"
joblib.dump(model, MODEL_PATH)

# %% [markdown]
# ## 2. The request schema: validate at the door
# Most production ML bugs are bad inputs, not bad models. A **pydantic** model declares the contract once: types, ranges, allowed values. FastAPI then parses each request against it and answers `422 Unprocessable Entity` (with the exact field at fault) *before* your code runs.
# Decisions worth copying:
# * `extra="forbid"`: a misspelled field (`tenure_month`) fails loudly instead of being ignored and defaulted.
# * `Literal[...]` for categories; a **before-validator** normalises case so `Month-to-month` and `month-to-month` both work (the messy casing of the source data must not leak to callers as errors).
# * Ranges with `ge`/`le` that are **wider than the training data** but still catch nonsense (negative tenure, a charge of 10 million).
# * Explicit `None` for values that may be unknown, rather than a sentinel such as `-1`.

# %%
class CustomerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = None
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float | None = Field(default=None, ge=0)
    support_calls: int = Field(ge=0, le=100)
    contract: Literal["month-to-month", "one year", "two year"]
    internet_service: Literal["DSL", "Fiber", "None"] | None = None       # the string "None" = no internet plan; JSON null = unknown
    paperless_billing: Literal["Yes", "No"]
    autopay: Literal["Yes", "No"]
    senior_citizen: Literal[0, 1]

    @field_validator("contract", mode="before")
    @classmethod
    def _normalise_contract(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

example = {"tenure_months": 5, "monthly_charges": 79.9, "total_charges": 399.5, "support_calls": 3, "contract": "Month-to-month",
           "internet_service": "Fiber", "paperless_billing": "Yes", "autopay": "No", "senior_citizen": 0}
print(CustomerIn(**example).model_dump()["contract"])

def first_error(**overrides):
    try:
        CustomerIn(**{**example, **overrides})
        return "valid"
    except ValidationError as e:
        err = e.errors()[0]
        return f"{'.'.join(map(str, err['loc']))}: {err['msg']}"

for label, kw in [("negative tenure", {"tenure_months": -3}), ("unknown contract", {"contract": "weekly"}),
                  ("text for a number", {"monthly_charges": "a lot"}), ("typo in a field name", {"tenure_month": 4}),
                  ("null total_charges", {"total_charges": None})]:
    print(f"{label:22s} -> {first_error(**kw)}")

# %% [markdown]
# Does the schema accept the real data? A schema that rejects legitimate traffic is also a bug, so replay every row of the training table through it:

# %%
rows = raw.drop(columns=["churned"]).astype(object).where(raw.drop(columns=["churned"]).notna(), None).to_dict("records")
bad = 0
for r in rows:
    try:
        CustomerIn(**r)
    except ValidationError:
        bad += 1
print(f"{len(rows)} real rows replayed, {bad} rejected")

# %% [markdown]
# ## 3. The app factory
# We build the app inside a function so each test (and each exercise) can get a fresh instance with its own model. Points to notice as you read:
# * **`lifespan`** runs once at startup: the model is loaded from disk **once**, not per request. `app.state.load_count` lets us prove it.
# * Endpoints are plain **`def`**, not `async def`. Scoring is CPU-bound; FastAPI runs plain `def` endpoints in a worker thread pool, whereas a slow `async def` body would block the event loop for every other request.
# * `/health` (liveness: the process is up) is separate from `/ready` (the model is loaded and requests can be served). Orchestrators like Kubernetes use them differently.
# * A middleware stamps every response with an `X-Request-ID` and the processing time, and counts requests.
# * A catch-all exception handler returns a **generic** 500 with a request id (details go to the log, never to the caller).
# * Every prediction is appended to a log (in memory here; a file or queue in production), because the **monitoring notebook needs the served inputs and outputs**.

# %%
class PredictionV1(BaseModel):
    customer_id: str | None
    churn_probability: float = Field(ge=0, le=1)
    churn_flag: bool
    model_version: str

class PredictionV2(BaseModel):                       # a breaking change: renamed field + new field
    customer_id: str | None
    score: float = Field(ge=0, le=1)
    risk_band: Literal["low", "medium", "high"]
    model_version: str

class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerIn] = Field(min_length=1, max_length=1000)

def to_frame(customers):
    return pd.DataFrame([c.model_dump(exclude={"customer_id"}) for c in customers])

def risk_band(p):
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"

def create_app(model_path, model_version="1.0.0", threshold=0.5):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = joblib.load(model_path)             # loaded ONCE per process
        app.state.load_count += 1
        yield
        app.state.model = None

    app = FastAPI(title="Churn service", version=model_version, lifespan=lifespan)
    app.state.load_count = 0
    app.state.request_count = 0
    app.state.prediction_log = []
    app.state.model = None

    @app.middleware("http")
    async def stamp(request: Request, call_next):
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        request.app.state.request_count += 1
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        print(f"[log] request {getattr(request.state, 'request_id', '?')} failed: {type(exc).__name__}: {exc}")   # the detail stays in the log
        return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": getattr(request.state, "request_id", None)})

    def score(customers, request: Request):
        proba = request.app.state.model.predict_proba(to_frame(customers))[:, 1]
        for c, p in zip(customers, proba):
            request.app.state.prediction_log.append({"request_id": request.state.request_id, **c.model_dump(), "probability": float(p), "model_version": model_version})
        return proba

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        if request.app.state.model is None:
            return JSONResponse(status_code=503, content={"status": "model not loaded"})
        return {"status": "ready", "model_version": model_version}

    @app.post("/v1/predict", response_model=PredictionV1)
    def predict_v1(customer: CustomerIn, request: Request):
        p = float(score([customer], request)[0])
        return PredictionV1(customer_id=customer.customer_id, churn_probability=p, churn_flag=p >= threshold, model_version=model_version)

    @app.post("/v1/predict/batch")
    def predict_batch(batch: BatchIn, request: Request):
        proba = score(batch.customers, request)
        return {"n": len(proba), "model_version": model_version,
                "predictions": [{"customer_id": c.customer_id, "churn_probability": float(p), "churn_flag": bool(p >= threshold)}
                                for c, p in zip(batch.customers, proba)]}

    @app.post("/v2/predict", response_model=PredictionV2)
    def predict_v2(customer: CustomerIn, request: Request):
        p = float(score([customer], request)[0])
        return PredictionV2(customer_id=customer.customer_id, score=p, risk_band=risk_band(p), model_version=model_version)

    return app

app = create_app(MODEL_PATH)
print("routes:", sorted(r.path for r in app.routes if r.path.startswith(("/v", "/h", "/r"))))

# %% [markdown]
# ## 4. Test it without a server: `TestClient`
# `TestClient` speaks to the app in-process (no port, no network), so API tests run in milliseconds. Use it as a context manager so the `lifespan` startup runs:

# %%
client = TestClient(app).__enter__()                       # in a test file you would write: with TestClient(app) as client:
atexit.register(client.__exit__, None, None, None)

print("health:", client.get("/health").json())
print("ready :", client.get("/ready").json())
r = client.post("/v1/predict", json={**example, "customer_id": "C-1"})
print(r.status_code, r.json())
print("headers:", {k: v for k, v in r.headers.items() if k.startswith("x-")})

# %% [markdown]
# *What just happened?* The request body was parsed and validated into a `CustomerIn`, the endpoint turned it into a one-row DataFrame, the pipeline (cleaning included) produced a probability, and the response was validated against `PredictionV1`.
# Now the failure paths, which are the ones callers actually hit. Look at how specific the 422 is:

# %%
r = client.post("/v1/predict", json={**example, "contract": "weekly", "tenure_months": -1})
print(r.status_code)
for err in r.json()["detail"]:
    print("-", ".".join(map(str, err["loc"])), "->", err["msg"])

print(client.post("/v1/predict", content=b"not json", headers={"content-type": "application/json"}).status_code, "for a body that is not JSON")
print(client.get("/v1/predict").status_code, "for the wrong HTTP method")

# %% [markdown]
# ## 5. Load once, prove it
# A classic mistake is `joblib.load(...)` inside the endpoint, which reloads the model on every call. Fire 50 requests and check the counters:

# %%
for _ in range(50):
    client.post("/v1/predict", json=example)
print("requests served:", app.state.request_count, "| model loads:", app.state.load_count)

# %% [markdown]
# ## 6. Batch endpoint and latency
# Latency is a **distribution**, so report percentiles, not the average: p50 is the typical request, p95/p99 is what your unluckiest users feel, and SLAs are written against the tail.
# We time 200 single requests, then compare **per-row cost** of single vs batched scoring. (Caveat, stated honestly: `TestClient` runs in-process, so it excludes network time and real server overhead; it measures our code path, which is the part we control. Measure again against the deployed service.)

# %%
def measure_latency(client, payload, n=200, path="/v1/predict"):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post(path, json=payload)
        times.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    a = np.array(times)
    return {"n": n, "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}

single = measure_latency(client, example, n=200)
print({k: round(v, 2) for k, v in single.items()})

# %%
batch_rows = [{k: v for k, v in r.items() if k not in ("customer_id",)} for r in rows[:500]]
t0 = time.perf_counter()
rb = client.post("/v1/predict/batch", json={"customers": batch_rows})
batch_ms = (time.perf_counter() - t0) * 1000
print(rb.status_code, "| rows:", rb.json()["n"], f"| {batch_ms:.0f} ms for the batch -> {batch_ms / 500:.2f} ms/row")
print(f"single requests: {single['p50_ms']:.2f} ms/row at the median -> batching is roughly {single['p50_ms'] / (batch_ms / 500):.0f}x cheaper per row")

# %% [markdown]
# Batching amortises the **fixed cost per call**. Here most of the single-row time is not the maths of a logistic regression but the per-call overhead of the DataFrame build plus the scikit-learn `ColumnTransformer`/pipeline machinery (profile before you believe me), and that overhead is paid once per batch instead of once per row. So offer a batch endpoint for callers that score many rows, and cap the batch size (`max_length=1000` above) so one caller cannot exhaust memory.
# Rules of thumb when latency is too high: first measure where the time goes (validation, feature build, model), then cache repeated inputs, use a lighter model, and only then scale out with more worker processes.

# %%
print("oversized batch:", client.post("/v1/predict/batch", json={"customers": [example] * 1001}).status_code, "| empty batch:", client.post("/v1/predict/batch", json={"customers": []}).status_code)

# %% [markdown]
# ## 7. Versioning: change without breaking callers
# Suppose the business wants a risk band and a renamed `score` field. Renaming a field **breaks** every existing caller. The safe rules:
#
# | Change | Safe for callers? | What to do |
# |---|---|---|
# | add an optional request field | yes | just do it |
# | add a response field | usually (callers must ignore unknown fields) | do it, document it |
# | rename/remove a field, change a type or meaning | **no** | new version path (`/v2`), keep `/v1` running |
#
# Both versions share one model here; run them side by side, announce a deprecation date for `/v1`, and watch its traffic fall before you remove it.

# %%
print("v1:", client.post("/v1/predict", json=example).json())
print("v2:", client.post("/v2/predict", json=example).json())

# %% [markdown]
# Also separate the **API version** (`/v1`) from the **model version** (`model_version` in every response). A new model behind the same contract is not a new API version, but callers and your monitoring both need to know which model answered.

# %% [markdown]
# ## 8. When things go wrong: a safe 500
# If the model itself raises, the caller must not see a stack trace (information leak), but you must be able to find the failure. The handler above returns a generic body plus the **request id**, and logs the detail. Simulate a broken model:

# %%
class BrokenModel:
    def predict_proba(self, X):
        raise RuntimeError("secret internal path /opt/models/v7 corrupted")

# (we inject the broken model into app.state instead of pickling a class defined in a notebook)
broken_app = create_app(MODEL_PATH)
with TestClient(broken_app, raise_server_exceptions=False) as bc:
    broken_app.state.model = BrokenModel()
    r = bc.post("/v1/predict", json=example, headers={"x-request-id": "demo-123"})
print(r.status_code, r.json())

# %% [markdown]
# The caller sees `internal error` and `demo-123`; the log line (printed above) has the real cause. Quote the request id in a support ticket and you can find the exact failure.

# %% [markdown]
# ## 9. Running it for real (shown, not run)
# In a terminal, `uvicorn` serves the app (the file would import `create_app`). Workers are processes, so a model held in memory is loaded once **per worker**; size the machine accordingly. The auto-generated interactive docs are at `/docs`.

# %% tags=["no-run"]
# uvicorn churn_service:app --host 0.0.0.0 --port 8000 --workers 2
#
# curl -s -X POST http://localhost:8000/v1/predict -H "Content-Type: application/json" \
#      -d '{"tenure_months": 5, "monthly_charges": 79.9, "support_calls": 3, "contract": "month-to-month",
#           "internet_service": "Fiber", "paperless_billing": "Yes", "autopay": "No", "senior_citizen": 0}'
#
# A quick load test with any tool you like, for example `hey -n 2000 -c 20 ...` or `wrk`; watch p95 and error rate, not the average.

# %% [markdown]
# ## Common mistakes
#
# 1. **Loading the model inside the endpoint.** It reloads on every request. Load once in `lifespan`.
# 2. **`async def` with blocking model code.** It blocks the event loop and stalls every other request. Use plain `def` for CPU-bound work (or `run_in_threadpool`).
# 3. **Trusting input.** Without validation, `"abc"` in a numeric field or a negative tenure reaches the model and gives a confident wrong answer. Validate types, ranges and categories, and forbid unknown fields.
# 4. **Returning a bare class label.** Return the probability, the threshold used and the model version; decisions can change, scores are reusable.
# 5. **Leaking errors.** A raw stack trace shows internals. Return a generic message and a request id; log the rest.
# 6. **Breaking the contract silently.** Renaming a response field in place breaks callers. Version the path.
# 7. **One health endpoint for everything.** Liveness (is the process up?) and readiness (can it serve?) need different answers; a service that is alive but has not loaded its model must not receive traffic.
# 8. **Reporting mean latency.** Users feel the tail. Report p50/p95/p99.
# 9. **Not logging predictions.** Without the served inputs and outputs you cannot monitor drift or compare with labels later.

# %% [markdown]
# ## Interview corner
#
# **1. How would you deploy a model as a REST service?**
# <details><summary>Answer</summary>
# Package the model with its preprocessing, load it once at startup, expose a validated prediction endpoint plus health/readiness endpoints, containerise it, run several workers behind a load balancer, log requests and predictions, and monitor latency, errors and data drift. Keep the API versioned and the model versioned separately.
# </details>
#
# **2. Batch, online, or streaming inference: how do you choose?**
# <details><summary>Answer</summary>
# Batch when predictions can be precomputed (nightly churn scores) - simplest and cheapest. Online (request/response) when the decision needs fresh inputs within milliseconds to seconds, like fraud at checkout. Streaming when events arrive continuously and you need low-latency stateful features. Start with the simplest that meets the latency requirement.
# </details>
#
# **3. Why `def` and not `async def` for a model endpoint in FastAPI?**
# <details><summary>Answer</summary>
# `async def` runs on the event loop; CPU-bound or blocking code inside it freezes all other requests. Plain `def` endpoints are run in a thread pool, so the loop stays responsive. (Pure-Python CPU work still competes for the GIL, which is why you scale with worker processes.)
# </details>
#
# **4. How do you roll out a new model version safely?**
# <details><summary>Answer</summary>
# Keep the API contract, return the model version in responses, and route a small share of traffic (canary) or run it in shadow mode, comparing predictions and latency with the current model before shifting all traffic. Keep the previous version deployable for fast rollback.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A safer batch schema
# `BatchIn` above allows 1 to 1000 customers. The business now wants: between **1 and 100** customers, and **no two customers may share the same non-null `customer_id`** in one batch (the same customer scored twice in a request is almost always a client bug).
# Write `StrictBatchIn(BaseModel)` (forbid extra fields) that enforces both. Use a `field_validator("customers")` (mode `"after"`) for the uniqueness check, raising `ValueError` with a message that contains `"duplicate"`.

# %% tags=["starter"]
class StrictBatchIn(BaseModel):
    ...

# %%
def cust(i):
    return {**example, "customer_id": f"C-{i}"}

assert len(StrictBatchIn(customers=[cust(i) for i in range(100)]).customers) == 100
for bad_payload, label in [({"customers": []}, "empty"), ({"customers": [cust(i) for i in range(101)]}, "101 customers"),
                           ({"customers": [cust(1), cust(1)]}, "duplicate ids"), ({"customers": [cust(1)], "extra": 1}, "extra field")]:
    try:
        StrictBatchIn(**bad_payload)
        raise AssertionError(f"{label}: should have been rejected")
    except ValidationError as e:
        if label == "duplicate ids":
            assert "duplicate" in str(e).lower(), "the error message should mention 'duplicate'"
assert len(StrictBatchIn(customers=[example, example]).customers) == 2, "customers WITHOUT an id may repeat"
print("Exercise 1 passed")

# %% tags=["solution"]
class StrictBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerIn] = Field(min_length=1, max_length=100)

    @field_validator("customers")
    @classmethod
    def _unique_ids(cls, customers):
        ids = [c.customer_id for c in customers if c.customer_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate customer_id in batch")
        return customers

# %% [markdown]
# ### Exercise 2: Add an endpoint to a running app
# Add `POST /v1/churn-flag` to the app passed in. It takes a `CustomerIn` body and a **query parameter** `threshold` (float, default 0.5, must be between 0 and 1 inclusive, otherwise FastAPI answers 422) and returns `{"churn": <bool>, "threshold": <float>}` where `churn` is `probability >= threshold`. Use `request.app.state.model`.
# Write `add_flag_route(app)` which registers the route (return nothing).

# %% tags=["starter"]
def add_flag_route(app):
    ...

# %%
app2 = create_app(MODEL_PATH)
add_flag_route(app2)
with TestClient(app2) as c2:
    p = c2.post("/v1/predict", json=example).json()["churn_probability"]
    r_low = c2.post("/v1/churn-flag?threshold=0.0", json=example)
    r_high = c2.post("/v1/churn-flag?threshold=1.0", json=example)
    r_default = c2.post("/v1/churn-flag", json=example)
    assert r_low.status_code == 200 and r_low.json() == {"churn": True, "threshold": 0.0}
    assert r_high.json() == {"churn": False, "threshold": 1.0}, "probability is below 1.0 so the flag must be False"
    assert r_default.json() == {"churn": bool(p >= 0.5), "threshold": 0.5}
    assert c2.post("/v1/churn-flag?threshold=1.5", json=example).status_code == 422, "threshold above 1 must be rejected"
    assert c2.post("/v1/churn-flag?threshold=-0.1", json=example).status_code == 422, "threshold below 0 must be rejected"
print("Exercise 2 passed")

# %% tags=["solution"]
def add_flag_route(app):
    @app.post("/v1/churn-flag")
    def churn_flag(customer: CustomerIn, request: Request, threshold: float = Query(0.5, ge=0, le=1)):
        p = float(request.app.state.model.predict_proba(to_frame([customer]))[:, 1][0])
        return {"churn": bool(p >= threshold), "threshold": threshold}

# %% [markdown]
# ### Exercise 3: Latency under a budget
# Your SLO says p95 must stay below **50 ms** for single predictions. Using `measure_latency` on a **fresh app** (`app3`, created for you), store in `lat` the result for 100 requests and in `within_budget` whether `p95_ms < 50`.
# Then write `explain` in a markdown cell: if it were over budget, which three things would you measure first?

# %%
app3 = create_app(MODEL_PATH)
c3 = TestClient(app3).__enter__()
atexit.register(c3.__exit__, None, None, None)

# %% tags=["starter"]
lat = ...
within_budget = ...

# %%
assert set(lat) >= {"n", "p50_ms", "p95_ms"} and lat["n"] == 100, "call measure_latency(c3, example, n=100)"
assert app3.state.request_count >= 100, "requests must really have been sent to app3"
assert lat["p95_ms"] >= lat["p50_ms"] > 0
assert within_budget == (lat["p95_ms"] < 50), "within_budget must match the p95 < 50 rule"
print(f"Exercise 3 passed: p50 {lat['p50_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms -> within budget: {within_budget}")

# %% tags=["solution"]
lat = measure_latency(c3, example, n=100)
within_budget = bool(lat["p95_ms"] < 50)

# %% [markdown]
# ### Exercise 4: Turn the prediction log into a DataFrame
# The app keeps every scored request in `app.state.prediction_log`. Write `log_frame(app)` returning a DataFrame with one row per logged prediction and columns that include `probability`, `model_version`, `contract` and `request_id`, and store in `mean_prob` the mean predicted probability in `app` (the first app; it served many requests above).
# *This is exactly the data the monitoring notebook will analyse: served inputs plus predictions.*

# %% tags=["starter"]
def log_frame(app):
    ...

mean_prob = ...

# %%
lf = log_frame(app)
assert len(lf) == len(app.state.prediction_log) and len(lf) > 100
assert {"probability", "model_version", "contract", "request_id"} <= set(lf.columns)
assert lf["probability"].between(0, 1).all()
assert abs(mean_prob - np.mean([e["probability"] for e in app.state.prediction_log])) < 1e-12
print(f"Exercise 4 passed: {len(lf)} logged predictions, mean probability {mean_prob:.3f}")

# %% tags=["solution"]
def log_frame(app):
    return pd.DataFrame(app.state.prediction_log)

mean_prob = float(log_frame(app)["probability"].mean())

# %% [markdown]
# ### Exercise 5 (open-ended): Review your own API
# Write the OpenAPI-style contract of a prediction service you would build at work: endpoint paths, the request fields with types/ranges, the error responses, which fields you would log for monitoring, and what would make you bump the version to `/v2`. Then list two requests you would deliberately send to try to break it.

# %% [markdown]
# ## Recap
# - A serving layer is a thin, strict wrapper: **pydantic** validates input (types, ranges, allowed values, `extra="forbid"`), the pipeline (with its own cleaning) predicts, the response is validated too.
# - Load the model **once** at startup; use plain `def` for CPU-bound scoring; separate **`/health`** (alive) from **`/ready`** (can serve).
# - Offer **batch** scoring (much cheaper per row) with a size cap; measure **p50/p95**, not the mean, and know what your test client does not include.
# - **Version the path** for breaking changes, and report the **model version** in every response.
# - Return a generic 500 plus a **request id**; log details and every prediction for later monitoring.
# - `TestClient` lets you test the whole API in-process with no server.
#
# **Go deeper:** the FastAPI documentation (fastapi.tiangolo.com): "Lifespan Events", "Request Body", "Testing"; the pydantic documentation on validators (docs.pydantic.dev).
#
# **Next:** [Data and model versioning](05_data_and_model_versioning.ipynb)
