"""Shared fixtures. Tests never touch datasets/ for correctness-critical checks: they use a small
synthetic table with the same messiness (casing, missing values, the literal "None" level), so CI works on a
fresh clone. One test class additionally runs on the real churn.csv when it is present."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config
from churn.train import run_training

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = ROOT / "datasets" / "churn.csv"
# The shipped floor in config.yaml is derived from the REAL file (see test_shipped_gate_floor_is_derived).
# The small synthetic fixture table has its own, weaker, baseline, so fixtures use a looser floor.
SYNTHETIC_AUC_FLOOR = 0.60


def make_raw_churn(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Raw-looking churn table (as written to CSV) with planted signal and planted dirt."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(0, 73, n)
    internet = rng.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.40, 0.25])
    monthly = np.round(rng.uniform(18, 90, n), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)
    calls = rng.poisson(1.6, n)
    paperless = rng.choice(["Yes", "No"], n)
    autopay = rng.choice(["Yes", "No"], n)
    senior = (rng.random(n) < 0.16).astype(int)
    z = (-0.3 + 1.1 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
         + 0.35 * calls - 0.03 * tenure + 0.5 * (internet == "Fiber") - 0.4 * (autopay == "Yes"))
    churned = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"T{100000 + i}" for i in range(n)], "tenure_months": tenure, "contract": contract,
        "internet_service": internet, "monthly_charges": monthly, "total_charges": total,
        "support_calls": calls, "paperless_billing": paperless, "autopay": autopay,
        "senior_citizen": senior, "churned": churned,
    })
    # plant the dirt the real file has
    low = rng.random(n) < 0.06
    df.loc[low, "contract"] = df.loc[low, "contract"].str.lower()
    df.loc[rng.random(n) < 0.03, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.03, "internet_service"] = np.nan
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw_churn()


@pytest.fixture(scope="session")
def churn_csv(tmp_path_factory, raw_df) -> Path:
    path = tmp_path_factory.mktemp("data") / "churn.csv"
    raw_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def cfg(churn_csv, tmp_path) -> dict:
    """The real config pointed at the synthetic CSV, tiny model, private output dir."""
    c = load_config(ROOT / "configs" / "config.yaml")
    c["paths"] = {"data": str(churn_csv), "output_dir": str(tmp_path / "out")}
    c["models"]["hgb"]["max_iter"] = 40
    c["mlflow"]["enabled"] = False
    c["gate"]["min_test_auc"] = SYNTHETIC_AUC_FLOOR
    return c


@pytest.fixture(scope="session")
def trained(tmp_path_factory, churn_csv) -> dict:
    """Train once per session; API/predict/monitor tests reuse it. Returns dir + metrics + cfg."""
    c = load_config(ROOT / "configs" / "config.yaml")
    out = tmp_path_factory.mktemp("model")
    c["paths"] = {"data": str(churn_csv), "output_dir": str(out)}
    c["models"]["hgb"]["max_iter"] = 40
    c["mlflow"]["enabled"] = False
    c["gate"]["min_test_auc"] = SYNTHETIC_AUC_FLOOR
    metrics = run_training(c)
    return {"dir": out, "metrics": metrics, "cfg": c}


@pytest.fixture()
def good_record() -> dict:
    return {"customer_id": "X1", "tenure_months": 5, "contract": "Month-to-month", "internet_service": "Fiber",
            "monthly_charges": 80.5, "total_charges": 402.5, "support_calls": 4, "paperless_billing": "Yes",
            "autopay": "No", "senior_citizen": 0}
