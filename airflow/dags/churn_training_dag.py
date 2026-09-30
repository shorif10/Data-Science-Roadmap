"""Churn retraining DAG:  validate -> train -> evaluate gate -> register.

STATUS: NOT EXECUTED. Apache Airflow is not installed in the environment where this project was
written, so this file has only been parsed (`ast`) by tests/test_infra.py, never run by a scheduler.
Treat it as a reviewed sketch: run it in your own Airflow 2.x/3.x before trusting it.

Assumptions when you deploy it
* The `churn` package is importable by the Airflow workers (pip install the project, or set PYTHONPATH).
* CHURN_PROJECT_ROOT points at the project root (default: two folders above this file).
* Tasks pass only small values (paths, ints) through XCom, never DataFrames.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

try:  # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

from airflow.exceptions import AirflowFailException

PROJECT_ROOT = Path(os.environ.get("CHURN_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
RUN_DIR = PROJECT_ROOT / "artifacts" / "airflow_runs"       # one folder per run: {{ run_id }}
REGISTRY = PROJECT_ROOT / "artifacts" / "registry"


@dag(
    dag_id="churn_training",
    description="Weekly retrain of the churn model with a release gate",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,                      # do not backfill every missed week
    max_active_runs=1,                  # never train two models into the registry at once
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["churn", "ml"],
)
def churn_training():
    @task
    def validate_data() -> int:
        """Fail early, and cheaply, when the input file is not what the model expects."""
        from churn.config import load_config, resolve
        from churn.data import load_dataset

        cfg = load_config(CONFIG)
        df = load_dataset(resolve(cfg, cfg["paths"]["data"]))   # raises DataValidationError on bad data
        return int(len(df))

    @task
    def train(n_rows: int, run_id: str | None = None) -> str:
        """Train into a run-specific folder so retries and concurrent runs never overwrite each other."""
        from churn.config import load_config
        from churn.train import run_training

        run_id = run_id or "manual"                                  # Airflow injects run_id from the task context
        out = RUN_DIR / run_id.replace(":", "_").replace("+", "_")
        run_training(load_config(CONFIG), output_dir=out)
        return str(out)

    @task
    def evaluate_gate(model_dir: str) -> str:
        """Stop the pipeline (no retry) when the new model fails the release gate in config.yaml."""
        metrics = json.loads((Path(model_dir) / "metrics.json").read_text(encoding="utf-8"))
        if not metrics["gate"]["passed"]:
            raise AirflowFailException(f"release gate failed: {metrics['gate']['checks']}")
        return model_dir

    @task
    def register(model_dir: str) -> str:
        """Copy the approved model into the registry and move `latest.json` (atomic replace)."""
        from churn.registry import register as register_model

        return str(register_model(model_dir, REGISTRY))

    register(evaluate_gate(train(validate_data())))


churn_training()
