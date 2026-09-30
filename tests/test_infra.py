"""Static checks for files that cannot be executed here (Docker, Airflow, DVC, GitHub Actions).

These tests prove the files are well-formed and consistent with the real code. They do NOT prove
the files work when a real Docker daemon / Airflow scheduler / DVC / GitHub runner uses them."""
import ast
import re
import tomllib

import yaml

from churn import train
from churn.config import load_config
from conftest import ROOT


def _yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_airflow_dag_parses_and_has_the_four_steps():
    src = (ROOT / "airflow" / "dags" / "churn_training_dag.py").read_text(encoding="utf-8")
    tree = ast.parse(src)  # syntax only: airflow is not installed, so the module is never imported
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"validate_data", "train", "evaluate_gate", "register"} <= funcs
    assert "NOT EXECUTED" in src
    assert "catchup=False" in src and "max_active_runs=1" in src


def test_airflow_dag_only_uses_functions_that_exist_in_the_package():
    from churn import registry

    src = (ROOT / "airflow" / "dags" / "churn_training_dag.py").read_text(encoding="utf-8")
    assert "from churn.registry import register" in src and callable(registry.register)
    assert "from churn.train import run_training" in src and callable(train.run_training)


def test_ci_workflow_is_valid_yaml_and_runs_pytest():
    ci = _yaml(".github/workflows/ci.yml")
    steps = ci["jobs"]["test"]["steps"]
    commands = " ".join(s.get("run", "") for s in steps)
    assert "pytest" in commands and "ruff check" in commands
    assert any("setup-python" in s.get("uses", "") for s in steps)
    assert "on" in ci or True in ci  # PyYAML (YAML 1.1) parses the bare key `on` as boolean True


def test_dvc_stage_matches_the_real_cli():
    stage = _yaml("dvc/dvc.yaml")["stages"]["train"]
    cmd = stage["cmd"]
    assert "churn.train" in cmd and "--params dvc/params.yaml" in cmd
    outs = {o for o in stage["outs"]} | {next(iter(m)) for m in stage["metrics"]}
    expected = {f"artifacts/model/{f}" for f in
                (train.MODEL_FILE, train.CARD_FILE, train.REFERENCE_FILE, train.METRICS_FILE)}
    assert outs == expected
    cfg = load_config(ROOT / "configs" / "config.yaml")
    assert cfg["paths"]["data"] in stage["deps"]
    params = _yaml("dvc/params.yaml")
    listed = stage["params"][0]["dvc/params.yaml"]
    assert set(listed) == set(params), "every tracked key must exist in params.yaml and vice versa"
    assert "--output-dir artifacts/model" in cmd


def test_dvc_params_overlay_produces_valid_config():
    merged = load_config(ROOT / "configs" / "config.yaml", overlay=ROOT / "dvc" / "params.yaml")
    train.check_costs(merged["costs"])
    assert merged["paths"] == load_config(ROOT / "configs" / "config.yaml")["paths"]


def test_dockerfile_is_slim_nonroot_and_layer_cached():
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    assert any(ln.startswith("FROM python:3.11-slim") for ln in lines)
    assert ":latest" not in text
    user_idx = max(i for i, ln in enumerate(lines) if ln.startswith("USER "))
    assert lines[user_idx] != "USER root"
    assert any(ln.startswith("HEALTHCHECK") for ln in lines) or "HEALTHCHECK" in text
    # requirements are copied and installed BEFORE the source is copied (cache-friendly order)
    order = [i for i, ln in enumerate(lines) if ln.startswith(("COPY docker/requirements", "RUN pip install -r", "COPY src"))]
    assert order == sorted(order) and len(order) == 3
    assert "NOT BUILT" in text
    assert "--no-cache-dir" in text or "PIP_NO_CACHE_DIR=1" in text


def test_serve_requirements_are_pinned_and_minimal():
    lines = [ln for ln in (ROOT / "docker" / "requirements-serve.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+==\d+(\.\d+)*", ln) for ln in lines), lines
    names = {ln.split("==")[0].lower() for ln in lines}
    assert {"scikit-learn", "fastapi", "pandas", "numpy", "uvicorn", "pydantic"} <= names
    assert not names & {"mlflow", "pytest", "matplotlib", "torch", "tensorflow", "httpx", "jupyterlab"}


def test_compose_file_is_consistent_with_dockerfile():
    compose = _yaml("docker/docker-compose.yml")
    api = compose["services"]["api"]
    assert api["build"]["dockerfile"] == "docker/Dockerfile"
    assert "8000:8000" in api["ports"]
    assert api["environment"]["CHURN_MODEL_PATH"] == "/models/model.joblib"
    assert any(v.endswith(":/models:ro") for v in api["volumes"])
    assert compose["services"]["train"]["profiles"] == ["train"]


def test_dockerignore_keeps_context_small():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for needle in (".venv/", "datasets/", "artifacts/"):
        assert needle in text


def test_pyproject_declares_minimal_runtime_deps():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = {re.split(r"[<>=]", d)[0].lower() for d in meta["project"]["dependencies"]}
    assert deps == {"numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyyaml"}
    assert {"fastapi", "uvicorn", "pydantic"} <= {re.split(r"[<>=]", d)[0] for d in meta["project"]["optional-dependencies"]["serve"]}
