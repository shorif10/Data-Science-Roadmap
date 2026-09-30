"""A deliberately tiny file-based model registry (what the Airflow DAG's last step calls).

    registry/
      1.0.0+ab12cd34/  model.joblib metrics.json model_card.md reference.csv
      latest.json      {"model_version": "...", "path": "...", "registered_at": "..."}

Versions are immutable folders; only `latest.json` moves, and it is replaced atomically, so a
serving container that reads it never sees a half-written pointer. For a real team use the MLflow
Model Registry or your cloud's registry; this shows the idea, not the product.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .train import CARD_FILE, METRICS_FILE, MODEL_FILE, REFERENCE_FILE

LATEST = "latest.json"


def register(model_dir: str | Path, registry_dir: str | Path, require_gate: bool = True) -> Path:
    """Copy a trained model directory into the registry and point `latest.json` at it."""
    model_dir, registry_dir = Path(model_dir), Path(registry_dir)
    metrics = json.loads((model_dir / METRICS_FILE).read_text(encoding="utf-8"))
    if require_gate and not metrics["gate"]["passed"]:
        raise RuntimeError(f"release gate failed, refusing to register: {metrics['gate']['checks']}")
    version = metrics["model_version"]
    dest = registry_dir / version.replace("+", "_")
    registry_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"version {version} is already registered (versions are immutable)")
    tmp = registry_dir / f".tmp-{version.replace('+', '_')}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for name in (MODEL_FILE, METRICS_FILE, CARD_FILE, REFERENCE_FILE):
        shutil.copy2(model_dir / name, tmp / name)
    tmp.rename(dest)
    pointer = registry_dir / (LATEST + ".tmp")
    pointer.write_text(json.dumps({"model_version": version, "path": dest.name,
                                   "registered_at": datetime.now(UTC).isoformat(timespec="seconds")}), encoding="utf-8")
    os.replace(pointer, registry_dir / LATEST)
    return dest


def latest(registry_dir: str | Path) -> Path:
    """Path to the current model directory."""
    registry_dir = Path(registry_dir)
    info = json.loads((registry_dir / LATEST).read_text(encoding="utf-8"))
    return registry_dir / info["path"]
