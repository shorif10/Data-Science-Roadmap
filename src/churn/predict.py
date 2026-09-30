"""Load a trained bundle and score customers, with the same validation as training."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import FEATURE_COLUMNS, ID_COLUMN, prepare_for_inference


class ChurnModel:
    """Thin wrapper around the artifact written by `churn.train`."""

    def __init__(self, bundle: dict):
        self._pipeline = bundle["pipeline"]
        self.threshold: float = float(bundle["threshold"])
        self.model_version: str = bundle["model_version"]
        self.model_name: str = bundle["model_name"]
        self.library_versions: dict = bundle.get("library_versions", {})

    @classmethod
    def load(cls, path: str | Path) -> ChurnModel:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path} (train one with `python -m churn.train`)")
        bundle = joblib.load(path)  # only load files you trained yourself: joblib/pickle can run code
        trained_with = bundle.get("library_versions", {}).get("sklearn")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"model trained with scikit-learn {trained_with} but this is {sklearn.__version__}; "
                "predictions may be wrong or loading may fail. Pin the same version.",
                stacklevel=2,
            )
        return cls(bundle)

    @property
    def pipeline(self):
        return self._pipeline

    def predict_proba(self, records: pd.DataFrame | dict | list[dict]) -> np.ndarray:
        """P(churn) for each record. Raises DataValidationError on invalid input."""
        X = prepare_for_inference(records)
        return self._pipeline.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def predict(self, records: pd.DataFrame | dict | list[dict]) -> list[dict]:
        """One dict per record: customer_id (if provided), churn_probability, churn_flag."""
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(
            [records] if isinstance(records, dict) else records)
        proba = self.predict_proba(frame)
        ids = frame[ID_COLUMN].tolist() if ID_COLUMN in frame.columns else [None] * len(proba)
        return [
            {"customer_id": i, "churn_probability": float(p), "churn_flag": bool(p >= self.threshold)}
            for i, p in zip(ids, proba, strict=True)
        ]
