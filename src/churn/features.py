"""Leakage-safe preprocessing and model pipelines.

Rule: *anything that learns from data lives inside the sklearn Pipeline*, so `fit` only ever
sees the training rows. Imputation medians, scaling means and category lists are therefore
learned from train and merely *applied* to validation, test and live traffic.
Stateless row-wise features (`engineer`) learn nothing and are safe anywhere.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

ENGINEERED_COLUMNS = ["is_first_year", "charge_gap"]


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise features (nothing is learned, so nothing can leak).

    * is_first_year: customers in their first 12 months are a classic churn segment.
    * charge_gap: total_charges minus tenure * monthly_charges. Near 0 for a customer who
      always paid today's price; non-zero after price changes. NaN when total_charges is NaN.
    """
    X = X.copy()
    X["is_first_year"] = (X["tenure_months"] <= 12).astype(float)
    X["charge_gap"] = X["total_charges"] - X["tenure_months"] * X["monthly_charges"]
    return X


def _names_after_engineer(transformer, input_features):
    return np.asarray([*input_features, *ENGINEERED_COLUMNS], dtype=object)


def build_preprocessor(engineered: bool = False, extra_numeric: Sequence[str] = ()) -> ColumnTransformer:
    """Impute + scale numbers, impute + one-hot categories. Unfitted.

    `extra_numeric` lists additional numeric columns created by a custom `engineer_fn`
    (see `build_pipeline`)."""
    numeric = list(NUMERIC_COLUMNS) + (ENGINEERED_COLUMNS if engineered else []) + list(extra_numeric)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        # add_indicator keeps "was missing" as a signal (156 total_charges are NaN)
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(name: str, params: dict | None = None, random_state: int = 42):
    """Build an unfitted classifier by name (`logistic` or `hgb`)."""
    params = dict(params or {})
    if name == "logistic":
        return LogisticRegression(random_state=random_state, **params)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=random_state, **params)
    raise ValueError(f"unknown model {name!r}; expected 'logistic' or 'hgb'")


def build_pipeline(estimator, engineered: bool = False, engineer_fn=None, extra_numeric: Sequence[str] = ()) -> Pipeline:
    """engineer -> preprocess -> classifier. Fit it on TRAIN rows only.

    To try your own feature: write `def my_engineer(X): X = engineer(X); X["new"] = ...; return X`
    and call `build_pipeline(est, engineer_fn=my_engineer, extra_numeric=["new"])`. `engineer_fn` must be
    stateless (row-wise); anything that learns from data belongs in the preprocessor, not here.
    Read feature names from `pipe.named_steps['prep'].get_feature_names_out()`. A pipeline that uses a
    function defined in a notebook cannot be reloaded elsewhere; move the function into features.py before
    you save a model."""
    steps = []
    if engineer_fn is not None:
        steps.append(("engineer", FunctionTransformer(engineer_fn)))
    elif engineered:
        steps.append(("engineer", FunctionTransformer(engineer, feature_names_out=_names_after_engineer)))
    steps += [("prep", build_preprocessor(engineered, extra_numeric)), ("model", estimator)]
    return Pipeline(steps)
