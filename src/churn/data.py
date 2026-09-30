"""Load, clean and validate churn data.

Three separate steps, on purpose:

* `load_raw`   reads the CSV without letting pandas guess what "missing" means.
* `clean`      fixes what is *fixable* (casing, whitespace, types) and never invents values.
* `validate`   raises one `DataValidationError` listing everything that is still wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "churned"
ID_COLUMN = "customer_id"

CONTRACT_LEVELS = ("Month-to-month", "One year", "Two year")
INTERNET_LEVELS = ("DSL", "Fiber", "None")
YES_NO = ("Yes", "No")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str  # "numeric" | "category"
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    allowed: tuple | None = None


# The schema is the contract between training, serving and monitoring.
SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec("tenure_months", "numeric", min=0, max=120),
    ColumnSpec("contract", "category", allowed=CONTRACT_LEVELS),
    # NaN = "we do not know". The string "None" = "customer has no internet" (a real level).
    ColumnSpec("internet_service", "category", nullable=True, allowed=INTERNET_LEVELS),
    ColumnSpec("monthly_charges", "numeric", min=0, max=1000),
    ColumnSpec("total_charges", "numeric", nullable=True, min=0, max=200_000),
    ColumnSpec("support_calls", "numeric", min=0, max=100),
    ColumnSpec("paperless_billing", "category", allowed=YES_NO),
    ColumnSpec("autopay", "category", allowed=YES_NO),
    ColumnSpec("senior_citizen", "numeric", min=0, max=1, allowed=(0, 1)),
)
FEATURE_COLUMNS: list[str] = [c.name for c in SCHEMA]
NUMERIC_COLUMNS: list[str] = [c.name for c in SCHEMA if c.kind == "numeric"]
CATEGORICAL_COLUMNS: list[str] = [c.name for c in SCHEMA if c.kind == "category"]

_CANONICAL = {
    "contract": {v.lower(): v for v in CONTRACT_LEVELS},
    "internet_service": {v.lower(): v for v in INTERNET_LEVELS},
    "paperless_billing": {v.lower(): v for v in YES_NO},
    "autopay": {v.lower(): v for v in YES_NO},
}


class DataValidationError(ValueError):
    """Raised with *all* problems found, so the caller fixes them in one pass."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(f"{len(problems)} data problem(s):\n" + "\n".join(f" - {p}" for p in problems))


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV as text-safe as possible.

    `keep_default_na=False` matters: pandas would otherwise turn the literal string "None"
    (a valid internet_service level) into NaN and silently merge every no-internet customer
    with the ones that are genuinely unknown. Only an empty field is treated as missing.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={ID_COLUMN: "str"})


def _canonicalise(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Strip whitespace, blank -> missing, fix casing of known levels; unknown values are kept."""

    def fix(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        return mapping.get(text.lower(), text) if text else np.nan

    return pd.Series([fix(v) for v in series], index=series.index, dtype="str")


def clean_with_report(df: pd.DataFrame, drop_duplicates: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fix what is fixable. Returns the cleaned frame and a small report of what changed.

    `drop_duplicates` is for training tables only: at serving time two identical requests are
    two customers and must produce two predictions."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    report: dict = {"rows_in": len(out), "casing_fixed": {}, "coerced_to_nan": {}, "duplicates_dropped": 0}

    for col, mapping in _CANONICAL.items():
        if col not in out.columns:
            continue
        fixed = _canonicalise(out[col], mapping)
        changed = int((fixed.notna() & out[col].notna() & (fixed != out[col])).sum())
        if changed:
            report["casing_fixed"][col] = changed
        out[col] = fixed

    for col in [c for c in [*NUMERIC_COLUMNS, TARGET] if c in out.columns]:
        before = out[col].isna()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        n_bad = int((out[col].isna() & ~before).sum())
        if n_bad:
            report["coerced_to_nan"][col] = n_bad

    if drop_duplicates:
        n0 = len(out)
        out = out.drop_duplicates().reset_index(drop=True)  # exact duplicate rows only
        report["duplicates_dropped"] = n0 - len(out)
    report["rows_out"] = len(out)
    return out, report


def clean(df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    return clean_with_report(df, drop_duplicates)[0]


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Raise DataValidationError listing every schema violation (returns None when valid)."""
    problems: list[str] = []
    needed = list(FEATURE_COLUMNS) + ([TARGET] if require_target else [])
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise DataValidationError([f"missing required column(s): {missing_cols}"])
    if len(df) == 0:
        raise DataValidationError(["no rows"])

    for spec in SCHEMA:
        s = df[spec.name]
        n_null = int(s.isna().sum())
        if n_null and not spec.nullable:
            problems.append(f"{spec.name}: {n_null} missing value(s) but the column is required")
        present = s.dropna()
        if spec.kind == "numeric":
            if not pd.api.types.is_numeric_dtype(s):
                problems.append(f"{spec.name}: expected numbers, got dtype {s.dtype} (run clean() first)")
                continue
            if spec.min is not None and (present < spec.min).any():
                problems.append(f"{spec.name}: {int((present < spec.min).sum())} value(s) below {spec.min}")
            if spec.max is not None and (present > spec.max).any():
                problems.append(f"{spec.name}: {int((present > spec.max).sum())} value(s) above {spec.max}")
            if not np.isfinite(present.to_numpy(dtype=float)).all():
                problems.append(f"{spec.name}: contains inf")
        if spec.allowed is not None and len(present):
            bad = sorted({str(v) for v in present[~present.isin(spec.allowed)].unique()})
            if bad:
                problems.append(f"{spec.name}: unexpected value(s) {bad[:5]}; allowed {list(spec.allowed)}")

    if require_target:
        t = df[TARGET]
        if t.isna().any():
            problems.append(f"{TARGET}: {int(t.isna().sum())} missing target value(s)")
        elif not t.isin([0, 1]).all():
            problems.append(f"{TARGET}: must be 0 or 1")
        if ID_COLUMN in df.columns and df[ID_COLUMN].duplicated().any():
            problems.append(f"{ID_COLUMN}: {int(df[ID_COLUMN].duplicated().sum())} duplicate id(s)")
    if problems:
        raise DataValidationError(problems)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Training entry point: load -> clean -> validate. Returns a clean DataFrame."""
    df = clean(load_raw(path), drop_duplicates=True)
    validate(df, require_target=True)
    return df


def prepare_for_inference(records: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    """Serving entry point: same cleaning and validation as training, no target needed.

    Returns a frame with exactly FEATURE_COLUMNS (customer_id, if given, is dropped;
    callers keep their own copy).
    """
    if isinstance(records, dict):
        records = pd.DataFrame([records])
    elif isinstance(records, list):
        records = pd.DataFrame(records)
    elif not isinstance(records, pd.DataFrame):
        raise DataValidationError([f"unsupported input type {type(records).__name__}"])
    cleaned = clean(records)
    validate(cleaned, require_target=False)
    return cleaned[FEATURE_COLUMNS]
