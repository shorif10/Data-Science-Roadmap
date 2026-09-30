"""Data drift monitoring: PSI and KS between a reference sample and new data.

    python -m churn.monitor --reference artifacts/model/reference.csv --current new.csv \
                            --config configs/config.yaml

Drift in the *inputs* is an early warning: labels (who really churned) arrive weeks later,
inputs arrive immediately. Drift does not prove the model got worse, and no drift does not
prove it is fine; it tells you where to look.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config
from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, clean, load_raw

_EPS = 1e-4  # floor for empty bins so log() is defined


def _psi_from_props(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, _EPS, None), np.clip(cur, _EPS, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference, current, bins: int = 10) -> float:
    """Population Stability Index with quantile bins learned from the REFERENCE only.

    Values outside the reference range fall in the first/last bin. NaNs are ignored here
    (missing-rate change is reported separately).
    """
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: compare share equal to the constant value
        edges = np.array([-np.inf, np.median(ref), np.inf])
    else:
        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    ref_p = np.histogram(ref, edges)[0] / max(len(ref), 1)
    cur_p = np.histogram(cur, edges)[0] / max(len(cur), 1)
    return _psi_from_props(ref_p, cur_p)


def psi_categorical(reference, current) -> float:
    """PSI over category shares (missing counts as its own category)."""
    ref = pd.Series(reference).astype("object").fillna("<missing>").value_counts(normalize=True)
    cur = pd.Series(current).astype("object").fillna("<missing>").value_counts(normalize=True)
    cats = sorted(set(ref.index) | set(cur.index))
    return _psi_from_props(ref.reindex(cats, fill_value=0).to_numpy(), cur.reindex(cats, fill_value=0).to_numpy())


def ks_numeric(reference, current) -> tuple[float, float]:
    """(KS statistic, p-value). Statistic = largest gap between the two empirical CDFs (0..1)."""
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    res = stats.ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def _status(psi: float, ks: float | None, th: dict) -> str:
    if psi >= th["psi_alert"] or (ks is not None and ks >= th["ks_alert"]):
        return "alert"
    return "warn" if psi >= th["psi_warn"] else "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, thresholds: dict,
                 numeric: list[str] | None = None, categorical: list[str] | None = None) -> pd.DataFrame:
    """One row per feature: PSI, KS (numeric), missing rates, status in {ok, warn, alert}.

    Attributes of the result: `overall` ('ok'|'warn'|'alert') and `n_rows` (reference, current).
    """
    numeric = NUMERIC_COLUMNS if numeric is None else numeric
    categorical = CATEGORICAL_COLUMNS if categorical is None else categorical
    ref, cur = clean(reference), clean(current)
    if len(cur) < thresholds.get("min_rows", 1) or len(ref) < thresholds.get("min_rows", 1):
        raise ValueError(f"need at least {thresholds['min_rows']} rows in both frames "
                         f"(reference {len(ref)}, current {len(cur)}); PSI on tiny samples is noise")
    bins = int(thresholds.get("bins", 10))
    rows = []
    for col in numeric:
        psi = psi_numeric(ref[col], cur[col], bins)
        ks, p = ks_numeric(ref[col], cur[col])
        rows.append({"feature": col, "type": "numeric", "psi": psi, "ks_stat": ks, "ks_pvalue": p,
                     "missing_ref": float(ref[col].isna().mean()), "missing_cur": float(cur[col].isna().mean()),
                     "status": _status(psi, ks, thresholds)})
    for col in categorical:
        psi = psi_categorical(ref[col], cur[col])
        rows.append({"feature": col, "type": "categorical", "psi": psi, "ks_stat": None, "ks_pvalue": None,
                     "missing_ref": float(ref[col].isna().mean()), "missing_cur": float(cur[col].isna().mean()),
                     "status": _status(psi, None, thresholds)})
    out = pd.DataFrame(rows)
    order = {"alert": 0, "warn": 1, "ok": 2}
    out["_rank"] = out["status"].map(order)
    out = out.sort_values(["_rank", "psi"], ascending=[True, False]).drop(columns="_rank").reset_index(drop=True)
    statuses = set(out["status"])
    out.attrs["overall"] = "alert" if "alert" in statuses else "warn" if "warn" in statuses else "ok"
    out.attrs["n_rows"] = (len(ref), len(cur))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="churn.monitor", description="PSI/KS drift report")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--fail-on-alert", action="store_true", help="exit code 3 when overall status is alert")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    report = drift_report(load_raw(args.reference), load_raw(args.current), cfg["drift"])
    if args.json:
        print(json.dumps({"overall": report.attrs["overall"], "features": report.to_dict("records")}, indent=2))
    else:
        print(report.round(4).to_string(index=False))
        print(f"\noverall: {report.attrs['overall']}  (rows reference/current: {report.attrs['n_rows']})")
    return 3 if args.fail_on_alert and report.attrs["overall"] == "alert" else 0


if __name__ == "__main__":
    sys.exit(main())
