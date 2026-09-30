"""Training CLI and library.

    python -m churn.train --config configs/config.yaml [--params dvc/params.yaml]
                          [--output-dir artifacts/model] [--fail-on-gate]

Flow (each step only uses data it is allowed to see):

1. load + validate the CSV
2. split into train / validation / test (stratified, or ordered by a column)
3. fit every candidate pipeline on TRAIN, score it on VALIDATION, keep the best by the configured metric
4. choose the decision threshold on VALIDATION probabilities by minimising the business cost matrix
5. evaluate the selected model once on TEST (test labels never influence any decision)
6. write model.joblib, metrics.json, model_card.md, reference.csv (and log to MLflow if enabled)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from . import __version__
from .config import load_config, resolve
from .data import FEATURE_COLUMNS, TARGET, load_dataset
from .features import build_pipeline, make_estimator

MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"
CARD_FILE = "model_card.md"
REFERENCE_FILE = "reference.csv"

THRESHOLD_GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)
SELECTION_METRICS = {"roc_auc": roc_auc_score, "average_precision": average_precision_score}


# --------------------------------------------------------------------------- splitting
def split_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test). `val_size` is a share of the *whole* dataset."""
    sp = cfg["split"]
    test_size, val_size, seed = float(sp["test_size"]), float(sp["val_size"]), int(sp["random_state"])
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("split.test_size and split.val_size must be in (0,1) and sum to less than 1")
    strategy = sp.get("strategy", "stratified")
    if strategy == "stratified":
        rest, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
        train, val = train_test_split(
            rest, test_size=val_size / (1 - test_size), stratify=rest[TARGET], random_state=seed
        )
    elif strategy == "ordered":
        # Out-of-time style: the LAST rows (by order_by) are test, the block before is validation.
        ordered = df.sort_values(sp.get("order_by", "customer_id"), kind="stable")
        n = len(ordered)
        n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
        train = ordered.iloc[: n - n_test - n_val]
        val = ordered.iloc[n - n_test - n_val : n - n_test]
        test = ordered.iloc[n - n_test :]
    else:
        raise ValueError(f"split.strategy must be 'stratified' or 'ordered', got {strategy!r}")
    return train, val, test


# --------------------------------------------------------------------------- cost logic
def check_costs(costs: dict) -> dict:
    c = {k: float(costs[k]) for k in ("tn", "fp", "fn", "tp")}
    if not (c["fn"] > c["tp"] and c["fp"] > c["tn"]):
        raise ValueError(
            "cost matrix must satisfy fn > tp (acting on a churner has to help) and fp > tn "
            f"(acting on a loyal customer has to cost something); got {c}"
        )
    return c


def theoretical_threshold(costs: dict) -> float:
    """Best threshold for *perfectly calibrated* probabilities: flag when p*(fn-tp) > (1-p)*(fp-tn)."""
    c = check_costs(costs)
    return (c["fp"] - c["tn"]) / ((c["fp"] - c["tn"]) + (c["fn"] - c["tp"]))


def confusion(y, flag) -> dict[str, int]:
    y, flag = np.asarray(y).astype(bool), np.asarray(flag).astype(bool)
    return {
        "tn": int((~y & ~flag).sum()),
        "fp": int((~y & flag).sum()),
        "fn": int((y & ~flag).sum()),
        "tp": int((y & flag).sum()),
    }


def cost_per_customer(y, flag, costs: dict) -> float:
    cm, c = confusion(y, flag), check_costs(costs)
    return float(sum(cm[k] * c[k] for k in cm) / max(len(y), 1))


def cost_curve(y, proba, costs: dict, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """Expected cost per customer at every threshold (used for choosing AND for plotting)."""
    proba = np.asarray(proba)
    rows = [{"threshold": float(t), "cost_per_customer": cost_per_customer(y, proba >= t, costs)} for t in grid]
    return pd.DataFrame(rows)


def choose_threshold(y, proba, costs: dict, grid=THRESHOLD_GRID) -> float:
    """Threshold with the lowest expected cost (first one wins a tie)."""
    curve = cost_curve(y, proba, costs, grid)
    return float(curve.loc[curve["cost_per_customer"].idxmin(), "threshold"])


# --------------------------------------------------------------------------- evaluation
def _safe_auc(y, p) -> float | None:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def evaluate(y, proba, threshold: float, costs: dict) -> dict[str, Any]:
    y = np.asarray(y)
    proba = np.asarray(proba)
    flag = proba >= threshold
    cm = confusion(y, flag)
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    none, everyone = np.zeros(len(y), bool), np.ones(len(y), bool)
    cost = cost_per_customer(y, flag, costs)
    cost_none = cost_per_customer(y, none, costs)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": _safe_auc(y, proba),
        "average_precision": float(average_precision_score(y, proba)) if y.sum() else None,
        "brier": float(brier_score_loss(y, proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flag_rate": float(flag.mean()),
        "confusion": cm,
        "cost_per_customer": cost,
        "cost_flag_none_per_customer": cost_none,
        "cost_flag_all_per_customer": cost_per_customer(y, everyone, costs),
        "saving_vs_flag_none_pct": float(100 * (cost_none - cost) / cost_none) if cost_none else 0.0,
    }


def subgroup_report(test: pd.DataFrame, proba, threshold: float, costs: dict) -> dict[str, list[dict]]:
    """Per-segment performance on the test split (contract, senior_citizen)."""
    out: dict[str, list[dict]] = {}
    proba = np.asarray(proba)
    for col in ("contract", "senior_citizen"):
        rows = []
        for level, idx in test.groupby(col).indices.items():
            y = test[TARGET].to_numpy()[idx]
            m = evaluate(y, proba[idx], threshold, costs)
            rows.append({"group": str(level), **{k: m[k] for k in
                         ("n", "positive_rate", "roc_auc", "recall", "precision", "flag_rate", "cost_per_customer")}})
        out[col] = rows
    return out


def evaluate_gate(metrics: dict, cfg: dict) -> dict[str, Any]:
    """Release gate: cheap, explicit checks a pipeline can act on. Returns {passed, checks}."""
    g = cfg.get("gate", {})
    t = metrics["test"]
    checks = []
    if "min_test_auc" in g:
        checks.append({"name": "test_roc_auc >= min_test_auc", "value": t["roc_auc"],
                       "threshold": g["min_test_auc"], "passed": t["roc_auc"] is not None and t["roc_auc"] >= g["min_test_auc"]})
    if "max_cost_ratio_vs_flag_none" in g:
        ratio = t["cost_per_customer"] / t["cost_flag_none_per_customer"] if t["cost_flag_none_per_customer"] else 0.0
        checks.append({"name": "cost / cost(flag none) <= max ratio", "value": ratio,
                       "threshold": g["max_cost_ratio_vs_flag_none"], "passed": ratio <= g["max_cost_ratio_vs_flag_none"]})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


# --------------------------------------------------------------------------- training
def _version(cfg: dict, df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    keep = {k: cfg.get(k) for k in ("split", "costs", "features", "models", "selection", "target")}
    h.update(json.dumps(keep, sort_keys=True, default=str).encode())
    return f"{cfg['model']['version']}+{h.hexdigest()[:8]}"


def run_training(cfg: dict, output_dir: str | Path | None = None, df: pd.DataFrame | None = None) -> dict:
    """Train, select, threshold, test once, write artifacts. Returns the metrics dict."""
    costs = check_costs(cfg["costs"])
    seed = int(cfg["split"]["random_state"])
    if df is None:
        df = load_dataset(resolve(cfg, cfg["paths"]["data"]))
    out_dir = Path(output_dir) if output_dir else resolve(cfg, cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = split_data(df, cfg)
    engineered = bool(cfg.get("features", {}).get("engineer", False))
    sel_metric = cfg["selection"]["metric"]
    if sel_metric not in SELECTION_METRICS:
        raise ValueError(f"selection.metric must be one of {sorted(SELECTION_METRICS)}")

    fitted, val_proba, candidates = {}, {}, {}
    for name in cfg["selection"]["candidates"]:
        pipe = build_pipeline(make_estimator(name, cfg["models"].get(name), seed), engineered)
        pipe.fit(train[FEATURE_COLUMNS], train[TARGET])
        proba = pipe.predict_proba(val[FEATURE_COLUMNS])[:, 1]
        fitted[name], val_proba[name] = pipe, proba
        candidates[name] = {
            "val_roc_auc": float(roc_auc_score(val[TARGET], proba)),
            "val_average_precision": float(average_precision_score(val[TARGET], proba)),
            "val_brier": float(brier_score_loss(val[TARGET], proba)),
        }
    key = "val_" + sel_metric
    selected = max(candidates, key=lambda n: candidates[n][key])  # max() keeps the first on ties

    threshold = choose_threshold(val[TARGET], val_proba[selected], costs)
    theo = theoretical_threshold(costs)
    model = fitted[selected]

    test_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]  # the single test evaluation
    test_metrics = evaluate(test[TARGET], test_proba, threshold, costs)
    # Reported for context only (lift over the baseline); never used to decide anything.
    baseline_name = cfg["selection"]["candidates"][0]
    baseline_test_auc = _safe_auc(test[TARGET], fitted[baseline_name].predict_proba(test[FEATURE_COLUMNS])[:, 1])

    version = _version(cfg, df)
    metrics: dict[str, Any] = {
        "model_version": version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected_model": selected,
        "selection_metric": sel_metric,
        "candidates": candidates,
        "baseline": {"name": baseline_name, "test_roc_auc": baseline_test_auc},
        "split": {
            "strategy": cfg["split"].get("strategy", "stratified"),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "positive_rate_train": float(train[TARGET].mean()),
            "positive_rate_val": float(val[TARGET].mean()),
            "positive_rate_test": float(test[TARGET].mean()),
        },
        "costs": costs,
        "threshold": {
            "value": threshold,
            "chosen_on": "validation",
            "theoretical_if_calibrated": theo,
            "val_cost_per_customer": cost_per_customer(val[TARGET], val_proba[selected] >= threshold, costs),
        },
        "test": test_metrics,
        "subgroups": subgroup_report(test, test_proba, threshold, costs),
    }
    metrics["gate"] = evaluate_gate(metrics, cfg)

    bundle = {
        "pipeline": model,
        "threshold": threshold,
        "model_version": version,
        "model_name": selected,
        "feature_columns": list(FEATURE_COLUMNS),
        "library_versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__, "python": platform.python_version(), "churn": __version__},
    }
    joblib.dump(bundle, out_dir / MODEL_FILE)
    (out_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / CARD_FILE).write_text(render_model_card(metrics, cfg), encoding="utf-8")
    train[FEATURE_COLUMNS].to_csv(out_dir / REFERENCE_FILE, index=False)

    if cfg.get("mlflow", {}).get("enabled", False):
        _log_to_mlflow(cfg, metrics, out_dir)
    return metrics


def _mlflow_uri(cfg: dict) -> str:
    """Make a relative sqlite:/// or file: URI relative to the project root instead of the CWD."""
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not re.match(r"sqlite:///(/|[A-Za-z]:)", uri):
        return "sqlite:///" + resolve(cfg, uri.removeprefix("sqlite:///")).resolve().as_posix()
    if uri.startswith("file:") and not uri.startswith("file:///"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # MLflow 3.x: file store is opt-in
        return resolve(cfg, uri.removeprefix("file:").removeprefix("./")).resolve().as_uri()
    return uri


def _log_to_mlflow(cfg: dict, metrics: dict, out_dir: Path) -> None:
    import mlflow  # imported lazily: only needed when tracking is enabled

    mlflow.set_tracking_uri(_mlflow_uri(cfg))
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        # explicit artifact location: otherwise MLflow drops an `mlruns/` folder in the CWD
        root = resolve(cfg, cfg["mlflow"].get("artifact_root", "artifacts/mlartifacts"))
        mlflow.create_experiment(name, artifact_location=root.resolve().as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=metrics["model_version"]):
        mlflow.log_params({"selected_model": metrics["selected_model"], "threshold": metrics["threshold"]["value"],
                           "split_strategy": metrics["split"]["strategy"],
                           **{f"cost_{k}": v for k, v in metrics["costs"].items()}})
        mlflow.log_params({f"{m}_{k}": v for m, p in cfg["models"].items() for k, v in p.items()})
        for name, c in metrics["candidates"].items():
            for k, v in c.items():
                mlflow.log_metric(f"{name}_{k}", v)
        for k in ("roc_auc", "average_precision", "brier", "precision", "recall", "cost_per_customer"):
            if metrics["test"][k] is not None:
                mlflow.log_metric(f"test_{k}", metrics["test"][k])
        mlflow.log_artifact(str(out_dir / METRICS_FILE))
        mlflow.log_artifact(str(out_dir / CARD_FILE))


# --------------------------------------------------------------------------- model card
def render_model_card(m: dict, cfg: dict) -> str:
    t, th, sp, c = m["test"], m["threshold"], m["split"], m["costs"]
    cand = "\n".join(f"| {n} | {v['val_roc_auc']:.3f} | {v['val_average_precision']:.3f} | {v['val_brier']:.3f} |"
                     for n, v in m["candidates"].items())
    groups = ""
    for col, rows in m["subgroups"].items():
        groups += f"\n**By `{col}`**\n\n| group | n | churn rate | AUC | recall | flag rate |\n|---|---|---|---|---|---|\n"
        for r in rows:
            auc = "n/a" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
            groups += f"| {r['group']} | {r['n']} | {r['positive_rate']:.1%} | {auc} | {r['recall']:.1%} | {r['flag_rate']:.1%} |\n"
    cm = t["confusion"]
    return f"""# Model card: customer churn classifier

*Generated by `churn.train` at {m['trained_at']}. Version `{m['model_version']}`.*

## Intended use
Rank current customers by probability of leaving so the retention team can decide whom to contact.
The score supports a human-run retention campaign; it is **not** for pricing, credit or any decision that
affects a customer's rights.

## Data
`{cfg['paths']['data']}`: {sp['n_train'] + sp['n_val'] + sp['n_test']:,} customers (synthetic teaching data, not real people).
Split `{sp['strategy']}`: train {sp['n_train']:,} / validation {sp['n_val']:,} / test {sp['n_test']:,}.
Churn rate train {sp['positive_rate_train']:.1%}, validation {sp['positive_rate_val']:.1%}, test {sp['positive_rate_test']:.1%}.

## Model
Selected: **{m['selected_model']}** (by validation `{m['selection_metric']}`). Candidates:

| model | val AUC | val avg precision | val Brier |
|---|---|---|---|
{cand}

## Decision rule
Flag a customer when predicted probability >= **{th['value']:.2f}**, chosen on the validation split by minimising the cost matrix
(TN {c['tn']:g}, FP {c['fp']:g}, FN {c['fn']:g}, TP {c['tp']:g} per customer).
If probabilities were perfectly calibrated the optimum would be {th['theoretical_if_calibrated']:.2f}.

## Test performance (evaluated once)
| metric | value |
|---|---|
| ROC AUC | {t['roc_auc']:.3f} (baseline {m['baseline']['name']}: {m['baseline']['test_roc_auc']:.3f}) |
| Average precision | {t['average_precision']:.3f} (churn rate {t['positive_rate']:.3f}) |
| Brier score | {t['brier']:.3f} |
| Precision / recall at threshold | {t['precision']:.1%} / {t['recall']:.1%} |
| Share of customers flagged | {t['flag_rate']:.1%} |
| Confusion (TN/FP/FN/TP) | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |
| Cost per customer | {t['cost_per_customer']:.2f} (contact nobody: {t['cost_flag_none_per_customer']:.2f}; contact everybody: {t['cost_flag_all_per_customer']:.2f}) |
| Saving vs contacting nobody | {t['saving_vs_flag_none_pct']:.1f}% |

## Performance by subgroup (test)
{groups}
## Release gate
{'PASSED' if m['gate']['passed'] else 'FAILED'}: """ + "; ".join(
        f"{k['name']} (value {k['value']:.3f}, limit {k['threshold']})" for k in m["gate"]["checks"]
    ) + f"""

## Limitations
* Trained on synthetic data with planted signal; real-world performance is unknown until measured on real customers.
* The threshold is optimal only for the cost matrix above (illustrative numbers). Changing costs changes the threshold.
* The cost matrix assumes 40% of contacted churners accept an offer; that rate is an assumption, not a measurement.
* The test set is a single random sample; expect roughly a few points of AUC uncertainty. Small subgroups are noisier still.
* The model predicts association, not causation: a high score does not mean a retention offer will work.
* Scores are not recalibrated; check the calibration plot before using them as literal probabilities.
* No timestamp exists in this dataset, so temporal robustness (train on past, predict future) was not measured.
"""


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="churn.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--params", default=None, help="optional YAML overlay merged over the config (e.g. dvc/params.yaml)")
    ap.add_argument("--output-dir", default=None, help="override paths.output_dir")
    ap.add_argument("--fail-on-gate", action="store_true", help="exit with code 2 if the release gate fails")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overlay=args.params)
    m = run_training(cfg, output_dir=args.output_dir)
    t = m["test"]
    print(f"model={m['selected_model']} version={m['model_version']} threshold={m['threshold']['value']:.2f}")
    print(f"test AUC={t['roc_auc']:.3f} recall={t['recall']:.1%} cost/customer={t['cost_per_customer']:.2f} "
          f"(flag none: {t['cost_flag_none_per_customer']:.2f}) gate={'PASS' if m['gate']['passed'] else 'FAIL'}")
    if args.fail_on_gate and not m["gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
