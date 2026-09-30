import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from churn import registry, train
from churn.config import load_config
from churn.data import FEATURE_COLUMNS, TARGET, load_dataset
from churn.features import build_pipeline, make_estimator
from churn.predict import ChurnModel
from conftest import REAL_CSV, ROOT

COSTS = {"tn": 0, "fp": 20, "fn": 200, "tp": 140}


# ---------------------------------------------------------------- splitting
@pytest.mark.parametrize("strategy", ["stratified", "ordered"])
def test_split_is_a_partition_with_expected_sizes(cfg, strategy):
    cfg["split"]["strategy"] = strategy
    df = load_dataset(cfg["paths"]["data"])
    tr, va, te = train.split_data(df, cfg)
    ids = [set(x["customer_id"]) for x in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(tr) + len(va) + len(te) == len(df)
    assert abs(len(te) / len(df) - 0.2) < 0.01 and abs(len(va) / len(df) - 0.2) < 0.01


def test_ordered_split_puts_latest_rows_in_test(cfg):
    cfg["split"]["strategy"] = "ordered"
    tr, va, te = train.split_data(load_dataset(cfg["paths"]["data"]), cfg)
    assert tr["customer_id"].max() < va["customer_id"].min() <= va["customer_id"].max() < te["customer_id"].min()


def test_split_rejects_bad_config(cfg):
    df = load_dataset(cfg["paths"]["data"])
    cfg["split"]["strategy"] = "random-ish"
    with pytest.raises(ValueError, match="strategy"):
        train.split_data(df, cfg)


# ---------------------------------------------------------------- cost logic
def test_theoretical_threshold_matches_formula():
    assert train.theoretical_threshold(COSTS) == pytest.approx(20 / (20 + 60))


def test_cost_matrix_sanity_is_enforced():
    with pytest.raises(ValueError, match="fn > tp"):
        train.check_costs({"tn": 0, "fp": 20, "fn": 100, "tp": 150})


def test_cost_per_customer_hand_computed():
    y = np.array([1, 1, 0, 0])
    flag = np.array([1, 0, 1, 0])  # tp, fn, fp, tn
    assert train.cost_per_customer(y, flag, COSTS) == pytest.approx((140 + 200 + 20 + 0) / 4)


def test_threshold_for_calibrated_scores_is_near_theory():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)  # perfectly calibrated by construction
    assert abs(train.choose_threshold(y, p, COSTS) - train.theoretical_threshold(COSTS)) < 0.03


def test_higher_false_negative_cost_never_raises_threshold(trained):
    va_p = pd.Series(np.random.default_rng(1).random(3000))
    y = (np.random.default_rng(2).random(3000) < va_p * 0.8).astype(int)
    thresholds = [train.choose_threshold(y, va_p, {**COSTS, "fn": fn}) for fn in (150, 200, 400, 800, 3000)]
    assert thresholds == sorted(thresholds, reverse=True), thresholds


# ---------------------------------------------------------------- end-to-end training
def test_training_writes_all_artifacts_and_valid_metrics(trained):
    out, m = trained["dir"], trained["metrics"]
    for name in (train.MODEL_FILE, train.METRICS_FILE, train.CARD_FILE, train.REFERENCE_FILE):
        assert (out / name).is_file(), name
    on_disk = json.loads((out / train.METRICS_FILE).read_text(encoding="utf-8"))
    assert on_disk["model_version"] == m["model_version"]
    assert m["selected_model"] in m["candidates"]
    best_val = max(c["val_roc_auc"] for c in m["candidates"].values())
    assert m["candidates"][m["selected_model"]]["val_roc_auc"] == best_val, "selection must use validation AUC"
    assert m["threshold"]["chosen_on"] == "validation"
    assert 0.02 <= m["threshold"]["value"] <= 0.98
    card = (out / train.CARD_FILE).read_text(encoding="utf-8")
    assert "Limitations" in card and m["model_version"] in card


def test_quality_gate_floor_comes_from_baselines_not_a_magic_number(trained, cfg):
    """Floor = best of (a) a naive one-rule score and (b) the logistic baseline, minus a small tolerance."""
    c = trained["cfg"]
    df = load_dataset(c["paths"]["data"])
    _, _, te = train.split_data(df, c)
    naive_score = te["contract"].map({"Month-to-month": 2, "One year": 1, "Two year": 0}) + 0.3 * te["support_calls"]
    naive_auc = roc_auc_score(te[TARGET], naive_score)
    m = trained["metrics"]
    baseline_auc = m["baseline"]["test_roc_auc"]
    test_auc = m["test"]["roc_auc"]
    assert test_auc > naive_auc, f"model ({test_auc:.3f}) must beat the naive rule ({naive_auc:.3f})"
    assert test_auc >= baseline_auc - 0.02, "selected model should not be clearly worse than the logistic baseline"
    assert m["test"]["cost_per_customer"] < m["test"]["cost_flag_none_per_customer"], "must beat contacting nobody"
    assert m["gate"]["passed"]


@pytest.mark.skipif(not REAL_CSV.exists(), reason="datasets/churn.csv not generated")
def test_shipped_gate_floor_is_derived_from_baselines():
    """The floor in config.yaml must sit above a naive one-rule score and at/below the lower 95% bootstrap
    bound of the logistic baseline (validation split, floored to 0.01). Re-derive it if data or split change."""
    c = load_config(ROOT / "configs" / "config.yaml")
    df = load_dataset(REAL_CSV)
    tr, va, _ = train.split_data(df, c)
    y = va[TARGET].to_numpy()
    naive = va["contract"].map({"Month-to-month": 2, "One year": 1, "Two year": 0}) + 0.3 * va["support_calls"]
    naive_auc = roc_auc_score(y, naive)
    pipe = build_pipeline(make_estimator("logistic", c["models"]["logistic"], 42)).fit(tr[FEATURE_COLUMNS], tr[TARGET])
    p = pipe.predict_proba(va[FEATURE_COLUMNS])[:, 1]
    rng = np.random.default_rng(0)
    boots = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    derived = np.floor(np.percentile(boots, 2.5) * 100) / 100
    floor = c["gate"]["min_test_auc"]
    assert floor > naive_auc, f"floor {floor} must be above the naive rule ({naive_auc:.3f})"
    assert floor == pytest.approx(derived), f"config floor {floor} != derived {derived}"


def test_training_is_reproducible(cfg, tmp_path):
    a = train.run_training(cfg, output_dir=tmp_path / "a")
    b = train.run_training(cfg, output_dir=tmp_path / "b")
    for m in (a, b):
        m.pop("trained_at")
    assert a == b


def test_cost_matrix_change_moves_the_threshold_and_version(cfg, tmp_path):
    lo = train.run_training(cfg, output_dir=tmp_path / "lo")
    cfg2 = copy.deepcopy(cfg)
    cfg2["costs"].update(fp=60.0, tp=180.0)  # expensive offer: contact fewer people
    hi = train.run_training(cfg2, output_dir=tmp_path / "hi")
    assert hi["threshold"]["value"] > lo["threshold"]["value"]
    assert hi["model_version"] != lo["model_version"], "config changes must change the version hash"


def test_test_set_is_not_used_for_selection(cfg, tmp_path, monkeypatch):
    """Scrambling test labels must not change which model or threshold is chosen."""
    base = train.run_training(cfg, output_dir=tmp_path / "base")
    original_split = train.split_data

    def scrambled(df, c):
        tr, va, te = original_split(df, c)
        te = te.copy()
        te[TARGET] = np.random.default_rng(0).permutation(te[TARGET].to_numpy())
        return tr, va, te

    monkeypatch.setattr(train, "split_data", scrambled)
    other = train.run_training(cfg, output_dir=tmp_path / "other")
    assert other["selected_model"] == base["selected_model"]
    assert other["threshold"]["value"] == base["threshold"]["value"]
    assert other["test"]["roc_auc"] != base["test"]["roc_auc"]


def test_gate_fails_with_impossible_floor(cfg, tmp_path):
    cfg["gate"]["min_test_auc"] = 0.999
    m = train.run_training(cfg, output_dir=tmp_path / "x")
    assert m["gate"]["passed"] is False
    with pytest.raises(RuntimeError, match="gate failed"):
        registry.register(tmp_path / "x", tmp_path / "reg")


def test_saved_model_predicts(trained):
    model = ChurnModel.load(trained["dir"] / train.MODEL_FILE)
    ref = pd.read_csv(trained["dir"] / train.REFERENCE_FILE, keep_default_na=False, na_values=[""])
    p = model.predict_proba(ref.head(25))
    assert p.shape == (25,) and ((p >= 0) & (p <= 1)).all()


def test_cli_with_overlay_and_fail_on_gate(cfg, tmp_path, capsys):
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    base = {k: v for k, v in cfg.items() if k != "_root"}
    (root / "configs" / "config.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    overlay = tmp_path / "params.yaml"
    overlay.write_text(yaml.safe_dump({"models": {"hgb": {"max_iter": 20}}, "gate": {"min_test_auc": 0.999}}))
    out = tmp_path / "cli_out"
    args = ["--config", str(root / "configs" / "config.yaml"), "--params", str(overlay), "--output-dir", str(out)]
    assert train.main(args) == 0
    assert "gate=FAIL" in capsys.readouterr().out
    assert train.main([*args, "--fail-on-gate"]) == 2
    assert (out / train.MODEL_FILE).is_file()


def test_registry_versions_are_immutable_and_latest_moves(trained, tmp_path):
    reg = tmp_path / "registry"
    dest = registry.register(trained["dir"], reg)
    assert registry.latest(reg) == dest and (dest / train.MODEL_FILE).is_file()
    with pytest.raises(FileExistsError):
        registry.register(trained["dir"], reg)
    assert not any(p.name.startswith(".tmp") for p in reg.iterdir())


def test_mlflow_logging_to_local_sqlite_store(cfg, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # a stray ./mlruns in the CWD would show up here
    cfg["mlflow"].update(enabled=True, tracking_uri=f"sqlite:///{(tmp_path / 'ml.db').as_posix()}",
                         artifact_root=str(tmp_path / "mlart"), experiment="pytest-churn")
    m = train.run_training(cfg, output_dir=tmp_path / "o")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=["pytest-churn"])
    assert len(runs) == 1
    assert runs.loc[0, "metrics.test_roc_auc"] == pytest.approx(m["test"]["roc_auc"])
    assert not (tmp_path / "mlruns").exists()
    mlflow.set_tracking_uri("")


def test_relative_mlflow_uri_is_anchored_to_project_root(cfg):
    uri = train._mlflow_uri({**cfg, "mlflow": {**cfg["mlflow"], "tracking_uri": "sqlite:///artifacts/mlflow.db"}})
    assert uri.startswith("sqlite:///") and uri.endswith("/artifacts/mlflow.db")
    assert cfg["_root"].replace("\\", "/") in uri


def test_default_config_is_loadable_and_consistent():
    c = load_config(ROOT / "configs" / "config.yaml")
    train.check_costs(c["costs"])
    assert set(c["selection"]["candidates"]) <= set(c["models"])
    assert c["drift"]["psi_warn"] < c["drift"]["psi_alert"]
