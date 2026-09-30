# %% [markdown]
# # CI/CD and Testing ML Systems
#
# **Why this matters.** Ordinary software breaks loudly: a failing test, a stack trace. ML systems break **quietly**: the code is fine, the tests that only check "it runs" are green, and yet a refactor flipped a feature's sign, a retrain on a bad extract lost 8 points of AUC, or the service now mis-handles an uppercase category. Continuous integration for ML must test **data, features, model behaviour and quality**, and gate releases on metrics, not just on "no exceptions".
# This notebook builds that safety net for a small churn project: real `pytest` suites run in a temp project, a **metric gate** that exits non-zero on regression, and a **GitHub Actions** workflow (shown and validated, not run).
#
# **You will learn**
# - what to test in an ML system and how it differs from testing ordinary code
# - write **data tests**, **feature tests** and **model behaviour tests** (invariance, directional, minimum-functionality)
# - build a **metric quality gate** for CI, pick its thresholds from noise, and watch it reject a bad model
# - prove your tests are not decorative by **sabotaging the model** and seeing them fail
# - read, write and validate a **GitHub Actions** workflow with `yaml.safe_load`, including the YAML gotchas
# - compare deployment strategies (canary, shadow, blue/green) and plan a rollback
#
# **Prerequisites:** [Testing, typing and pytest](../01_Python/12_testing_typing_pytest.ipynb), [Model packaging and registry](03_model_packaging_and_registry.ipynb) and [Testing pipelines](../19_Data_Pipelines/05_testing_monitoring_and_alerting_pipelines.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic). The project is generated in a temp folder and tested with the real `pytest` through `subprocess`.

# %%
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA = (Path("../datasets") / "churn.csv").resolve()
TMP = Path(tempfile.mkdtemp(prefix="mlci_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
PROJ = TMP / "churn_project"
(PROJ / "churnml").mkdir(parents=True)
(PROJ / "tests").mkdir()
print("project folder:", PROJ.name)

# %% [markdown]
# ## 1. What do we test in an ML system?
# Four layers, cheapest first. Each catches a different class of bug:
#
# | Layer | Example question | Typical failure it catches |
# |---|---|---|
# | **Data** | do the columns, types, ranges and keys look right? is the label rate plausible? do train and test share no customers? | a broken extract, a schema change, a leak |
# | **Features / code** | does `clean()` give the same output twice, handle unseen categories, avoid mutating its input? | refactors, off-by-one, NaN handling |
# | **Model behaviour** | does the prediction ignore irrelevant fields and casing (**invariance**)? does more support calls not *lower* churn risk (**directional**)? are obvious cases scored sensibly (**minimum functionality**)? | silently wrong models with fine aggregate metrics |
# | **Quality gate** | is the new model at least as good as the champion and above a floor? | regressions from retraining |
#
# This follows the "behavioural testing" idea popularised by the CheckList paper for NLP: a single accuracy number hides failures that specific, cheap, targeted tests expose.
# Two more principles: tests must be **deterministic** (seed everything; a flaky test trains people to ignore red), and **fast** (train on a small, fixed sample in CI; run the full retrain nightly).

# %% [markdown]
# ## 2. The project under test
# A small package with the parts every ML repo has: data loading and validation, feature cleaning, a model, a training CLI, and a gate. We write it to a temp folder (this is the code a real repository would contain). Read `features.py` and `gate.py` in particular.

# %%
def write(rel, body):
    path = PROJ / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())

write("churnml/__init__.py", "")
write("churnml/data.py", '''
    import pandas as pd

    EXPECTED_COLUMNS = ["customer_id", "tenure_months", "contract", "internet_service", "monthly_charges", "total_charges",
                        "support_calls", "paperless_billing", "autopay", "senior_citizen", "churned"]

    def load(path):
        # 'None' in internet_service is a real category, not a missing value
        return pd.read_csv(path, keep_default_na=False, na_values=[""])

    def validate(df):
        """Return a list of problems; an empty list means the data is fit to train on."""
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            return [f"missing columns: {missing}"]
        problems = []
        if df["customer_id"].duplicated().any():
            problems.append(f"{int(df['customer_id'].duplicated().sum())} duplicate customer_id values")
        if (df["tenure_months"] < 0).any():
            problems.append("negative tenure_months")
        if not df["churned"].isin([0, 1]).all():
            problems.append("churned must be 0 or 1")
        rate = df["churned"].mean()
        if not 0.05 < rate < 0.95:
            problems.append(f"implausible churn rate {rate:.2f}")
        return problems
''')
write("churnml/features.py", '''
    import pandas as pd

    NUM = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
    CAT = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Raw rows -> model features. Never mutates its input."""
        out = df[NUM + CAT].copy()
        out["contract"] = out["contract"].astype("str").str.strip().str.lower()
        out["internet_service"] = out["internet_service"].fillna("unknown")
        for c in ("paperless_billing", "autopay", "senior_citizen"):
            out[c] = out[c].astype("str")
        return out
''')
write("churnml/model.py", '''
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    from churnml.features import CAT, NUM, clean

    def build_pipeline(C=1.0, seed=42):
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ])
        return Pipeline([("clean", FunctionTransformer(clean)), ("pre", pre),
                         ("clf", LogisticRegression(C=C, max_iter=1000, random_state=seed))])

    def split_frame(df, seed=42):
        return train_test_split(df, test_size=0.25, stratify=df["churned"], random_state=seed)

    def train_and_evaluate(df, C=1.0, seed=42):
        train, test = split_frame(df, seed)
        pipe = build_pipeline(C, seed).fit(train, train["churned"])
        auc = roc_auc_score(test["churned"], pipe.predict_proba(test)[:, 1])
        return pipe, {"auc": round(float(auc), 4), "n_train": int(len(train)), "n_test": int(len(test))}
''')
write("churnml/train.py", '''
    import argparse, json, sys
    from pathlib import Path
    import joblib
    from churnml import data, model

    def main(argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--data", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--C", type=float, default=1.0)
        args = ap.parse_args(argv)
        df = data.load(args.data)
        problems = data.validate(df)
        if problems:                                   # never train on data that fails validation
            print("DATA VALIDATION FAILED:", "; ".join(problems))
            return 2
        pipe, metrics = model.train_and_evaluate(df, C=args.C)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out / "model.joblib")
        (out / "metrics.json").write_text(json.dumps({**metrics, "C": args.C}))
        print("trained:", metrics)
        return 0

    if __name__ == "__main__":
        sys.exit(main())
''')
write("churnml/gate.py", '''
    import argparse, json, sys

    def evaluate_gate(candidate, baseline=None, min_auc=0.70, max_drop=0.01):
        """Return (ok, reasons). Fails below an absolute floor, or when worse than the champion by more than max_drop."""
        reasons, auc = [], candidate["auc"]
        if auc < min_auc:
            reasons.append(f"auc {auc:.4f} is below the absolute floor {min_auc}")
        if baseline is not None and auc < baseline["auc"] - max_drop:
            reasons.append(f"auc {auc:.4f} regressed by more than {max_drop} against the champion {baseline['auc']:.4f}")
        return (not reasons), reasons

    def main(argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--candidate", required=True)
        ap.add_argument("--baseline")
        ap.add_argument("--min-auc", type=float, default=0.70)
        ap.add_argument("--max-drop", type=float, default=0.01)
        a = ap.parse_args(argv)
        cand = json.load(open(a.candidate))
        base = json.load(open(a.baseline)) if a.baseline else None
        ok, reasons = evaluate_gate(cand, base, a.min_auc, a.max_drop)
        print("GATE PASSED" if ok else "GATE FAILED: " + "; ".join(reasons))
        return 0 if ok else 1

    if __name__ == "__main__":
        sys.exit(main())
''')
print(sorted(p.relative_to(PROJ).as_posix() for p in PROJ.rglob("*.py")))

# %% [markdown]
# ## 3. Data, feature and behaviour tests
# First the shared fixtures. A `conftest.py` provides the data and a trained model **once per test session** (training per test would make the suite slow). It also supports an environment variable that **sabotages** the model on purpose; we will use it to prove the tests can fail.

# %%
write("tests/conftest.py", '''
    import os
    import pytest
    from churnml import data, model as m

    @pytest.fixture(scope="session")
    def raw_df():
        return data.load(os.environ["CHURN_DATA"])

    @pytest.fixture(scope="session")
    def split(raw_df):
        return m.split_frame(raw_df)

    @pytest.fixture(scope="session")
    def model(split):
        train, _ = split
        if os.environ.get("CHURN_SABOTAGE") == "tenure_reversed":      # used only to prove the tests can fail
            train = train.assign(tenure_months=train["tenure_months"].max() - train["tenure_months"])
        return m.build_pipeline().fit(train, train["churned"])
''')
write("tests/test_data.py", '''
    from churnml import data

    def test_real_data_passes_validation(raw_df):
        assert data.validate(raw_df) == []

    def test_validation_catches_duplicate_ids(raw_df):
        bad = raw_df.copy()
        bad.loc[1, "customer_id"] = bad.loc[0, "customer_id"]
        assert any("duplicate" in p for p in data.validate(bad))

    def test_validation_catches_negative_tenure(raw_df):
        bad = raw_df.copy()
        bad.loc[0, "tenure_months"] = -1
        assert "negative tenure_months" in data.validate(bad)

    def test_validation_reports_missing_columns(raw_df):
        assert data.validate(raw_df.drop(columns=["churned"]))[0].startswith("missing columns")

    def test_no_customer_appears_in_both_splits(split):
        train, test = split
        assert set(train["customer_id"]).isdisjoint(test["customer_id"])

    def test_label_rate_is_similar_in_both_splits(split):
        train, test = split
        assert abs(train["churned"].mean() - test["churned"].mean()) < 0.02
''')
write("tests/test_features.py", '''
    import pandas as pd
    from churnml.features import CAT, NUM, clean

    def test_clean_returns_exactly_the_model_columns(raw_df):
        assert list(clean(raw_df.head(20)).columns) == NUM + CAT

    def test_clean_does_not_mutate_its_input(raw_df):
        before = raw_df.head(50).copy()
        clean(before)
        pd.testing.assert_frame_equal(before, raw_df.head(50))

    def test_clean_is_deterministic(raw_df):
        pd.testing.assert_frame_equal(clean(raw_df.head(50)), clean(raw_df.head(50)))

    def test_contract_casing_is_normalised(raw_df):
        assert set(clean(raw_df)["contract"]) == {"month-to-month", "one year", "two year"}

    def test_missing_internet_service_becomes_unknown_but_None_stays_a_category(raw_df):
        out = clean(raw_df)
        assert out["internet_service"].notna().all()
        assert {"None", "unknown"} <= set(out["internet_service"])
''')
write("tests/test_model_behaviour.py", '''
    import numpy as np
    import pandas as pd

    BASE = {"tenure_months": 24, "monthly_charges": 60.0, "total_charges": 1440.0, "support_calls": 1, "contract": "one year",
            "internet_service": "DSL", "paperless_billing": "Yes", "autopay": "Yes", "senior_citizen": 0}

    def proba(model, **overrides):
        return float(model.predict_proba(pd.DataFrame([{**BASE, **overrides}]))[:, 1][0])

    # ---- sanity -------------------------------------------------------------------------------------
    def test_outputs_are_probabilities(model, raw_df):
        p = model.predict_proba(raw_df.head(200))[:, 1]
        assert np.all((p >= 0) & (p <= 1)) and not np.isnan(p).any()

    # ---- invariance: changes that must NOT change the prediction --------------------------------------
    def test_invariant_to_contract_casing(model):
        assert abs(proba(model, contract="Month-to-month") - proba(model, contract="month-to-month")) < 1e-12

    def test_invariant_to_row_order(model, raw_df):
        sample = raw_df.sample(100, random_state=1)
        a = model.predict_proba(sample)[:, 1]
        b = model.predict_proba(sample.iloc[::-1])[:, 1][::-1]
        assert np.allclose(a, b)

    def test_invariant_to_customer_id(model, raw_df):
        sample = raw_df.head(100)
        a = model.predict_proba(sample)[:, 1]
        b = model.predict_proba(sample.assign(customer_id="X"))[:, 1]
        assert np.allclose(a, b)

    # ---- directional: changes that must move the prediction the right way -----------------------------
    def test_more_support_calls_do_not_lower_churn_risk(model, raw_df):
        rows = raw_df.sample(300, random_state=2)
        low = model.predict_proba(rows)[:, 1]
        high = model.predict_proba(rows.assign(support_calls=rows["support_calls"] + 3))[:, 1]
        assert np.mean(high >= low - 1e-9) >= 0.95

    def test_longer_tenure_does_not_raise_churn_risk(model, raw_df):
        rows = raw_df.sample(300, random_state=3)
        short = model.predict_proba(rows)[:, 1]
        long = model.predict_proba(rows.assign(tenure_months=rows["tenure_months"] + 12))[:, 1]
        assert np.mean(long <= short + 1e-9) >= 0.95

    # ---- minimum functionality: obvious cases ---------------------------------------------------------
    def test_obvious_high_risk_scores_above_obvious_low_risk(model):
        risky = proba(model, contract="month-to-month", tenure_months=2, support_calls=6, autopay="No")
        safe = proba(model, contract="two year", tenure_months=60, support_calls=0, autopay="Yes")
        assert risky > safe + 0.2

    # ---- performance floor ----------------------------------------------------------------------------
    def test_beats_a_constant_predictor_by_a_clear_margin(model, split):
        from sklearn.metrics import roc_auc_score
        _, test = split
        assert roc_auc_score(test["churned"], model.predict_proba(test)[:, 1]) >= 0.65     # a dummy scores 0.5
''')
write("tests/test_gate.py", '''
    from churnml.gate import evaluate_gate

    def test_passes_when_better_than_champion():
        assert evaluate_gate({"auc": 0.82}, {"auc": 0.80})[0] is True

    def test_fails_below_absolute_floor():
        ok, reasons = evaluate_gate({"auc": 0.60})
        assert not ok and "floor" in reasons[0]

    def test_tolerates_small_drop_but_not_large_one():
        assert evaluate_gate({"auc": 0.795}, {"auc": 0.80}, max_drop=0.01)[0] is True
        assert evaluate_gate({"auc": 0.77}, {"auc": 0.80}, max_drop=0.01)[0] is False
''')
print(sorted(p.name for p in (PROJ / "tests").glob("*.py")))

# %% [markdown]
# Three kinds of behaviour test deserve a name, because they generalise to any model:
# * **Invariance**: perturb something that *should not matter* (casing, an ID column, row order) and assert the output is unchanged.
# * **Directional expectation**: perturb something with a known direction of effect (more support calls should not reduce churn risk) and assert the output moves that way, allowing a small tolerance fraction for non-monotone models.
# * **Minimum functionality**: a few obvious cases any sane model must get right.
#
# Now run the suite. The helper uses the real `pytest` in a subprocess, exactly as CI would, and returns the exit code (the thing a CI system looks at):

# %%
def run_pytest(args=(), extra_env=None):
    env = {**os.environ, "CHURN_DATA": str(DATA), "PYTHONPATH": str(PROJ), "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "-rf", *args],
                          cwd=PROJ, capture_output=True, text=True, env=env, timeout=300)
    return proc.returncode, proc.stdout

code, out = run_pytest()
print("exit code:", code)
print(out.strip().splitlines()[-1])

# %% [markdown]
# All green. Is that meaningful? A test suite you have never seen fail is unproven. So: **sabotage the model** and check the tests notice.
# `CHURN_SABOTAGE=tenure_reversed` trains the model on data where tenure is flipped (long becomes short), a plausible real bug (a wrong join, a sign error in a feature). Aggregate accuracy would still look respectable, because the model can learn the reversed relationship perfectly well.

# %%
code, out = run_pytest(extra_env={"CHURN_SABOTAGE": "tenure_reversed"})
print("exit code with the sabotaged model:", code)
for line in out.splitlines():
    if line.startswith("FAILED"):
        print(" ", line.split(" - ")[0])
print(out.strip().splitlines()[-1])

# %% [markdown]
# Look at which tests failed: the **directional tenure test** names the cause (longer tenure now *raises* churn risk), and the performance floor also trips, because the flipped feature hurts a model that has to predict the real, unflipped test set. What would *not* have caught it: a test that only checks "the model trains and returns probabilities". That is the point of behaviour tests.
# A second sabotage, in the **code** this time: someone "simplifies" `features.clean` and drops the lower-casing. The invariance test should fail:

# %%
features = PROJ / "churnml" / "features.py"
original = features.read_text()
features.write_text(original.replace('.astype("str").str.strip().str.lower()', '.astype("str").str.strip()'))
code, out = run_pytest()
print("exit code with the casing bug:", code)
for line in out.splitlines():
    if line.startswith("FAILED"):
        print(" ", line.split(" - ")[0])
features.write_text(original)
print("after restoring the file, exit code:", run_pytest()[0])

# %% [markdown]
# Two different bugs, caught by two different layers of tests. In CI this is exactly what a pull request would see: a red check naming the failing behaviour, before anything reaches production.

# %% [markdown]
# ## 4. The metric quality gate
# Tests check *behaviour*; the **gate** checks *quality* of a newly trained model against a floor and against the current champion, and **fails the pipeline with a non-zero exit code** when it regresses. The gate code is in `churnml/gate.py` (above). Let us use it through the same CLI a CI job would call:

# %%
def cli(*args):
    env = {**os.environ, "PYTHONPATH": str(PROJ), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-m", *args], cwd=PROJ, capture_output=True, text=True, env=env, timeout=300)
    return proc.returncode, (proc.stdout.strip() or proc.stderr.strip()[-300:])

print("train champion :", cli("churnml.train", "--data", str(DATA), "--out", str(TMP / "champion"), "--C", "1.0"))
print("train candidate:", cli("churnml.train", "--data", str(DATA), "--out", str(TMP / "cand_good"), "--C", "0.5"))
print("train bad cand.:", cli("churnml.train", "--data", str(DATA), "--out", str(TMP / "cand_bad"), "--C", "1e-9"))

# %%
for name in ("cand_good", "cand_bad"):
    code, msg = cli("churnml.gate", "--candidate", str(TMP / name / "metrics.json"), "--baseline", str(TMP / "champion" / "metrics.json"))
    print(f"{name:10s} exit code {code} | {msg}")

# %% [markdown]
# *What just happened?* The reasonable candidate (`C=0.5`) passed; the candidate with an absurdly strong regularisation (`C=1e-9`) was rejected with a reason, and the **exit code 1** is what makes a CI job turn red and block the merge or deploy.
# Look closely at the reason: the bad candidate was caught by the comparison with the **champion**, not by the absolute floor. Run the gate again without a baseline:

# %%
code, msg = cli("churnml.gate", "--candidate", str(TMP / "cand_bad" / "metrics.json"))
print(f"cand_bad, floor only (no champion): exit code {code} | {msg}")

# %% [markdown]
# A crippled model still ranks customers surprisingly well (very strong regularisation shrinks the weights but mostly keeps their direction, so AUC, which only needs the ranking, barely collapses), so it sails past a fixed floor. **A floor alone is not a quality gate**: you need the relative comparison against what is live as well.
# Also note `churnml.train` refuses# Also note `churnml.train` refuses to train at all if data validation fails, so a broken extract never produces a "model". Try it on a corrupted file:

# %%
corrupt = pd.read_csv(DATA, keep_default_na=False)
corrupt.loc[0, "customer_id"] = corrupt.loc[1, "customer_id"]          # duplicate key
corrupt.loc[5, "tenure_months"] = -4                                      # impossible value
corrupt_path = TMP / "corrupt.csv"
corrupt.to_csv(corrupt_path, index=False)
print(cli("churnml.train", "--data", str(corrupt_path), "--out", str(TMP / "never_written")))

# %% [markdown]
# Exit code 2, a clear message, and no model directory. Different failure modes get different exit codes, which lets a pipeline react differently (bad data: alert the data owner; regression: alert the modeller).

# %% [markdown]
# ### Choosing the gate's thresholds honestly
# Two numbers decide whether the gate is useful: the **floor** and the **tolerated drop**. If `max_drop` is smaller than the natural noise of your evaluation, the gate will reject good models at random; if it is huge, it will pass real regressions. Measure the noise: bootstrap the test set and see how much AUC moves by chance.

# %%
import importlib
sys.path.insert(0, str(PROJ))
model_mod = importlib.import_module("churnml.model")
data_mod = importlib.import_module("churnml.data")
df_all = data_mod.load(DATA)
pipe, metrics = model_mod.train_and_evaluate(df_all)
_, test = model_mod.split_frame(df_all)
scores = pipe.predict_proba(test)[:, 1]
y = test["churned"].to_numpy()

from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(42)
boot = []
for _ in range(300):
    idx = rng.integers(0, len(y), len(y))
    boot.append(roc_auc_score(y[idx], scores[idx]))
boot = np.array(boot)
print(f"test AUC {metrics['auc']} on {len(y)} rows | bootstrap sd {boot.std():.4f} | 95% interval {np.percentile(boot, 2.5):.3f} to {np.percentile(boot, 97.5):.3f}")
print(f"a max_drop of 0.01 is about {0.01 / boot.std():.1f} bootstrap standard deviations of one model's AUC")

# The gate compares two models on the SAME rows, so what matters is the noise of the DIFFERENCE (paired bootstrap):
other = model_mod.build_pipeline(C=0.5).fit(model_mod.split_frame(df_all)[0], model_mod.split_frame(df_all)[0]["churned"])
scores_b = other.predict_proba(test)[:, 1]
diffs = []
for _ in range(300):
    idx = rng.integers(0, len(y), len(y))
    diffs.append(roc_auc_score(y[idx], scores_b[idx]) - roc_auc_score(y[idx], scores[idx]))
diffs = np.array(diffs)
print(f"paired bootstrap sd of (C=0.5 minus C=1.0) AUC difference: {diffs.std():.4f} -> a 0.01 tolerance is {0.01 / diffs.std():.0f} sd of that difference")

# %% [markdown]
# Read those two lines as an engineer. The noise of a single AUC is large relative to 0.01, but two similar models scored on the **same rows** move together, so the noise of their *difference* is much smaller (hence paired comparison is the right way to gate). If `max_drop` were below roughly one or two standard deviations of that paired difference, a model that is truly identical would sometimes "regress" and block a release for nothing. Use a larger evaluation set or repeated-seed averaging to shrink the noise further.
# And a *floor* should be tied to something meaningful: a dummy baseline plus a margin (as in the `0.65` test above), or the business's minimum useful performance, not a number copied from the last run.
#
# Averages can also hide a failing segment. A model with a healthy overall AUC might be poor for one contract type. Exercise 1 asks you to build a **slice gate**.

# %% [markdown]
# ## 5. CI/CD in GitHub Actions (shown, validated, not run)
# A workflow is a YAML file in `.github/workflows/`. It says *when* to run (`on`), and lists **jobs** made of **steps**; `needs` orders jobs. Below: lint and tests on every push and pull request, then a training-plus-gate job, then an image build that only runs on the main branch.
# We cannot run GitHub Actions here, but we can parse the file and check it for common mistakes.

# %%
WORKFLOW = textwrap.dedent("""
    name: ml-ci

    on:
      pull_request:
      push:
        branches: [main]

    permissions:
      contents: read

    concurrency:
      group: ci-${{ github.ref }}
      cancel-in-progress: true

    jobs:
      test:
        runs-on: ubuntu-latest
        timeout-minutes: 15
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: "3.11"
              cache: pip
          - run: pip install -r requirements.txt pytest
          - name: Unit, data and behaviour tests
            run: pytest -q
            env:
              CHURN_DATA: datasets/churn.csv

      train-and-gate:
        needs: test
        runs-on: ubuntu-latest
        timeout-minutes: 20
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: "3.11"
              cache: pip
          - run: pip install -r requirements.txt
          - name: Train candidate (fails on invalid data)
            run: python -m churnml.train --data datasets/churn.csv --out artifacts/candidate
          - name: Quality gate against the champion
            run: python -m churnml.gate --candidate artifacts/candidate/metrics.json --baseline champion/metrics.json --max-drop 0.01
          - uses: actions/upload-artifact@v4
            with:
              name: candidate-model
              path: artifacts/candidate

      build-image:
        if: github.ref == 'refs/heads/main'
        needs: train-and-gate
        runs-on: ubuntu-latest
        timeout-minutes: 20
        steps:
          - uses: actions/checkout@v4
          - uses: docker/build-push-action@v6
            with:
              context: .
              push: false
              tags: churn-service:${{ github.sha }}
""").lstrip()

wf = yaml.safe_load(WORKFLOW)
print("top-level keys:", list(wf))
print("jobs:", list(wf["jobs"]))

# %% [markdown]
# Look at the top-level keys. There is no `on` key: there is a **`True`** key. YAML 1.1 (which PyYAML implements, and which some tooling inherits) reads the bare word `on` as the boolean true! GitHub reads it as the word `on`, but if *your* script parses workflows with PyYAML, you must handle both. This is the same family of gotcha as the unquoted `22:22` port in the Docker notebook.

# %%
print(True in wf, "on" in wf)
triggers = wf.get("on", wf.get(True))
print("triggers:", triggers)

# %%
def check_workflow(wf):
    problems = []
    triggers = wf.get("on", wf.get(True))
    if triggers is None:
        problems.append("no trigger ('on') defined")
    if "permissions" not in wf:
        problems.append("no top-level permissions: the default token may be broader than needed")
    jobs = wf.get("jobs", {})
    if not jobs:
        problems.append("no jobs")
    for name, job in jobs.items():
        if "runs-on" not in job:
            problems.append(f"{name}: missing runs-on")
        if "timeout-minutes" not in job:
            problems.append(f"{name}: no timeout-minutes (a hung job blocks for the default six hours)")
        needs = job.get("needs", [])
        for n in ([needs] if isinstance(needs, str) else needs):
            if n not in jobs:
                problems.append(f"{name}: needs unknown job '{n}'")
        for i, step in enumerate(job.get("steps", []), 1):
            label = f"{name} step {i}"
            if "run" not in step and "uses" not in step:
                problems.append(f"{label}: has neither run nor uses")
            uses = step.get("uses")
            if uses and not uses.startswith(("./", "docker://")):
                if "@" not in uses or uses.split("@", 1)[1] in ("main", "master", "latest", "HEAD"):
                    problems.append(f"{label}: action '{uses}' is not pinned to a version")
            blob = json.dumps({k: v for k, v in step.items() if k in ("run", "env", "with")})
            if re.search(r"(password|token|secret|api[_-]?key)\\W{1,4}[A-Za-z0-9/+_-]{8,}", blob, re.I) and "secrets." not in blob:
                problems.append(f"{label}: looks like a hard-coded secret; use ${{{{ secrets.NAME }}}}")
    return problems

print("our workflow:", check_workflow(wf))

broken = yaml.safe_load(textwrap.dedent("""
    name: bad
    on: push
    jobs:
      train:
        needs: [tests]
        steps:
          - uses: actions/checkout@main
          - run: python deploy.py --api-token abcd1234efgh5678
          - name: nothing to do
""").lstrip())
for p in check_workflow(broken):
    print("-", p)

# %% [markdown]
# Design choices in the good workflow worth stealing: **least-privilege** `permissions`; `concurrency` cancelling superseded runs; a `timeout-minutes` on every job; cached pip installs; the gate as its own visible step so a red X says exactly *why*; and uploaded artifacts so a reviewer can download the candidate model.
# Actions referenced by **tag** (`@v4`) are convenient but tags can be moved; the strictest practice pins a **commit SHA**. We do not show real SHAs because we cannot verify them offline.

# %% [markdown]
# ## 6. Continuous *delivery*: releasing a model safely
# CI answers "is this change acceptable?". CD answers "how do we put it in front of users without a bad day?" Patterns, from cheap to careful:
#
# | Strategy | How | Protects against | Cost |
# |---|---|---|---|
# | **Smoke test after deploy** | call `/health` and one known prediction; fail the deploy if wrong | a service that does not even start | tiny |
# | **Blue/green** | run new next to old, switch traffic in one step, switch back if needed | slow or risky rollbacks | double infrastructure briefly |
# | **Shadow** | new model receives a copy of live traffic, its answers are logged not used | bad predictions reaching users | compute + analysis effort |
# | **Canary** | send 1-5% of live traffic to the new model, widen if metrics hold | blast radius of a bad model | traffic routing, monitoring |
#
# Whatever you choose, **rollback must be a single, rehearsed action** (move the registry alias back, redeploy the previous image tag) and **monitoring must be watching the new version** from minute one (version in every response, as in the serving notebook).
# A smoke test is just a function you can write today:

# %%
def smoke_test(predict_fn, known_case, expected_min, expected_max):
    """After a deploy: the service must answer a known case with a probability in a sane range."""
    p = predict_fn(known_case)
    return expected_min <= p <= expected_max, p

ok, p = smoke_test(lambda row: float(pipe.predict_proba(pd.DataFrame([row]))[:, 1][0]),
                   {"tenure_months": 2, "monthly_charges": 80.0, "total_charges": 160.0, "support_calls": 6, "contract": "month-to-month",
                    "internet_service": "Fiber", "paperless_billing": "Yes", "autopay": "No", "senior_citizen": 0}, 0.5, 1.0)
print("smoke test passed:", ok, "| probability:", round(p, 3))

# %% [markdown]
# ## Common mistakes
#
# 1. **Testing only that code runs.** "Trains without error" says nothing about quality. Add behaviour tests and a metric gate.
# 2. **A gate with no noise analysis.** A threshold tighter than the evaluation noise blocks good models at random; one that is too loose lets regressions through. Measure the noise.
# 3. **Comparing against a stale or absent champion.** Regress against what is *currently live*, fetched from the registry, not against a number in a config file nobody updates.
# 4. **Averages hiding failing segments.** Gate on slices (new customers, each contract type) as well as overall.
# 5. **Flaky tests** from unseeded randomness, time, or network. Seed everything; inject clocks; never depend on the internet.
# 6. **Slow CI** (full retrain on every commit). Train on a fixed small sample in CI; run the full retrain and large-scale evaluation nightly or on demand.
# 7. **Secrets in workflow files or logs.** Use the platform's secret store and never echo them.
# 8. **Unpinned actions and dependencies.** A moved tag or a new library release changes your build without a commit. Pin versions.
# 9. **No rollback rehearsal.** The first time you roll back should not be during an incident.
# 10. **Skipping data tests in CI.** Most incidents are data, not code; validate the data before training and fail loudly.

# %% [markdown]
# ## Interview corner
#
# **1. How is testing an ML system different from testing ordinary software?**
# <details><summary>Answer</summary>
# Behaviour depends on data as well as code, outputs are statistical, and failures are often silent (a model that runs but is wrong). So you add data validation, feature tests, model behaviour tests (invariance, directional, minimum functionality), and metric quality gates on top of unit and integration tests, and keep everything deterministic and fast enough for CI.
# </details>
#
# **2. What goes into a CI pipeline for a model?**
# <details><summary>Answer</summary>
# Lint and type checks, unit tests, data validation, a small deterministic training run, behaviour tests and a metric gate against the current champion and an absolute floor, then build and scan the serving image, deploy to staging with a smoke test, and promote via canary or blue/green with monitoring and a rehearsed rollback.
# </details>
#
# **3. What are invariance and directional tests?**
# <details><summary>Answer</summary>
# Invariance: perturb something that should not matter (casing, an ID, row order, irrelevant wording) and assert the prediction is unchanged. Directional: change an input in a way with a known effect (more late payments should not lower default risk) and assert the output moves accordingly. Both catch failures that aggregate metrics average away.
# </details>
#
# **4. How do you decide a quality-gate threshold?**
# <details><summary>Answer</summary>
# Tie the floor to a baseline and to business need, and set the tolerated regression relative to the noise of the evaluation (bootstrap or repeated seeds), using paired comparisons on the same test rows. Also gate important slices, not just the average, and review thresholds as data and expectations change.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A slice gate
# An overall AUC can look fine while one group is badly served. Write `failing_slices(y_true, y_score, groups, min_auc)` that computes the ROC-AUC **within each group** (skip groups that contain only one class, since AUC is undefined there) and returns a dict `{group: auc}` for the groups whose AUC is **below** `min_auc`.
# Then use it on the trained churn model's test rows, grouped by `contract` (after cleaning), with `min_auc=0.65`, and store the result in `slice_failures` (it may be empty, which is a fine, reportable result).

# %% tags=["starter"]
def failing_slices(y_true, y_score, groups, min_auc):
    ...

slice_failures = ...

# %%
yt = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1])
ys = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.8, 0.2, 0.5, 0.6])            # group a: perfect, group b: reversed, group c: one class only
gr = np.array(["a", "a", "a", "a", "b", "b", "b", "b", "c", "c"])
assert failing_slices(yt, ys, gr, 0.65) == {"b": 0.0}, failing_slices(yt, ys, gr, 0.65)
assert failing_slices(yt, ys, gr, -1) == {}, "nothing is below -1"
contract_groups = model_mod.clean(test)["contract"].to_numpy()
expected = {g: roc_auc_score(y[contract_groups == g], scores[contract_groups == g]) for g in np.unique(contract_groups)
            if len(np.unique(y[contract_groups == g])) == 2}
expected = {g: a for g, a in expected.items() if a < 0.65}
assert set(slice_failures) == set(expected) and all(abs(slice_failures[g] - expected[g]) < 1e-12 for g in expected), (slice_failures, expected)
print("Exercise 1 passed: slices below 0.65 ->", slice_failures or "none")

# %% tags=["solution"]
def failing_slices(y_true, y_score, groups, min_auc):
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(y_true[m])) < 2:
            continue
        auc = roc_auc_score(y_true[m], y_score[m])
        if auc < min_auc:
            out[str(g)] = float(auc)
    return out

slice_failures = failing_slices(y, scores, model_mod.clean(test)["contract"].to_numpy(), 0.65)

# %% [markdown]
# ### Exercise 2: Write a behaviour test that catches a sabotaged model
# Write the **text of a pytest file** `MY_TEST` that uses the fixtures `model` and `raw_df` from `conftest.py` and contains one test, `test_contract_effect_not_reversed`, asserting a **directional expectation about `contract`**:
# for at least 95% of a sample of 300 rows, changing `contract` to `"month-to-month"` should give a churn probability **greater than or equal to** changing it to `"two year"`.
# The cell below writes your file to the project and runs `pytest` twice: normally (your test must pass) and with `CHURN_SABOTAGE=contract_swapped` (your test must **fail**: in that mode the model is trained with the labels' relationship to `contract` reversed).
# *Hint: use `rows.assign(contract="month-to-month")` and `model.predict_proba(...)[:, 1]`, like the tests in `test_model_behaviour.py`. The `clean` step lower-cases the contract, so either casing works.*

# %%
conftest = PROJ / "tests" / "conftest.py"
patched = conftest.read_text().replace(
    '''    return m.build_pipeline().fit(train, train["churned"])''',
    '''    if os.environ.get("CHURN_SABOTAGE") == "contract_swapped":
        swap = {"Month-to-month": "Two year", "month-to-month": "two year", "Two year": "Month-to-month", "two year": "month-to-month"}
        train = train.assign(contract=train["contract"].replace(swap))
    return m.build_pipeline().fit(train, train["churned"])''')
assert patched != conftest.read_text(), "conftest patch did not apply"
conftest.write_text(patched)
print("conftest now supports CHURN_SABOTAGE=contract_swapped")

# %% tags=["starter"]
MY_TEST = '''
'''

# %%
(PROJ / "tests" / "test_mine.py").write_text(textwrap.dedent(MY_TEST).lstrip())
normal_code, normal_out = run_pytest(["tests/test_mine.py"])
sabotage_code, sabotage_out = run_pytest(["tests/test_mine.py"], extra_env={"CHURN_SABOTAGE": "contract_swapped"})
assert "test_contract_effect_not_reversed" in MY_TEST, "name your test test_contract_effect_not_reversed"
assert normal_code == 0, f"your test must PASS on the healthy model:\n{normal_out[-600:]}"
assert sabotage_code == 1, f"your test must FAIL (exit code 1, a real assertion failure) on the sabotaged model; got {sabotage_code}:\n{sabotage_out[-600:]}"
print("Exercise 2 passed: passes on the healthy model, fails on the sabotaged one")

# %% tags=["solution"]
MY_TEST = '''
    import numpy as np

    def test_contract_effect_not_reversed(model, raw_df):
        rows = raw_df.sample(300, random_state=5)
        mtm = model.predict_proba(rows.assign(contract="month-to-month"))[:, 1]
        two = model.predict_proba(rows.assign(contract="two year"))[:, 1]
        assert np.mean(mtm >= two - 1e-9) >= 0.95
'''

# %% [markdown]
# ### Exercise 3: Unpinned actions
# Write `unpinned_actions(workflow_dict)` returning a sorted list of strings `"<job>: <uses>"` for every step that uses a **remote** action (not starting with `./` or `docker://`) which is **not pinned**: no `@ref` at all, or a ref of `main`, `master`, `latest` or `HEAD`. Run it on `broken` (from above) and on `wf`, storing the results in `bad_refs` and `good_refs`.

# %% tags=["starter"]
def unpinned_actions(workflow_dict):
    ...

bad_refs = ...
good_refs = ...

# %%
assert bad_refs == ["train: actions/checkout@main"], bad_refs
assert good_refs == [], good_refs
sample = {"jobs": {"a": {"steps": [{"uses": "actions/cache"}, {"uses": "./local-action"}, {"uses": "docker://alpine:3.19"}, {"uses": "org/tool@v2"}, {"uses": "org/x@latest"}, {"run": "echo hi"}]}}}
assert unpinned_actions(sample) == ["a: actions/cache", "a: org/x@latest"], unpinned_actions(sample)
print("Exercise 3 passed")

# %% tags=["solution"]
def unpinned_actions(workflow_dict):
    found = []
    for job_name, job in workflow_dict.get("jobs", {}).items():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if not uses or uses.startswith(("./", "docker://")):
                continue
            ref = uses.split("@", 1)[1] if "@" in uses else None
            if ref is None or ref in ("main", "master", "latest", "HEAD"):
                found.append(f"{job_name}: {uses}")
    return sorted(found)

bad_refs = unpinned_actions(broken)
good_refs = unpinned_actions(wf)

# %% [markdown]
# ### Exercise 4 (open-ended): Write the CI plan for your project
# For a model you own, list (in a markdown cell): the data tests you would run, three invariance or directional tests that encode domain knowledge, the gate's metric, floor and tolerance (with how you measured the noise), the slices you would gate, the deployment strategy and the exact rollback command. Then write the first of those tests as real code.

# %% [markdown]
# ## Recap
# - ML CI tests four layers: **data**, **features/code**, **model behaviour** (invariance, directional, minimum functionality) and a **metric quality gate**; everything seeded and fast.
# - A test suite must be **proven able to fail**: sabotaging the model or code and seeing the right test go red is how you know the tests are not decorative.
# - The **gate** exits non-zero on regression (vs the champion and an absolute floor); choose the tolerated drop relative to **evaluation noise** (bootstrap), compare on the same rows, and gate **slices** too. Invalid data should stop training outright.
# - A GitHub Actions workflow is YAML: jobs, steps, `needs`; parse and lint it (mind `on` parsing as `True`, unpinned actions, missing timeouts, hard-coded secrets).
# - Deploy with smoke tests, then blue/green, shadow or canary, with a rehearsed one-step **rollback** and monitoring on the new version from the first minute.
#
# **Go deeper:** the GitHub Actions documentation (docs.github.com/actions), "Workflow syntax" and "Security hardening"; the pytest documentation on fixtures (docs.pytest.org); Ribeiro et al., "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList" (ACL 2020) for the behavioural testing idea; Breck et al., "The ML Test Score: A Rubric for ML Production Readiness" (Google, 2017).
#
# **Next:** this module is complete. Continue with [21 LLMs and RAG](../21_LLMs_and_RAG/).
