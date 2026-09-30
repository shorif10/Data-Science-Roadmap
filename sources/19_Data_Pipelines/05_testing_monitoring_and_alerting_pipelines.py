# %% [markdown]
# # Testing, Monitoring and Alerting Pipelines
#
# **Why this matters.** The worst pipeline failures do not crash. The job is green, the table is there, and the numbers are quietly wrong: a source started sending nulls, a join fanned out, yesterday's partition is empty.
# Someone finds out from a dashboard three weeks later. Two habits prevent most of that: **tests** that catch logic bugs before deploy, and **monitors plus alerts** that catch data bugs after deploy, routed to a human who can act.
#
# **You will learn**
# - structure a pipeline so its transformations are unit-testable, and run `pytest` against a real package
# - write a **data contract** (schema, types, nullability, ranges, allowed values) and detect breaking changes
# - build monitors: **row count**, **freshness**, **null rate**, compared with a rolling baseline instead of a fixed threshold
# - turn monitor results into **alerts** with severity, de-duplication and cooldown, and avoid alert fatigue
# - measure an **SLA/SLO** (freshness target, error budget) from run history
#
# **Prerequisites:** [Data quality and validation](../18_Data_Engineering/04_data_quality_and_validation.ipynb), [Build an ETL pipeline](02_build_an_etl_pipeline_in_python.ipynb) and [Orchestration and DAGs](03_orchestration_and_dags.ipynb).
#
# > Data used: `orders.csv` and `churn.csv` (synthetic, see `docs/DATASETS.md`). Everything runs in a temp folder.

# %%
import atexit
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="pipe_test_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])   # 'None' internet plan is a real category
print(orders.shape, churn.shape)

# %% [markdown]
# ## 1. Make the logic testable
# The single most useful design rule: **separate pure transformations from I/O**. A function `clean_orders(df) -> df` can be tested with a five-row DataFrame in milliseconds. A function that reads S3, transforms, and writes to a warehouse can only be tested with S3 and a warehouse.
# Keep the I/O in a thin shell, keep the decisions in pure functions.
#
# We will write a tiny package to a temp directory (so we can run the real `pytest` on it), with one pure function and one idempotent loader.

# %%
PKG = TMP / "mypipe_project"
(PKG / "mypipe").mkdir(parents=True)
(PKG / "tests").mkdir()

(PKG / "mypipe" / "__init__.py").write_text("")
(PKG / "mypipe" / "transforms.py").write_text(textwrap.dedent('''
    import pandas as pd

    VALID_STATUS = {"delivered", "shipped", "cancelled", "returned"}

    def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise status text, drop duplicate order ids (keep last), drop rows with an unknown status."""
        out = df.copy()
        out["status"] = out["status"].str.strip().str.lower()
        out = out.drop_duplicates("order_id", keep="last")
        return out[out["status"].isin(VALID_STATUS)].reset_index(drop=True)

    def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
        """Orders per day, with missing days filled with 0 so downstream code never sees gaps."""
        counts = df.groupby("order_date").size()
        full = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
        return counts.reindex(full, fill_value=0).rename_axis("order_date").reset_index(name="n_orders")
'''))
(PKG / "mypipe" / "load.py").write_text(textwrap.dedent('''
    import sqlite3

    def load_orders(con: sqlite3.Connection, rows: list) -> int:
        """Idempotent load keyed on order_id."""
        con.execute("CREATE TABLE IF NOT EXISTS orders (order_id INTEGER PRIMARY KEY, status TEXT)")
        with con:
            con.executemany(
                "INSERT INTO orders VALUES (?, ?) ON CONFLICT(order_id) DO UPDATE SET status = excluded.status", rows)
        return con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
'''))
print(sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py")))

# %% [markdown]
# ## 2. Write the tests, run them with `pytest`
# Three kinds of test cover most pipeline bugs:
# * **Unit tests** of transformations on tiny hand-made inputs, including the ugly cases (duplicates, blanks, empty input).
# * **Idempotency tests**: run the load twice, assert the table is unchanged.
# * **Property/invariant tests**: rules that must hold for any input (no duplicate keys, row count never increases after cleaning).

# %%
(PKG / "tests" / "test_transforms.py").write_text(textwrap.dedent('''
    import sqlite3
    import pandas as pd
    import pytest
    from mypipe.transforms import clean_orders, daily_counts
    from mypipe.load import load_orders

    @pytest.fixture
    def raw():
        return pd.DataFrame({
            "order_id": [1, 2, 2, 3, 4],
            "order_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-03", "2025-01-03"]),
            "status": ["Delivered ", "shipped", "DELIVERED", "weird", "cancelled"],
        })

    def test_status_is_normalised(raw):
        assert set(clean_orders(raw)["status"]) <= {"delivered", "shipped", "cancelled", "returned"}

    def test_duplicates_removed_keeping_last(raw):
        out = clean_orders(raw)
        assert out["order_id"].is_unique
        assert out.loc[out["order_id"] == 2, "status"].item() == "delivered"   # the later record wins

    def test_unknown_status_dropped(raw):
        assert 3 not in clean_orders(raw)["order_id"].tolist()

    def test_cleaning_never_adds_rows(raw):
        assert len(clean_orders(raw)) <= len(raw)

    def test_daily_counts_fills_gaps(raw):
        counts = daily_counts(clean_orders(raw))
        assert counts["order_date"].is_monotonic_increasing
        assert counts.set_index("order_date")["n_orders"].to_dict() == {
            pd.Timestamp("2025-01-01"): 2, pd.Timestamp("2025-01-02"): 0, pd.Timestamp("2025-01-03"): 1}

    def test_empty_input_is_handled():
        empty = pd.DataFrame({"order_id": [], "order_date": pd.to_datetime([]), "status": pd.Series([], dtype="str")})
        assert clean_orders(empty).empty

    def test_load_is_idempotent():
        con = sqlite3.connect(":memory:")
        rows = [(1, "shipped"), (2, "delivered")]
        assert load_orders(con, rows) == 2
        assert load_orders(con, rows) == 2          # the second run changes nothing
'''))

def run_pytest(args=()):
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", *args],
                          cwd=PKG, capture_output=True, text=True, timeout=120, env={**__import__("os").environ, "PYTHONPATH": str(PKG), "PYTHONDONTWRITEBYTECODE": "1"})
    return proc.returncode, proc.stdout

code, out = run_pytest()
print("exit code:", code)
print("\n".join(out.strip().splitlines()[-12:]))

# %% [markdown]
# Exit code `0` means every test passed; that number is what CI looks at. Did we write useful tests, or only green ones? Break the code on purpose and see whether a test notices (this is the idea behind **mutation testing**):

# %%
transforms = PKG / "mypipe" / "transforms.py"
original = transforms.read_text()
transforms.write_text(original.replace('.drop_duplicates("order_id", keep="last")', '.drop_duplicates("order_id", keep="first")'))   # subtle bug
code, out = run_pytest()
print("exit code with the bug:", code)
print([line for line in out.splitlines() if line.startswith(("FAILED", "E   "))][:4])
transforms.write_text(original)                       # restore
print("after restoring:", run_pytest()[0])

# %% [markdown]
# *What just happened?* A one-word change (`first` vs `last`) was caught by `test_duplicates_removed_keeping_last`. If no test had failed, that would tell you the suite does not pin down that behaviour.
# Good habits: name the test after the rule, test one rule per test, keep fixtures tiny and readable, and write a failing test **first** when you fix a bug so it cannot come back.

# %% [markdown]
# ## 3. Data contracts
# Tests check your code. A **data contract** checks the *data you are handed* (or hand to others): an explicit, versioned agreement about columns, types, nullability, ranges and allowed values. The producer promises it; the consumer validates it at the boundary. Keep the contract as data, not as scattered `if`s:

# %%
CONTRACT = {
    "name": "churn_customers", "version": "1.0.0",
    "columns": {
        "customer_id":     {"dtype": "str",   "nullable": False, "unique": True},
        "tenure_months":   {"dtype": "int",   "nullable": False, "min": 0, "max": 120},
        "contract":        {"dtype": "str",   "nullable": False, "allowed": ["Month-to-month", "One year", "Two year"]},
        "monthly_charges": {"dtype": "float", "nullable": False, "min": 0, "max": 500},
        "total_charges":   {"dtype": "float", "nullable": True,  "min": 0},
        "churned":         {"dtype": "int",   "nullable": False, "allowed": [0, 1]},
    },
}

def kind(series):
    if pd.api.types.is_bool_dtype(series): return "bool"
    if pd.api.types.is_integer_dtype(series): return "int"
    if pd.api.types.is_float_dtype(series): return "float"
    return "str"

def validate_contract(df, contract):
    """Return a list of human-readable violations (empty list = the data honours the contract)."""
    problems = []
    for col, rule in contract["columns"].items():
        if col not in df.columns:
            problems.append(f"{col}: column missing")
            continue
        s = df[col]
        if kind(s) != rule["dtype"] and not (rule["dtype"] == "float" and kind(s) == "int"):
            problems.append(f"{col}: dtype {kind(s)} but contract says {rule['dtype']}")
        n_null = int(s.isna().sum())
        if n_null and not rule["nullable"]:
            problems.append(f"{col}: {n_null} nulls but not nullable")
        if rule.get("unique") and s.duplicated().any():
            problems.append(f"{col}: {int(s.duplicated().sum())} duplicate values")
        if "allowed" in rule:
            bad = set(s.dropna().unique()) - set(rule["allowed"])
            if bad:
                problems.append(f"{col}: values outside the allowed set: {sorted(map(str, bad))[:3]}")
        if kind(s) in ("int", "float"):
            if "min" in rule and (s < rule["min"]).any():
                problems.append(f"{col}: {int((s < rule['min']).sum())} values below {rule['min']}")
            if "max" in rule and (s > rule["max"]).any():
                problems.append(f"{col}: {int((s > rule['max']).sum())} values above {rule['max']}")
    return problems

for p in validate_contract(churn, CONTRACT):
    print("-", p)

# %% [markdown]
# The real churn data already violates the contract, and that is useful: the `contract` column has inconsistent casing (`month-to-month` vs `Month-to-month`), planted on purpose.
# This is exactly how a contract earns its keep: the violation is reported with a count **at the boundary**, instead of surfacing as a mysterious drop in a model's accuracy. Apply the fix the contract points to and validate again:

# %%
fixed = churn.assign(contract=churn["contract"].str.strip().str.lower().map(
    {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}))
print("violations after normalising:", validate_contract(fixed, CONTRACT))

# %% [markdown]
# ### Breaking vs non-breaking changes
# A contract is also how a producer knows what they may change. Compare two versions and classify the diff:

# %%
def diff_contracts(old, new):
    """Classify schema changes. Removing/retyping a column or making it stricter breaks consumers; adding an optional column does not."""
    breaking, safe = [], []
    for col, rule in old["columns"].items():
        if col not in new["columns"]:
            breaking.append(f"removed column {col}")
            continue
        nr = new["columns"][col]
        if nr["dtype"] != rule["dtype"]:
            breaking.append(f"{col}: dtype {rule['dtype']} -> {nr['dtype']}")
        if rule["nullable"] is False and nr["nullable"] is True:
            breaking.append(f"{col}: became nullable")
        if "allowed" in rule and "allowed" in nr and set(nr["allowed"]) - set(rule["allowed"]):
            breaking.append(f"{col}: new values {sorted(set(nr['allowed']) - set(rule['allowed']))} may surprise consumers")
    for col, nr in new["columns"].items():
        if col not in old["columns"]:
            (safe if nr["nullable"] else breaking).append(f"added column {col} ({'nullable' if nr['nullable'] else 'required'})")
    return {"breaking": breaking, "safe": safe}

import copy
v2 = copy.deepcopy(CONTRACT)
v2["columns"]["support_calls"] = {"dtype": "int", "nullable": True}                  # add optional column: fine
v2["columns"]["total_charges"]["dtype"] = "str"                                       # retype: breaking
v2["columns"]["contract"]["allowed"].append("Three year")                             # widen enum: consumers may not handle it
del v2["columns"]["churned"]                                                          # remove: breaking
diff_contracts(CONTRACT, v2)

# %% [markdown]
# Run `diff_contracts` in CI on the producer's pull request and a breaking change needs a version bump and a conversation *before* it ships, not an incident after.

# %% [markdown]
# ## 4. Monitors: what to measure after deploy
# Tests run once per code change. **Monitors** run on every data load, because data changes without anyone touching the code. Start with four, they catch most real incidents:
#
# | Monitor | Question | Typical failure it catches |
# |---|---|---|
# | **Volume** | did we load about the usual number of rows? | a failed upstream extract, a bad filter, a fan-out join |
# | **Freshness** | how old is the newest record? | a stuck scheduler, a source that stopped publishing |
# | **Null rate** | did the share of missing values jump? | a renamed source field, a broken parser |
# | **Distribution** | did a key metric move implausibly? | a unit change, a duplicated load |
#
# A fixed threshold ("fewer than 100 rows is bad") goes stale; compare with a **rolling baseline** of recent healthy runs. We will keep run history in a table, the way real tools do.

# %%
# The real orders table has only ~7 orders a day, far too few for volume monitoring to be meaningful (Poisson noise swamps everything).
# So we SIMULATE the daily row count of a busy events feed: about 2,000 rows a day, quieter on weekends. The mechanics are identical on real counts.
rng = np.random.default_rng(42)
days90 = pd.date_range("2025-03-01", periods=91, freq="D")
weekday_factor = np.array([1.0, 1.05, 1.05, 1.0, 0.95, 0.6, 0.55])          # Mon..Sun
expected = 2000 * weekday_factor[days90.dayofweek]
daily = pd.DataFrame({"order_date": days90, "n_orders": rng.poisson(expected)})
hist = daily.copy()
print(len(hist), "days; median volume:", int(hist["n_orders"].median()), "rows/day; weekend median:", int(hist.loc[hist["order_date"].dt.dayofweek >= 5, "n_orders"].median()))

# inject an incident into a copy so we know the ground truth
incidents = hist.copy()
incidents.loc[60, "n_orders"] = int(incidents.loc[60, "n_orders"] * 0.15)          # extract half-failed: 15% of normal volume
print("incident day:", incidents.loc[60, "order_date"].strftime("%Y-%m-%d (%A)"), "->", int(incidents.loc[60, "n_orders"]), "rows")

# %%
def robust_bounds(history, k=4.0):
    """Median +/- k * scaled MAD: resistant to the very outliers we are hunting."""
    med = np.median(history)
    mad = np.median(np.abs(history - med)) * 1.4826
    mad = max(mad, 0.05 * med, 1e-9)                   # floor so a perfectly flat history does not alarm on any wobble
    return med - k * mad, med + k * mad

def volume_monitor(series, window=28, k=4.0):
    """For each day, compare with bounds computed from the previous `window` days only (no peeking)."""
    results = []
    for i in range(window, len(series)):
        lo, hi = robust_bounds(series.iloc[i - window:i].to_numpy(), k)
        v = series.iloc[i]
        results.append({"i": i, "value": int(v), "lo": round(lo, 1), "hi": round(hi, 1), "ok": bool(lo <= v <= hi)})
    return pd.DataFrame(results)

vol = volume_monitor(incidents["n_orders"])
flagged = vol[~vol["ok"]]
print(f"checked {len(vol)} days, flagged {len(flagged)}")
flagged.assign(date=incidents.loc[flagged["i"], "order_date"].dt.strftime("%Y-%m-%d").values)

# %% [markdown]
# The monitor caught the injected incident (Wednesday, 316 rows) but look at the other flags: nearly all are **weekends**. Real volume has a weekly rhythm, so a baseline that mixes all seven weekdays treats every normal Saturday as an anomaly.
# That is the honest cost of monitoring: a monitor with a bad baseline cries wolf, and people stop listening. The fix is a baseline that matches the seasonality (compare a Saturday with previous Saturdays); you will build it in Exercise 3.
# Why median/MAD instead of mean/standard deviation? One earlier outlier inflates a standard deviation and hides the next incident. And always count **false alarms** as well as catches:

# %%
false_alarms = flagged[flagged["i"] != 60]
weekend_share = (incidents.loc[false_alarms["i"], "order_date"].dt.dayofweek >= 5).mean()
print(f"false alarms: {len(false_alarms)} of {len(vol) - 1} normal days, {weekend_share:.0%} of them on weekends")

# %% [markdown]
# Some of those may be genuine weekday or seasonal wobble, which is the real-world cost of monitoring: tighten `k` and you miss incidents, loosen it and you page people for nothing. There is no free setting; choose by the cost of each error and review it periodically.
#
# Now freshness and null rate, which are simpler:

# %%
def freshness_monitor(latest_ts, now, max_age_hours):
    age = (now - latest_ts).total_seconds() / 3600
    return {"monitor": "freshness", "ok": age <= max_age_hours, "detail": f"newest record is {age:.1f} h old (limit {max_age_hours} h)"}

def null_rate_monitor(df, column, baseline_rate, tolerance=0.02):
    rate = float(df[column].isna().mean())
    return {"monitor": f"null_rate[{column}]", "ok": rate <= baseline_rate + tolerance, "detail": f"{rate:.1%} null (baseline {baseline_rate:.1%} +{tolerance:.0%})"}

now = datetime(2025, 6, 1, 6, 0)
print(freshness_monitor(datetime(2025, 5, 31, 23, 50), now, max_age_hours=12))
print(freshness_monitor(datetime(2025, 5, 29, 23, 50), now, max_age_hours=12))
# churn: total_charges has a small, known share of missing values
print(null_rate_monitor(churn, "total_charges", baseline_rate=float(churn["total_charges"].isna().mean())))
print(null_rate_monitor(churn.assign(total_charges=churn["total_charges"].where(rng.random(len(churn)) > 0.3)), "total_charges", baseline_rate=float(churn["total_charges"].isna().mean())))

# %% [markdown]
# ## 5. From monitor results to alerts
# A monitor produces a *result*; an **alert** is a decision to interrupt a person. Keep them separate, because most results should **not** page anyone. A workable rule set:
#
# | Severity | Meaning | Route | Example |
# |---|---|---|---|
# | `page` | needs a human now, business impact likely | on-call phone | fact table empty, SLA about to be breached |
# | `ticket` | fix this week | issue tracker | null rate drifting up |
# | `info` | record only | dashboard | small volume dip inside tolerance |
#
# Two features separate a usable alerting system from a noisy one: **de-duplication** (one incident = one alert) and **cooldown** (do not repeat the same alert every five minutes). Alert fatigue is a real failure mode: people mute noisy channels, and then miss the real page.

# %%
class AlertManager:
    def __init__(self, cooldown_minutes=60):
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_sent = {}               # alert key -> time
        self.outbox = []                  # stand-in for Slack / PagerDuty / email

    def fire(self, key, severity, message, at):
        last = self.last_sent.get(key)
        if last is not None and at - last < self.cooldown:
            return False                  # suppressed: same alert still in cooldown
        self.last_sent[key] = at
        self.outbox.append({"at": at.strftime("%H:%M"), "severity": severity, "key": key, "message": message})
        return True

am = AlertManager(cooldown_minutes=60)
t0 = datetime(2025, 6, 1, 6, 0)
sent = [am.fire("orders.volume", "page", "orders volume 15% of baseline", t0 + timedelta(minutes=5 * i)) for i in range(6)]   # same failing check every 5 min
sent.append(am.fire("orders.volume", "page", "orders volume 15% of baseline", t0 + timedelta(minutes=70)))                 # still failing an hour later
sent.append(am.fire("orders.nulls", "ticket", "customer_id null rate 32%", t0 + timedelta(minutes=10)))                   # different key
print("sent flags:", sent)
pd.DataFrame(am.outbox)

# %% [markdown]
# Eight evaluations produced three messages: one per distinct problem, plus a reminder after the cooldown. Good alert messages say **what** is wrong, **how bad** (observed vs expected), **where** to look (dashboard or runbook link), and **who** owns it. "Check failed" is not an alert; "orders volume 15% of the 28-day median, runbook: ..." is.

# %% [markdown]
# ## 6. SLAs, SLOs and error budgets
# An **SLA** is a promise to consumers ("the daily sales table is ready by 07:00"); an **SLO** is your internal target for meeting it (say 95% of days); the **error budget** is how many misses that target allows. Measure it from run history, not feelings:

# %%
rng = np.random.default_rng(42)
days = pd.date_range("2025-03-01", periods=90, freq="D")
finish_hour = 5.0 + rng.gamma(2.0, 0.45, len(days))               # when the table became ready, in hours after midnight
finish_hour[[20, 21, 55]] += 4                                     # three bad days
runs = pd.DataFrame({"day": days, "ready_at_hour": finish_hour.round(2)})
SLA_HOUR = 7.0
runs["met_sla"] = runs["ready_at_hour"] <= SLA_HOUR

slo_target = 0.95
achieved = runs["met_sla"].mean()
budget_days = int(np.floor((1 - slo_target) * len(runs)))
missed = int((~runs["met_sla"]).sum())
print(f"SLA met on {achieved:.1%} of {len(runs)} days (SLO {slo_target:.0%})")
print(f"error budget: {budget_days} missed days allowed, {missed} used, {budget_days - missed} left")
print("p50 / p95 ready time (hours):", runs["ready_at_hour"].quantile([0.5, 0.95]).round(2).tolist())

# %% [markdown]
# Reading it: the **p95** ready time shows how close you usually run to the deadline; if p95 is near 07:00, a small slowdown breaches the SLA, so alert on a **leading** indicator (for example "the job has not finished by 06:30") rather than only on the breach itself.
# When the error budget is spent, the team's priority shifts from new features to reliability; that is what makes an SLO useful instead of decorative.

# %% [markdown]
# ## Common mistakes
#
# 1. **Testing only the happy path.** Empty input, duplicates, nulls, a single row and a schema with an extra column are where pipelines really break.
# 2. **Tests that need production**: network, real warehouse, today's date. Flaky tests get ignored. Use tiny fixtures, temp databases and injected clocks (note how `freshness_monitor` takes `now` as a parameter).
# 3. **Fixed thresholds that never get revisited.** Volume grows, the threshold goes stale, the alert either fires constantly or never.
# 4. **Alerting on every failed check.** Without severity, de-duplication and cooldown, on-call learns to ignore the channel.
# 5. **Monitoring only the job status.** A green run that loaded zero rows is still a failure: monitor the data, not just the process.
# 6. **Contracts that live in a wiki.** If it is not validated automatically at the boundary, it is a wish.
# 7. **No owner and no runbook.** An alert nobody knows how to act on is noise with a delay.

# %% [markdown]
# ## Interview corner
#
# **1. How do you test a data pipeline?**
# <details><summary>Answer</summary>
# Layers: unit tests for pure transformations (edge cases, duplicates, nulls, empty input); idempotency tests (run twice, same result); integration tests on a small sample through the real steps against a throwaway database;
# schema/contract checks at boundaries; and in production, monitors for volume, freshness and distribution. Keep I/O thin so most logic is unit-testable.
# </details>
#
# **2. What is a data contract?**
# <details><summary>Answer</summary>
# An explicit, versioned agreement between a data producer and its consumers covering schema, types, nullability, allowed values, semantics, freshness and ownership, validated automatically. It moves breakage discovery from downstream dashboards to the producer's CI or the ingestion boundary.
# </details>
#
# **3. Your volume monitor pages you twice a week and it is always a false alarm. What do you do?**
# <details><summary>Answer</summary>
# Look at why it fires (weekday seasonality, holidays, growth). Compare with a same-weekday baseline or model seasonality, use robust statistics, widen bands where cost of a miss is low, route low-confidence signals to a ticket instead of a page, and track the precision of the alert so you can tune it.
# </details>
#
# **4. SLA vs SLO vs SLI?**
# <details><summary>Answer</summary>
# SLI is the measurement (fraction of days the table was ready by 07:00). SLO is your target for it (95%). SLA is the external promise, often with consequences, usually looser than the SLO so you have margin. The gap between 100% and the SLO is the error budget.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Write the test that would have caught it
# The function below is meant to return the **share of delivered orders** among non-cancelled orders, but has a bug. Write `test_delivered_share` (a plain function using `assert`) that **fails on `buggy_share` and passes on `correct_share`** for a hand-made DataFrame, then store the two outcomes in `buggy_ok` and `correct_ok` (True if the test passed on that function).

# %%
def buggy_share(df):
    return (df["status"] == "delivered").mean()             # cancelled orders wrongly stay in the denominator

def correct_share(df):
    active = df[df["status"] != "cancelled"]
    return (active["status"] == "delivered").mean()

def passes(test, fn):
    try:
        test(fn)
        return True
    except AssertionError:
        return False

# %% tags=["starter"]
def test_delivered_share(fn):
    ...

buggy_ok = passes(test_delivered_share, buggy_share)
correct_ok = passes(test_delivered_share, correct_share)

# %%
assert buggy_ok is False, "your test must FAIL on the buggy implementation (it passed, so it does not pin the behaviour down)"
assert correct_ok is True, "your test must PASS on the correct implementation"
print("Exercise 1 passed: the test separates the two implementations")

# %% tags=["solution"]
def test_delivered_share(fn):
    df = pd.DataFrame({"status": ["delivered", "delivered", "shipped", "cancelled", "cancelled"]})
    assert abs(fn(df) - 2 / 3) < 1e-9            # 2 delivered out of 3 non-cancelled orders

buggy_ok = passes(test_delivered_share, buggy_share)
correct_ok = passes(test_delivered_share, correct_share)

# %% [markdown]
# ### Exercise 2: A contract for orders
# Write a contract `ORDERS_CONTRACT` in the same format as `CONTRACT` for `orders.csv` (columns `order_id`, `customer_id`, `order_date`, `status`, `channel`). `order_id` must be an integer, unique and not null; `status` must be limited to the statuses that really occur; `order_date` is left out (it is parsed as text in this load).
# Then validate the real data and store the violation list in `orders_violations`. Finally, store in `broken_violations` the violations you get when you validate a copy of the orders with `status` set to `"refunded"` for one row.
# *Hint: read the allowed statuses from the data when you define the contract only if you are sure they are right; here use the four listed in `docs/DATASETS.md`.*

# %% tags=["starter"]
ORDERS_CONTRACT = ...
orders_violations = ...
broken_violations = ...

# %%
assert orders_violations == [], f"the real orders should satisfy a sensible contract, got {orders_violations}"
assert len(broken_violations) == 1 and broken_violations[0].startswith("status"), f"expected one status violation, got {broken_violations}"
assert "order_id" in ORDERS_CONTRACT["columns"] and ORDERS_CONTRACT["columns"]["order_id"].get("unique") is True
print("Exercise 2 passed:", broken_violations)

# %% tags=["solution"]
ORDERS_CONTRACT = {
    "name": "orders", "version": "1.0.0",
    "columns": {
        "order_id":    {"dtype": "int", "nullable": False, "unique": True, "min": 1},
        "customer_id": {"dtype": "int", "nullable": False},
        "status":      {"dtype": "str", "nullable": False, "allowed": ["delivered", "shipped", "cancelled", "returned"]},
        "channel":     {"dtype": "str", "nullable": False},
    },
}
orders_violations = validate_contract(orders, ORDERS_CONTRACT)
bad = orders.copy()
bad.loc[0, "status"] = "refunded"
broken_violations = validate_contract(bad, ORDERS_CONTRACT)

# %% [markdown]
# ### Exercise 3: A weekday-aware volume monitor
# Orders have a weekly rhythm, so comparing Monday with the previous 28 days of all weekdays is noisy. Write `same_weekday_bounds(series_by_date, day, weeks=8, k=4.0)` that returns `(lo, hi)` computed with `robust_bounds` from **the same weekday in the previous `weeks` weeks** (strictly before `day`).
# `series_by_date` is a Series indexed by `Timestamp`.

# %% tags=["starter"]
def same_weekday_bounds(series_by_date, day, weeks=8, k=4.0):
    ...

# %%
s = daily.set_index("order_date")["n_orders"]
day = pd.Timestamp("2025-05-12")                              # a Monday
lo, hi = same_weekday_bounds(s, day, weeks=8, k=4.0)
prior = s[[day - pd.Timedelta(weeks=w) for w in range(1, 9)]]
assert (lo, hi) == robust_bounds(prior.to_numpy(), 4.0), "use exactly the 8 previous same-weekday values"
assert day not in prior.index, "never include the day being checked"
# run it as a monitor over the incident series: it should catch the incident and cry wolf far less than the mixed baseline
inc = incidents.set_index("order_date")["n_orders"]
alarms = []
for i in range(28, len(inc)):
    d = inc.index[i]
    lo_i, hi_i = same_weekday_bounds(inc, d)
    if not (lo_i <= inc.iloc[i] <= hi_i):
        alarms.append(i)
assert 60 in alarms, "the injected incident (index 60) must still be flagged"
assert len(alarms) - 1 < len(false_alarms) / 2, f"expected far fewer false alarms than the mixed-weekday monitor ({len(false_alarms)}), got {len(alarms) - 1}"
print(f"Exercise 3 passed: {len(alarms) - 1} false alarms instead of {len(false_alarms)}; Monday bounds {lo:.1f} .. {hi:.1f}, actual {int(s[day])}")

# %% tags=["solution"]
def same_weekday_bounds(series_by_date, day, weeks=8, k=4.0):
    idx = [day - pd.Timedelta(weeks=w) for w in range(1, weeks + 1)]
    history = series_by_date.reindex(idx).dropna().to_numpy()
    return robust_bounds(history, k)

# %% [markdown]
# ### Exercise 4: An alert rule with escalation
# Extend alerting. Write `escalate(alerts, now, ticket_age_hours=24)`. `alerts` is a list of dicts `{"key", "severity", "opened_at"}` for still-open problems. Return a **new list** of the same dicts where any `ticket` alert open for **more than** `ticket_age_hours` has its severity changed to `"page"`; everything else is unchanged and the input list must not be modified.

# %% tags=["starter"]
def escalate(alerts, now, ticket_age_hours=24):
    ...

# %%
now = datetime(2025, 6, 2, 12, 0)
alerts = [
    {"key": "a", "severity": "ticket", "opened_at": datetime(2025, 6, 1, 9, 0)},    # 27 h old -> escalate
    {"key": "b", "severity": "ticket", "opened_at": datetime(2025, 6, 2, 9, 0)},    # 3 h old  -> stay
    {"key": "c", "severity": "info",   "opened_at": datetime(2025, 5, 20, 9, 0)},   # old but info -> stay
    {"key": "d", "severity": "ticket", "opened_at": datetime(2025, 6, 1, 12, 0)},   # exactly 24 h -> stay (strictly more than)
]
snapshot = copy.deepcopy(alerts)
out = escalate(alerts, now)
assert [a["severity"] for a in out] == ["page", "ticket", "info", "ticket"], [a["severity"] for a in out]
assert alerts == snapshot, "do not mutate the input"
print("Exercise 4 passed")

# %% tags=["solution"]
def escalate(alerts, now, ticket_age_hours=24):
    out = []
    for a in alerts:
        age_h = (now - a["opened_at"]).total_seconds() / 3600
        out.append({**a, "severity": "page"} if a["severity"] == "ticket" and age_h > ticket_age_hours else dict(a))
    return out

# %% [markdown]
# ### Exercise 5 (open-ended): Design the monitoring for one real pipeline
# Take a pipeline you run or know. List its five most likely failure modes, and for each write: the monitor that would catch it, its threshold and how you chose it, the alert severity, and the first step in the runbook.
# Then pick the one you would be most embarrassed to miss and write the test or monitor for it in code.

# %% [markdown]
# ## Recap
# - Split **pure transformations** from I/O, then unit-test the pure part with edge cases, and test idempotency by running the load twice; run `pytest` and use its exit code in CI.
# - A **data contract** is a versioned, machine-checked agreement about schema, nullability, ranges and values; diff contracts to spot breaking changes before they ship.
# - Monitor the **data** (volume, freshness, null rate, distribution) against a **rolling robust baseline**, and track false positives.
# - Turn results into **alerts** with severity, de-duplication, cooldown and an owner; fatigue is a failure mode, not a detail.
# - Measure an **SLO** from history, watch the p95 and the **error budget**, and alert on leading indicators.
#
# **Go deeper:** the pytest documentation (docs.pytest.org), especially fixtures and parametrization; the Google SRE book chapter "Service Level Objectives" (sre.google/sre-book); the Great Expectations documentation for a production validation framework (not installed here).
#
# **Next:** this module is complete. Continue with [MLOps: lifecycle and maturity](../20_MLOps/01_mlops_lifecycle_and_maturity.ipynb).
