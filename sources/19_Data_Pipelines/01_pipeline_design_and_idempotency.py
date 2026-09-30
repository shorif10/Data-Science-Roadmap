# %% [markdown]
# # Pipeline Design and Idempotency
#
# **Why this matters.** Every pipeline you write will be run more than once: it will fail halfway and be restarted, a colleague will trigger a backfill, the scheduler will double-fire, data will arrive late.
# If a re-run corrupts your tables (duplicates, half-written files, a missing day) you will spend your week on repairs. Pipelines designed for re-runs turn those events into non-events.
# This notebook is the design vocabulary the rest of the module builds on.
#
# **You will learn**
# - think of a pipeline as a DAG of tasks with clear inputs and outputs
# - make a step **idempotent** (safe to re-run) and **deterministic** (same inputs, same outputs)
# - write files and tables **atomically** so a crash never leaves half a result
# - handle **late-arriving data** with a lookback window, and detect **schema drift**
#
# **Prerequisites:** [APIs and ingestion](../18_Data_Engineering/05_apis_and_ingestion.ipynb) (upserts, idempotent loads) and [Data quality](../18_Data_Engineering/04_data_quality_and_validation.ipynb).
#
# > Data used: the synthetic shop tables (`orders.csv`, `order_items.csv`, `products.csv`). See `docs/DATASETS.md`.

# %%
import atexit
import os
import shutil
import sqlite3
import tempfile
from graphlib import TopologicalSorter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="pipe_design_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
lines = items.merge(orders, on="order_id").merge(products, on="product_id")
lines["revenue"] = lines["quantity"] * lines["unit_price"]
lines = lines[lines["status"] == "delivered"][["order_id", "order_date", "channel", "revenue"]].reset_index(drop=True)
print(lines.shape, lines["order_date"].min().date(), "->", lines["order_date"].max().date())

# %% [markdown]
# ## 1. A pipeline is a DAG
# A pipeline is a set of **tasks** (extract, clean, aggregate, publish), each reading some inputs and producing outputs, plus **dependencies** ("clean needs extract to finish first").
# Because dependencies never loop, the structure is a **directed acyclic graph** (DAG). Writing it down as data, not as a tangle of `if` statements, lets a machine work out the order, run independent tasks in parallel,
# and skip everything downstream of a failure.

# %%
dag = {                                   # task -> the tasks it depends on
    "extract_orders": set(),
    "extract_items": set(),
    "clean": {"extract_orders", "extract_items"},
    "daily_revenue": {"clean"},
    "quality_checks": {"daily_revenue"},
    "publish": {"quality_checks"},
}
order = list(TopologicalSorter(dag).static_order())
print(" -> ".join(order))

# %% [markdown]
# *What just happened?* `graphlib.TopologicalSorter` (in the standard library) returned an order in which every task comes after all of its dependencies. Both extracts have no dependencies, so they could run in parallel.
# Notebook 03 builds a runner around exactly this. First, the properties each task must have to be safely re-run.

# %% [markdown]
# ## 2. Idempotency: the property that saves you
# A task is **idempotent** if running it twice with the same inputs leaves the world in the same state as running it once. Compare two ways to write a daily aggregate into a table.

# %%
def daily_revenue(run_date):
    """Pure function: revenue per channel for ONE day, computed from source data."""
    day = lines[lines["order_date"] == pd.Timestamp(run_date)]
    return day.groupby("channel", as_index=False)["revenue"].sum().assign(run_date=str(run_date))[["run_date", "channel", "revenue"]]


def load_append(con, run_date):
    """BROKEN: appends every time it runs."""
    daily_revenue(run_date).to_sql("daily_revenue", con, if_exists="append", index=False)


con = sqlite3.connect(":memory:")
load_append(con, "2024-12-20")
load_append(con, "2024-12-20")               # the scheduler fired twice, or an engineer clicked "retry"
print(pd.read_sql("SELECT run_date, channel, round(revenue, 2) AS revenue FROM daily_revenue", con))

# %% [markdown]
# Every channel appears twice and any total over this table is now **double**. Nothing errored. The standard fix is **delete-then-insert for the partition, inside one transaction** (or `MERGE`/upsert on a key, or
# overwriting a date partition file). The unit of replacement is the **partition** the run is responsible for:

# %%
con.execute("DROP TABLE daily_revenue")
con.execute("CREATE TABLE daily_revenue (run_date TEXT, channel TEXT, revenue REAL, PRIMARY KEY (run_date, channel))")


def load_idempotent(con, run_date):
    df = daily_revenue(run_date)
    with con:                                                    # one transaction: all or nothing
        con.execute("DELETE FROM daily_revenue WHERE run_date = ?", (str(run_date),))
        con.executemany("INSERT INTO daily_revenue VALUES (?, ?, ?)", df.itertuples(index=False, name=None))


for _ in range(3):
    load_idempotent(con, "2024-12-20")
print(pd.read_sql("SELECT run_date, channel, round(revenue, 2) AS revenue FROM daily_revenue", con))

# %% [markdown]
# Run it 3 times, get the same 3 rows. The primary key adds a safety net: a bug that tried to insert duplicates would raise an error instead of silently doubling numbers.
#
# **Three habits that make idempotency work**
# 1. **Parameterise by the logical date**, never `datetime.now()`. `daily_revenue("2024-12-20")` always means that day, whether it runs at midnight or in a backfill next month.
# 2. **Deterministic transforms**: same input rows in, same output rows out (no `random` without a seed, no `LIMIT` without `ORDER BY`, no reading "whatever is in the folder right now").
# 3. **Replace, do not append** at the granularity of a partition or key.

# %% [markdown]
# ## 3. Atomic writes: never leave half a result
# What if the process dies *while writing*? A naive write leaves a truncated file that the next reader happily loads. The fix: write to a temporary file in the same folder, then **rename** it into place.
# On the same filesystem a rename is atomic: readers see the old complete file or the new complete file, never a mix.

# %%
def naive_write(path, text, crash_after=None):
    with open(path, "w") as f:
        for i, line in enumerate(text.splitlines(keepends=True)):
            if crash_after is not None and i == crash_after:
                raise RuntimeError("process killed mid-write")
            f.write(line)


def atomic_write(path, text, crash_after=None):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        naive_write(tmp, text, crash_after)
        os.replace(tmp, path)                       # atomic swap
    finally:
        tmp.unlink(missing_ok=True)                 # clean up if we crashed before the swap


report = TMP / "report.csv"
good = "day,revenue\n" + "".join(f"2024-12-{d:02d},{d * 100}\n" for d in range(1, 8))
naive_write(report, "old,complete\n1,1\n")
try:
    naive_write(report, good, crash_after=3)
except RuntimeError:
    pass
print("naive write, after crash:", report.read_text().splitlines())

atomic_write(report, "old,complete\n1,1\n")
try:
    atomic_write(report, good, crash_after=3)
except RuntimeError:
    pass
print("atomic write, after crash:", report.read_text().splitlines(), "| leftover temp files:", [p.name for p in TMP.glob("*.tmp")])

# %% [markdown]
# The naive version destroyed the old file and left a broken new one; the atomic version still shows the old complete file. Object stores (S3, GCS) make single-object writes atomic already; for a *directory of many files*
# the same idea is "write to `_staging/`, then move", or use a table format with transactions (Delta, Iceberg). Databases give you this via transactions, as in `load_idempotent`.

# %% [markdown]
# ## 4. Late-arriving data
# Real data is messy in time: an order placed on Monday may reach the warehouse on Thursday (mobile app offline, partner batch). Two clocks matter:
#
# * **event time**: when it happened (`order_date`);
# * **processing / arrival time**: when your pipeline saw it.
#
# If each daily run only computes "today's" partition and never looks back, late rows are lost from history. We simulate it: about 10% of orders arrive 1 to 5 days late.

# %%
rng = np.random.default_rng(42)
delay_days = np.where(rng.random(len(lines)) < 0.10, rng.integers(1, 6, len(lines)), 0)
late = lines.assign(arrived_at=lines["order_date"] + pd.to_timedelta(delay_days, unit="D"))
print(f"{(delay_days > 0).mean():.1%} of lines arrive late; max delay {delay_days.max()} days")

by_pair = late.groupby(["order_date", "arrived_at"])["revenue"].sum()
parts = {d: s.droplevel(0) for d, s in by_pair.groupby(level=0)}            # order_date -> revenue by arrival date
truth = pd.Series({d: s.sum() for d, s in parts.items()})
all_run_dates = pd.date_range(late["order_date"].min(), late["arrived_at"].max())


def run_daily(lookback):
    """Simulate a daily pipeline. On run date D it (re)computes partitions D-lookback..D using rows that have arrived by D."""
    table = {}
    for D in all_run_dates:
        for k in range(lookback + 1):
            d = D - pd.Timedelta(days=k)
            if d in parts:
                s = parts[d]
                table[d] = s[s.index <= D].sum()             # overwrite the partition (idempotent), never append
    return pd.Series(table).reindex(truth.index).fillna(0)


for lb in (0, 2, 5):
    result = run_daily(lb)
    wrong_days = int((~np.isclose(result, truth)).sum())
    print(f"lookback {lb} days: {wrong_days:3d} of {len(truth)} daily totals wrong, missing revenue = {(truth - result).sum():10.2f}")

# %% [markdown]
# *What just happened?* With no lookback, every late row is missing from its day forever. A lookback of 2 days repairs only part of it (compare the missing revenue above), and a lookback at least as long as the **maximum lateness** (5 days here) makes history exact.
# This is why real pipelines **recompute the last N partitions on every run**; it only works cleanly *because* the load is idempotent (recomputing a partition replaces it). The trade-off: a longer lookback costs more compute,
# so choose it from measured lateness (for instance, the 99th percentile of `arrived_at - event_time`), not by guessing. Rows arriving later than the lookback need a periodic full or monthly backfill.

# %% [markdown]
# ## 5. Schema drift
# The upstream team renames a column, adds one, or changes a type, and you find out when a dashboard breaks. **Schema drift** is best handled by comparing each incoming batch to an expected schema (a small *data contract*)
# and applying a policy: additive changes are usually warnings; missing columns and type changes fail the run.

# %%
EXPECTED = {"order_id": "int64", "order_date": "str", "channel": "str", "amount": "float64"}


def classify_drift(expected, df):
    actual = {c: str(t) for c, t in df.dtypes.items()}
    return {"added": sorted(set(actual) - set(expected)),
            "missing": sorted(set(expected) - set(actual)),
            "type_changed": sorted(c for c in expected if c in actual and actual[c] != expected[c])}


batch_ok = pd.DataFrame({"order_id": [1, 2], "order_date": ["2025-01-01", "2025-01-02"], "channel": ["web", "store"], "amount": [9.5, 20.0]})
batch_added = batch_ok.assign(coupon=["A", None])
batch_renamed = batch_ok.rename(columns={"amount": "total_amount"})
batch_retyped = batch_ok.assign(amount=["9.50", "20.00"])
for name, b in [("ok", batch_ok), ("new column", batch_added), ("renamed column", batch_renamed), ("amount became text", batch_retyped)]:
    print(f"{name:20s}", classify_drift(EXPECTED, b))

# %% [markdown]
# A **rename** shows up as *one missing plus one added* column: a human must decide it is a rename, so the safe automated reaction is to fail on the missing column. Policy in code:
# `if drift["missing"] or drift["type_changed"]: raise; elif drift["added"]: log a warning and drop or pass through`. Combine this with the full quality checks from module 18.04.

# %% [markdown]
# ## Common mistakes
#
# 1. **`if_exists="append"` in a job that can be re-run.** Duplicates with no error. Replace the partition or upsert on a key.
# 2. **`datetime.now()` inside transformation logic.** Re-running yesterday's job then computes today's data. Pass the logical date in.
# 3. **Writing directly to the final path.** A crash leaves a corrupt file that downstream jobs consume. Write to a temp path and rename.
# 4. **Deleting the partition in one step and inserting in another** without a transaction: a crash between them leaves the day empty.
# 5. **Only processing "today".** Late data silently vanishes; add a lookback based on measured lateness.
# 6. **Ignoring upstream schema changes** until something downstream throws a cryptic error. Check the contract at ingestion.
# 7. **Non-deterministic steps** (unordered `LIMIT`, unseeded sampling, dict/set ordering assumptions) that make two runs differ, so you cannot tell whether a change is real.

# %% [markdown]
# ## Interview corner
#
# **1. What does it mean for a pipeline to be idempotent and how do you achieve it?**
# <details><summary>Answer</summary>
# Re-running with the same inputs gives the same final state. Achieve it by parameterising runs by a logical date, making transforms deterministic, and replacing (overwrite partition, upsert/MERGE on a key)
# rather than appending, inside a transaction or via atomic rename.
# </details>
#
# **2. How would you handle data that arrives 3 days late?**
# <details><summary>Answer</summary>
# Distinguish event time from arrival time; recompute a lookback window of partitions (a bit longer than the observed maximum lateness) on every run, which is safe because loads are idempotent; monitor lateness
# and backfill anything beyond the window; tell consumers that recent days are provisional.
# </details>
#
# **3. A job crashed halfway through writing output. How do you make sure nobody reads a partial result?**
# <details><summary>Answer</summary>
# Write to a staging location and atomically publish (rename or transactional commit); readers only see complete versions. Add a success marker or manifest so downstream tasks start only after a completed write.
# </details>
#
# **4. An upstream team renames a column. How should your pipeline react?**
# <details><summary>Answer</summary>
# Detect it at ingestion by comparing to an expected schema, fail (or quarantine the batch) on missing or retyped columns, warn on additive ones, and alert the owner. Coordinating changes through versioned data contracts is better
# than discovering them from a broken dashboard.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Fix the duplicate-prone loader
# The table `channel_totals(run_date TEXT, channel TEXT, revenue REAL)` has no key. Write `load_channel_totals(con, run_date)` that loads `daily_revenue(run_date)` **idempotently** (delete the partition and insert, in a transaction), so
# running it several times never duplicates rows.

# %% tags=["starter"]
def load_channel_totals(con, run_date):
    ...

# %%
c1 = sqlite3.connect(":memory:")
c1.execute("CREATE TABLE channel_totals (run_date TEXT, channel TEXT, revenue REAL)")
for d in ["2024-12-20", "2024-12-21", "2024-12-20", "2024-12-20"]:
    load_channel_totals(c1, d)
n = c1.execute("SELECT count(*) FROM channel_totals").fetchone()[0]
exp_n = len(daily_revenue("2024-12-20")) + len(daily_revenue("2024-12-21"))
assert n == exp_n, f"expected {exp_n} rows after loading two distinct days repeatedly, found {n}"
tot = c1.execute("SELECT sum(revenue) FROM channel_totals WHERE run_date = '2024-12-20'").fetchone()[0]
assert abs(tot - daily_revenue("2024-12-20")["revenue"].sum()) < 1e-6, "revenue for the day is wrong"
print("Exercise 1 passed")

# %% tags=["solution"]
def load_channel_totals(con, run_date):
    df = daily_revenue(run_date)
    with con:
        con.execute("DELETE FROM channel_totals WHERE run_date = ?", (str(run_date),))
        con.executemany("INSERT INTO channel_totals VALUES (?, ?, ?)", df.itertuples(index=False, name=None))

# %% [markdown]
# ### Exercise 2: An atomic JSON writer
# Write `atomic_write_json(path, obj, crash=False)` that writes `obj` as JSON atomically (temp file in the same folder, then `os.replace`). If `crash=True` it must raise `RuntimeError` **after writing part of the temp file
# and before the swap**, leaving the original file untouched and no temp files behind.

# %% tags=["starter"]
import json


def atomic_write_json(path, obj, crash=False):
    ...

# %%
target = TMP / "state.json"
atomic_write_json(target, {"version": 1})
try:
    atomic_write_json(target, {"version": 2, "big": list(range(1000))}, crash=True)
    raise AssertionError("crash=True must raise RuntimeError")
except RuntimeError:
    pass
assert json.loads(target.read_text()) == {"version": 1}, "the original file must survive a crash"
assert not list(TMP.glob("state.json*.tmp")) and not list(TMP.glob("*.tmp")), "no temp files may be left behind"
atomic_write_json(target, {"version": 3})
assert json.loads(target.read_text()) == {"version": 3}, "a successful write must replace the file"
print("Exercise 2 passed")

# %% tags=["solution"]
import json


def atomic_write_json(path, obj, crash=False):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        text = json.dumps(obj)
        with open(tmp, "w") as f:
            f.write(text[: len(text) // 2])
            if crash:
                raise RuntimeError("simulated crash mid-write")
            f.write(text[len(text) // 2:])
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

# %% [markdown]
# ### Exercise 3: How long a lookback?
# Using `late` (columns `order_date`, `arrived_at`), compute the lateness in whole days of each line (`arrived_at - order_date`). Store `p99_lateness` (the 99th percentile, as an int, with `np.ceil`) and
# `lookback_needed` = the smallest lookback (in days) for which `run_daily(lookback)` has **zero wrong days**. Find it by trying 0..7 with `run_daily` and `np.isclose` against `truth`.

# %% tags=["starter"]
p99_lateness = ...
lookback_needed = ...

# %%
lat = (late["arrived_at"] - late["order_date"]).dt.days
assert p99_lateness == int(np.ceil(np.percentile(lat, 99))), "99th percentile of lateness in days, rounded up"
assert lookback_needed == int(lat.max()), f"the lookback that repairs everything equals the maximum lateness ({int(lat.max())} days)"
print(f"Exercise 3 passed: p99 lateness {p99_lateness} days, full repair needs lookback {lookback_needed}")

# %% tags=["solution"]
lat = (late["arrived_at"] - late["order_date"]).dt.days
p99_lateness = int(np.ceil(np.percentile(lat, 99)))
lookback_needed = next(lb for lb in range(8) if np.isclose(run_daily(lb), truth).all())

# %% [markdown]
# ### Exercise 4: Drift policy
# Write `drift_policy(expected, df)` that returns `"fail"` if any expected column is missing or has a changed type, `"warn"` if columns were only added, and `"ok"` otherwise. Reuse `classify_drift`.

# %% tags=["starter"]
def drift_policy(expected, df):
    ...

# %%
assert drift_policy(EXPECTED, batch_ok) == "ok"
assert drift_policy(EXPECTED, batch_added) == "warn"
assert drift_policy(EXPECTED, batch_renamed) == "fail", "a rename looks like a missing column, so it must fail"
assert drift_policy(EXPECTED, batch_retyped) == "fail"
assert drift_policy(EXPECTED, batch_added.assign(amount=["1", "2"])) == "fail", "a blocking problem beats a warning"
print("Exercise 4 passed")

# %% tags=["solution"]
def drift_policy(expected, df):
    d = classify_drift(expected, df)
    if d["missing"] or d["type_changed"]:
        return "fail"
    return "warn" if d["added"] else "ok"

# %% [markdown]
# ### Exercise 5 (open-ended): Audit a pipeline you know
# Take a script or notebook you have written (or the `load_append` function above) and list every place it is **not** safe to re-run: appends, `now()` calls, direct writes, hidden state. For each, write the one-line fix.
# Put your list in a markdown cell.

# %% [markdown]
# ## Recap
# - A pipeline is a **DAG** of tasks; declare dependencies as data so order, parallelism and failure handling can be automated.
# - **Idempotent** = safe to re-run: parameterise by logical date, keep transforms deterministic, replace partitions or upsert (in a transaction), never blind-append.
# - **Atomic writes** (temp file + rename, or a transaction) mean readers never see half a result.
# - **Late data**: recompute a lookback window sized from measured lateness; it is only safe because loads are idempotent.
# - **Schema drift**: compare each batch to a contract; fail on missing/retyped columns, warn on added ones.
#
# **Go deeper:** the Python `graphlib` docs, the SQLite documentation on transactions and `UPSERT`, and the "Idempotency" and "Backfill" sections of the Apache Airflow documentation (airflow.apache.org/docs).
#
# **Next:** [Build an ETL pipeline in Python](02_build_an_etl_pipeline_in_python.ipynb)
