# %% [markdown]
# # Build an ETL Pipeline in Python
#
# **Why this matters.** "Write a pipeline that loads the shop CSVs into a database" is a classic take-home and a real first-month task. The difference between a script and a pipeline
# is everything around the transformation: configuration you can change without editing code, logs you can read at 3 a.m., retries for flaky inputs, checks that stop bad data from being published, outputs that are safe to
# re-run, and a record of what each run did. This notebook builds all of that in about 150 lines of ordinary Python.
#
# **You will learn**
# - structure a pipeline as small **extract / transform / load** functions with a thin runner
# - drive it from **config** (YAML), log with a run id, and retry flaky steps with backoff
# - load to **SQLite and partitioned Parquet** atomically and idempotently
# - write a **run manifest** (what ran, how long, row counts, input and output fingerprints) and use it to prove a re-run changed nothing
# - watch a failure stop the pipeline without touching the published data
#
# **Prerequisites:** [Pipeline design and idempotency](01_pipeline_design_and_idempotency.ipynb).
#
# > Data used: `customers.csv`, `products.csv`, `orders.csv`, `order_items.csv` (synthetic). Everything is written to a temporary folder that is deleted at the end.

# %%
import atexit
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="etl_"))
_ = atexit.register(shutil.rmtree, TMP, ignore_errors=True)

# %% [markdown]
# ## 1. Configuration
# Anything that changes between environments (paths, thresholds, retry counts) belongs in config, not in code. YAML is the usual format; we parse it into a typed object so typos fail early.
# (Secrets never go in this file; read them from environment variables.)

# %%
CONFIG_YAML = f"""
pipeline: shop_daily
logical_date: 2025-05-30
source_dir: {DATA.resolve().as_posix()}
output_dir: {(TMP / "out").as_posix()}
retry: {{max_attempts: 3, base_delay_s: 0.01}}
min_rows: {{orders: 1000, order_items: 1000, customers: 100, products: 10}}
"""


@dataclass(frozen=True)
class Config:
    pipeline: str
    logical_date: str
    source_dir: Path
    output_dir: Path
    max_attempts: int
    base_delay_s: float
    min_rows: dict

    @classmethod
    def from_yaml(cls, text):
        raw = yaml.safe_load(text)
        return cls(raw["pipeline"], str(raw["logical_date"]), Path(raw["source_dir"]), Path(raw["output_dir"]),
                   raw["retry"]["max_attempts"], raw["retry"]["base_delay_s"], raw["min_rows"])


cfg = Config.from_yaml(CONFIG_YAML)
print(cfg.pipeline, cfg.logical_date, cfg.max_attempts, cfg.min_rows)

# %% [markdown]
# ## 2. Logging with a run id
# `print` is for notebooks. Pipelines use the `logging` module: levels (INFO for progress, WARNING for oddities, ERROR for failures), timestamps in the file, and a **run id** on every line so you can pull out one run from a busy log.

# %%
class RunIdFilter(logging.Filter):
    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        record.run_id = self.run_id
        return True


def make_logger(run_id, log_file):
    log = logging.getLogger(f"etl.{run_id}")
    log.setLevel(logging.INFO)
    log.propagate = False
    log.handlers.clear()
    for h, fmt in [(logging.StreamHandler(sys.stdout), "%(levelname)-7s [%(run_id)s] %(message)s"),
                   (logging.FileHandler(log_file), "%(asctime)s %(levelname)s [%(run_id)s] %(message)s")]:
        h.setFormatter(logging.Formatter(fmt))
        h.addFilter(RunIdFilter(run_id))
        log.addHandler(h)
    return log


demo = make_logger("demo0001", TMP / "demo.log")
demo.info("pipeline starting")
demo.warning("3 rows had a missing customer id")

# %% [markdown]
# ## 3. Retries for flaky steps
# Reading from a network share or an API fails transiently. A decorator keeps retry logic out of the business code. It retries only the exceptions you name (never `Exception`, which would also retry bugs) and gives up after a fixed number of attempts.

# %%
def retry(max_attempts, base_delay_s, retry_on=(ConnectionError, TimeoutError), sleep=time.sleep):
    def wrap(fn):
        def inner(*args, log=None, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    if attempt == max_attempts:
                        raise
                    if log:
                        log.warning("%s failed (%s), attempt %d/%d, retrying", fn.__name__, exc, attempt, max_attempts)
                    sleep(base_delay_s * 2 ** (attempt - 1))
        return inner
    return wrap


calls = {"n": 0}

@retry(max_attempts=3, base_delay_s=0.0)
def flaky_read():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("share unavailable")
    return "data"

print(flaky_read(log=demo), "after", calls["n"], "calls")

# %% [markdown]
# ## 4. Extract
# Extract only *reads and validates the shape* of the sources. It does not clean anything (that is transform's job) so that a bad source is reported as a source problem.

# %%
REQUIRED = {"customers": ["customer_id", "country", "signup_date", "is_premium"],
            "products": ["product_id", "product_name", "category", "unit_price", "unit_cost"],
            "orders": ["order_id", "customer_id", "order_date", "status", "channel"],
            "order_items": ["item_id", "order_id", "product_id", "quantity"]}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def read_source(source_dir, name):
    return pd.read_csv(Path(source_dir) / f"{name}.csv")


def validate_source(tables, cfg):
    for name, df in tables.items():
        missing = [c for c in REQUIRED[name] if c not in df.columns]
        if missing:
            raise ValueError(f"source '{name}' is missing columns {missing}")
        if len(df) < cfg.min_rows[name]:
            raise ValueError(f"source '{name}' has {len(df)} rows, expected at least {cfg.min_rows[name]}")


def extract(cfg, log, reader=read_source):
    reader = retry(cfg.max_attempts, cfg.base_delay_s)(reader)
    tables = {name: reader(cfg.source_dir, name, log=log) for name in REQUIRED}
    validate_source(tables, cfg)
    log.info("extracted %s", {k: len(v) for k, v in tables.items()})
    return tables

# %% [markdown]
# ## 5. Transform
# Pure functions from DataFrames to DataFrames: no I/O, no globals, so they are trivial to unit test (notebook 05). We produce three outputs from the four sources:
# a line-level fact table, daily sales by channel, and a customer summary.

# %%
def clean(tables):
    orders = tables["orders"].assign(order_date=lambda d: pd.to_datetime(d["order_date"]),
                                     status=lambda d: d["status"].str.strip().str.lower())
    orders = orders.drop_duplicates("order_id", keep="last")
    items = tables["order_items"].query("quantity > 0").drop_duplicates("item_id", keep="last")
    customers = tables["customers"].assign(signup_date=lambda d: pd.to_datetime(d["signup_date"]))
    return {"orders": orders, "order_items": items, "customers": customers, "products": tables["products"]}


def build_fact_lines(t):
    f = (t["order_items"].merge(t["orders"], on="order_id", how="inner", validate="many_to_one")
         .merge(t["products"], on="product_id", how="inner", validate="many_to_one"))
    f["revenue"] = (f["quantity"] * f["unit_price"]).round(2)
    f["cost"] = (f["quantity"] * f["unit_cost"]).round(2)
    f["order_month"] = f["order_date"].dt.strftime("%Y-%m")
    return f[["item_id", "order_id", "order_date", "order_month", "customer_id", "product_id", "category", "channel", "status",
              "quantity", "revenue", "cost"]].sort_values("item_id").reset_index(drop=True)


def build_daily_sales(fact):
    d = fact[fact["status"] == "delivered"]
    out = d.groupby(["order_date", "channel"], as_index=False).agg(orders=("order_id", "nunique"), units=("quantity", "sum"), revenue=("revenue", "sum"))
    out["revenue"] = out["revenue"].round(2)
    return out.assign(order_date=out["order_date"].dt.strftime("%Y-%m-%d"))


def build_customer_summary(fact, customers):
    d = fact[fact["status"] == "delivered"]
    s = d.groupby("customer_id", as_index=False).agg(n_orders=("order_id", "nunique"), revenue=("revenue", "sum"),
                                                    first_order=("order_date", "min"), last_order=("order_date", "max"))
    s["revenue"] = s["revenue"].round(2)
    s = customers[["customer_id", "country", "is_premium"]].merge(s, on="customer_id", how="left")
    s[["n_orders"]] = s[["n_orders"]].fillna(0).astype(int)
    s["revenue"] = s["revenue"].fillna(0.0)
    for c in ("first_order", "last_order"):
        s[c] = s[c].dt.strftime("%Y-%m-%d")
    return s


def transform(tables, log):
    t = clean(tables)
    fact = build_fact_lines(t)
    outputs = {"fact_order_lines": fact, "daily_sales": build_daily_sales(fact), "customer_summary": build_customer_summary(fact, t["customers"])}
    # quality gates: stop before anything is published
    assert fact["item_id"].is_unique, "fact_order_lines.item_id must be unique"
    assert len(fact) == len(t["order_items"]), f"grain check failed: {len(fact)} fact rows vs {len(t['order_items'])} items (fan-out or dropped rows)"
    log.info("transformed: %s", {k: len(v) for k, v in outputs.items()})
    return outputs

# %% [markdown]
# Two gates live inside `transform`: a uniqueness check and a **grain/reconciliation check** (the fact table must have as many rows as the items table; a wrong join would change that). In production, put these in a reusable
# suite as in module 18. A failed gate raises, so the load step never runs.

# %% [markdown]
# ## 6. Load: atomic and idempotent
# * **SQLite** (stands in for a warehouse): write each table to a `_new` staging name, then drop-and-rename inside one transaction. Readers see the old table or the new table, never a gap.
# * **Parquet**: write the partitioned dataset into a staging folder, then swap folders. (Each rename of a directory is atomic on one filesystem, but our two-step swap leaves a very short moment with no final folder; a truly gap-free swap needs a pointer or manifest file, and object stores need a manifest or a table format instead.)
#
# Both **replace** the full output, so running twice equals running once.

# %%
def load_sqlite(outputs, db_path, log):
    con = sqlite3.connect(db_path)
    try:
        for name, df in outputs.items():
            df.to_sql(f"{name}_new", con, if_exists="replace", index=False)
        with con:                                               # one transaction for all swaps
            for name in outputs:
                con.execute(f"DROP TABLE IF EXISTS {name}")
                con.execute(f"ALTER TABLE {name}_new RENAME TO {name}")
    finally:
        con.close()
    log.info("loaded %d tables into %s", len(outputs), Path(db_path).name)


def publish_dir(staging, final):
    final, old = Path(final), Path(str(final) + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        os.replace(final, old)
    os.replace(staging, final)
    shutil.rmtree(old, ignore_errors=True)


def load_parquet(fact, out_dir, log):
    staging, final = out_dir / "fact_order_lines.staging", out_dir / "fact_order_lines"
    shutil.rmtree(staging, ignore_errors=True)
    fact.to_parquet(staging, partition_cols=["order_month"], index=False)
    publish_dir(staging, final)
    log.info("published %d parquet partitions", len(list(final.iterdir())))

# %% [markdown]
# ## 7. The run manifest
# A **manifest** is a small JSON document written at the end of every run: what ran, with which inputs, how long each step took, how many rows came out, and a fingerprint of each output. It answers
# "did last night's run do what I think it did?" and "did re-running change anything?" without opening a single table.

# %%
def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def df_fingerprint(df):
    """Order-independent content hash: same rows => same hash."""
    d = df.sort_values(list(df.columns)).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes()).hexdigest()[:16]


class PipelineFailed(RuntimeError):
    pass


STEPS = ["extract", "transform", "load"]


def run_pipeline(cfg, reader=read_source):
    run_id = uuid.uuid4().hex[:8]
    out = cfg.output_dir
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    log = make_logger(run_id, out / "pipeline.log")
    manifest = {"run_id": run_id, "pipeline": cfg.pipeline, "logical_date": cfg.logical_date, "status": "running",
                "inputs": {n: sha256_file(Path(cfg.source_dir) / f"{n}.csv") for n in REQUIRED if (Path(cfg.source_dir) / f"{n}.csv").exists()},
                "steps": {s: {"status": "pending"} for s in STEPS}, "outputs": {}}
    state = {}

    def step(name, fn):
        t0 = time.perf_counter()
        try:
            state[name] = fn()
            manifest["steps"][name] = {"status": "ok", "seconds": round(time.perf_counter() - t0, 3)}
        except Exception as exc:
            manifest["steps"][name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            raise

    try:
        log.info("start %s for %s", cfg.pipeline, cfg.logical_date)
        step("extract", lambda: extract(cfg, log, reader))
        step("transform", lambda: transform(state["extract"], log))
        outputs = state["transform"]
        def load():
            load_sqlite(outputs, out / "warehouse.db", log)
            load_parquet(outputs["fact_order_lines"], out, log)
        step("load", load)
        manifest["outputs"] = {n: {"rows": len(df), "fingerprint": df_fingerprint(df)} for n, df in outputs.items()}
        manifest["status"] = "success"
        log.info("success")
    except Exception as exc:
        manifest["status"] = "failed"
        for s in STEPS:
            if manifest["steps"][s]["status"] == "pending":
                manifest["steps"][s]["status"] = "skipped"
        log.error("FAILED: %s", exc)
        raise PipelineFailed(str(exc)) from exc
    finally:
        atomic_write_json(out / "manifests" / f"run_{run_id}.json", manifest)
        atomic_write_json(out / "manifests" / "latest.json", manifest)
    return manifest


m1 = run_pipeline(cfg)

# %% [markdown]
# *What just happened?* Extract, transform and load ran in order, each logging progress with the run id, and the manifest was written even though we did not ask for it explicitly: it lives in a `finally` so failed runs leave one too. Let's read it.

# %%
print(json.dumps({k: m1[k] for k in ("run_id", "status", "inputs", "steps")}, indent=1)[:900])
pd.DataFrame(m1["outputs"]).T

# %% [markdown]
# Check the published data with an independent path (read the Parquet folder and SQLite, and reconcile with the raw CSVs):

# %%
con = sqlite3.connect(cfg.output_dir / "warehouse.db")
print("tables:", [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")])
sql_rev = con.execute("SELECT round(sum(revenue), 2) FROM daily_sales").fetchone()[0]
pq_rev = pd.read_parquet(cfg.output_dir / "fact_order_lines", filters=[("status", "==", "delivered")])["revenue"].sum().round(2)
o, i, p = (pd.read_csv(DATA / f"{n}.csv") for n in ("orders", "order_items", "products"))
raw = i.merge(o, on="order_id").merge(p, on="product_id").query("status == 'delivered'")
raw_rev = round((raw["quantity"] * raw["unit_price"]).sum(), 2)
print("delivered revenue  sqlite:", sql_rev, "| parquet:", pq_rev, "| raw csv:", raw_rev)
con.close()

# %% [markdown]
# Three independent routes, one number. Now the idempotency test: **run it again**, and compare the manifests' output fingerprints (not the run ids or timings, which legitimately differ).

# %%
m2 = run_pipeline(cfg)
same = {n: m1["outputs"][n]["fingerprint"] == m2["outputs"][n]["fingerprint"] for n in m1["outputs"]}
print("run ids differ:", m1["run_id"] != m2["run_id"], "| identical output fingerprints:", same)
con = sqlite3.connect(cfg.output_dir / "warehouse.db")
print("rows in daily_sales after two runs:", con.execute("SELECT count(*) FROM daily_sales").fetchone()[0], "= rows in manifest:", m2["outputs"]["daily_sales"]["rows"])
con.close()

# %% [markdown]
# Same inputs, same fingerprints, same row counts, and leftover staging objects cleaned up:

# %%
print(sorted(p.name for p in cfg.output_dir.iterdir()))

# %% [markdown]
# ## 8. When things go wrong
# **Flaky input**: the first two reads fail with a connection error; the retry wrapper absorbs them.

# %%
attempts = {"n": 0}

def flaky_reader(source_dir, name):
    attempts["n"] += 1
    if attempts["n"] <= 2:
        raise ConnectionError("network share timed out")
    return read_source(source_dir, name)

m3 = run_pipeline(cfg, reader=flaky_reader)
print("status:", m3["status"], "| read attempts:", attempts["n"], "(4 tables + 2 failures)")

# %% [markdown]
# **Bad input**: a source file arrives without the `quantity` column. The pipeline must fail at extract, skip the rest, and leave the previously published data untouched. We point a copy of the config at a corrupted copy of the sources.

# %%
bad_src = TMP / "bad_source"
bad_src.mkdir()
for n in REQUIRED:
    shutil.copy(DATA / f"{n}.csv", bad_src / f"{n}.csv")
pd.read_csv(DATA / "order_items.csv").drop(columns="quantity").to_csv(bad_src / "order_items.csv", index=False)

bad_cfg = Config(cfg.pipeline, cfg.logical_date, bad_src, cfg.output_dir, cfg.max_attempts, cfg.base_delay_s, cfg.min_rows)
before = m2["outputs"]["fact_order_lines"]["fingerprint"]
try:
    run_pipeline(bad_cfg)
except PipelineFailed as e:
    print("pipeline failed as intended:", e)

last = json.loads((cfg.output_dir / "manifests" / "latest.json").read_text())
print({k: v["status"] for k, v in last["steps"].items()}, "| run status:", last["status"])
con = sqlite3.connect(cfg.output_dir / "warehouse.db")
still = df_fingerprint(pd.read_sql("SELECT * FROM daily_sales", con))
con.close()
print("published daily_sales still matches the last good run's fingerprint:", still == m2["outputs"]["daily_sales"]["fingerprint"])

# %% [markdown]
# The manifest of the failed run says *which step failed and why*, the later steps are marked `skipped`, and the warehouse still holds the last good data. That is the contract of a healthy pipeline:
# **fail early, fail loudly, never publish partial or bad results.** (Exercise 3 makes the "untouched" claim rigorous with fingerprints.)

# %% [markdown]
# ## Common mistakes
#
# 1. **Hard-coded paths and thresholds** scattered in code. Put them in config; one place to change, one place to review.
# 2. **`except Exception:` around everything** with a retry. It retries bugs and hides them. Retry named transient errors only.
# 3. **Transformations that do I/O** (reading files inside `clean`). They become untestable; keep I/O in extract and load.
# 4. **Loading before validating.** Put quality gates between transform and load, and make a failed gate stop the run.
# 5. **`to_sql(if_exists="append")`** in a re-runnable job. Replace or upsert instead.
# 6. **No run record.** Without logs and a manifest with a run id you cannot answer "what happened on Tuesday?".
# 7. **Testing only the happy path.** Corrupt an input on purpose, as we did, and confirm the failure is clean.

# %% [markdown]
# ## Interview corner
#
# **1. Walk me through how you would structure an ETL job.**
# <details><summary>Answer</summary>
# Separate extract (read and validate source shape, with retries), transform (pure functions, unit-testable, with quality gates) and load (atomic, idempotent writes). Drive it from config, log with a run id,
# write a run manifest, and make failures stop the run before anything is published.
# </details>
#
# **2. How do you know a re-run did not change your data?**
# <details><summary>Answer</summary>
# Fingerprint each output (an order-independent content hash) and row counts in the manifest; identical inputs must produce identical fingerprints. Idempotent loads (replace, not append) and deterministic transforms make that true.
# </details>
#
# **3. What should be retried and what should not?**
# <details><summary>Answer</summary>
# Transient infrastructure errors (timeouts, connection resets, 429/503) with backoff and a cap. Not validation errors, schema problems, or bugs: retrying those just repeats the failure and delays the alert.
# </details>
#
# **4. Why write to a staging location first?**
# <details><summary>Answer</summary>
# So consumers never see a half-written result: the swap to the final name is one atomic operation (rename or transaction). If the job crashes, the old complete output remains and the staging leftovers are cleaned up next run.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Add a product summary output
# Write `build_product_summary(fact)` returning one row per `product_id` with columns `product_id`, `category`, `units` (sum of quantity) and `revenue` (sum of revenue, rounded to 2), counting **delivered** lines only,
# sorted by `revenue` descending then `product_id`. Test it on `fact = build_fact_lines(clean(extract(cfg, demo)))`.

# %% tags=["starter"]
def build_product_summary(fact):
    ...

# %%
fact = build_fact_lines(clean(extract(cfg, demo)))
ps = build_product_summary(fact)
d = fact[fact["status"] == "delivered"]
exp = (d.groupby(["product_id", "category"], as_index=False).agg(units=("quantity", "sum"), revenue=("revenue", "sum"))
       .assign(revenue=lambda x: x["revenue"].round(2)).sort_values(["revenue", "product_id"], ascending=[False, True]).reset_index(drop=True))
assert list(ps.columns) == ["product_id", "category", "units", "revenue"], "columns: product_id, category, units, revenue"
assert len(ps) == len(exp), f"expected {len(exp)} products"
assert ps.reset_index(drop=True)[["product_id", "units"]].astype("int64").equals(exp[["product_id", "units"]].astype("int64")), "product_id/units differ or sort order is wrong"
assert abs(ps["revenue"].sum() - exp["revenue"].sum()) < 0.01
print("Exercise 1 passed")

# %% tags=["solution"]
def build_product_summary(fact):
    d = fact[fact["status"] == "delivered"]
    out = d.groupby(["product_id", "category"], as_index=False).agg(units=("quantity", "sum"), revenue=("revenue", "sum"))
    out["revenue"] = out["revenue"].round(2)
    return out.sort_values(["revenue", "product_id"], ascending=[False, True]).reset_index(drop=True)

# %% [markdown]
# ### Exercise 2: A stricter source validator
# `validate_source` checks columns and row counts. Write `validate_keys(tables)` that raises `ValueError` (message mentioning the table name) if `orders.order_id`, `customers.customer_id` or `products.product_id` contains duplicates, or if any
# `order_items.order_id` does not exist in `orders`. It must return `None` on good data.

# %% tags=["starter"]
def validate_keys(tables):
    ...

# %%
good = {n: read_source(DATA, n) for n in REQUIRED}
assert validate_keys(good) is None, "good data must pass"
dup = dict(good, orders=pd.concat([good["orders"], good["orders"].head(1)]))
try:
    validate_keys(dup)
    raise AssertionError("duplicate order_id must raise")
except ValueError as e:
    assert "orders" in str(e), "message should name the table"
orphan = dict(good, order_items=pd.concat([good["order_items"], pd.DataFrame({"item_id": [10**6], "order_id": [10**6], "product_id": [1], "quantity": [1]})]))
try:
    validate_keys(orphan)
    raise AssertionError("an item pointing to a missing order must raise")
except ValueError as e:
    assert "order_items" in str(e)
print("Exercise 2 passed")

# %% tags=["solution"]
def validate_keys(tables):
    for table, key in [("orders", "order_id"), ("customers", "customer_id"), ("products", "product_id")]:
        if tables[table][key].duplicated().any():
            raise ValueError(f"duplicate {key} in '{table}'")
    orphans = ~tables["order_items"]["order_id"].isin(tables["orders"]["order_id"])
    if orphans.any():
        raise ValueError(f"{int(orphans.sum())} rows in 'order_items' reference a missing order")

# %% [markdown]
# ### Exercise 3: Prove the failed run changed nothing
# Write `outputs_identical(manifest_a, manifest_b)` that returns `True` if both manifests have the same output table names, row counts **and** fingerprints. Then compute `failed_run_safe`: compare `m2` (the last good run)
# with a fresh **successful** run `m4 = run_pipeline(cfg)` executed *after* the failed run. It should be `True`, showing the failure did not corrupt anything.

# %% tags=["starter"]
def outputs_identical(manifest_a, manifest_b):
    ...

failed_run_safe = ...

# %%
assert outputs_identical(m1, m2) is True
changed = json.loads(json.dumps(m2))
changed["outputs"]["daily_sales"]["fingerprint"] = "0" * 16
assert outputs_identical(m2, changed) is False, "a different fingerprint must be detected"
changed2 = json.loads(json.dumps(m2))
changed2["outputs"]["daily_sales"]["rows"] += 1
assert outputs_identical(m2, changed2) is False, "a different row count must be detected"
assert failed_run_safe is True, "after the failed run, a good run should reproduce the same outputs as before"
print("Exercise 3 passed")

# %% tags=["solution"]
def outputs_identical(manifest_a, manifest_b):
    return manifest_a["outputs"] == manifest_b["outputs"]

m4 = run_pipeline(cfg)
failed_run_safe = outputs_identical(m2, m4)

# %% [markdown]
# ### Exercise 4 (open-ended): Add a data-quality gate
# Add a check to the pipeline (inside `transform`, or as a new step between transform and load) that stops the run when the data looks wrong, for example: **total delivered revenue of any single day in `daily_sales`
# is more than 5 times that channel's median day**, or any other business rule you would want as a gate. Decide whether it should fail the run or only warn, and write the rule, the threshold and your reasoning in a markdown cell.
# Then test it by corrupting a copy of the sources (as we did with `bad_src`) and confirm the manifest shows the failure.

# %% [markdown]
# ## Recap
# - Structure: **extract** (read, validate shape, retry), **transform** (pure, tested, with quality gates), **load** (atomic, idempotent), plus a thin **runner**.
# - **Config** in YAML, typed on load; **logging** with a run id; **retries** only for named transient errors, with backoff and a cap.
# - Load by staging and swapping (transaction for SQLite, directory rename for files); re-runs replace, never append.
# - A **manifest** per run (inputs hash, step status and timings, output row counts and fingerprints) is cheap observability and proves idempotency.
# - Failures must stop the run early, be recorded, and leave published data untouched.
#
# **Go deeper:** the Python `logging` HOWTO, the pandas `DataFrame.to_parquet` and `merge(validate=...)` docs, and the "Twelve-Factor App" page on config (12factor.net).
#
# **Next:** [Orchestration and DAGs](03_orchestration_and_dags.ipynb)
